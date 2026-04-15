from __future__ import annotations
import os
import random
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import omni.usd
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera
from pxr import Gf, Usd, UsdGeom

from .drawer_cfg import DrawerEnvCfg
from lehome.devices.action_process import preprocess_device_action
from lehome.utils.rendering import apply_default_render_settings_drawer


def _resolve_assets_root() -> str:
    """Resolve the assets root directory without hard-coded absolute paths."""
    env_root = os.environ.get("LEHOME_ASSETS_ROOT")
    if env_root:
        return str(Path(env_root).expanduser())

    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "Assets"
        if candidate.is_dir():
            return str(candidate)

    return str(Path("Assets").resolve())


DEFAULT_ASSETS_ROOT = _resolve_assets_root()
DEFAULT_SCENE_USD = os.environ.get(
    "LEHOME_DRAWER_SCENE_USD",
    str(Path(DEFAULT_ASSETS_ROOT) / "benchmark" / "scenes" / "benchmark_scene1.usd"),
)

class DrawerEnv(DirectRLEnv):
    cfg: DrawerEnvCfg

    def __init__(self, cfg: DrawerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos
        self.last_print_time = 0.0
        self._cache_drawer_local_bounds()
        self.drawer_initial_x_min = None
        self.drawer_initial_x_max = None
        self.full_distance = None
        self.set_drawer_position(-0.1, "Drawer001_joint", immediate=True)

        self._drawer_close_threshold = 0.02


    def _setup_scene(self):

        apply_default_render_settings_drawer()

        self.robot = Articulation(self.cfg.robot)
        self.behind_camera = TiledCamera(self.cfg.behind_camera)
        self.front_camera = TiledCamera(self.cfg.front_camera)

        cfg = sim_utils.UsdFileCfg(usd_path=DEFAULT_SCENE_USD)

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.drawer = Articulation(self.cfg.drawer)
        # Cache drawer joint indices once to avoid repeated list scans / prints.
        self._drawer_joint_idx_map: dict[str, int] = {}
        try:
            joint_names = list(getattr(self.drawer, "joint_names", []))
            for i, n in enumerate(joint_names):
                self._drawer_joint_idx_map[str(n)] = i
        except Exception:
            self._drawer_joint_idx_map = {}

        # add articulation to scene
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["drawer"] = self.drawer
        self.scene.sensors["behind_camera"] = self.behind_camera
        self.scene.sensors["front_camera"] = self.front_camera
        
        self.joint_num = 9
        self.scores = 0
        self.part = 0
        self.full_marks = 2
        self.contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

    def set_drawer_position(
        self,
        position: float | torch.Tensor,
        joint_name: str,
        env_ids: torch.Tensor | None = None,
        immediate: bool = False,
    ):
        """Set drawer joint position.

        Args:
            position: Target position (meters). Scalar or tensor of shape (num_envs,)
                or (len(env_ids),).
            joint_name: Drawer joint name.
            env_ids: Environment indices. None means all envs.
            immediate: If True, write directly to simulation (for resets). If False,
                set a joint target (for smooth control).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if isinstance(position, (int, float)):
            position = torch.full((len(env_ids),), float(position), device=self.device, dtype=torch.float32)
        elif isinstance(position, torch.Tensor):
            if position.dim() == 0:
                position = position.unsqueeze(0).expand(len(env_ids))
            position = position.to(device=self.device)

        joint_names = self.drawer.joint_names
        # Avoid spamming logs; cache index when possible.
        if not hasattr(self, "_drawer_joint_idx_map"):
            self._drawer_joint_idx_map = {}
        if joint_name in self._drawer_joint_idx_map:
            self._drawer_joint_idx = self._drawer_joint_idx_map[joint_name]
        else:
            # Fallback to a slow lookup, then cache it.
            self._drawer_joint_idx = joint_names.index(joint_name)
            self._drawer_joint_idx_map[joint_name] = self._drawer_joint_idx

        if immediate:
            joint_pos = self.drawer.data.joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            joint_pos[:, self._drawer_joint_idx] = position
            self.drawer.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            joint_pos = self.drawer.data.joint_pos[env_ids].clone()
            joint_pos[:, self._drawer_joint_idx] = position
            self.drawer.set_joint_position_target(joint_pos, env_ids=env_ids)

    def _get_drawer_joint_pos(self, joint_name: str = "joint_1") -> torch.Tensor:
        """Return current drawer joint position for a given joint name as (num_envs,) tensor."""
        joint_pos = self.drawer.data.joint_pos  # (num_envs, num_joints)
        if joint_pos.numel() == 0:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        idx = None
        if hasattr(self, "_drawer_joint_idx_map") and joint_name in self._drawer_joint_idx_map:
            idx = self._drawer_joint_idx_map[joint_name]
        else:
            try:
                idx = list(getattr(self.drawer, "joint_names", [])).index(joint_name)
                if not hasattr(self, "_drawer_joint_idx_map"):
                    self._drawer_joint_idx_map = {}
                self._drawer_joint_idx_map[joint_name] = idx
            except Exception:
                idx = 0
        return joint_pos[:, idx]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _cache_drawer_local_bounds(self):
        """Cache local AABBs for drawer bodies from USD (utility/debug)."""
        stage = omni.usd.get_context().get_stage()
        self.drawer_local_bounds = []
        
        for body_name in self.drawer.body_names:
            prim_path = f"/World/Object/drawer/{body_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                print(f"[Warn] Prim not found for body '{body_name}' at {prim_path}")
                local_min = (-0.01, -0.01, -0.01)
                local_max = (0.01, 0.01, 0.01)
            else:
                bbox_cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy],
                    useExtentsHint=True,
                    ignoreVisibility=True
                )
                bound = bbox_cache.ComputeLocalBound(prim)
                range_ = bound.GetRange()
                if range_.IsEmpty():
                    print(f"[Warn] Empty bounding box for {prim_path}")
                    local_min = (-0.01, -0.01, -0.01)
                    local_max = (0.01, 0.01, 0.01)
                else:
                    min_vec = range_.GetMin()
                    max_vec = range_.GetMax()
                    local_min = (min_vec[0], min_vec[1], min_vec[2])
                    local_max = (max_vec[0], max_vec[1], max_vec[2])

            self.drawer_local_bounds.append((local_min, local_max))

    def get_prim_dimensions(self, prim_path: str):
        if not hasattr(self, "drawer_local_bounds"):
            print("[Error] drawer_local_bounds not initialized!")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        body_states = self.drawer.data.body_state_w  # (num_envs, num_bodies, 13)
        if body_states.shape[0] == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        body_states = body_states.squeeze(0)  # (num_bodies, 13)

        global_min = Gf.Vec3f(float('inf'), float('inf'), float('inf'))
        global_max = Gf.Vec3f(float('-inf'), float('-inf'), float('-inf'))

        for i, (local_min, local_max) in enumerate(self.drawer_local_bounds):
            if i >= body_states.shape[0]:
                break
            pos = body_states[i, :3].cpu().numpy()
            quat = body_states[i, 3:7].cpu().numpy()  # w, x, y, z

            # Simplified: ignore rotation (reasonable for small drawer rotations).
            world_min = Gf.Vec3f(local_min[0] + pos[0], local_min[1] + pos[1], local_min[2] + pos[2])
            world_max = Gf.Vec3f(local_max[0] + pos[0], local_max[1] + pos[1], local_max[2] + pos[2])

            global_min = Gf.Vec3f(
                min(global_min[0], world_min[0]),
                min(global_min[1], world_min[1]),
                min(global_min[2], world_min[2])
            )
            global_max = Gf.Vec3f(
                max(global_max[0], world_max[0]),
                max(global_max[1], world_max[1]),
                max(global_max[2], world_max[2])
            )

        return global_max[0], global_max[1], global_max[2], global_min[0], global_min[1], global_min[2]
    
    def _get_observations(self) -> dict:
        # Handle case where self.actions might not be initialized yet (before first step)
        if hasattr(self, 'actions'):
            action = self.actions.squeeze(0)
        else:
            # Return zero action if actions not yet set
            action = torch.zeros(self.cfg.action_space, device=self.device)

        # Safely handle joint_pos: ensure at least self.joint_num joints, pad if needed
        current_joint_pos = self.robot.data.joint_pos
        if current_joint_pos.shape[1] >= self.joint_num:
            joint_pos = current_joint_pos[:, :self.joint_num].squeeze(0)
        else:
            # Pad with zeros if fewer than self.joint_num joints
            padded = torch.zeros(current_joint_pos.shape[0], self.joint_num, device=current_joint_pos.device)
            padded[:, :current_joint_pos.shape[1]] = current_joint_pos
            joint_pos = padded.squeeze(0)

        current_time = time.time()
        if current_time - self.last_print_time >= 5.0:
            x_max, y_max, z_max, x_min, y_min, z_min = self.get_prim_dimensions("/World/Object/drawer")
            joint_dis = float(self.drawer.data.joint_pos[0, 0].item())
            # print(f"Joint Pos: {joint_dis}")

            # print(f"[Drawer Dimensions] Max: ({x_max:.4f}, {y_max:.4f}, {z_max:.4f}), "
            #     f"Min: ({x_min:.4f}, {y_min:.4f}, {z_min:.4f})")
            
            self.last_print_time = current_time

        # === Safe camera data access ===
        # Default fallback images

        behind_camera_rgb = self.behind_camera.data.output["rgb"]
        front_camera_rgb = self.front_camera.data.output["rgb"]
        top_camera_rgb = self.top_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Dense shaping: encourage closing the drawer (joint_1 -> 0).
        # Reward in [-1, 1]: 1 when closed, approaching -1 as it stays open.
        drawer_q = self._get_drawer_joint_pos("joint_1")
        # Normalize by initial opening (0.1m). Clamp to keep bounded.
        open_norm = torch.clamp(drawer_q.abs() / 0.1, 0.0, 1.0)
        reward = 1.0 - 2.0 * open_norm  # open -> -1, closed -> +1
        # Small bonus for being within close threshold.
        success = drawer_q.abs() <= self._drawer_close_threshold
        reward = reward + success.to(reward.dtype) * 0.5
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Success if drawer is "closed" (slider joint near 0).
        # drawer_q = self._get_drawer_joint_pos("joint_1")
        # success_tensor = drawer_q.abs() <= self._drawer_close_threshold
        joint_dis = float(self.drawer.data.joint_pos[0, 0].item())

        success = joint_dis >= -0.05
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None,joint:float=-0.1) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        drawer_root_state = self.drawer.data.default_root_state[env_ids].clone()
        self.drawer.write_root_state_to_sim(drawer_root_state, env_ids=env_ids)

        drawer_joint_pos = self.drawer.data.default_joint_pos[env_ids]
        drawer_joint_pos.fill_(0.0)
        self.drawer.write_joint_position_to_sim(
            drawer_joint_pos, joint_ids=None, env_ids=env_ids
        )
        # Task design: start slightly open.
        self.set_drawer_position(joint, "Drawer001_joint", immediate=True)

        self.drawer_reset_state = np.array(
            drawer_root_state.cpu().detach(), dtype=np.float32
        )

        self.drawer_joint_reset_state = np.array(
            drawer_joint_pos.cpu().detach(), dtype=np.float32
        )

        
    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        """Cache default root/joint states for logging and replay."""
        robot_state = self.robot.data.default_root_state
        self.robot_reset_state = np.array(
            robot_state.cpu().detach(), dtype=np.float32
        )
        drawer_state = self.drawer.data.default_root_state
        self.drawer_reset_state = np.array(
            drawer_state.cpu().detach(), dtype=np.float32
        )
        
        drawer_joint_state = self.drawer.data.default_joint_pos
        self.drawer_joint_reset_state = np.array(
            drawer_joint_state.cpu().detach(), dtype=np.float32
        )

    def get_all_pose(self):
        """Return cached robot/drawer states for logging."""
        if not hasattr(self, "robot_reset_state"):
            self.initialize_obs()
        return {
            "robot": self.robot_reset_state,
            "drawer_root": self.drawer_reset_state,
            "drawer_joint": self.drawer_joint_reset_state,
        }
    
    def get_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return None
    
    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        """Restore robot and drawer states from a recorded pose dict."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        if pose and "robot" in pose and pose["robot"] is not None:
            robot_tensor = torch.tensor(
                pose["robot"], dtype=torch.float32, device=self.device
            )
            self.robot.write_root_state_to_sim(robot_tensor, env_ids=env_ids)
        
        if pose and "drawer_root" in pose and pose["drawer_root"] is not None:
            drawer_tensor = torch.tensor(
                pose["drawer_root"], dtype=torch.float32, device=self.device
            )
            self.drawer.write_root_state_to_sim(drawer_tensor, env_ids=env_ids)
        
        # NOTE: recorded datasets use key `drawer_joint` (not `drawerjoint`).
        # We keep backward-compat by accepting both.
        drawer_joint_value = None
        if pose:
            if "drawer_joint" in pose:
                drawer_joint_value = pose.get("drawer_joint")
            elif "drawerjoint" in pose:
                drawer_joint_value = pose.get("drawerjoint")

        if drawer_joint_value is not None:
            # Normalize shape to (num_envs, num_drawer_joints) and handle mismatch gracefully.
            drawer_joint_tensor = torch.tensor(drawer_joint_value, dtype=torch.float32, device=self.device)
            if drawer_joint_tensor.dim() == 0:
                drawer_joint_tensor = drawer_joint_tensor.view(1, 1)
            elif drawer_joint_tensor.dim() == 1:
                drawer_joint_tensor = drawer_joint_tensor.unsqueeze(0)

            num_envs = len(env_ids)
            if drawer_joint_tensor.shape[0] == 1 and num_envs > 1:
                drawer_joint_tensor = drawer_joint_tensor.expand(num_envs, -1).clone()

            # Current asset joint count (can differ across drawer USDs / dataset versions)
            num_joints = int(self.drawer.data.joint_pos.shape[1])
            if drawer_joint_tensor.shape[1] > num_joints:
                # Dataset has extra joints -> try to preserve joint indexing semantics.
                # Common case observed in Drawer_Orin2: dataset stores 4 values (joint_0..joint_3)
                # but current asset exposes 3 joints (joint_1..joint_3). In that case, we must
                # DROP the first value to avoid shifting joint_2->joint_3, etc.
                extra = int(drawer_joint_tensor.shape[1] - num_joints)
                start_col = 0
                if extra == 1:
                    try:
                        import re
                        joint_names = getattr(self.drawer, "joint_names", [])
                        nums = []
                        for n in joint_names:
                            m = re.match(r"^joint_(\d+)$", str(n))
                            if m:
                                nums.append(int(m.group(1)))
                        # If current joints start at 1 (no joint_0), assume dataset includes joint_0
                        # and shift by one.
                        if len(nums) > 0 and min(nums) == 1:
                            start_col = 1
                    except Exception:
                        start_col = 0
                # Fallback: keep first N
                drawer_joint_tensor = drawer_joint_tensor[:, start_col : start_col + num_joints]
            elif drawer_joint_tensor.shape[1] < num_joints:
                # Dataset has fewer joints -> pad zeros
                pad = torch.zeros(
                    (drawer_joint_tensor.shape[0], num_joints - drawer_joint_tensor.shape[1]),
                    dtype=drawer_joint_tensor.dtype,
                    device=drawer_joint_tensor.device,
                )
                drawer_joint_tensor = torch.cat([drawer_joint_tensor, pad], dim=1)

            self.drawer.write_joint_position_to_sim(drawer_joint_tensor, joint_ids=None, env_ids=env_ids)
        else:
            drawer_joint_pos = self.drawer.data.default_joint_pos[env_ids]
            drawer_joint_pos.fill_(0.0)
            self.drawer.write_joint_position_to_sim(
                drawer_joint_pos, joint_ids=None, env_ids=env_ids
            )

        # Task design: enforce a consistent initial opening (close-drawer task).
        self.set_drawer_position(-0.25, "Drawer001_joint", immediate=True)