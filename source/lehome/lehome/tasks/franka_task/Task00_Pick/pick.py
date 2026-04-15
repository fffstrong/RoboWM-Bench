from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from collections.abc import Sequence

import numpy as np
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, TiledCamera
from pxr import Usd, UsdGeom

from .pick_cfg import PickEnvCfg
from lehome.utils.success_checker import success_checker_pick
from lehome.devices.action_process import preprocess_device_action


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
    "LEHOME_PICK_SCENE_USD",
    str(Path(DEFAULT_ASSETS_ROOT) / "benchmark" / "scenes" / "benchmark_scene1.usd"),
)

class PickEnv(DirectRLEnv):
    """Direct RL environment for the Franka pick task."""

    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        self.scores = 0
        self.part = 0
        # Keep the original initialization order/values to avoid affecting any
        # downstream code that might read these before the end of `_setup_scene`.
        self.full_marks = 4
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.top_camera2 = TiledCamera(self.cfg.top_camera2)

        cfg = sim_utils.UsdFileCfg(usd_path=DEFAULT_SCENE_USD)

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.object_A_names = [
            "white_low",
            "white_high",
            "banana",
            "brown_cup",
            "button",
        ]
        self.object_A_map: dict[str, RigidObject] = {}
        for name in self.object_A_names:
            obj = RigidObject(getattr(self.cfg, name))
            self.object_A_map[name] = obj
            self.scene.rigid_objects[name] = obj

        self.object_A_name = random.choice(self.object_A_names)
        self.object_A = self.object_A_map[self.object_A_name]

        self.texture_cfg = {
            "enable": True,
            "folder": "Assets/textures/surface/solid_color",
            "min_id": 0,
            "max_id": 9,
            "prim_path": "/World/Scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7",
        }
        self.light_cfg = {
            "enable": False,
            "prim_path": "/World/Light",
            "intensity_range": [500, 3000],
            "color_range": [0.0, 1.0],
        }


        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["top_camera2"] = self.top_camera2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=800, color=(0.7, 0.7, 0.7))
        light_cfg.func("/World/Light", light_cfg)

        self.ori_z = 0
        self.joint_num = 9

        # Preserve the original final scoring configuration.
        self.scores = 0
        self.part = 0
        self.full_marks = 2

        self.contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)
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
        total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(
        self,
        env_ids: Sequence[int] | None,
        name: str = None,
        xyz: list = None,
        quat: list = None,
        name2: str = None,
        xyz2: list = None,
        quat2: list = None,
    ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()

        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )
        if name is not None:
            obj = self.object_A_map.get(name)
            if obj is None:
                print(f"[Reset][Warn] Object name '{name}' not found in object_A_map. Using random selection.")
                self.object_A_name = random.choice(self.object_A_names)
                self.object_A = self.object_A_map[self.object_A_name]
            else:
                self.object_A_name = name
                self.object_A = obj
        else:
            self.object_A_name = random.choice(self.object_A_names)
            self.object_A = self.object_A_map[self.object_A_name]

        for obj_name, obj in self.object_A_map.items():
            obj_state = obj.data.default_root_state[env_ids].clone()
            if obj is self.object_A and xyz is not None and quat is not None:
                
                obj_state[:, 0:3] = torch.tensor(xyz, device=obj_state.device)
                obj_state[:, 3:7] = torch.tensor(quat, device=obj_state.device)
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            if obj is self.object_A:
                self.object_A_reset_state = np.array(
                    obj_state.cpu().detach(), dtype=np.float32
                )

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        """Cache the current simulation root states for external logging/replay."""
        robot_state = self.robot.data.root_state_w.clone()
        self.robot_reset_state = np.array(
            robot_state.cpu().detach(), dtype=np.float32
        )
        object_A_state = self.object_A.data.root_state_w.clone()
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )

    def get_all_pose(self):
        """Return current world poses for the robot and the active object."""
        return {
            "robot": self.robot.data.root_state_w.clone().cpu().numpy(),
            "object_A": self.object_A.data.root_state_w.clone().cpu().numpy(),
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        pose_tensor = torch.tensor(
            pose["robot"], dtype=torch.float32, device=self.device
        )
        self.robot.write_root_state_to_sim(pose_tensor, env_ids=env_ids)
        pose_tensor = torch.tensor(
            pose["object_A"], dtype=torch.float32, device=self.device
        )
        self.object_A.write_root_state_to_sim(pose_tensor, env_ids=env_ids)

    def get_gripper_poses(self):
        """Return the current gripper pose in world coordinates."""
        if not hasattr(self, "gripper_idx"):
            self.gripper_idx = self.robot.find_bodies("gripper")[0][0]

        left_body_states = self.robot.data.body_state_w

        gripper_poses = {
            "left_pos": left_body_states[0, self.gripper_idx, 0:3],
            "left_quat": left_body_states[0, self.gripper_idx, 3:7],
        }

        return gripper_poses
    
    def get_rigid_body_dimensions(self):
        """Read a prim's local AABB dimensions from USD (utility/debug)."""
        stage = omni.usd.get_context().get_stage()
        
        prim_path = f"/World/Object/dustpan"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"[Warn] Prim not found for body 'b_cups' at {prim_path}")
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
            
        width = local_max[0] - local_min[0]
        height = local_max[1] - local_min[1]
        depth = local_max[2] - local_min[2]

        print(f"x: {width}, y: {height}, z: {depth}")
        return width, height, depth
    
    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = success_checker_pick(self.object_A, self.ori_z)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success
    
    def get_pose(self) -> None:
        self.ori_z = self.object_A.data.root_pos_w[0, 2].item()
        return None
    
