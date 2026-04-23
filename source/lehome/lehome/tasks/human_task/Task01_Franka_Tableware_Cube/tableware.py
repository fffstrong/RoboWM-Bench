from __future__ import annotations
from typing import Any
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera
from pxr import UsdShade, Sdf
import random
from lehome.utils.success_checker import success_checker_lifted
from .tableware_cfg import TablewareEnvCfg
from lehome.assets.scenes.byobu_table import BYOBU_TABLE_USD_PATH
from lehome.devices.action_process import preprocess_device_action
from isaaclab.controllers import DifferentialIKController
import numpy as np
import os


class TablewareEnv(DirectRLEnv):
    cfg: TablewareEnvCfg

    def __init__(self, cfg: TablewareEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

        self.ik_controller = DifferentialIKController(
            self.cfg.ik_controller, num_envs=self.scene.num_envs, device=self.device
        )

        self.ee_frame_name = "panda_hand"
        try:
            self.ee_idx = self.robot.find_bodies(self.ee_frame_name)[0][0]
        except (IndexError, RuntimeError):
            self.ee_idx = self.robot.num_bodies - 1
            print(f"[Warn] {self.ee_frame_name} not found, using last body.")

        self.lift_time = torch.zeros(
            self.scene.num_envs, dtype=torch.float32, device=self.device
        )

    def _setup_scene(self):

        self.scores = 0
        self.part = 0
        self.full_marks = 4

        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.top_camera2 = TiledCamera(self.cfg.top_camera2)

        cfg = sim_utils.UsdFileCfg(usd_path=f"{BYOBU_TABLE_USD_PATH}")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.object_A_name = "cube"
        self.object_A = RigidObject(getattr(self.cfg, self.object_A_name))

        self.scene.rigid_objects["object_A"] = self.object_A
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
        light_cfg = sim_utils.DomeLightCfg(intensity=900, color=(0.7, 0.7, 0.7))
        light_cfg.func("/World/Light", light_cfg)
        self.joint_num = 9

        self.scores = 0
        self.part = 0
        self.full_marks = 2
        self.contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7]
        ee_pos_curr = ee_pose_w[:, 0:3]
        ee_quat_curr = ee_pose_w[:, 3:7]

        ik_command = self.actions[:, :6]  # Shape: (num_envs, 6)

        self.ik_controller.set_command(
            ik_command, ee_pos=ee_pos_curr, ee_quat=ee_quat_curr
        )

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx, :, :]
        joint_pos_curr = self.robot.data.joint_pos

        joint_pos_target = self.ik_controller.compute(
            ee_pos_curr, ee_quat_curr, jacobian, joint_pos_curr
        )

        gripper_pos_target = self.actions[:, 6].view(-1, 1).repeat(1, 2)

        if joint_pos_target.shape[1] == 7:
            full_joint_target = torch.cat([joint_pos_target, gripper_pos_target], dim=1)
        else:
            full_joint_target = joint_pos_target
            full_joint_target[:, 7:9] = gripper_pos_target

        self.robot.set_joint_position_target(full_joint_target)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": top_camera_depth.cpu().detach().numpy().copy(),
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.ik_controller.reset(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()

        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        object_A_pos = self.object_A.data.default_root_state[env_ids].clone()

        self.object_A.write_root_state_to_sim(object_A_pos, env_ids=env_ids)

        object_A_state = object_A_pos
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )

        self.lift_time[env_ids] = 0.0

    def _randomize_table038_texture(self):
        """Randomize Table038 texture based on config."""
        if not self.texture_cfg.get("enable", False):
            return

        folder = self.texture_cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(self.texture_cfg.get("min_id", 1))
        max_id = int(self.texture_cfg.get("max_id", 1))
        shader_path = self.texture_cfg.get("prim_path", "")

        if not folder or not os.path.exists(folder):
            print(f"[Reset][Warn] Texture folder not found: {folder}")
            return
        if not shader_path:
            print("[Reset][Warn] No prim_path provided for texture randomization")
            return

        stage = self.scene.stage
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            print(f"[Reset][Warn] Shader prim not found at {shader_path}")
            return

        shader = UsdShade.Shader(shader_prim)
        idx = random.randint(min_id, max_id)
        tex_path = os.path.join(folder, f"{idx}.png")

        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            print("[Reset][Warn] No texture input found on shader")
            return

        tex_input.Set(Sdf.AssetPath(tex_path))
        # print(f"[Reset] Texture randomized -> {tex_path}")

    def _randomize_light(self):
        """Randomize DomeLight attributes based on config."""
        if not self.light_cfg.get("enable", False):
            return

        prim_path = self.light_cfg.get("prim_path", "/World/Light")
        intensity_range = self.light_cfg.get("intensity_range", [800, 2000])
        color_range = self.light_cfg.get("color_range", [0.0, 1.0])

        stage = self.scene.stage
        light_prim = stage.GetPrimAtPath(prim_path)
        if not light_prim.IsValid():
            print(f"[Reset][Warn] Light prim not found at {prim_path}")
            return

        intensity = random.uniform(*intensity_range)
        color = tuple(random.uniform(color_range[0], color_range[1]) for _ in range(3))

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        robot_state = self.robot.data.root_state_w.clone()
        self.robot_reset_state = np.array(robot_state.cpu().detach(), dtype=np.float32)
        object_A_state = self.object_A.data.root_state_w.clone()
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )

    def get_all_pose(self):

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
        """
        Retrieves the current world-space poses (position and orientation)
        for both the left and right grippers.

        Returns:
            dict: A dictionary containing tensors for 'left_pos', 'left_quat',
                'right_pos', and 'right_quat'.
        """
        # 1. Identify the body indices for the grippers if not already cached
        # find_bodies returns a tuple of (indices, names)
        if not hasattr(self, "gripper_idx"):
            self.gripper_idx = self.robot.find_bodies("gripper")[0][0]

        # 2. Access the body_state_w tensor: [num_envs, num_bodies, 13]
        # The 13 dimensions represent: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, lin_vel, ang_vel]
        left_body_states = self.robot.data.body_state_w

        # 3. Extract Position [0:3] and Quaternion [3:7] for the specific gripper index
        # Using index 0 for the first environment (num_envs=1)
        gripper_poses = {
            "left_pos": left_body_states[0, self.gripper_idx, 0:3],
            "left_quat": left_body_states[0, self.gripper_idx, 3:7],
        }

        return gripper_poses

    def _get_success(self) -> torch.Tensor:
        episode_success = torch.zeros(
            self.scene.num_envs, dtype=torch.bool, device=self.device
        )

        for env_id in range(self.scene.num_envs):
            # Get the initial Z coordinate of the object (this value has been stored in self.object_A_reset_state in _reset_idx)
            ori_z = self.object_A_reset_state[env_id, 2].item()

            # Use checker to determine if the Z-axis of the current frame is higher than the initial Z-axis position
            is_lifted = success_checker_lifted(self.object_A, ori_z, env_id=env_id)

            if is_lifted:
                # Accumulate step time (self.step_dt represents the time elapsed per physics step)
                self.lift_time[env_id] += self.step_dt
            else:
                # Once put down or dropped midway, reset the timer to zero
                self.lift_time[env_id] = 0.0

            # If the accumulated lifted time exceeds 2.0 seconds, it is considered a success
            if self.lift_time[env_id] >= 2.0:
                episode_success[env_id] = True

        return episode_success
