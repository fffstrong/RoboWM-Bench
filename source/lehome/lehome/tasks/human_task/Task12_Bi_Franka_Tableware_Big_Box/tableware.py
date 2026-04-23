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
from pxr import Usd, UsdShade, Sdf, UsdGeom
import random
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

        self.ik_controller_left = DifferentialIKController(
            self.cfg.ik_controller_left,
            num_envs=self.scene.num_envs,
            device=self.device,
        )
        self.ik_controller_right = DifferentialIKController(
            self.cfg.ik_controller_right,
            num_envs=self.scene.num_envs,
            device=self.device,
        )

        self.ee_frame_name = "panda_hand"
        try:
            self.ee_idx_l = self.left_arm.find_bodies(self.ee_frame_name)[0][0]
            self.ee_idx_r = self.right_arm.find_bodies(self.ee_frame_name)[0][0]
        except (IndexError, RuntimeError):
            self.ee_idx_l = self.left_arm.num_bodies - 1
            self.ee_idx_r = self.right_arm.num_bodies - 1
            print(
                f"[Warn] Could not find body named '{self.ee_frame_name}' in left or right arm. Defaulting to last body index: {self.ee_idx_l} (left), {self.ee_idx_r} (right)"
            )

    def _setup_scene(self):

        self.scores = 0
        self.part = 0
        self.full_marks = 4

        self.left_arm = Articulation(self.cfg.left_arm)
        self.right_arm = Articulation(self.cfg.right_arm)

        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.top_camera2 = TiledCamera(self.cfg.top_camera2)

        cfg = sim_utils.UsdFileCfg(usd_path=f"{BYOBU_TABLE_USD_PATH}")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.object_A_name = "big_box"
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
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["top_camera2"] = self.top_camera2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.7, 0.7, 0.7))
        light_cfg.func("/World/Light", light_cfg)
        self.joint_num = 9

        self.scores = 0
        self.part = 0
        self.full_marks = 2
        self.contact_sensor_left = ContactSensor(cfg=self.cfg.contact_sensor_left)
        self.contact_sensor_right = ContactSensor(cfg=self.cfg.contact_sensor_right)
        self.scene.sensors["contact_sensor_left"] = self.contact_sensor_left
        self.scene.sensors["contact_sensor_right"] = self.contact_sensor_right

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # ================= left arm control =================
        ee_pose_w_l = self.left_arm.data.body_state_w[:, self.ee_idx_l, 0:7]
        ee_pos_curr_l = ee_pose_w_l[:, 0:3]
        ee_quat_curr_l = ee_pose_w_l[:, 3:7]

        ik_cmd_l = self.actions[:, 0:6]
        self.ik_controller_left.set_command(
            ik_cmd_l, ee_pos=ee_pos_curr_l, ee_quat=ee_quat_curr_l
        )
        jac_l = self.left_arm.root_physx_view.get_jacobians()[:, self.ee_idx_l, :, :]
        joint_pos_target_l = self.ik_controller_left.compute(
            ee_pos_curr_l, ee_quat_curr_l, jac_l, self.left_arm.data.joint_pos
        )

        gripper_target_l = self.actions[:, 6].view(-1, 1).repeat(1, 2)
        if joint_pos_target_l.shape[1] == 7:
            full_joint_target_l = torch.cat(
                [joint_pos_target_l, gripper_target_l], dim=1
            )
        else:
            full_joint_target_l = joint_pos_target_l.clone()
            full_joint_target_l[:, 7:9] = gripper_target_l

        self.left_arm.set_joint_position_target(full_joint_target_l)

        # ================= right arm control =================
        ee_pose_w_r = self.right_arm.data.body_state_w[:, self.ee_idx_r, 0:7]
        ee_pos_curr_r = ee_pose_w_r[:, 0:3]
        ee_quat_curr_r = ee_pose_w_r[:, 3:7]

        ik_cmd_r = self.actions[:, 7:13]
        self.ik_controller_right.set_command(
            ik_cmd_r, ee_pos=ee_pos_curr_r, ee_quat=ee_quat_curr_r
        )
        jac_r = self.right_arm.root_physx_view.get_jacobians()[:, self.ee_idx_r, :, :]
        joint_pos_target_r = self.ik_controller_right.compute(
            ee_pos_curr_r, ee_quat_curr_r, jac_r, self.right_arm.data.joint_pos
        )

        gripper_target_r = self.actions[:, 13].view(-1, 1).repeat(1, 2)
        if joint_pos_target_r.shape[1] == 7:
            full_joint_target_r = torch.cat(
                [joint_pos_target_r, gripper_target_r], dim=1
            )
        else:
            full_joint_target_r = joint_pos_target_r.clone()
            full_joint_target_r[:, 7:9] = gripper_target_r

        self.right_arm.set_joint_position_target(full_joint_target_r)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = self.left_arm.data.joint_pos
        right_joint_pos = self.right_arm.data.joint_pos

        joint_pos_l = torch.cat(
            [left_joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        ).squeeze(0)
        joint_pos_r = torch.cat(
            [right_joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        ).squeeze(0)
        joint_pos_combined = torch.cat([joint_pos_l, joint_pos_r], dim=0)

        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos_combined.cpu().detach().numpy(),
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
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        self.ik_controller_left.reset(env_ids)
        self.ik_controller_right.reset(env_ids)

        # 2. reset left arm
        joint_pos_l = self.left_arm.data.default_joint_pos[env_ids]
        self.left_arm.write_joint_position_to_sim(joint_pos_l, env_ids=env_ids)

        # 3. left right arm
        joint_pos_r = self.right_arm.data.default_joint_pos[env_ids]
        self.right_arm.write_joint_position_to_sim(joint_pos_r, env_ids=env_ids)

        left_arm_root_state = self.left_arm.data.default_root_state[env_ids].clone()
        right_arm_root_state = self.right_arm.data.default_root_state[env_ids].clone()

        self.left_arm_reset_state = np.array(
            left_arm_root_state.cpu().detach(), dtype=np.float32
        )

        self.right_arm_reset_state = np.array(
            right_arm_root_state.cpu().detach(), dtype=np.float32
        )

        object_A_pos = self.object_A.data.default_root_state[env_ids].clone()

        self.object_A.write_root_state_to_sim(object_A_pos, env_ids=env_ids)

        object_A_state = object_A_pos
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )

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

        # print(f"[Reset] Light randomized -> intensity={intensity:.1f}, color={color}")

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.left_arm_reset_state = np.array(
            self.left_arm.data.root_state_w.clone().cpu().detach(), dtype=np.float32
        )
        self.right_arm_reset_state = np.array(
            self.right_arm.data.root_state_w.clone().cpu().detach(), dtype=np.float32
        )
        self.object_A_reset_state = np.array(
            self.object_A.data.root_state_w.clone().cpu().detach(), dtype=np.float32
        )

    def get_all_pose(self):
        return {
            "left_arm": self.left_arm.data.root_state_w.clone().cpu().numpy(),
            "right_arm": self.right_arm.data.root_state_w.clone().cpu().numpy(),
            "object_A": self.object_A.data.root_state_w.clone().cpu().numpy(),
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        pose_tensor_l = torch.tensor(
            pose["left_arm"], dtype=torch.float32, device=self.device
        )
        self.left_arm.write_root_state_to_sim(pose_tensor_l, env_ids=env_ids)

        pose_tensor_r = torch.tensor(
            pose["right_arm"], dtype=torch.float32, device=self.device
        )
        self.right_arm.write_root_state_to_sim(pose_tensor_r, env_ids=env_ids)

        pose_tensor_A = torch.tensor(
            pose["object_A"], dtype=torch.float32, device=self.device
        )
        self.object_A.write_root_state_to_sim(pose_tensor_A, env_ids=env_ids)

    def get_gripper_poses(self):
        """
        Retrieves the current world-space poses (position and orientation)
        for both the left and right grippers.

        Returns:
            dict: A dictionary containing tensors for 'left_pos', 'left_quat',
                'right_pos', and 'right_quat'.
        """
        if not hasattr(self, "gripper_idx_l"):
            self.gripper_idx_l = self.left_arm.find_bodies("gripper")[0][0]
            self.gripper_idx_r = self.right_arm.find_bodies("gripper")[0][0]

        left_states = self.left_arm.data.body_state_w
        right_states = self.right_arm.data.body_state_w

        gripper_poses = {
            "left_pos": left_states[0, self.gripper_idx_l, 0:3],
            "left_quat": left_states[0, self.gripper_idx_l, 3:7],
            "right_pos": right_states[0, self.gripper_idx_r, 0:3],
            "right_quat": right_states[0, self.gripper_idx_r, 3:7],
        }
        return gripper_poses

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Final success flag of the task: all steps have been completed (self.part == 4)
        if self.part == 4:
            success_tensor = torch.ones_like(self.episode_length_buf, dtype=torch.bool)
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def get_part_scores(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        object_A_default_pos = self.object_A.data.default_root_state[env_ids, :3]
        object_A_pos = self.object_A.data.root_pos_w[env_ids]

        lift_height_threshold = 0.08
        place_distance_threshold = 0.05

        # 1: Touch left
        if self.part == 0:
            left_force = torch.norm(
                self.contact_sensor_left.data.net_forces_w[env_ids], dim=-1
            ).sum(dim=-1)
            if (left_force > 10.0).all():
                self.part = 1
                self.scores += 1
            return

        # 2: Touch right
        if self.part == 1:
            right_force = torch.norm(
                self.contact_sensor_right.data.net_forces_w[env_ids], dim=-1
            ).sum(dim=-1)
            if (right_force > 10.0).all():
                self.part = 2
                self.scores += 1
            return

        # 3: Lift
        if self.part == 2:
            box_lifted = (
                object_A_pos[:, 2] - object_A_default_pos[:, 2]
            ) > lift_height_threshold
            if box_lifted.all():
                self.part = 3
                self.scores += 1
            return

        # 4: Place
        if self.part == 3:
            dist_place = torch.norm(object_A_pos - object_A_default_pos, dim=-1)
            if (dist_place < place_distance_threshold).all():
                self.part = 4
                self.scores += 1
            return
