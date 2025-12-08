from __future__ import annotations
import os
import torch

# from dataclasses import MISSING
from typing import Any, Dict, List, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from .loft_water_cfg import LoftWaterEnvCfg
from ..base.base_env import BaseEnv
from ..base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action
from omegaconf import OmegaConf
import numpy as np
from lehome.assets.object.fluid import FluidObject


class LoftWaterEnv(BaseEnv):
    """Environment inheriting from base LW_Loft environment with additional features."""

    cfg: BaseEnvCfg | LoftWaterEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | LoftWaterEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.arm.data.joint_pos

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to get base scene (LW_Loft + robot + camera)
        super()._setup_scene()
        self.arm = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        self.object = FluidObject(
            env_id=0,
            env_origin=torch.zeros(1, 3),
            prim_path="/World/Object/fluid_items/fluid_items_1",
            usd_path=os.getcwd() + "/Assets/scenes/LW_Loft/water.usdc",
            config=OmegaConf.load(
                os.getcwd()
                + "/source/lehome/lehome/tasks/livingroom/config_file/fluid.yaml"
            ),
        )
        self.bowl = RigidObject(self.cfg.bowl)
        self.scene.rigid_objects["bowl"] = self.bowl

        # add articulation to scene
        self.scene.articulations["robot"] = self.arm

        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.arm.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        ).squeeze(0)

        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.wrist_rgb": wrist_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_successes(self) -> torch.Tensor:
        successes = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return successes

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.arm._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.arm.data.default_joint_pos[env_ids]
        self.object.reset()
        bowl_pos = self.bowl.data.default_root_state[env_ids].clone()
        # 在与 bowl_pos 相同的设备上创建随机值
        rand_vals_1 = torch.empty(len(env_ids), 2, device=bowl_pos.device).uniform_(-0.05, 0.05)
        # 为 bowl 添加随机位置扰动
        random_bowl_pos = bowl_pos.clone()
        random_bowl_pos[..., :2] += rand_vals_1
        random_bowl_pos[..., 7:] = 0.0  # 重置速度
        self.bowl.write_root_state_to_sim(random_bowl_pos, env_ids=env_ids)
        self.arm.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()
        # RigidObject 使用 reset() 进行初始化
        self.bowl.reset()

    def get_all_pose(self):
        poses = {}
        poses.update(self.object.get_all_pose())  # {'cup': ...}
        # 从 RigidObject 获取 bowl 的位姿 (位置 + 四元数)
        bowl_root_state = self.bowl.data.root_state_w[0]  # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        bowl_pose = torch.cat([bowl_root_state[:3], bowl_root_state[3:7]]).cpu().numpy()  # pos + quat
        poses.update({"bowl": bowl_pose})
        return poses

    def set_all_pose(self, pose, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.bowl._ALL_INDICES
        self.object.set_all_pose(pose)
        # 为 RigidObject 设置位姿
        if "bowl" in pose:
            bowl_pose = pose["bowl"]
            # 构造 root_state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            bowl_root_state = self.bowl.data.default_root_state[env_ids].clone()
            if isinstance(bowl_pose, np.ndarray):
                bowl_pose = torch.from_numpy(bowl_pose).float()
            if len(bowl_pose) >= 7:  # pos(3) + quat(4)
                bowl_root_state[..., :3] = bowl_pose[:3]  # position
                bowl_root_state[..., 3:7] = bowl_pose[3:7]  # quaternion
            bowl_root_state[..., 7:] = 0.0  # 速度归零
            self.bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
