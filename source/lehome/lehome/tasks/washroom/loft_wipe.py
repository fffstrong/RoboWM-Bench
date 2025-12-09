from __future__ import annotations
import os
import torch
from dataclasses import MISSING
from typing import Any, Dict, List, Sequence
import isaaclab.sim as sim_utils
import torch
from collections.abc import Sequence
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from .loft_wipe_cfg import LoftWipeEnvCfg
from ..base.base_env import BaseEnv
from ..base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action
from omegaconf import OmegaConf
import numpy as np
from lehome.assets.object.fluid import FluidObject
from lehome.assets.object.Garment import GarmentObject


class LoftWipeEnv(BaseEnv):
    """Environment inheriting from base LW_Loft environment with additional features."""

    cfg: BaseEnvCfg | LoftWipeEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | LoftWipeEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to get base scene (LW_Loft + robot + camera)
        super()._setup_scene()
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.towel = GarmentObject(
            prim_path="/World/Objects/Towel",
            usd_path=os.getcwd() + "/Assets/objects/Towel/towel.usd",
            visual_usd_path=os.getcwd() + "/Assets/Material/Garment/linen_Blue.usd",
            config=OmegaConf.load(
                os.getcwd()
                + "/source/lehome/lehome/tasks/washroom/config_file/particle_towel_cfg.yaml"
            ),
        )
        self.object = FluidObject(
            env_id=0,
            env_origin=torch.zeros(1, 3),
            prim_path="/World/Object/fluid_items/fluid_items_1",
            usd_path=os.getcwd() + "/Assets/scenes/LW_Loft/water.usdc",
            config=OmegaConf.load(
                os.getcwd()
                + "/source/lehome/lehome/tasks/washroom/config_file/fluid.yaml"
            ),
            use_container=False,
        )

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:

        self.robot.set_joint_position_target(self.actions)

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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        wrist_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.object.reset(soft=True)
        self.robot.write_joint_position_to_sim(
            wrist_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.towel.reset()
        self.object.reset(soft=True)

    def _get_successes(self) -> torch.Tensor:  # TODO: define success condition
        successes = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return successes

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()
        self.towel.initialize()

    def get_all_pose(self):
        # TODO: return real pose
        return {"abc": [1, 23, 4, 5, 6, 7]}

    def set_all_pose(self, pose, env_ids: Sequence[int] | None):
        # TODO: set real pose
        # self.object.set_all_pose(pose)
        pass
