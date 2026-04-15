from __future__ import annotations
import os
import random
import torch

# from dataclasses import MISSING
from typing import Any, Dict, List, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from .pour_water_cfg import PourWaterEnvCfg
from ...base.base_env import BaseEnv
from ...base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action
from omegaconf import OmegaConf
import numpy as np
from lehome.assets.object.fluid import FluidObject
from lehome.utils.success_checker import success_checker_pour
from pxr import UsdShade, Sdf


class PourWaterEnv(BaseEnv):
    """Environment inheriting from base LW_Loft environment with additional features."""

    cfg: BaseEnvCfg | PourWaterEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | PourWaterEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos
        self.table_texture_cfg = {
            "enable": True,
            "folder": "Assets/textures/surface/solid_color",
            "min_id": 0,
            "max_id": 9,
            "prim_path": "/World/Scene/Table062/Looks/M_Table062_001/Shader",
        }

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to get base scene (LW_Loft + robot + camera)
        # super()._setup_scene()
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        from lehome.utils.rendering import apply_default_render_settings
        apply_default_render_settings()
        from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
        import isaaclab.sim as sim_utils
        cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_WITH_ORANGE_USD_PATH}")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )
        self.object = FluidObject(
            env_id=0,
            env_origin=torch.zeros(1, 3),
            prim_path="/World/Object/fluid_items/fluid_items_1",
            usd_path=os.getcwd() + "/Assets/benchmark/object/water.usdc",
            config=OmegaConf.load(
                "/home/feng/lehome_1/source/lehome/lehome/tasks/franka_task/Task07_PourWater/config_file/fluid.yaml"
            ),
            use_container=False,
        )
        self.bowl = RigidObject(self.cfg.bowl)
        self.scene.rigid_objects["bowl"] = self.bowl
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        self.joint_num=9
        self.scores=0
        self.part=0
        self.full_marks=2
        from isaaclab.sensors import ContactSensor,ContactSensorCfg
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

    def _get_success(self) -> torch.Tensor:
        success = success_checker_pour(self.object)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(
                self.episode_length_buf, dtype=torch.bool, device=self.device
            )
        return success_tensor

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.object.reset()
        bowl_pos = self.bowl.data.default_root_state[env_ids].clone()

        self.bowl.write_root_state_to_sim(bowl_pos, env_ids=env_ids)

        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        # self._randomize_table_texture()

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()
        # RigidObject use reset() to initialize
        self.bowl.reset()
        self._randomize_table_texture()

    def get_all_pose(self):
        poses = {}
        poses.update(self.object.get_all_pose())  # {'cup': ...}
        # get bowl pose from RigidObject (position + quaternion)
        bowl_root_state = self.bowl.data.root_state_w[
            0
        ]  # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        bowl_pose = (
            torch.cat([bowl_root_state[:3], bowl_root_state[3:7]]).cpu().numpy()
        )  # pos + quat
        poses.update({"bowl": bowl_pose})
        return poses

    def _randomize_table_texture(self):
        """随机切换桌面纹理。"""
        cfg = self.table_texture_cfg
        if not cfg.get("enable", False):
            return

        folder = cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(cfg.get("min_id", 1))
        max_id = int(cfg.get("max_id", 1))
        shader_path = cfg.get("prim_path", "")

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

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.bowl._ALL_INDICES
        self.object.set_all_pose(pose)
        # set pose for RigidObject
        if "bowl" in pose:
            bowl_pose = pose["bowl"]
            # construct root_state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            bowl_root_state = self.bowl.data.default_root_state[env_ids].clone()
            if isinstance(bowl_pose, np.ndarray):
                bowl_pose = torch.from_numpy(bowl_pose).float()
            elif isinstance(bowl_pose, (list, tuple)):
                bowl_pose = torch.tensor(bowl_pose, dtype=torch.float32, device=bowl_root_state.device)
            if len(bowl_pose) >= 7:  # pos(3) + quat(4)
                bowl_root_state[..., :3] = bowl_pose[:3]  # position
                bowl_root_state[..., 3:7] = bowl_pose[3:7]  # quaternion
            bowl_root_state[..., 7:] = 0.0  # reset velocity
            self.bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
