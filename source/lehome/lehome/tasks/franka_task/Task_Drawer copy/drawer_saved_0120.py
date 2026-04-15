from __future__ import annotations
import torch
from typing import Any, Dict, List

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera
from pxr import Usd, UsdShade, Sdf, UsdGeom, Gf
import random

from .drawer_cfg import DrawerEnvCfg
from lehome.devices.action_process import preprocess_device_action
import numpy as np
import os

import omni.usd

from lehome.utils.rendering import apply_default_render_settings

class DrawerEnv(DirectRLEnv):
    cfg: DrawerEnvCfg

    def __init__(self, cfg: DrawerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        # Set joint_pos reference after robot is created in _setup_scene (called by super().__init__)
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        # Make viewport/rendering defaults consistent across environments.
        apply_default_render_settings()

        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        scene_cfg.func(
            "/World/Scene",
            scene_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        # Microwave - 铰链物体
        # 注意：配置中的 init_state.joint_pos 会在创建时自动设置铰链关节位置
        self.drawer = Articulation(self.cfg.drawer)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["drawer"] = self.drawer
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        # Handle case where self.actions might not be initialized yet (before first step)
        if hasattr(self, 'actions'):
            action = self.actions.squeeze(0)
        else:
            # Return zero action if actions not yet set
            action = torch.zeros(self.cfg.action_space, device=self.device)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)
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

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: 实现抽屉任务的成功判断
        success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return success_tensor

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 重置机器人关节位置
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        # ===== 全局 XY 平移：对 robot / drawer 施加相同的 (x, y) 偏移 =====
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        global_xy = torch.zeros(len(env_ids), 2, device=robot_root_state.device)
        global_xy[:, 0].uniform_(-0.1, 0.1)  # 仅在 x 轴平移，y 保持不动
        robot_root_state[..., :2] += global_xy
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
        # 记录本次随机后的机器人位姿
        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        # 重置微波炉位置和铰链关节
        drawer_root_state = self.drawer.data.default_root_state[env_ids].clone()
        drawer_root_state[..., :2] += global_xy
        self.drawer.write_root_state_to_sim(drawer_root_state, env_ids=env_ids)

        # 🔹 重置微波炉的铰链关节位置（关闭状态）
        # 获取微波炉的默认关节位置
        drawer_joint_pos = self.drawer.data.default_joint_pos[env_ids]
        # 设置所有关节为 0（关闭状态）
        drawer_joint_pos.fill_(0.0)
        self.drawer.write_joint_position_to_sim(
            drawer_joint_pos, joint_ids=None, env_ids=env_ids
        )

        # 记录微波炉的初始状态
        self.drawer_reset_state = np.array(
            drawer_root_state.cpu().detach(), dtype=np.float32
        )
        # 记录微波炉的关节状态
        self.drawer_joint_reset_state = np.array(
            drawer_joint_pos.cpu().detach(), dtype=np.float32
        )

        # Apply randomization if enabled in config
        if hasattr(self, "texture_cfg") and self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()

        if hasattr(self, "light_cfg") and self.light_cfg.get("enable", False):
            self._randomize_light()

    def _randomize_table038_texture(self):
        """Randomize Table038 texture based on config."""
        if not hasattr(self, "texture_cfg") or not self.texture_cfg.get("enable", False):
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

    def _randomize_light(self):
        """Randomize DomeLight attributes based on config."""
        if not hasattr(self, "light_cfg") or not self.light_cfg.get("enable", False):
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
        """初始化观测值，保存机器人和微波炉的初始状态用于后续恢复。"""
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
        """返回机器人和抽屉的当前姿态，用于记录。"""
        # 确保在 get_all_pose 之前已经初始化了 obs
        if not hasattr(self, 'robot_reset_state'):
            self.initialize_obs()
        return {
            "robot": self.robot_reset_state,
            "drawer_root": self.drawer_reset_state,
            "drawer_joint": self.drawer_joint_reset_state,
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        """设置机器人和抽屉的姿态，用于从记录中恢复。"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # 设置机器人姿态
        if pose and "robot" in pose and pose["robot"] is not None:
            robot_tensor = torch.tensor(
                pose["robot"], dtype=torch.float32, device=self.device
            )
            self.robot.write_root_state_to_sim(robot_tensor, env_ids=env_ids)
        
        # 设置抽屉根状态
        if pose and "drawer_root" in pose and pose["drawer_root"] is not None:
            drawer_tensor = torch.tensor(
                pose["drawer_root"], dtype=torch.float32, device=self.device
            )
            self.drawer.write_root_state_to_sim(drawer_tensor, env_ids=env_ids)
        
        # 设置抽屉关节状态
        if pose and "drawerjoint" in pose and pose["drawer_joint"] is not None:
            drawer_joint_tensor = torch.tensor(
                pose["drawer_joint"], dtype=torch.float32, device=self.device
            )
            self.drawer.write_joint_position_to_sim(
                drawer_joint_tensor, joint_ids=None, env_ids=env_ids
            )
        else:
            # 如果没有提供关节状态，则设置为关闭状态（0）
            drawer_joint_pos = self.drawer.data.default_joint_pos[env_ids]
            drawer_joint_pos.fill_(0.0)
            self.drawer.write_joint_position_to_sim(
                drawer_joint_pos, joint_ids=None, env_ids=env_ids
            )

