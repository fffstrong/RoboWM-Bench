from __future__ import annotations
import os
import torch

# from dataclasses import MISSING
from typing import Any, Dict, List, Sequence
import random

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from isaaclab.envs import DirectRLEnv
from .swaptrash_cfg import SwapRubbishEnvCfg
from lehome.devices.action_process import preprocess_device_action
import numpy as np
from lehome.utils.success_checker import success_checker_rubbish
from pxr import Usd, UsdShade, Sdf
import isaaclab.sim as sim_utils
from lehome.utils.rendering import apply_default_render_settings


class SwapRubbishEnv(DirectRLEnv):
    """Single-arm swap rubbish environment (right arm only)."""

    cfg: SwapRubbishEnvCfg

    def __init__(
        self,
        cfg: SwapRubbishEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        """Setup the scene with robot, cameras, and objects."""
        # Make viewport/rendering defaults consistent across environments.
        apply_default_render_settings()

        # Load scene USD
        from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
        cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome_1/Assets/scenes/kitchen_with_orange/scene_v1.usd")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Robot (single arm - right arm)
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        self.food_rubbish01 = RigidObject(self.cfg.food_rubbish01)
        self.food_rubbish02 = RigidObject(self.cfg.food_rubbish02)
        # self.food_rubbish03 = RigidObject(self.cfg.food_rubbish03)
 
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["food_rubbish01"] = self.food_rubbish01
        self.scene.rigid_objects["food_rubbish02"] = self.food_rubbish02
        # self.scene.rigid_objects["food_rubbish03"] = self.food_rubbish03
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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

        pass 

        food_rubbish01_pos = self.food_rubbish01.data.root_pos_w
        food_rubbish02_pos = self.food_rubbish02.data.root_pos_w
        # food_rubbish03_pos = self.food_rubbish03.data.root_pos_w
        
        success = success_checker_rubbish(
            food_rubbish01_pos=food_rubbish01_pos,
            food_rubbish02_pos=food_rubbish02_pos,
            # food_rubbish03_pos=food_rubbish03_pos,
        )
        
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        
        return success_tensor

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # 获取默认初始位置和姿态（更安全的方式）
        food_rubbish01_pos = self.food_rubbish01.data.default_root_state[env_ids, :3].clone()
        food_rubbish01_quat = self.food_rubbish01.data.default_root_state[env_ids, 3:7].clone()
        
        food_rubbish02_pos = self.food_rubbish02.data.default_root_state[env_ids, :3].clone()
        food_rubbish02_quat = self.food_rubbish02.data.default_root_state[env_ids, 3:7].clone()
        
        # food_rubbish03_pos = self.food_rubbish03.data.default_root_state[env_ids, :3].clone()
        # food_rubbish03_quat = self.food_rubbish03.data.default_root_state[env_ids, 3:7].clone()
        

        # 归一化四元数，确保有效性
        food_rubbish01_quat = food_rubbish01_quat / torch.norm(food_rubbish01_quat, dim=-1, keepdim=True)
        food_rubbish02_quat = food_rubbish02_quat / torch.norm(food_rubbish02_quat, dim=-1, keepdim=True)
        # food_rubbish03_quat = food_rubbish03_quat / torch.norm(food_rubbish03_quat, dim=-1, keepdim=True)

        # 创建零速度
        zero_velocity = torch.zeros((len(env_ids), self.joint_num), device=self.device)

        # 重建完整的root_state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        food_rubbish01_state = torch.cat([food_rubbish01_pos, food_rubbish01_quat, zero_velocity], dim=-1)
        food_rubbish02_state = torch.cat([food_rubbish02_pos, food_rubbish02_quat, zero_velocity], dim=-1)
        # food_rubbish03_state = torch.cat([food_rubbish03_pos, food_rubbish03_quat, zero_velocity], dim=-1)

        # 重置机械臂为竖直向上姿态（所有关节为0），避免碰撞到前面的垃圾物品
        vertical_up_pose = torch.zeros((len(env_ids), self.joint_num), device=self.device)
        joint_pos = vertical_up_pose.clone()

        # 如果需要添加随机位置噪声（当前禁用）
        # rand_vals = torch.empty(len(env_ids), 2, device=self.device).uniform_(-0.05, 0.05)
        # food_rubbish01_state[:, :2] += rand_vals
        # food_rubbish02_state[:, :2] += rand_vals
        # food_rubbish03_state[:, :2] += rand_vals

        # 写入仿真环境 - 恢复到初始位置和姿态，速度为0
        self.food_rubbish01.write_root_state_to_sim(food_rubbish01_state, env_ids=env_ids)
        self.food_rubbish02.write_root_state_to_sim(food_rubbish02_state, env_ids=env_ids)
        # self.food_rubbish03.write_root_state_to_sim(food_rubbish03_state, env_ids=env_ids)

        # 重置机器人关节位置为竖直向上姿态
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        # 应用随机化（如果在配置中启用）
        if self.cfg.use_random:
            self._randomize_texture()
            self._randomize_light()

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        """初始化观测值，保存垃圾的初始状态用于后续恢复。"""
        # 保存三个垃圾物体的初始位置状态
        self.food_rubbish01_reset_state = self.food_rubbish01.data.root_state_w.cpu().detach().numpy()
        self.food_rubbish02_reset_state = self.food_rubbish02.data.root_state_w.cpu().detach().numpy()
        # self.food_rubbish03_reset_state = self.food_rubbish03.data.root_state_w.cpu().detach().numpy()

    def get_all_pose(self):
        """返回所有垃圾物体的当前姿态，用于记录。"""
        return {
            "food_rubbish01": self.food_rubbish01_reset_state if hasattr(self, 'food_rubbish01_reset_state') else None,
            "food_rubbish02": self.food_rubbish02_reset_state if hasattr(self, 'food_rubbish02_reset_state') else None,
            # "food_rubbish03": self.food_rubbish03_reset_state if hasattr(self, 'food_rubbish03_reset_state') else None,
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        """设置所有垃圾物体的姿态，用于从记录中恢复。"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # 如果提供了姿态数据，恢复垃圾物体的位置
        if pose and "food_rubbish01" in pose and pose["food_rubbish01"] is not None:
            food_rubbish01_tensor = torch.tensor(
                pose["food_rubbish01"], dtype=torch.float32, device=self.device
            )
            self.food_rubbish01.write_root_state_to_sim(food_rubbish01_tensor, env_ids=env_ids)
        
        if pose and "food_rubbish02" in pose and pose["food_rubbish02"] is not None:
            food_rubbish02_tensor = torch.tensor(
                pose["food_rubbish02"], dtype=torch.float32, device=self.device
            )
            self.food_rubbish02.write_root_state_to_sim(food_rubbish02_tensor, env_ids=env_ids)
        
        # if pose and "food_rubbish03" in pose and pose["food_rubbish03"] is not None:
        #     food_rubbish03_tensor = torch.tensor(
        #         pose["food_rubbish03"], dtype=torch.float32, device=self.device
        #     )
        #     self.food_rubbish03.write_root_state_to_sim(food_rubbish03_tensor, env_ids=env_ids)
        

    def _randomize_texture(self):
        """Randomize textures for rubbish scene."""
        folder = os.getcwd() + "/Assets/textures/surface/seen"
        min_id = 0
        max_id = 999
        # TODO: 更新为垃圾分类场景的实际shader路径
        shader_paths = [
            # 示例路径，需要根据实际场景修改
            # "/World/Rubbish/DesktopDustpan/Looks/material/Shader",
            # "/World/Scene/Floor/Looks/material/Shader",
        ]
        if not folder or not os.path.exists(folder):
            print(f"[Reset][Warn] Texture folder not found: {folder}")
            return
        stage = self.scene.stage
        success = False
        for shader_path in shader_paths:
            shader_prim = stage.GetPrimAtPath(shader_path)
            if not shader_prim.IsValid():
                print(f"[Reset][Warn] Shader prim not found at {shader_path}")
                continue
            shader = UsdShade.Shader(shader_prim)
            idx = random.randint(min_id, max_id)
            tex_path = os.path.join(folder, f"{idx}.png")
            tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
            if not tex_input:
                print(
                    f"[Reset][Warn] No texture input found on shader at {shader_path}"
                )
                continue
            tex_input.Set(Sdf.AssetPath(tex_path))
            print(f"[Reset][Info] Applied texture {tex_path} to {shader_path}")
            success = True
        if not success:
            print("[Reset][Warn] No valid shader prims found, nothing randomized")

    def _randomize_light(self):  ##!!! 需要更改 todo
        """Randomize DomeLight attributes based on config."""
        prim_path = "/World/Scene/RangeHood017/RectLight_01"
        intensity_range = [500, 8000]
        color_range = [0.0, 1.0]

        stage = self.scene.stage
        light_prim = stage.GetPrimAtPath(prim_path)
        if not light_prim.IsValid():
            print(f"[Reset][Warn] Light prim not found at {prim_path}")
            return
        intensity = random.uniform(*intensity_range)
        color = tuple(random.uniform(color_range[0], color_range[1]) for _ in range(3))

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)
