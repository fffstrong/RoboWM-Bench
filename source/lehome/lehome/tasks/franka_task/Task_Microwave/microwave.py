from __future__ import annotations
import torch
import time
from dataclasses import MISSING
from typing import Any, Dict, List

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera
from pxr import Usd, UsdShade, Sdf, Gf, UsdPhysics
import random

from .microwave_cfg import MicrowaveEnvCfg
from lehome.devices.action_process import preprocess_device_action
import numpy as np
import os

import omni.usd

from lehome.utils.rendering import apply_default_render_settings

class MicrowaveEnv(DirectRLEnv):
    cfg: MicrowaveEnvCfg

    def __init__(self, cfg: MicrowaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        # Set joint_pos reference after robot is created in _setup_scene (called by super().__init__)
        self.joint_pos = self.robot.data.joint_pos
        # Note: microwave_joint_pos will be set after microwave is created in _setup_scene
        # Real-time pose printing (throttled to avoid spamming the console)
        self._last_pose_print_time: float = 0.0
        self._pose_print_every_s: float = 1.0  # set smaller (e.g. 0.05) if you want near-per-step prints

    def _setup_scene(self):
        # Make viewport/rendering defaults consistent across environments.
        apply_default_render_settings()

        self.robot = Articulation(self.cfg.robot)
        # self.top_camera = TiledCamera(self.cfg.top_camera)
        # self.top_camera_2 = TiledCamera(self.cfg.top_camera_2)
        # self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.behind_camera = TiledCamera(self.cfg.behind_camera)
        self.front_camera = TiledCamera(self.cfg.front_camera)

        from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
        cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_WITH_ORANGE_USD_PATH}")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Microwave is spawned from cfg.microwave (teleop_record will rebuild env to switch USD).
        self.microwave = Articulation(self.cfg.microwave)

        # 设置微波炉关节的 damping 参数
        self._set_microwave_joint_damping()

        # Ensure robot starts from default joint pose immediately (task-local fix, avoids global script edits).
        try:
            env_ids = self.robot._ALL_INDICES
            joint_pos = self.robot.data.default_joint_pos[env_ids]
            self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
            self.robot.set_joint_position_target(joint_pos)
        except Exception:
            pass

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # Expose microwave under stable key "microwave"
        self.scene.articulations["microwave"] = self.microwave
        # self.scene.sensors["top_camera"] = self.top_camera
        # self.scene.sensors["top_camera_2"] = self.top_camera_2
        # self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.sensors["behind_camera"] = self.behind_camera
        self.scene.sensors["front_camera"] = self.front_camera
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=800, color=(0.75, 0.75, 0.75))
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
        # Handle case where self.actions might not be initialized yet (before first step)
        if hasattr(self, 'actions'):
            action = self.actions.squeeze(0)
        else:
            # Return zero action if actions not yet set
            action = torch.zeros(self.cfg.action_space, device=self.device)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)

        microwave_joint_pos_data = self.microwave.data.joint_pos.clone()
        microwave_joint_pos = microwave_joint_pos_data.squeeze(0)

        # --- Real-time print (throttled) ---
        now = time.time()
        if (now - self._last_pose_print_time) >= self._pose_print_every_s:
            try:
                mw_j_np = microwave_joint_pos.detach().cpu().numpy()
                print(f"[Pose] microwave_joints={mw_j_np}")
            except Exception:
                pass
            self._last_pose_print_time = now

        # top_camera_rgb = self.top_camera.data.output["rgb"]
        # top_camera_rgb_2 = self.top_camera_2.data.output["rgb"]
        # top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        # top_camera_depth_2 = self.top_camera_2.data.output["depth"].squeeze()
        # wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        behind_camera_rgb = self.behind_camera.data.output["rgb"]
        front_camera_rgb = self.front_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            # "observation.microwave_joint_pos": microwave_joint_pos.cpu().detach().numpy(),
            # "observation.images.top_rgb": top_camera_rgb.cpu()
            # .detach()
            # .numpy()
            # .squeeze(),
            # "observation.images.top_rgb_2": top_camera_rgb_2.cpu()
            # .detach()
            # .numpy()
            # .squeeze(),
            # "observation.images.wrist_rgb": wrist_camera_rgb.cpu()
            # .detach()
            # .numpy()
            # .squeeze(),
            # "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
            # "observation.top_depth_2": top_camera_depth_2.cpu().detach().numpy(),
            "observation.images.behind_rgb": behind_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.front_rgb": front_camera_rgb.cpu()
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

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Success: the first microwave joint has moved away from 0 by a small threshold.
        # Note: different microwaves can have different joint DOF counts, so we guard for empty joints.
        success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        joint_pos = self.microwave.data.joint_pos  # (num_envs, num_dof)
        if joint_pos.numel() > 0 and joint_pos.shape[-1] > 0:
            success_tensor = torch.abs(joint_pos[:, 0]) > 0.01
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # 重置机器人关节位置
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        # ===== 全局 XY 平移：对 robot / microwave 施加相同的 (x, y) 偏移 =====
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
        # 记录本次随机后的机器人位姿
        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        # 重置微波炉位置和铰链关节
        microwave_root_state = self.microwave.data.default_root_state[env_ids].clone()
        self.microwave.write_root_state_to_sim(microwave_root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        
        # 1. 设置当前关节位置
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        
        # 2. ⭐ 设置关节目标位置（告诉 ImplicitActuator 保持在这个位置）
        self.robot.set_joint_position_target(joint_pos)
        
        # 记录本次随机后的机器人位姿，便于后续 get/set
        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        # 重置微波炉的铰链关节位置（关闭状态）
        # 获取微波炉的默认关节位置
        microwave_joint_pos = self.microwave.data.default_joint_pos[env_ids]
        # 设置所有关节为 0（关闭状态）
        microwave_joint_pos.fill_(0.0)
        self.microwave.write_joint_position_to_sim(
            microwave_joint_pos, joint_ids=None, env_ids=env_ids
        )


        # 确保微波炉关节可以自由移动 - 设置关节速度为0但不锁定
        microwave_joint_vel = self.microwave.data.default_joint_vel[env_ids]
        microwave_joint_vel.fill_(0.0)
        self.microwave.write_joint_velocity_to_sim(
            microwave_joint_vel, joint_ids=None, env_ids=env_ids
        )

        # 记录微波炉的初始状态
        self.microwave_reset_state = np.array(
            microwave_root_state.cpu().detach(), dtype=np.float32
        )
        # 记录微波炉的关节状态
        self.microwave_joint_reset_state = np.array(
            microwave_joint_pos.cpu().detach(), dtype=np.float32
        )

        # # Apply randomization if enabled in config
        # if hasattr(self, "texture_cfg") and self.texture_cfg.get("enable", False):
        #     self._randomize_table038_texture()

        # if hasattr(self, "light_cfg") and self.light_cfg.get("enable", False):
        #     self._randomize_light()

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

    def _set_microwave_joint_damping(self):
        """设置微波炉关节的 damping 参数"""
        stage = self.scene.stage
        microwave_prim_path = self.microwave.cfg.prim_path
        
        # 获取微波炉的根 prim
        microwave_prim = stage.GetPrimAtPath(microwave_prim_path)
        if not microwave_prim.IsValid():
            return
        
        # 铰链参数调整
        door_damping_value = 0.00015
        door_joint_friction_value = 0.0
        door_drive_stiffness_value = 0.0
        door_drive_max_force_value = 0.005
        joints_found = []
        
        # 递归查找所有关节
        def set_joint_damping(prim, damping_value):
            """递归设置关节的 damping"""
            # 检查是否是关节
            joint = UsdPhysics.Joint(prim)
            if joint:
                joints_found.append(prim.GetPath())
                joint_type = prim.GetTypeName()
                # keep silent by default (this function may run on every env creation)
                
                # 判断关节类型，选择对应的 drive 类型
                # 旋转关节使用 "angular"，线性关节使用 "linear"
                drive_type = "angular"  # 默认使用 angular
                if "Prismatic" in joint_type or "Linear" in joint_type:
                    drive_type = "linear"
                
                try:
                    # Skip fixed joints (no meaningful "feel" to tune)
                    if "FixedJoint" in joint_type:
                        return

                    # Only tune the microwave DOOR joint by default.
                    # This prevents making other joints (e.g., tray/turntable) feel unexpectedly heavy.
                    prim_path_str = str(prim.GetPath())
                    if "microjoint" not in prim_path_str:
                        return

                    # 先应用 DriveAPI（如果已存在则获取，不存在则创建）
                    drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)

                    # ---- Make the joint as passive/free as possible ----
                    # For articulated props (microwave door / tray), we typically don't want servo effects.
                    # Non-zero stiffness/maxForce can make the joint feel "heavy" even if damping is tiny.
                    try:
                        stiff_attr = drive.GetStiffnessAttr()
                        if stiff_attr:
                            prev = stiff_attr.Get()
                            stiff_attr.Set(door_drive_stiffness_value)
                            print(f"[Microwave]   - Drive stiffness: {prev} -> {door_drive_stiffness_value}")
                    except Exception:
                        pass

                    try:
                        maxf_attr = drive.GetMaxForceAttr()
                        if maxf_attr:
                            prev = maxf_attr.Get()
                            maxf_attr.Set(door_drive_max_force_value)
                            print(f"[Microwave]   - Drive maxForce: {prev} -> {door_drive_max_force_value}")
                    except Exception:
                        pass

                    # ---- Try to reduce PhysX joint friction (Coulomb/static friction) ----
                    # This is often the main reason for "sticky" feeling without rebound.
                    try:
                        fr_attr = prim.GetAttribute("physxJoint:friction")
                        if fr_attr and fr_attr.IsValid():
                            prev = fr_attr.Get()
                            fr_attr.Set(door_joint_friction_value)
                            print(f"[Microwave]   - Joint friction: {prev} -> {door_joint_friction_value}")
                    except Exception:
                        pass
                    
                    # 设置 damping 属性
                    damping_attr = drive.GetDampingAttr()
                    if damping_attr:
                        damping_attr.Set(door_damping_value)
                    else:
                        # 如果没有该属性，创建它
                        damping_attr = drive.CreateDampingAttr(door_damping_value)
                        
                except Exception:
                    pass
            
            # 递归处理子 prim
            for child in prim.GetChildren():
                set_joint_damping(child, damping_value)
        
        # Walk and tune joints (only door joint is affected by the filters above)
        set_joint_damping(microwave_prim, door_damping_value)
        
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
        microwave_state = self.microwave.data.default_root_state
        self.microwave_reset_state = np.array(
            microwave_state.cpu().detach(), dtype=np.float32
        )
        # 保存微波炉的关节状态
        microwave_joint_state = self.microwave.data.default_joint_pos
        self.microwave_joint_reset_state = np.array(
            microwave_joint_state.cpu().detach(), dtype=np.float32
        )

    def get_all_pose(self):
        """返回机器人和微波炉的当前姿态，用于记录。"""
        # 确保在 get_all_pose 之前已经初始化了 obs
        if not hasattr(self, 'robot_reset_state'):
            self.initialize_obs()
        return {
            "robot": self.robot_reset_state,
            "microwave_root": self.microwave_reset_state,
            "microwave_joint": self.microwave_joint_reset_state,
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        """设置机器人和微波炉的姿态，用于从记录中恢复。"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # 设置机器人姿态
        if pose and "robot" in pose and pose["robot"] is not None:
            robot_tensor = torch.tensor(
                pose["robot"], dtype=torch.float32, device=self.device
            )
            self.robot.write_root_state_to_sim(robot_tensor, env_ids=env_ids)
        
        # 设置微波炉根状态
        if pose and "microwave_root" in pose and pose["microwave_root"] is not None:
            microwave_tensor = torch.tensor(
                pose["microwave_root"], dtype=torch.float32, device=self.device
            )
            self.microwave.write_root_state_to_sim(microwave_tensor, env_ids=env_ids)
        
        # 设置微波炉关节状态（优先使用记录值；若没有则回到关闭状态 0）
        if pose and "microwave_joint" in pose and pose["microwave_joint"] is not None:
            microwave_joint_tensor = torch.tensor(
                pose["microwave_joint"], dtype=torch.float32, device=self.device
            )
            # Ensure 2D shape (num_envs, num_joints)
            if microwave_joint_tensor.ndim == 1:
                microwave_joint_tensor = microwave_joint_tensor.unsqueeze(0)
            
            # Handle joint count mismatch (record vs sim)
            sim_num_joints = self.microwave.data.joint_pos.shape[1]
            if microwave_joint_tensor.shape[1] != sim_num_joints:
                print(f"[Warn] Microwave joint mismatch in set_all_pose: record={microwave_joint_tensor.shape[1]}, sim={sim_num_joints}. Truncating.")
                microwave_joint_tensor = microwave_joint_tensor[:, :sim_num_joints]

            self.microwave.write_joint_position_to_sim(
                microwave_joint_tensor, joint_ids=None, env_ids=env_ids
            )
        else:
            microwave_joint_pos = self.microwave.data.default_joint_pos[env_ids]
            microwave_joint_pos.fill_(0.0)
            self.microwave.write_joint_position_to_sim(
                microwave_joint_pos, joint_ids=None, env_ids=env_ids
            )

