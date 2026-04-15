from __future__ import annotations
import torch
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
from pxr import Usd, UsdShade, Sdf, UsdGeom, Gf
import random

from .tableware_cfg import TablewareEnvCfg
from lehome.utils.success_checker import success_checker_bowlinplate
from lehome.devices.action_process import preprocess_device_action
import numpy as np
import os

import omni.usd

from lehome.utils.rendering import apply_default_render_settings, setup_default_lighting


class TablewareEnv(DirectRLEnv):
    cfg: TablewareEnvCfg

    def __init__(self, cfg: TablewareEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        # Make viewport/rendering defaults consistent across environments.
        apply_default_render_settings()

        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        # self.top_camera2 = TiledCamera(self.cfg.top_camera2)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)

        # 使用配置中的场景路径（可通过 --scene_usd 参数覆盖）
        print(f"[DEBUG TablewareEnv._setup_scene] Loading scene from: {self.cfg.path_scene}")
        scene_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        scene_cfg.func(
            "/World/Scene",
            scene_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # 从 GarmentEnvCfg 中的 object_A / object_B 组随机选择一个配置
        object_A_names = [
            "b_cups",
            "bowl1",
            "bowl2",
            "mug",
            "pitcher_base",
            "coffeecup028",
            "cup002",
            "cup012",
            "cup030",
        ]
        self.object_B_names = [
            # "plate",
            # "plate_scale0_8",
            "plate_scale1_2",
        ]

        # object_A is selected from config so replay scripts can control it.
        self.object_A_name = getattr(self.cfg, "object_A_name", "b_cups")
        if self.object_A_name not in object_A_names:
            print(
                f"[Warn TablewareEnv._setup_scene] Invalid cfg.object_A_name={self.object_A_name}, "
                f"fallback to b_cups. Valid options: {object_A_names}"
            )
            self.object_A_name = "b_cups"

        self.object_B_map: dict[str, RigidObject] = {}
        for name in self.object_B_names:
            obj = RigidObject(getattr(self.cfg, name))
            self.object_B_map[name] = obj
            self.scene.rigid_objects[name] = obj
        self.object_B_name = random.choice(self.object_B_names)
        self.object_B = self.object_B_map[self.object_B_name]
        self.object_A = RigidObject(getattr(self.cfg, self.object_A_name))
        self.scene.rigid_objects["object_A"] = self.object_A

        # rigidapple 暂时禁用W
        # self.rigidapple = RigidObject(self.cfg.rigidapple)
        # self.scene.rigid_objects["rigidapple"] = self.rigidapple

 
        # self.object.config = {
        #     # 其他配置...
        #     "texture_randomization": {
        #         "enable": True,
        #         "folder": "Assets/textures/surface/seen",
        #         "min_id": 1,
        #         "max_id": 10000,
        #         "prim_path": "/World/Scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7",
        #     },
        #     "ligh_randomization":{
        #         "enable":True,
        #         "prim_path":"World/Light",
        #         "intensity_range":[500,3000],
        #         "color_range":[0.0,1.0]
        #     }
        # }


        # self.texture_cfg = self.object.config.get("texture_randomization", {})
        # self.light_cfg = self.object.config.get("light_randomization", {})
        self.texture_cfg = {
                "enable": True,
                "folder": "Assets/textures/surface/seen",
                "min_id": 1,
                "max_id": 10000,
                "prim_path": "/World/Scene/Table038/looks/M_Table038a/UsdPreviewSurface/________7/________7",
            }
        self.light_cfg = {
                "enable": False,
                "prim_path": "/World/Light",
                "intensity_range": [500, 3000],
                "color_range": [0.0, 1.0]
            }
        # self.plate_cfg ={
        #         "enable":False,
        #         "folder":"Assets/LW_Loft/Loft/Plate017/texture/",
        #         "prim_path": "/World/Object/rigidplate/Looks/MI_Plate017_001/Shader"
        # }
        # self.bowl_cfg ={
        #         "enable":False,
        #         "folder":"Assets/LW_Loft/Loft/Bowl016/texture/",
        #         "prim_path": "/World/Object/rigidbowl/Looks/MI_Bowl016/Shader"
        # }
        # self.apple_cfg ={
        #         "enable":False,
        #         "folder":"/home/shu/code/lehome/Assets/cosmos_assets/apple",
        #         "prim_path": "/World/Object/rigidapple/Looks/apple/Shader"
        # }


        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["top_camera"] = self.top_camera
        # self.scene.sensors["top_camera2"] = self.top_camera2
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        # add lights
        # 增加Default模拟光源
        setup_default_lighting()
        # light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
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
        success = success_checker_bowlinplate(self.object_A, self.object_B)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success
    

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES   
        super()._reset_idx(env_ids)

        # Replay mode: disable all reset-time randomization for strict replay.
        # This flag can be set externally (e.g. by replay scripts) via `env.replay_mode = True`.
        replay_mode = getattr(self, "replay_mode", False)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        # ===== 全局 XY 平移：对 robot / object_A / object_B 施加相同的 (x, y) 偏移 =====
        # 这样三者的相对位置保持不变，但整体在桌面上随机平移。
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        global_xy = torch.zeros(len(env_ids), 2, device=robot_root_state.device)
        if not replay_mode:
            global_xy[:, 0].uniform_(-0.1, 0.1)  # 仅在 x 轴平移，y 保持不动
        
        # [DEBUG] Check randomization status
        print(f"[DEBUG TablewareEnv._reset_idx] replay_mode={replay_mode}")
        print(f"[DEBUG TablewareEnv._reset_idx] global_xy (offset)={global_xy[0].cpu().numpy()}")

        # [DEBUG] Check randomization status
        print(f"[DEBUG TablewareEnv._reset_idx] replay_mode={replay_mode}")
        print(f"[DEBUG TablewareEnv._reset_idx] global_xy (offset)={global_xy[0].cpu().numpy()}")

        robot_root_state[..., :2] += global_xy
        self.robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids)
        
        # ========================================================================
        # 关键修复：确保关节位置在设置根状态后仍然正确
        # ========================================================================
        # 问题1：write_root_state_to_sim() 可能会重置关节位置
        # 问题2：如果只设置当前位置，ImplicitActuator会驱动关节到默认目标（通常是0）
        # 解决：在设置根状态后，同时设置当前位置和目标位置
        # ========================================================================
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        
        # 1. 设置当前关节位置
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        
        # 2. ⭐ 设置关节目标位置（告诉 ImplicitActuator 保持在这个位置）
        self.robot.set_joint_position_target(joint_pos)
        
        # 记录本次随机后的机器人位姿，便于后续 get/set
        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        # 随机选择本回合的 object_B
        if not replay_mode:
            self.object_B_name = random.choice(self.object_B_names)
            self.object_B = self.object_B_map[self.object_B_name]

        # 将未选中的盘子移出场景，避免干扰
        for name, obj in self.object_B_map.items():
            if name == self.object_B_name:
                continue
            off_state = obj.data.default_root_state[env_ids].clone()
            off_state[..., :3] = torch.tensor([0.0, 0.0, -10.0], device=off_state.device)
            obj.write_root_state_to_sim(off_state, env_ids=env_ids)

        object_A_pos = self.object_A.data.default_root_state[env_ids].clone()
        object_B_pos = self.object_B.data.default_root_state[env_ids].clone()
        object_A_pos[..., :2] += global_xy  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        object_B_pos[..., :2] += global_xy  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # rigidapple_pos = self.rigidapple.data.default_root_state[env_ids].clone()
        # 在全局随机平移的基础上，再对 object_A / object_B 叠加各自的小范围随机 (x, y) 噪声
        if replay_mode:
            rand_object_A = torch.zeros(len(env_ids), 2, device=object_A_pos.device)
            rand_object_B = torch.zeros(len(env_ids), 2, device=object_B_pos.device)
        else:
            rand_object_A = torch.empty(len(env_ids), 2, device=object_A_pos.device).uniform_(
                -0.02, 0.02
            )
            rand_object_B = torch.empty(len(env_ids), 2, device=object_B_pos.device).uniform_(
                -0.03, 0.03
            )
        # rand_apple = torch.empty(len(env_ids), 2, device="cuda").uniform_(-0.02, 0.02)
        # plate (root state)
        random_object_A_pos = object_A_pos.clone()
        random_object_A_pos[..., :2] += rand_object_A  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        self.object_A.write_root_state_to_sim(random_object_A_pos, env_ids=env_ids)
        random_object_B_pos = object_B_pos.clone()
        random_object_B_pos[..., :2] += rand_object_B  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        self.object_B.write_root_state_to_sim(random_object_B_pos, env_ids=env_ids)
        # random_apple_pos = rigidapple_pos.clone()
        # random_apple_pos[..., :2] += rand_apple  
        # self.rigidapple.write_root_state_to_sim(random_apple_pos, env_ids=env_ids)


        # object_A_state = self.object_A.data.default_root_state
        # self.object_A_reset_state = np.array(
        #     object_A_state.cpu().detach(), dtype=np.float32
        # )


        object_A_state = random_object_A_pos
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )

        object_B_state = random_object_B_pos
        self.object_B_reset_state = np.array(
            object_B_state.cpu().detach(), dtype=np.float32
        )

        
        # Apply randomization if enabled in config
        if (not replay_mode) and self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()

        if (not replay_mode) and self.light_cfg.get("enable", False):
            self._randomize_light()
        
        # if self.bowl_cfg.get("enable", False):
        #     self._randomize_bowl_texture()     

        # if self.plate_cfg.get("enable", False):
        #     self._randomize_plate_texture()        



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


    # def _randomize_plate_texture(self):
    #     """Randomize  plate and  texture based on config."""
    #     if not self.plate_cfg.get("enable", False):
    #         return

    #     folder = self.plate_cfg.get("folder", "")
    #     if not os.path.isabs(folder):
    #         folder = os.path.join(os.getcwd(), folder)

    #     # 🔹1. 列出该文件夹下所有图片文件（只要后缀是 png/jpg/jpeg 就行）
    #     all_files = os.listdir(folder)
    #     tex_files = [
    #         f for f in all_files
    #         if f.lower().endswith((".png", ".jpg", ".jpeg"))
    #     ]

    #     if not tex_files:
    #         print(f"[Reset][Warn] No texture files found in folder: {folder}")
    #         return

    #     # 🔹2. 随机选一个文件名
    #     filename = random.choice(tex_files)
    #     tex_path = os.path.join(folder, filename)

    #     shader_path = self.plate_cfg.get("prim_path", "")

    #     if not folder or not os.path.exists(folder):
    #         print(f"[Reset][Warn] Texture folder not found: {folder}")
    #         return
    #     if not shader_path:
    #         print("[Reset][Warn] No prim_path provided for texture randomization")
    #         return


    #     stage = self.scene.stage
    #     shader_prim = stage.GetPrimAtPath(shader_path)
    #     if not shader_prim.IsValid():
    #         print(f"[Reset][Warn] Shader prim not found at {shader_path}")
    #         return

    #     shader = UsdShade.Shader(shader_prim)
    #     tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
    #     if not tex_input:
    #         print("[Reset][Warn] No texture input found on shader")
    #         return

    #     tex_input.Set(Sdf.AssetPath(tex_path))
    #     # print(f"[Reset] Texture randomized -> {tex_path}")


    # def _randomize_bowl_texture(self):
    #     """Randomize bowl and  texture based on config."""
    #     if not self.bowl_cfg.get("enable", False):
    #         return

    #     folder = self.bowl_cfg.get("folder", "")
    #     if not os.path.isabs(folder):
    #         folder = os.path.join(os.getcwd(), folder)

    #     # 🔹1. 列出该文件夹下所有图片文件（只要后缀是 png/jpg/jpeg 就行）
    #     all_files = os.listdir(folder)
    #     tex_files = [
    #         f for f in all_files
    #         if f.lower().endswith((".png", ".jpg", ".jpeg"))
    #     ]

    #     if not tex_files:
    #         print(f"[Reset][Warn] No texture files found in folder: {folder}")
    #         return

    #     # 🔹2. 随机选一个文件名
    #     filename = random.choice(tex_files)
    #     tex_path = os.path.join(folder, filename)

    #     shader_path = self.bowl_cfg.get("prim_path", "")

    #     if not folder or not os.path.exists(folder):
    #         print(f"[Reset][Warn] Texture folder not found: {folder}")
    #         return
    #     if not shader_path:
    #         print("[Reset][Warn] No prim_path provided for texture randomization")
    #         return


    #     stage = self.scene.stage
    #     shader_prim = stage.GetPrimAtPath(shader_path)
    #     if not shader_prim.IsValid():
    #         print(f"[Reset][Warn] Shader prim not found at {shader_path}")
    #         return

    #     shader = UsdShade.Shader(shader_prim)
    #     tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
    #     if not tex_input:
    #         print("[Reset][Warn] No texture input found on shader")
    #         return

    #     tex_input.Set(Sdf.AssetPath(tex_path))
    #     # print(f"[Reset] Texture randomized -> {tex_path}")

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
        robot_state = self.robot.data.default_root_state
        self.robot_reset_state = np.array(
            robot_state.cpu().detach(), dtype=np.float32
        )
        object_A_state = self.object_A.data.default_root_state
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )
        object_B_state = self.object_B.data.default_root_state
        self.object_B_reset_state = np.array(
            object_B_state.cpu().detach(), dtype=np.float32
        )
        # apple_state = self.rigidapple.data.default_root_state
        # self.apple_reset_state = np.array(
        #     apple_state.cpu().detach(), dtype=np.float32        
        # )
        

    def get_all_pose(self):
        # rigidapple 暂不参与 pose 记录
        return {
            "robot": self.robot_reset_state,
            "object_A": self.object_A_reset_state,
            "object_B": self.object_B_reset_state,
            # "rigid_apple": self.apple_reset_state,
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
        pose_tensor = torch.tensor(
            pose["object_B"], dtype=torch.float32, device=self.device
        )
        self.object_B.write_root_state_to_sim(pose_tensor, env_ids=env_ids)
        # pose_tensor = torch.tensor(
        #     pose["rigid_apple"], dtype=torch.float32, device=self.device
        # )
        # self.rigidapple.write_root_state_to_sim(pose_tensor, env_ids=env_ids)