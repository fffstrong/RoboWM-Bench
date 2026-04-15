from __future__ import annotations
import os
import torch

# from dataclasses import MISSING
from typing import Any, Dict, List, Sequence
import random

from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.sensors import TiledCamera
from .burger_cfg import BurgerEnvCfg
from ...base.base_env import BaseEnv
from ...base.base_env_cfg import BaseEnvCfg
from lehome.devices.action_process import preprocess_device_action
import numpy as np
from lehome.utils.success_checker import success_checker_burger
from pxr import Usd, UsdShade, Sdf
import isaaclab.sim as sim_utils

class BurgerEnv(BaseEnv):
    """Environment inheriting from base LW_Loft environment with additional features."""

    cfg: BaseEnvCfg | BurgerEnvCfg

    def __init__(
        self,
        cfg: BaseEnvCfg | BurgerEnvCfg,
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
        from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
        cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_WITH_ORANGE_USD_PATH}")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.robot = Articulation(self.cfg.robot)

        self.top_camera = TiledCamera(self.cfg.top_camera)

        self.burger_beef = DeformableObject(self.cfg.burger_beef)
        self.burger_board = RigidObject(self.cfg.burger_board)
        self.burger_plate = RigidObject(self.cfg.burger_plate)
        self.burger_bread2 = RigidObject(self.cfg.burger_bread2)
        self.burger_cheese = DeformableObject(self.cfg.burger_cheese)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["burger_plate"] = self.burger_plate
        self.scene.rigid_objects["burger_bread2"] = self.burger_bread2

        self.scene.rigid_objects["burger_board"] = self.burger_board
        self.scene.deformable_objects["burger_beef"] = self.burger_beef
        self.scene.deformable_objects["burger_cheese"] = self.burger_cheese
        self.scene.sensors["top_camera"] = self.top_camera

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
        self.robot.set_joint_position_target(self.actions[:, :])

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
            # "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            # "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
        }
        return observations

    def get_obs(self, photo_dir: str,json_path:str):
        from pathlib import Path
        output_dir = Path(photo_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        max_idx = -1
        for path in output_dir.glob("frame_*.png"):
            try:
                idx = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            if idx > max_idx:
                max_idx = idx
        frame_idx = max_idx + 1
        image_path = output_dir / f"frame_{frame_idx:06d}.png"
        top_camera_rgb = self.top_camera.data.output["rgb"]
        rgb = top_camera_rgb.cpu().detach().numpy().squeeze()
        import imageio.v2 as imageio
        if rgb is not None:
            imageio.imwrite(image_path, rgb)
        object_A_name = "burger_beef"
        object_A_pos = self.burger_beef.data.root_pos_w[0]
        
        # 将物体信息写入JSONL文件
        import json
        from pathlib import Path
        
        # 准备数据
        pose_data = {
            "name": object_A_name,
            "xyz": object_A_pos[:3].tolist(),  # 前三维度的xyz
            "quat": [1.0, 0.0, 0.0,0.0]  # 固定四元数 (wxyz)
        }
        
        # 写入JSONL文件
        jsonl_path = Path(json_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pose_data, ensure_ascii=False) + "\n")   

    def _get_rewards(self) -> torch.Tensor:
        total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        beef_pos = self.burger_beef.data.root_pos_w  # deformable 中心
        plate_pos = self.burger_plate.data.root_pos_w  # rigid 中心
        success = success_checker_burger(beef_pos=beef_pos, plate_pos=plate_pos)
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
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        # defalut position
        burger_plate_pos = self.burger_plate.data.default_root_state[env_ids].clone()
        burger_bread2_pos = self.burger_bread2.data.default_root_state[env_ids].clone()
        burger_cheese_pos = self.burger_cheese.data.default_nodal_state_w[
            env_ids
        ].clone()
        burger_beef_pos = self.burger_beef.data.default_nodal_state_w[env_ids].clone()
        burger_board_pos = self.burger_board.data.default_root_state[env_ids].clone()

        joint_pos = self.robot.data.default_joint_pos[env_ids]


        # random position noise (x,y)
        # rand_vals = torch.empty(1, 1, 2, device="cuda").uniform_(-0.1, 0.1)
        rand_vals_1 = torch.empty(len(env_ids), 2, device="cuda").uniform_(-0.15, 0.15)

        # plate (root state)
        random_plate_pos = burger_plate_pos.clone()
        # random_plate_pos[..., :2] += rand_vals_1

        # bread2 (root state)
        random_bread2_pos = burger_bread2_pos.clone()
        # random_bread2_pos[..., :2] += rand_vals_1

        # cheese (nodal state)
        random_cheese_pos = burger_cheese_pos.clone()
        # random_cheese_pos[..., :2] += rand_vals_1

        # beef (nodal state)
        random_beef_pos = burger_beef_pos.clone()
        # random_beef_pos[..., :2] += rand_vals_1

        random_board_pos = burger_board_pos.clone()
        # random_board_pos[..., :2] += rand_vals_1

        # sim
        self.burger_plate.write_root_state_to_sim(random_plate_pos, env_ids=env_ids)
        self.burger_board.write_root_state_to_sim(random_board_pos, env_ids=env_ids)
        self.burger_bread2.write_root_state_to_sim(random_bread2_pos, env_ids=env_ids)
        self.burger_cheese.write_nodal_state_to_sim(random_cheese_pos, env_ids=env_ids)
        self.burger_beef.write_nodal_state_to_sim(random_beef_pos, env_ids=env_ids)

        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )


        # Save the beef state after reset
        burger_state = self.burger_beef.data.nodal_state_w
        self.burger_reset_state = np.array(
            burger_state.cpu().detach(), dtype=np.float32
        )
        # Apply randomization if enabled in config
        # if self.cfg.use_random:
        #     self._randomize_texture()
        #     self._randomize_light()

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        burger_state = self.burger_beef.data.nodal_state_w
        self.burger_reset_state = np.array(
            burger_state.cpu().detach(), dtype=np.float32
        )

    def get_all_pose(self):
        return {"burger_beef": self.burger_reset_state}

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        pose_tensor = torch.tensor(
            pose["burger_beef"], dtype=torch.float32, device=self.device
        )
        self.burger_beef.write_nodal_state_to_sim(pose_tensor, env_ids=env_ids)

    def _randomize_texture(self):
        """Randomize textures Looks and Kitchen_Cabinet002 independently."""
        folder = os.getcwd() + "/Assets/textures/surface/seen"
        min_id = 0
        max_id = 999
        shader_paths = [
            "/World/Burger/burger_board/Looks/material_Burger_ChoppingBlock_material/Shader",
            "/World/Scene/Kitchen_Cabinet002/Looks/_23/UsdPreviewSurface/Map__34/Map__34",
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

    def _randomize_light(self):
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
