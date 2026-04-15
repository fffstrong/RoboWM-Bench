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
import json

from .pick_cfg import PickEnvCfg
from lehome.utils.success_checker import success_checker_pick
from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
from lehome.devices.action_process import preprocess_device_action
# from lehome.assets.object.Garment import GarmentObject
from lehome.assets.object.fluid import BowlObject
from omegaconf import OmegaConf
import numpy as np
import os
import omni.usd



class PickEnv(DirectRLEnv):
    cfg: PickEnvCfg

    def __init__(self, cfg: PickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):

        self.scores=0
        self.part=0
        self.full_marks=4


        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.top_camera2 = TiledCamera(self.cfg.top_camera2)

        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome_1/Assets/benchmark/scenes/benchmark_scene.usd")
        cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome_1/Assets/benchmark/scenes/benchmark_scene1.usd")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.object_A_names = ["white_low", "white_high", "banana", "brown_cup"]
        self.object_A_map: dict[str, RigidObject] = {}
        for name in self.object_A_names:
            obj = RigidObject(getattr(self.cfg, name))
            self.object_A_map[name] = obj
            self.scene.rigid_objects[name] = obj

        self.object_A_name = random.choice(self.object_A_names)
        self.object_A = self.object_A_map[self.object_A_name]

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
                "color_range": [0.0, 1.0]
            }


        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["top_camera2"] = self.top_camera2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=800, color=(0.7, 0.7, 0.7))
        light_cfg.func("/World/Light", light_cfg)

        self.ori_z=0
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
        # top_camera2_rgb = self.top_camera2.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        # wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            # "observation.top_depth": top_camera_depth.cpu().detach().numpy().copy(),
            # "observation.images.top_camera2_rgb": top_camera2_rgb.cpu()
            # .detach()
            # .numpy()
            # .squeeze(),
            # "observation.images.wrist_rgb": wrist_camera_rgb.cpu()
            # .detach()
            # .numpy()
            # .squeeze(),
            # "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
        }
        return observations

    # def _get_observations(self) -> dict:
    #     action = self.actions.squeeze(0)
    #     joint_pos = torch.cat(
    #         [self.joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
    #     )
    #     joint_pos = joint_pos.squeeze(0)
    #     # print("action:", action.cpu().detach().numpy())
    #     # print("joint_pos:", joint_pos.cpu().detach().numpy())
    #     top_camera_rgb = self.top_camera.data.output["rgb"]
    #     segmentation = self.top_camera.data.output.get("instance_id_segmentation_fast")
    #     segmentation_info = self.top_camera.data.info.get("instance_id_segmentation_fast")
    #     if segmentation is None or segmentation_info is None:
    #         segmentation = self.top_camera.data.output.get("instance_segmentation_fast")
    #         segmentation_info = self.top_camera.data.info.get("instance_segmentation_fast")

    #     if segmentation is not None and segmentation_info is not None:
    #         id_to_labels = segmentation_info.get("idToLabels", {})
    #         target_prefix = "/World/Robot/Robot"

    #         def _extract_prim_path(label_value):
    #             if isinstance(label_value, str):
    #                 return label_value
    #             if isinstance(label_value, dict):
    #                 for key in ("primPath", "path", "label"):
    #                     value = label_value.get(key)
    #                     if isinstance(value, str):
    #                         return value
    #             if isinstance(label_value, (list, tuple)):
    #                 for value in label_value:
    #                     if isinstance(value, str) and value.startswith("/"):
    #                         return value
    #             return None

    #         if segmentation.ndim >= 4 and segmentation.shape[-1] == 4:
    #             target_colors = []
    #             for instance_id, label_value in id_to_labels.items():
    #                 prim_path = _extract_prim_path(label_value)
    #                 if prim_path is None or not (prim_path == target_prefix or prim_path.startswith(target_prefix + "/")):
    #                     continue
    #                 if isinstance(instance_id, (list, tuple)) and len(instance_id) == 4:
    #                     target_colors.append([int(v) for v in instance_id])

    #             seg_mask = torch.zeros(segmentation.shape[:-1], dtype=torch.bool, device=segmentation.device)
    #             for color in target_colors:
    #                 color_tensor = torch.tensor(color, dtype=segmentation.dtype, device=segmentation.device).view(
    #                     *([1] * (segmentation.ndim - 1)), 4
    #                 )
    #                 seg_mask |= (segmentation == color_tensor).all(dim=-1)
    #         else:
    #             target_ids = []
    #             for instance_id, label_value in id_to_labels.items():
    #                 prim_path = _extract_prim_path(label_value)
    #                 if prim_path is None or not (prim_path == target_prefix or prim_path.startswith(target_prefix + "/")):
    #                     continue
    #                 if isinstance(instance_id, (int, np.integer)):
    #                     target_ids.append(int(instance_id))
    #                 elif isinstance(instance_id, str) and instance_id.strip().lstrip("-").isdigit():
    #                     target_ids.append(int(instance_id))

    #             if target_ids:
    #                 seg_ids = torch.tensor(target_ids, device=segmentation.device, dtype=segmentation.dtype)
    #                 seg_mask = torch.isin(segmentation, seg_ids)
    #             else:
    #                 seg_mask = torch.zeros_like(segmentation, dtype=torch.bool)

    #         seg_mask = seg_mask.to(dtype=torch.bool)
    #         if seg_mask.shape[-1] == 1:
    #             seg_mask = seg_mask.expand_as(top_camera_rgb[..., :1]).expand_as(top_camera_rgb)
    #         elif seg_mask.ndim == top_camera_rgb.ndim - 1:
    #             seg_mask = seg_mask.unsqueeze(-1).expand_as(top_camera_rgb)
    #         masked_rgb = top_camera_rgb.clone()
    #         masked_rgb[~seg_mask] = 0
    #         top_camera_rgb = masked_rgb
    #     # top_camera2_rgb = self.top_camera2.data.output["rgb"]
    #     # wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
    #     observations = {
    #         "action": action.cpu().detach().numpy(),
    #         "observation.state": joint_pos.cpu().detach().numpy(),
    #         "observation.images.top_rgb": top_camera_rgb.cpu()
    #         .detach()
    #         .numpy()
    #         .squeeze(),
    #     }
    #     return observations


    def _get_rewards(self) -> torch.Tensor:
        total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(
        self,
        env_ids: Sequence[int] | None,
        name: str = None,
        xyz: list = None,
        quat: list = None,
        name2: str = None,
        xyz2: list = None,
        quat2: list = None,
    ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        name = None
        xyz = None
        quat  = None
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()

        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )
        if name is not None:
            obj = self.object_A_map.get(name)
            if obj is None:
                print(f"[Reset][Warn] Object name '{name}' not found in object_A_map. Using random selection.")
                self.object_A_name = random.choice(self.object_A_names)
                self.object_A = self.object_A_map[self.object_A_name]
            else:
                self.object_A_name = name
                self.object_A = obj
        else:
            self.object_A_name = random.choice(self.object_A_names)
            self.object_A = self.object_A_map[self.object_A_name]

        for obj_name, obj in self.object_A_map.items():
            obj_state = obj.data.default_root_state[env_ids].clone()
            if obj is self.object_A and xyz is not None and quat is not None:
                
                obj_state[:, 0:3] = torch.tensor(xyz, device=obj_state.device)
                obj_state[:, 3:7] = torch.tensor(quat, device=obj_state.device)
                # self.ori_z= obj_state[0,2].item()
            # if name2 is not None and obj_name == name2 and xyz2 is not None and quat2 is not None:
            #     obj_state[:, 0:3] = torch.tensor(xyz2, device=obj_state.device)
            #     obj_state[:, 3:7] = torch.tensor(quat2, device=obj_state.device)
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
            if obj is self.object_A:
                self.object_A_reset_state = np.array(
                    obj_state.cpu().detach(), dtype=np.float32
                )

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        # 记录当前仿真中的 root_state（已包含 _reset_idx 随机平移/选物体），而不是 cfg 默认值
        robot_state = self.robot.data.root_state_w.clone()
        self.robot_reset_state = np.array(
            robot_state.cpu().detach(), dtype=np.float32
        )
        object_A_state = self.object_A.data.root_state_w.clone()
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )


    def get_all_pose(self):
        # rigidapple 暂不参与 pose 记录
        return {
            # 直接读取当前仿真状态，确保与 _reset_idx 的随机结果一致
            "robot": self.robot.data.root_state_w.clone().cpu().numpy(),
            "object_A": self.object_A.data.root_state_w.clone().cpu().numpy(),
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
    
    def get_rigid_body_dimensions(self):
        """从 USD 静态几何中读取 drawer 各 body 的局部包围盒，仅调用一次"""
        stage = omni.usd.get_context().get_stage()
        
        # 构造 prim 路径，假设 link prim 在 /World/Object/drawer 下
        # 注意：实际路径可能不同，需根据你的 USD 结构调整
        prim_path = f"/World/Object/b_cups"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"[Warn] Prim not found for body 'b_cups' at {prim_path}")
            # 使用默认小包围盒避免崩溃
            local_min = (-0.01, -0.01, -0.01)
            local_max = (0.01, 0.01, 0.01)
        else:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),  # 静态模型用 Default
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
                ignoreVisibility=True
            )
            bound = bbox_cache.ComputeLocalBound(prim)
            range_ = bound.GetRange()
            if range_.IsEmpty():
                print(f"[Warn] Empty bounding box for {prim_path}")
                local_min = (-0.01, -0.01, -0.01)
                local_max = (0.01, 0.01, 0.01)
            else:
                min_vec = range_.GetMin()
                max_vec = range_.GetMax()
                local_min = (min_vec[0], min_vec[1], min_vec[2])
                local_max = (max_vec[0], max_vec[1], max_vec[2])
            
        # 计算尺寸
        width = local_max[0] - local_min[0]  # X 方向
        height = local_max[1] - local_min[1]  # Y 方向
        depth = local_max[2] - local_min[2]  # Z 方向

        print(f"x: {width}, y: {height}, z: {depth}")
        return width, height, depth
    
    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = success_checker_pick(self.object_A, self.ori_z)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success
    
    def get_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.ori_z = self.object_A.data.root_pos_w[0, 2].item()
        return None
    
    def get_part_scores(self, env_ids: Sequence[int] | None = None):
        object_A_default = self.object_A.data.default_root_state[env_ids].clone()
        object_A_pos = self.object_A.data.root_pos_w[env_ids]   
        if self.part==0:
            #判断物体接触
            if 1:
                self.part+=1
                self.scores+=1
            return self.part, self.scores
        
        if self.part==1:
            #判断拿起
            if True:
                self.part+=1
                self.scores+=1
            return self.part, self.scores
        
        if self.part==2:
            #判断移动并对准
            success = False
            if success:
                self.part+=1
                self.scores+=1
            return self.part, self.scores
        
        if self.part==3:
            #判断放下
            success = False
            if success:
                self.part+=1
                self.scores+=1
            return self.part, self.scores
    
    def get_first_frame(self, env_ids: Sequence[int] | None = None,json_root: str = None,img_root: str = None):
        top_camera_rgb = self.top_camera.data.output["rgb"]
        if img_root is None:
            raise ValueError("img_root cannot be None")

        rgb_np = top_camera_rgb.detach().cpu().numpy()
        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]

        if rgb_np.dtype != np.uint8:
            if np.issubdtype(rgb_np.dtype, np.floating):
                if rgb_np.max() <= 1.0:
                    rgb_np = np.clip(rgb_np, 0.0, 1.0) * 255.0
                else:
                    rgb_np = np.clip(rgb_np, 0.0, 255.0)
            rgb_np = rgb_np.astype(np.uint8)

        os.makedirs(img_root, exist_ok=True)
        existing_indices = []
        for filename in os.listdir(img_root):
            stem, _ = os.path.splitext(filename)
            if stem.isdigit():
                existing_indices.append(int(stem))

        next_index = max(existing_indices) + 1 if existing_indices else 0
        image_path = os.path.join(img_root, f"{next_index}.png")

        from PIL import Image

        Image.fromarray(rgb_np).save(image_path)

        if json_root is not None:
            if str(json_root).endswith(".jsonl"):
                jsonl_path = json_root
                jsonl_dir = os.path.dirname(jsonl_path)
                if jsonl_dir:
                    os.makedirs(jsonl_dir, exist_ok=True)
            else:
                os.makedirs(json_root, exist_ok=True)
                jsonl_path = os.path.join(json_root, "pose.jsonl")

            root_state = self.object_A.data.root_state_w[0].detach().cpu()
            pose_record = {
                "name": self.object_A_name,
                "xyz": root_state[0:3].tolist(),
                "quat": root_state[3:7].tolist(),
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pose_record, ensure_ascii=False) + "\n")


