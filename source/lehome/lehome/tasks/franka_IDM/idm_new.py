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

from .idm_cfg_new import IDMEnvCfg
from lehome.utils.success_checker import success_checker_bowlinplate
from lehome.assets.scenes.kitchen import KITCHEN_NOTABLE_USD_PATH
from lehome.devices.action_process import preprocess_device_action
# from lehome.assets.object.Garment import GarmentObject
from lehome.assets.object.fluid import BowlObject
from omegaconf import OmegaConf
import numpy as np
import os

import numpy as np
import omni.usd
from isaaclab.sensors import TiledCameraCfg

def assign_id():
    if random.random() < 0.4:  # 40% 的概率
        return 0
    else:  # 60% 的概率均匀分布在 1-20
        return random.randint(1, 20)

def rotation_matrix(axis, angle):
    """
    生成绕指定轴旋转的旋转矩阵
    :param axis: 旋转轴 ('x', 'y', 'z')
    :param angle: 旋转角度（度）
    :return: 旋转矩阵 (3x3)
    """
    angle = np.radians(angle)  # 将角度转换为弧度
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

def quaternion_to_rotation_matrix(q):
    """
    将四元数 (w, x, y, z) 转换为旋转矩阵
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (w, x, y, z) 格式
    """
    trace = np.trace(R)
    if trace > 0:
        w = np.sqrt(1.0 + trace) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) / 2.0
            w = (R[2, 1] - R[1, 2]) / (4.0 * x)
            y = (R[0, 1] + R[1, 0]) / (4.0 * x)
            z = (R[0, 2] + R[2, 0]) / (4.0 * x)
        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) / 2.0
            w = (R[0, 2] - R[2, 0]) / (4.0 * y)
            x = (R[0, 1] + R[1, 0]) / (4.0 * y)
            z = (R[1, 2] + R[2, 1]) / (4.0 * y)
        else:
            z = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) / 2.0
            w = (R[1, 0] - R[0, 1]) / (4.0 * z)
            x = (R[0, 2] + R[2, 0]) / (4.0 * z)
            y = (R[1, 2] + R[2, 1]) / (4.0 * z)
    
    return np.array([w, x, y, z])

def random_rotation_matrix(max_angle_deg):
    """
    生成一个随机旋转矩阵，旋转角度在 [0, max_angle_deg] 范围内
    """
    angle_x = random.uniform(-max_angle_deg, max_angle_deg)
    angle_y = random.uniform(-max_angle_deg, max_angle_deg)
    angle_z = random.uniform(-max_angle_deg, max_angle_deg)

    R_x = rotation_matrix('x', angle_x)
    R_y = rotation_matrix('y', angle_y)
    R_z = rotation_matrix('z', angle_z)

    # 旋转顺序：Z -> Y -> X
    return R_z @ R_y @ R_x


class IDMEnv(DirectRLEnv):
    cfg: IDMEnvCfg

    def __init__(self, cfg: IDMEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos

    def _setup_scene(self):
        # cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_NOTABLE_USD_PATH}")
        cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome_1/Assets/scenes/kitchen_with_orange/scene_notable.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        self.top_camera_list = []
        self.top_camera_0 = TiledCamera(self.cfg.top_camera)
        self.scene.sensors[f"top_camera_0"] = self.top_camera_0


        self.scores=0
        self.part=0
        self.full_marks=4


        self.robot = Articulation(self.cfg.robot)

        # self.object_A_name = "banana"
        # self.object_A = RigidObject(getattr(self.cfg, self.object_A_name))
        # self.scene.rigid_objects["object_A"] = self.object_A
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=800, color=(0.7, 0.7, 0.7))
        light_cfg.func("/World/Light", light_cfg)

        self.joint_num=13

        self.scores=0
        self.part=0
        self.full_marks=2

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
        # print("action:", action.cpu().detach().numpy())
        # print("joint_pos:", joint_pos.cpu().detach().numpy())
        top_camera_rgb = self.top_camera_0.data.output["rgb"]
        segmentation = self.top_camera_0.data.output.get("instance_id_segmentation_fast")
        segmentation_info = self.top_camera_0.data.info.get("instance_id_segmentation_fast")
        if segmentation is None or segmentation_info is None:
            segmentation = self.top_camera_0.data.output.get("instance_segmentation_fast")
            segmentation_info = self.top_camera_0.data.info.get("instance_segmentation_fast")

        if segmentation is not None and segmentation_info is not None:
            id_to_labels = segmentation_info.get("idToLabels", {})
            target_ids = [
                int(instance_id)
                for instance_id, prim_path in id_to_labels.items()
                if prim_path == "/World/Robot/Robot" or prim_path.startswith("/World/Robot/Robot/")
            ]

            if target_ids:
                seg_ids = torch.tensor(target_ids, device=segmentation.device, dtype=segmentation.dtype)
                seg_mask = torch.isin(segmentation, seg_ids)
            else:
                seg_mask = torch.zeros_like(segmentation, dtype=torch.bool)

            seg_mask = seg_mask.to(dtype=torch.bool)
            if seg_mask.shape[-1] == 1:
                seg_mask = seg_mask.expand_as(top_camera_rgb[..., :1]).expand_as(top_camera_rgb)
            masked_rgb = top_camera_rgb.clone()
            masked_rgb[~seg_mask] = 0
            top_camera_rgb = masked_rgb
        # top_camera2_rgb = self.top_camera2.data.output["rgb"]
        # wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
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

    def _reset_idx(self, env_ids: Sequence[int] | None,joint=None):

        base_pos = np.array([3.6 - 0.08763833, -2.6 - 0.74637488, 0.4 + 0.62755654])
        base_rot = (-0.47446683, 0.83054371, 0.2426624, -0.16184353)
        base_rot_matrix = quaternion_to_rotation_matrix(base_rot)

        # 60% 概率直接使用 base_pos 和 base_rot
        if random.random() < 1:
            new_pos = base_pos
            new_rot_quaternion = base_rot
        else:
            # 随机平移
            random_translation = np.array([
                random.uniform(-0.03, 0.03),  # x 方向随机移动 0-5cm
                random.uniform(-0.03, 0.03),  # y 方向随机移动 0-5cm
                random.uniform(-0.03, 0.03),  # z 方向随机移动 0-5cm
            ])
            new_pos = base_pos + random_translation

            # 随机旋转
            random_rotation = random_rotation_matrix(6)  # 随机旋转 0-10 度
            new_rot_matrix = random_rotation @ base_rot_matrix
            new_rot_quaternion = rotation_matrix_to_quaternion(new_rot_matrix)

        # 将位置和方向转换为 torch.Tensor
        positions = torch.tensor([new_pos], dtype=torch.float32)  # (1, 3)
        orientations = torch.tensor([new_rot_quaternion], dtype=torch.float32)  # (1, 4)

        # 设置摄像头的位姿
        self.top_camera_0.set_world_poses(
            positions=positions,
            orientations=orientations,  # wxyz
            env_ids=None,  # 对所有摄像头生效
        )

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if joint is not None:
            joint_pos = joint
        else:
            joint_pos = self.robot.data.default_joint_pos[env_ids]

        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )
        if joint is not None:
            robot_root_state = joint
        else:
            robot_root_state = self.robot.data.default_root_state[env_ids].clone()

        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        # object_A_pos = self.object_A.data.default_root_state[env_ids].clone()

        # self.object_A.write_root_state_to_sim(object_A_pos, env_ids=env_ids)

        # object_A_state = object_A_pos
        # self.object_A_reset_state = np.array(
        #     object_A_state.cpu().detach(), dtype=np.float32
        # )

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

    def get_all_pose(self):
        # rigidapple 暂不参与 pose 记录
        return {
            # 直接读取当前仿真状态，确保与 _reset_idx 的随机结果一致
            "robot": self.robot.data.root_state_w.clone().cpu().numpy(),
        }

    def set_all_pose(self, pose, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        pose_tensor = torch.tensor(
            pose["robot"], dtype=torch.float32, device=self.device
        )
        self.robot.write_root_state_to_sim(pose_tensor, env_ids=env_ids)
        

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
        prim_path = f"/World/Object/banana"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"[Warn] Prim not found for body at {prim_path}")
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
        success = False
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

