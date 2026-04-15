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
from isaacsim.core.utils.prims import is_prim_path_valid
from .faucet_cfg import FaucetEnvCfg
from lehome.utils.success_checker import success_checker_pull
# from lehome.utils.success_checker import success_checker_push
from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
from lehome.devices.action_process import preprocess_device_action
# from lehome.assets.object.Garment import GarmentObject
from lehome.assets.object.fluid import BowlObject
from omegaconf import OmegaConf
import numpy as np
import os
import math

import omni.usd

from isaaclab.sensors import ContactSensor,ContactSensorCfg
import logging
from lehome.utils.logger import get_logger

# Create logger for this module with DEBUG level
logger = get_logger(__name__)

class FaucetEnv(DirectRLEnv):
    cfg: FaucetEnvCfg

    def __init__(self, cfg: FaucetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos
        # Debug: print a faucet joint angle from observations.
        self._debug_obs_step = 0
        self._debug_print_every = int(os.environ.get("LEHOME_DEBUG_PRINT_EVERY", "30"))
        # self._debug_faucet_joint_prim = os.environ.get(
        #     "LEHOME_DEBUG_FAUCET_JOINT_PRIM",
        #     "/World/Object/faucet/E_switch_1/joint_0",
        # )
        self._debug_faucet_joint_idx = 0

    def _resolve_faucet_joint_index(self, joint_prim_path: str) -> int | None:
        """Resolve articulation joint index from a USD joint prim path or name."""
        joint_names_raw = getattr(self.object_A, "joint_names", None)
        if not joint_names_raw:
            return None
        joint_names = [str(n) for n in joint_names_raw]

        # Try exact matches first.
        if joint_prim_path in joint_names:
            return joint_names.index(joint_prim_path)

        # Try common name variants derived from prim path.
        leaf = joint_prim_path.split("/")[-1] if joint_prim_path else ""
        tail2 = "/".join(joint_prim_path.split("/")[-2:]) if joint_prim_path else ""
        candidates = [leaf, tail2]
        for cand in candidates:
            if not cand:
                continue
            if cand in joint_names:
                return joint_names.index(cand)

        # Fallback: suffix / substring match (helps when names are prefixed).
        for idx, name in enumerate(joint_names):
            if leaf and name.endswith(leaf):
                return idx
            if tail2 and name.endswith(tail2):
                return idx
            if joint_prim_path and joint_prim_path in name:
                return idx
        return None

    def _setup_scene(self):


        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.top_camera2 = TiledCamera(self.cfg.top_camera2)


        cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_WITH_ORANGE_USD_PATH}")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )


        self.object_A_name = "faucet"
        self.object_A = Articulation(getattr(self.cfg, self.object_A_name))
        self.scene.articulations["object_A"] = self.object_A

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        self.scene.sensors["top_camera"] = self.top_camera
        # self.scene.sensors["top_camera2"] = self.top_camera2

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=900, color=(0.7, 0.7, 0.7))
        light_cfg.func("/World/Light", light_cfg)

        self.joint_num=9

        self.scores=0
        self.part=0
        self.full_marks=2

        
        # 使用配置对象初始化 ContactSensor
        self.contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        # self._debug_obs_step += 1
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)

        # Debug print faucet joint angle (radians and degrees).
        # if self._debug_print_every > 0 and (self._debug_obs_step % self._debug_print_every == 0):
        #     try:
        #         if self._debug_faucet_joint_idx is None:
        #             self._debug_faucet_joint_idx = self._resolve_faucet_joint_index(
        #                 self._debug_faucet_joint_prim
        #             )
        #             print("joint_index:",self._debug_faucet_joint_idx)
        #             if self._debug_faucet_joint_idx is None:
        #                 joint_names = [str(n) for n in getattr(self.object_A, "joint_names", [])]
        #                 logger.warning(
        #                     f"[FaucetEnv] Cannot resolve joint index for '{self._debug_faucet_joint_prim}'. "
        #                     f"Available joint_names: {joint_names}"
        #                 )
        #         if self._debug_faucet_joint_idx is not None:
        #             ang_rad = float(self.object_A.data.joint_pos[0, self._debug_faucet_joint_idx].item())
        #             ang_deg = ang_rad * 180.0 / math.pi
        #             logger.info(
        #                 f"[FaucetEnv] faucet joint '{self._debug_faucet_joint_prim}' "
        #                 f"(idx={self._debug_faucet_joint_idx}) pos={ang_rad:.6f} rad ({ang_deg:.2f} deg)"
        #             )
        #     except Exception as e:
        #         logger.warning(f"[FaucetEnv] Failed to debug-print faucet joint angle: {e}")

        top_camera_rgb = self.top_camera.data.output["rgb"]
        # top_camera2_rgb = self.top_camera2.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            # "observation.images.top_camera2_rgb": top_camera2_rgb.cpu()
            # .detach()
            # .numpy()
            # .squeeze(),
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
        object_A_name = self.object_A_name
        object_A_pos = self.object_A.data.root_pos_w[0]
        
        # 将物体信息写入JSONL文件
        import json
        from pathlib import Path
        
        # 准备数据
        pose_data = {
            "name": object_A_name,
            "xyz": object_A_pos[:3].tolist(),  # 前三维度的xyz
            "quat": [0.0, 0.0, 0.7071068,0.7071068]  # 固定四元数 (wxyz)
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

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        object_A_pos = self.object_A.data.default_root_state[env_ids].clone()

        self.object_A.write_root_state_to_sim(object_A_pos, env_ids=env_ids)


        object_A_state = object_A_pos
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )


    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        joint_names = getattr(self.object_A, "joint_names", [])
        print(joint_names)
        # 记录当前仿真中的 root_state（已包含 _reset_idx 随机平移/选物体），而不是 cfg 默认值
        robot_state = self.robot.data.default_root_state
        self.robot_reset_state = np.array(
            robot_state.cpu().detach(), dtype=np.float32
        )

        object_A_state = self.object_A.data.default_root_state.clone()
        self.object_A_reset_state = np.array(
            object_A_state.cpu().detach(), dtype=np.float32
        )
        object_A_joint_state = self.object_A.data.default_joint_pos
        self.object_A_joint_reset_state = np.array(
            object_A_joint_state.cpu().detach(), dtype=np.float32
        )


    def get_all_pose(self):
        # rigidapple 暂不参与 pose 记录
        if not hasattr(self, 'robot_reset_state'):
            self.initialize_obs()
        return {
            "robot": self.robot_reset_state,
            "object_A_root": self.object_A_reset_state,
            "object_A_joint": self.object_A_joint_reset_state,
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

        object_A_joint_value = None
        if pose:
            if "object_A_joint" in pose:
                object_A_joint_value = pose.get("object_A_joint")
            elif "object_A_joint" in pose:
                object_A_joint_value = pose.get("object_A_joint")

        if object_A_joint_value is not None:
            object_A_joint_tensor = torch.tensor(object_A_joint_value, dtype=torch.float32, device=self.device)
            if object_A_joint_tensor.dim() == 0:
                object_A_joint_tensor = object_A_joint_tensor.view(1, 1)
            elif object_A_joint_tensor.dim() == 1:
                object_A_joint_tensor = object_A_joint_tensor.unsqueeze(0)
            num_envs = len(env_ids)
            if object_A_joint_tensor.shape[0] == 1 and num_envs > 1:
                object_A_joint_tensor = object_A_joint_tensor.expand(num_envs, -1).clone()

            # Current asset joint count (can differ across drawer USDs / dataset versions)
            num_joints = int(self.object_A.data.joint_pos.shape[1])
            if object_A_joint_tensor.shape[1] > num_joints:
                extra = int(object_A_joint_tensor.shape[1] - num_joints)
                start_col = 0
                if extra == 1:
                    try:
                        import re
                        joint_names = getattr(self.object_A, "joint_names", [])
                        print(joint_names)
                        nums = []
                        for n in joint_names:
                            m = re.match(r"^joint_(\d+)$", str(n))
                            if m:
                                nums.append(int(m.group(1)))
                        # If current joints start at 1 (no joint_0), assume dataset includes joint_0
                        # and shift by one.
                        if len(nums) > 0 and min(nums) == 1:
                            start_col = 1
                    except Exception:
                        start_col = 0
                # Fallback: keep first N
                object_A_joint_tensor = object_A_joint_tensor[:, start_col : start_col + num_joints]
            elif object_A_joint_tensor.shape[1] < num_joints:
                # Dataset has fewer joints -> pad zeros
                pad = torch.zeros(
                    (object_A_joint_tensor.shape[0], num_joints - object_A_joint_tensor.shape[1]),
                    dtype=object_A_joint_tensor.dtype,
                    device=object_A_joint_tensor.device,
                )
                object_A_joint_tensor = torch.cat([object_A_joint_tensor, pad], dim=1)

            self.object_A.write_joint_position_to_sim(object_A_joint_tensor, joint_ids=None, env_ids=env_ids)
        else:
            # 如果没有提供关节状态，则设置为关闭状态（0）
            object_A_joint_pos = self.object_A.data.default_joint_pos[env_ids]
            object_A_joint_pos.fill_(0.0)
            self.object_A.write_joint_position_to_sim(
                object_A_joint_pos, joint_ids=None, env_ids=env_ids
            )


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
        stage = omni.usd.get_context().get_stage()
        
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
    
    def get_part_scores(self, env_ids: Sequence[int] | None = None):
        if self.part==0:
            #判断物体接触
            last_two = self.scene.sensors["contact_sensor"][:, -2:, :]  # 取最后两个接触点的力向量

            # 计算每个三维向量的模长
            magnitudes = torch.norm(last_two, dim=2)  # 计算每个三维向量的模长

            # 判断模长是否大于阈值
            is_above_threshold = magnitudes > 0.1

            if is_above_threshold[0] or is_above_threshold[1]:
                self.part+=1
                self.scores+=1
            return self.part, self.scores
        if self.part==1:
            #判断移动
            success = success_checker_pull(self.object_A)
            if success:
                self.part+=1
                self.scores+=1
            return self.part, self.scores
        
    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        # success = success_checker_pull(self.object_A)
        ang_rad = float(self.object_A.data.joint_pos[0, self._debug_faucet_joint_idx].item())
        ang_deg = ang_rad * 180.0 / math.pi
        print("ang_deg:",ang_deg)
        success = ang_deg > 40
        print("success:",success)
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success
    
    def get_contact(self):
        print(torch.max(self.scene.sensors["contact_sensor"].data.net_forces_w).item())
        # print(self.contact_sensor.data.force_matrix_w)
        # print(self.contact_sensor.data.force_matrix_w_history)
        # print(self.contact_sensor.data.net_forces_w)
        # print(self.contact_sensor.data.net_forces_w_history)
        return self.scene.sensors["contact_sensor"]
    
    def create_object(self):
        if self.object_A is not None:
            self.delete_object()

        prim_name=self.object_A_name
        prim_path = f"/World/Object/{prim_name}"

        try:
            if is_prim_path_valid(prim_path):
                print(
                    f"Prim path {prim_path} still exists, deleting before creation"
                )
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                if hasattr(self, "sim") and self.sim is not None:
                    for _ in range(5):
                        self.sim.step(render=False)

        except Exception as e:
            print(
                f"Could not delete existing prim (may not exist): {e}"
            )

        # Create new garment object
        try:
            print(
                f"Creating Object at prim_path: {prim_path}"
            )
            self.object_A_name = random.choice(self.object_A_names)
            self.object_A = RigidObject(getattr(self.cfg, self.object_A_name))
            self.scene.rigid_objects["object_A"] = self.object_A
            print("Object created successfully")
        except Exception as e:
            print(f"Failed to create Object: {e}")

    def delete_object(self):
        if self.object_A is None:
            return
        try:
            # Try to get prim_path from object first, then fallback to garment_name-based path
            if hasattr(self.object_A, "prim_path") and self.object_A.prim_path:
                prim_path = self.object_A.prim_path
            else:

                prim_name = self.object_A_name
                prim_path = f"/World/Object/{prim_name}"

            if is_prim_path_valid(prim_path):
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                print(f"Deleted prim at {prim_path}")
            else:
                print(
                    f"Prim path {prim_path} is not valid, skipping deletion"
                )

        except Exception as e:
            print(f"Failed to delete garment object: {e}")
            import traceback

            traceback.print_exc()

        self.object_A = None