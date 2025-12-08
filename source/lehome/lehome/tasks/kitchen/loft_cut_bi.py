from __future__ import annotations
import torch
from typing import Any, Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from pxr import Sdf, UsdGeom, Usd, Gf
from ..base.base_env_cfg import BaseEnvCfg
from .loft_cut_bi_cfg import LoftCutEnvCfg
from ..base.base_env import BaseEnv
from lehome.utils.success_checker import success_checker_cut
from lehome.devices.action_process import preprocess_device_action
from lehome.utils.cutMeshNode import cutMeshNode
from lehome.utils.collision_checker import Collision_Checker
from types import SimpleNamespace
import omni
from isaacsim.core.utils.prims import delete_prim
import transforms3d as t3
import numpy as np
from numpy.random import default_rng
from lehome.utils.random_position import randomize_pose
import os


class LoftCutEnv(BaseEnv):
    """Environment inheriting from base LW_Loft environment with additional features."""

    cfg: BaseEnvCfg | LoftCutEnvCfg

    def __init__(
        self, cfg: BaseEnvCfg | LoftCutEnvCfg, render_mode: str | None = None, **kwargs
    ):
        # self.base_t = (-0.1, -0.1, 0.7)
        self.base_t = (3.6912, -6.15, 0.84059)
        self.base_q_wxyz = (-0.23287, -0.02628, 0.02471, 0.97184)
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos
        # Assume that there is a prim path in the stage
        self.dummy_db = SimpleNamespace()
        self.dummy_db.inputs = SimpleNamespace()
        self.dummy_db.inputs.cut_mesh_path = "/World/Scene/Sausage001/Sausage001"
        self.dummy_db.inputs.knife_mesh_path = (
            "/World/Robot/Right_Robot/gripper/Knife/Knife/Cube"
        )
        self.dummy_db.internal_state = cutMeshNode.internal_state()
        self.dummy_db.inputs.cutEventIn = False
        self.stage = omni.usd.get_context().get_stage()
        self.collision_checker = Collision_Checker(stage=self.stage)
        self.last_if_collision = False

    def _setup_scene(self):
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to get base scene (LW_Loft + robot + camera)
        super()._setup_scene()
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)

        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        cfg = sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/scenes//LW_Loft/Loft/Sausage001/Sausage001.usd"
        )
        cfg.func(
            "/World/Scene/Sausage001",
            cfg,
            translation=self.base_t,
            orientation=self.base_q_wxyz,
        )

        # add articulation to scene
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera

    def _apply_action(self) -> None:

        if_collision, _, _ = self.collision_checker.meshes_aabb_collide()
        if if_collision == self.last_if_collision and if_collision:
            if_collision = False
        else:
            self.last_if_collision = if_collision

        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])
        self.dummy_db.inputs.cutEventIn = if_collision
        cutMeshNode.compute(self.dummy_db)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        left_camera_rgb = self.left_camera.data.output["rgb"]
        right_camera_rgb = self.right_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
        }
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        sausage_count = 0
        sausage_prim = self.stage.GetPrimAtPath("/World/Scene/Sausage001/Sausage001")
        sausage_count = len(sausage_prim.GetChildren())
        success = success_checker_cut(sausage_count)
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

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]
        self.left_arm.write_joint_position_to_sim(
            left_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.right_arm.write_joint_position_to_sim(
            right_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.object_reset("/World/Scene/Sausage001")

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def object_reset(self, sausage_path=""):

        # sausage_path = self.stage.GetPrimAtPath(sausage_path)
        delete_prim(sausage_path)
        t_new, q_new = randomize_pose(
            base_translation=self.base_t,
            base_quat_wxyz=self.base_q_wxyz,
            trans_range={"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (0.0, 0)},
            axis="z",
            deg_range=20,
            axis_space="local",
            rng=default_rng(),
        )
        cfg = sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/scenes/LW_Loft/Loft/Sausage001/Sausage001.usd"
        )
        cfg.func(
            "/World/Scene/Sausage001",
            cfg,
            translation=t_new,
            orientation=q_new,
        )

    def initialize_obs(self):
        pass

    def get_all_pose(self):
        timeline = omni.timeline.get_timeline_interface()
        current_time = timeline.get_current_time()
        usd_current_time = Usd.TimeCode(current_time)
        prim = self.stage.GetPrimAtPath("/World/Scene/Sausage001")

        sausage_xform = UsdGeom.Xformable(prim)
        sausage_trans = sausage_xform.GetLocalTransformation(usd_current_time)

        trans = sausage_trans.ExtractTranslation()
        rot = sausage_trans.ExtractRotationMatrix()
        trans_np = np.array([float(trans[i]) for i in range(3)])
        rot_np = np.array(
            [[rot[i, j] for j in range(3)] for i in range(3)], dtype=float
        )
        w, x, y, z = t3.quaternions.mat2quat(rot_np)
        rot_xyzw = np.array([w, x, y, z])
        return {"sausage_trans": trans_np, "sausage_rot": rot_xyzw}

    def set_all_pose(self, pose_dict: dict):
        """Move the sausage to the specified position and posture"""
        trans = pose_dict["sausage_trans"]  # numpy array [x, y, z]
        quat = pose_dict["sausage_rot"]  # numpy array [w, x, y, z]

        prim = self.stage.GetPrimAtPath("/World/Scene/Sausage001")
        if not prim.IsValid():
            raise RuntimeError("Prim at /World/Scene/Sausage001 not found.")

        xform = UsdGeom.Xformable(prim)

        # Convert to Gf ​
        t_gf = Gf.Vec3d(*trans)
        q_gf = Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

        # transform matrix
        m = Gf.Matrix4d()
        m.SetRotate(q_gf)
        m.SetTranslate(t_gf)

        xform.ClearXformOpOrder()  # clear old transform
        xform.AddTransformOp().Set(m)
