from __future__ import annotations
import torch
from typing import Any, Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera
from pxr import Sdf, UsdGeom, Usd, Gf
from ...base.base_env_cfg import BaseEnvCfg
from .sausage_cfg import SausageEnvCfg
from ...base.base_env import BaseEnv
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


class SausageEnv(BaseEnv):
    """Environment inheriting from base LW_Loft environment with additional features."""

    cfg: BaseEnvCfg | SausageEnvCfg

    def __init__(
        self, cfg: BaseEnvCfg | SausageEnvCfg, render_mode: str | None = None, **kwargs
    ):
        # self.base_t = (-0.1, -0.1, 0.7)
        self.base_t = (2.86, -1.79989, 0.63848)
        # self.base_q_wxyz = (-0.23287, -0.02628, 0.02471, 0.97184)
        self.base_q_wxyz = (1, 0, 0, 0)
        super().__init__(cfg, render_mode, **kwargs)
        # Additional initialization specific to this environment

        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos
        # Assume that there is a prim path in the stage
        self.dummy_db = SimpleNamespace()
        self.dummy_db.inputs = SimpleNamespace()
        self.dummy_db.inputs.cut_mesh_path = "/World/Scene/Sausage001/Sausage001"
        self.dummy_db.inputs.knife_mesh_path = (
            "/World/Robot/Robot/panda_rightfinger/Knife/Knife/Cube"
        )
        self.dummy_db.internal_state = cutMeshNode.internal_state()
        self.dummy_db.inputs.cutEventIn = False
        self.stage = omni.usd.get_context().get_stage()
        self.collision_checker = Collision_Checker(stage=self.stage)
        self.last_if_collision = False

    def _setup_scene(self):
        
        """Setup the scene by calling parent method and adding additional assets."""
        # Call parent setup to get base scene (LW_Loft + robot + camera)
        # super()._setup_scene()
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)

        from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
        cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_WITH_ORANGE_USD_PATH}")
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/feng/lehome/Assets/LW_Loft/Scene_lw_room.usd")

        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )
        cfg = sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/scenes/LW_Loft/Loft/Sausage001/Sausage001.usd"
        )
        cfg.func(
            "/World/Scene/Sausage001",
            cfg,
            translation=self.base_t,
            orientation=self.base_q_wxyz,
        )
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["top_camera"] = self.top_camera
        light_cfg = sim_utils.DomeLightCfg(intensity=700, color=(0.75, 0.75, 0.75))
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

        # if_collision, _, _ = self.collision_checker.meshes_aabb_collide()
        # if if_collision == self.last_if_collision and if_collision:
        #     if_collision = False
        # else:
        #     self.last_if_collision = if_collision

        self.robot.set_joint_position_target(self.actions)
        # self.dummy_db.inputs.cutEventIn = if_collision
        # cutMeshNode.compute(self.dummy_db)

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
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.object_reset("/World/Scene/Sausage001")

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def object_reset(self, sausage_path=""):

        # sausage_path = self.stage.GetPrimAtPath(sausage_path)
        delete_prim(sausage_path)
        self.t_new, self.q_new = randomize_pose(
            base_translation=self.base_t,
            base_quat_wxyz=self.base_q_wxyz,
            # trans_range={"x": (2.9601299999999995, 2.9701299999999995), "y": (-1.9165999999999996, -1.9065999999999996), "z": (0.86748, 0.86749)},
            trans_range={"x": (-0.1, 0.1), "y": (-0.1 ,0.1), "z": (0, 0.0001)},
            axis="z",
            deg_range=20,
            axis_space="local",
            rng=default_rng(),
        )
        cfg = sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome/Assets/LW_Loft/Loft/Sausage001/Sausage001.usd"
        )
        cfg.func(
            "/World/Scene/Sausage001",
            cfg,
            translation=self.t_new,
            orientation=self.q_new,
        )

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
        
        # 将物体信息写入JSONL文件
        import json
        from pathlib import Path
        
        def _to_jsonable_vec(x: Any) -> list[float]:
            """Convert numpy/torch/sequence vectors to a JSON-serializable list[float]."""
            # torch tensor
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            # numpy array / numpy scalar / python scalar / sequence
            arr = np.asarray(x, dtype=float).reshape(-1)
            return arr.tolist()
        
        # 准备数据
        pose_data = {
            "name": "SAUSAGE",
            "xyz": _to_jsonable_vec(self.t_new),  # xyz
            "quat": _to_jsonable_vec(self.q_new),  # 四元数 (wxyz)
        }
        
        # 写入JSONL文件
        jsonl_path = Path(json_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pose_data, ensure_ascii=False) + "\n")   


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
