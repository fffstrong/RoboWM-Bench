from __future__ import annotations
import torch
from dataclasses import MISSING
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_45_CFG,SO101_FOLLOWER_CFG
import os
import os.path


@configclass
class MicrowaveEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0  # [N]
    action_space = 6
    observation_space = 6
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg
    )

    # robot
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.05636, -0.385, 0.49844),
            joint_pos={
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": -1.5,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            }
        ),
    )
    # wrist_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/Robot/Robot/gripper/wrist_camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(-0.001, 0.1, -0.04),
    #         rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
    #         convention="ros",
    #     ),  # wxyz
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=36.5,
    #         focus_distance=400.0,
    #         horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
    #         clipping_range=(0.01, 50.0),
    #         lock_camera=True,
    #     ),
    #     width=960,
    #     height=720,
    #     update_period=1 / 30.0,  # 30FPS
    # )
    # top_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/top_camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.02199, -1.02033, 1.18161),
    #         # 为了让 IsaacSim UI 中显示为 [0.84479, 0.53504, 0.00414, 0.00653]
    #         # 需要在此处写入“反解”后的 wxyz
    #         rot=[0.53504, -0.84479, -0.00653, 0.00414],
    #         convention="ros",
    #     ),  # wxyz
    #     data_types=["rgb", "depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=28.7,
    #         focus_distance=400.0,
    #         horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
    #         clipping_range=(0.01, 50.0),
    #         lock_camera=True,
    #     ),
    #     width=960,
    #     height=720,
    # )
    # top_camera_2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/top_camera_2",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.96219, -0.33163, 0.99578),
    #         # 为了让 IsaacSim UI 中显示为 [0.66182, 0.45071, 0.3372, 0.49513]
    #         # 需要在此处写入“反解”后的 wxyz
    #         rot=(0.45071, -0.66182, -0.49513, 0.3372),
    #         convention="ros",
    #     ),  # wxyz
    #     data_types=["rgb", "depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=28.7,
    #         focus_distance=400.0,
    #         horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
    #         clipping_range=(0.01, 50.0),
    #         lock_camera=True,
    #     ),
    #     width=960,
    #     height=720,
    # )
    # behind_pos = ( 1.00274, -1.11842, 1.03989)
    # behind_quat_wxyz = (0.74274, 0.57294, 0.2138, 0.27269)
    
    # front_pos = (-1.03607,  -1.23903,  0.99387)
    # front_quat_wxyz = (0.71865,0.60586,-0.21655, -0.26379)

    # spawn_cfg = sim_utils.PinholeCameraCfg(
    #     focal_length=40,
    #     focus_distance=400.0,
    #     f_stop=0.0,
    #     horizontal_aperture=38.11,
    #     clipping_range=(0.01, 50.0),
    #     lock_camera=True,
    # )
    

    behind_pos = ( 0.86486, -0.98468, 0.92848)
    behind_quat_wxyz = (0.75602, 0.56958, 0.19621, 0.25594)
    
    front_pos = (-0.81168,  -0.93751,  1.01816)
    front_quat_wxyz = (0.77754,0.52891,-0.18759, -0.28371)

    # Top camera (user provided pose)
    top_pos = (0.04404,  -1.15658,  1.58668)
    top_quat_wxyz = (0.91466, 0.4039, 0.00879, 0.01357)

    # Extra cameras (user provided poses)
    extera1_pos = (-0.50384,  -0.94294,  0.81994)
    extera1_quat_wxyz = (0.75189, 0.62288, -0.13582, -0.16806)

    extera2_pos = (-0.50146,  -0.82873,  0.91945)
    extera2_quat_wxyz = (0.77284, 0.57473, -0.15705, -0.21849)

    extera3_pos = (0.78512,  -0.76534,  0.93371)
    extera3_quat_wxyz = (0.75295, 0.53876, 0.22358, 0.30467)
    
    spawn_cfg = sim_utils.PinholeCameraCfg(
        focal_length=54,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=38.11,
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )
    

    # behind_camera = TiledCameraCfg(
    #     prim_path="/World/camera_behind",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=behind_pos,
    #         rot=behind_quat_wxyz,
    #         convention="opengl",
    #     ),
    #     data_types=["rgb"],
    #     spawn=spawn_cfg,
    #     width=640,
    #     height=480,
    #     update_period=0.0,
    # )
    
    top_camera = TiledCameraCfg(
        prim_path="/World/camera_top",
        offset=TiledCameraCfg.OffsetCfg(
            pos=top_pos,
            rot=top_quat_wxyz,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=spawn_cfg,
        width=640,
        height=480,
        update_period=0.0,
    )

    microwave_usd_path: str = os.getcwd() + "/Assets/objects/Articulated/Microwave/Microwave011/Microwave011.usd"
    _mw_name = os.path.basename(microwave_usd_path)
    _mw_is_lw = _mw_name.startswith("Microwave") and _mw_name.endswith(".usd")
    microwave_family: str = "LW" if _mw_is_lw else "Sapien"
    # Fixed microwave USD (no random/record switching).
    # Sapien和LW微波炉资产差异初始化
    _mw_scale = (0.8, 0.8, 0.8) if _mw_is_lw else (0.003, 0.003, 0.003)
    _mw_rot = (1, 0.0, 0.0, 0) if _mw_is_lw else (0.70711, 0.0, 0.0, 0.70711)
    # For Sapien assets, joints often come with drive settings that make physical opening very hard.
    # We disable joint drives at spawn time (applies to all joints under the articulation).
    _mw_joint_drive_props = None if _mw_is_lw else sim_utils.JointDrivePropertiesCfg(
        drive_type="force",
        stiffness=0.0,
        damping=0.0,
        max_effort=0.0,
        max_velocity=100.0,
    )

    microwave: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path=microwave_usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,  # 固定根链接，防止微波炉移动或下落
            ),
            joint_drive_props=_mw_joint_drive_props,
            scale=_mw_scale,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.06969, 0.18, 0.67137),
            rot=_mw_rot,
            joint_pos={
                ".*": 0.0,  # 默认所有关节为 0（关闭状态）
            },
            joint_vel={
                ".*": 0.0,
            },
        ),
        actuators={},
        soft_joint_pos_limit_factor=1.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
    
    # scene USD path (可以通过 --scene_usd 参数覆盖)
    path_scene: str = "/home/lightwheel/Projects/Lehome_Marble/Assets/scenes/Marble/Scene_01_LoftwithKitchen/Scene_01_LoftwithKitchen.usd"

