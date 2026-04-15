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
from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG
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
    from lehome.assets.robots.franka import FRANKA_CFG
    robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_CFG.init_state.replace(
            # pos=(1.4, -2.6, 0.5),
            pos=(1.14, -2.9, 0.7),
            # rot=(0.382683, 0, 0, 0.923880),
            # rot=(0.0, 0, 0, 1),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),  # (pos=(2.7, -2.76, 0.21),
        # rot=(0.707, 0.0, 0.0, 0.707) )
    )
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/Robot/Robot/.*",  # 匹配所有机器人部件
            filter_prim_paths_expr=["/World/Object/*"], 
            update_period=0.0,  # 传感器更新频率
            history_length=1, 
            debug_vis=True
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
    behind_pos = ( 1.00274, -1.11842, 1.03989)
    behind_quat_wxyz = (0.74274, 0.57294, 0.2138, 0.27269)
    
    front_pos = (-1.03607,  -1.23903,  0.99387)
    front_quat_wxyz = (0.71865,0.60586,-0.21655, -0.26379)
    
    spawn_cfg = sim_utils.PinholeCameraCfg(
        focal_length=40,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=38.11,
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )
    
    behind_camera = TiledCameraCfg(
        prim_path="/World/camera_behind",
        offset=TiledCameraCfg.OffsetCfg(
            pos=behind_pos,
            rot=behind_quat_wxyz,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=spawn_cfg,
        width=640,
        height=480,
        update_period=0.0,
    )
    
    front_camera = TiledCameraCfg(
        prim_path="/World/camera_front",
        offset=TiledCameraCfg.OffsetCfg(
            pos=front_pos,
            rot=front_quat_wxyz,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=spawn_cfg,
        width=640,
        height=480,
        update_period=0.0,
    )

    microwave_usd_path: str = "/home/feng/lehome_1/Assets/benchmark/object/Microwave035/Microwave035.usd"
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
            pos=(1.61, -2.3, 0.67137),
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
    # path_scene: str = KITCHEN_WITH_ORANGE_USD_PATH

