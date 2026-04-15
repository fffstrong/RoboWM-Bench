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

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG
import os


@configclass
class DrawerEnvCfg(DirectRLEnvCfg):
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

    from lehome.assets.robots.franka import FRANKA_CFG
    robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_CFG.init_state.replace(
            # pos=(2.24213, -1.7366, 0.62748),
            pos=(3.41213, -1.7366, 0.62748),
            rot=(0, 0.0, 0, 1)
            # rot=(1, 0.0, 0, 0),#wxyz
        ),  
    )
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/Robot/Robot/.*",  # 匹配所有机器人部件
            filter_prim_paths_expr=["/World/Object/*"], 
            update_period=0.0,  # 传感器更新频率
            history_length=1, 
            debug_vis=True
        )
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/Robot/Robot/.*",  # 匹配所有机器人部件
            filter_prim_paths_expr=["/World/Object/*"], 
            update_period=0.0,  # 传感器更新频率
            history_length=1, 
            debug_vis=False
        )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.41213-0.14382973, -1.7366-0.77309364, 0.62748+0.7135533),
            # rot=(-0.47446683, 0.83054371, 0.2426624, -0.16184353),
            rot=(-0.50096300, 0.82463646, 0.23517914, -0.11705365),
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth", "instance_id_segmentation_fast"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.4139,  # Physical focal length (mm)
            focus_distance=400.0,  # Focus distance (mm)
            horizontal_aperture=2.7700,  # Sensor width (mm)
            clipping_range=(0.1, 50.0),  # Near/far clipping planes (m)
            lock_camera=True,
        ),
        width=1280,
        height=720,
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
    #         pos=(-1.6, 3.1, 1.2),
    #         rot=(0, 0, -0.86603, 0.5),
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

    behind_pos = (-2.335, 3.36957, 0.91474)
    behind_quat_wxyz = (-0.20326, -0.17309, 0.62481, 0.73372)

    front_pos = (-0.80109, 3.14487, 0.86252)
    front_quat_wxyz = (0.27657, 0.24828, 0.62018, 0.69083)
    
    spawn_cfg = sim_utils.PinholeCameraCfg(
        focal_length=40,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=38.11,
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )
    
    behind_camera = TiledCameraCfg(
        prim_path="/World/camera_behindhaha",
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
        prim_path="/World/camera_fronthaha",
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


    # drawer - 铰链物体，使用 ArticulationCfg
    # 注意：微波炉的铰链关节不需要主动控制，所以 actuators 设置为空字典
    # scale=(0.8, 0.8, 0.8) 会同时缩放外观和物理属性（碰撞体、质量等）
    drawer: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path= "/home/feng/lehome_1/Assets/benchmark/object/StorageBin022/StorageBin022.usd",  # 无盖

            # scale=(0.3, 0.3, 0.3),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.66, -1.79989, 0.70848),
            # rot=(0.17365, 0.0, 0.0, -0.98481),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_pos={
                ".*": 0.0,  # 默认所有关节为 0（关闭状态）
            },
            joint_vel={
                ".*": 0.0,
            },
        ),
        actuators={},  # 微波炉铰链关节不需要主动控制，设置为空字典
        soft_joint_pos_limit_factor=1.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(3.5, -4.7, 1.7), lookat=(3.9, 1.2, -1))
    

    # drawer_max_strokes: Dict[str, float] = None  # type: ignore
    # default_drawer_stroke: float = 0.15

    # def __post_init__(self):
    #     if self.drawer_max_strokes is None:
    #         self.drawer_max_strokes = {
    #             "drawer001_joint": 0.15,
    #             "drawer002_joint": 0.15,
    #             "drawer003_joint": 0.15,
    #         }