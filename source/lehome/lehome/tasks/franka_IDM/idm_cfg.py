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

from lehome.assets.robots.franka import FRANKA_CFG
from lehome.assets.robots.droid import FRANKA_ROBOTIQ_GRIPPER_CFG
import os


@configclass
class IDMEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0  # [N]
    action_space = 9
    observation_space = 9
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50, render_interval=decimation, render=render_cfg
    )

    # robot
    # robot: ArticulationCfg = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(
    #     prim_path="/World/Robot/Robot",
    #     init_state=FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.replace(
    #         # pos=(2.43, -2.6, 0.4),
    #         pos=(3.6, -2.6, 0.4),
    #         # pos=(3.41213, -1.7366, 0.62748),
    #         rot=(0, 0.0, 0, 1)
    #         # rot=(1, 0.0, 0, 0),#wxyz
    #     ),  
    # )
    robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_CFG.init_state.replace(
            # pos=(2.43, -2.6, 0.4),
            pos=(3.6, -2.6, 0.4),
            # pos=(3.41213, -1.7366, 0.62748),
            rot=(0, 0.0, 0, 1)
            # rot=(1, 0.0, 0, 0),#wxyz
        ),  
    )
    # from isaaclab.sensors import ContactSensor,ContactSensorCfg
    # contact_sensor_cfg = ContactSensorCfg(
    #         prim_path="/World/Robot/Robot/.*",  # 匹配所有机器人部件
    #         filter_prim_paths_expr=["/World/Object/*"], 
    #         update_period=0.0,  # 传感器更新频率
    #         history_length=1, 
    #         debug_vis=False
    #     )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.6-0.14382973, -2.6-0.77309364, 0.4+0.7135533),
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
    top_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera2",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.3, -2.6, 1.2),
            rot=[0.43642, -0.71327, -0.51472, 0.18932],
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )

    # banana: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/Object/banana",
    #     spawn=sim_utils.UsdFileCfg( 
    #         usd_path="/home/feng/lehome_1/Assets/benchmark/object/banana/banana.usd"
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(11, 11, 11),
    #         rot=(1, 0.0, 0.0, 0), 
    #     ),
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(3.5, -4.7, 1.4), lookat=(3.9, 1.2, -1))
