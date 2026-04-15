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
from lehome.assets.robots.franka import FRANKA_CFG
# from lehome.assets.robots.droid import DROID_CFG
import os
from isaaclab.sensors import ContactSensor,ContactSensorCfg
'''
robot pos=(1.4, -2.6, 0.5),
cups pos=(1.55, -2.3, 0.575),
plate pos=(1.65, -2.5, 0.585),
camera pos=(1.6, -3.2, 1.0),
'''

@configclass
class FaucetEnvCfg(DirectRLEnvCfg):
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

    # top_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/Robot/Right_Robot/base/top_camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(3.41213-0.01763833, -1.7366-0.74637488, 0.62748+0.72755654),
    #         rot=(-0.47446683, 0.83054371, 0.2426624, -0.16184353),
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
    #     width=1280,
    #     height=960,
    # )

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
        width=960,
        height=720,
    )

    faucet: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/faucet",
        spawn=sim_utils.UsdFileCfg(
            usd_path= "/home/feng/lehome_1/Assets/benchmark/object/faucet/faucet_4/model_faucet_4.usd",
            # /home/glzn/new/lehome_1/Assets/benchmark/object/faucet/faucet_4/model_faucet_4.usd joint_idx=1
            # /home/glzn/new/lehome_1/Assets/benchmark/object/faucet/faucet_7/model_faucet_7.usda joint_idx=0
            scale=(1.0, 1.0, 1.0), 
            # scale=(0.3, 0.3, 0.3),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # pos=(2.86, -1.79989, 0.62848),
            pos=(2.795586347579956, -1.8095694780349731, 0.6284800171852112),
            # rot=(0.17365, 0.0, 0.0, -0.98481),
            # rot = (0.7071, 0.0, 0.0, 0.7071),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
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