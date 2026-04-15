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

    # robot
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.05636, -0.33987, 0.49844)
        ),
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=960,
        height=720,
        update_period=1 / 30.0,  # 30FPS
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.6, -3.2, 1.0),
            rot=[0.5225, -0.85264, 0, 0],
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

    # drawer - 铰链物体，使用 ArticulationCfg
    # 注意：微波炉的铰链关节不需要主动控制，所以 actuators 设置为空字典
    # scale=(0.8, 0.8, 0.8) 会同时缩放外观和物理属性（碰撞体、质量等）
    drawer: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Articulated/Nightstand052/Nightstand052.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.03563, 0.23754, 0.74874), 
            rot=(1.0, 0.0, 0.0, 0.0),
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

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
    
    # scene USD path (可以通过 --scene_usd 参数覆盖)
    path_scene: str = os.getcwd() + "/Assets/scenes/LW_Loft/LW_Loft_bedroom.usd"

