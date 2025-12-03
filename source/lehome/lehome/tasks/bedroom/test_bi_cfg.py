from __future__ import annotations
import torch
from dataclasses import MISSING
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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


@configclass
class MarbelEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 60
    # action_scale = 1.0  # [N]
    action_scale = 1.0  # [N]
    action_space = 12
    observation_space = 12
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg
    )

    # robot
    left_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.12816, 0.74377, -0.42),
            rot=(0.98481, 0.0, 0.0, 0.217365),
        ),
        #     pos=(1.25, -2.3, 0.5)
        # ),  # (pos=(2.7, -2.76, 0.21),
        # # rot=(0.707, 0.0, 0.0, 0.707) )
    )
    right_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-0.13434, 0.61914, -0.42),
            rot=(0.98481, 0.0, 0.0, 0.217365),
        ),
        #     pos=(1.55, -2.3, 0.5)
        # ),  # (pos=(2.7, -3.11, 0.21),
        # # rot=(0.707, 0.0, 0.0, 0.707) )
    )
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Left_Robot/gripper/left_wrist_camera",
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
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
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
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.225, -0.5, 0.6),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
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
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
