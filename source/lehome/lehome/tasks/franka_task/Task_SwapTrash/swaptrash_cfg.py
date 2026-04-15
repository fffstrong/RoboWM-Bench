from __future__ import annotations


import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg

# from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG, SO101_BROOM_CFG
from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG
from isaaclab.sim import SimulationCfg
import os


@configclass
class SwapRubbishEnvCfg(DirectRLEnvCfg):
    """Environment configuration for single-arm swap rubbish task (right arm only)."""

    # env
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0  # [N]
    action_space = 6
    observation_space = 6
    state_space = 0

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="FXAA"
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 80, render_interval=decimation, render=render_cfg)

    # robot - 单臂（右臂），初始姿态设置为竖直向上，避免碰撞
    from lehome.assets.robots.franka import FRANKA_CFG
    # robot
    robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_CFG.init_state.replace(
            pos=(3.41213, -1.7366, 0.62748),
            rot=(0, 0.0, 0, 1)
        ),  
    )
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/Robot/Robot/.*",  # 匹配所有机器人部件
            filter_prim_paths_expr=["/World/Rubbish/*"], 
            update_period=0.0,  # 传感器更新频率
            history_length=1, 
            debug_vis=True
        )
    
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.6, -3.2, 1.0),
            rot=[0.5225, -0.85264,0, 0],
            # pos=(2.6, -3.0, 1.8),
            # rot=[0.43642, -0.71327, -0.51472, 0.18932],
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

    #object_A

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/benchmark/object/cube/cube.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(12, 12, 12),
            rot=(1, 0.0, 0.0, 0), 
        ),
    )
    brown_plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/brown_plate",
        spawn=sim_utils.UsdFileCfg( 
            usd_path="/home/feng/lehome_1/Assets/benchmark/object/brown_plate/brown_plate.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(11, 11, 11),
            rot=(1, 0.0, 0.0, 0), 
        ),
    )
    brown_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/brown_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/benchmark/object/brown_cup/brown_cup.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10, 10, 10),
            rot=(1, 0.0, 0.0, 0),  # 180° around z-axis (xyzw)
        ),
    )

    #object_B

    trash_can: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/trash_can",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/benchmark/object/trash_can/trash_can.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(15, 15, 15),
            rot=(1, 0.0, 0.0, 0),  # 180° around z-axis (xyzw)
        ),
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )
    # viewer
    viewer = ViewerCfg(eye=(3.5, -4.7, 1.7), lookat=(3.9, 1.2, -1))


    use_random: bool = False
