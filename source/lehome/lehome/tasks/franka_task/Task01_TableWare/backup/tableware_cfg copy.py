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

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG,SO101_FOLLOWER_45_CFG
import os


@configclass
class TablewareEnvCfg(DirectRLEnvCfg):
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
    # ========================================================================
    # 关键配置：必须显式设置 joint_pos
    # ========================================================================
    # 问题：使用 .replace() 时，如果不显式传入 joint_pos，会使用类默认值 {".*": 0.0}
    # 解决：显式设置 joint_pos，确保机械臂初始为蜷缩状态而非展开状态
    # ========================================================================
    robot: ArticulationCfg = SO101_FOLLOWER_45_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_45_CFG.init_state.replace(
            pos=(-0.13, -0.15, 0.49844),
            # ⭐ 必须显式设置，否则 .replace() 会使用默认值导致关节全为0（展开状态）
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
            rot=[0.5225, -0.85264,0, 0],
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

    # ---------------------------------------------------------------------
    # Runtime selection
    # ---------------------------------------------------------------------
    # Select which configured rigid object to instantiate as object_A in the environment.
    # This is used by replay / dataset rebuilding scripts to control the object type.
    object_A_name: str = "b_cups"
    # top_camera2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/top_camera2",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.6, -3.2, 1.0),
    #         rot=[0.5225, -0.85264,0, 0],
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
    #     height=704,
    # )





    #object_A

    bowl1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/bowl1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/bowl1/Bowl016.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,  0.53844),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    bowl2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/bowl2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/bowl2/bowl.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,  0.53844),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    b_cups: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/b_cups",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/b_cups/b_cups.usd"
            # + "/Assets/objects/Cup/Cup008/Cup008.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,0.55344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    coffeecup028: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/coffeecup028",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/container/CoffeeCup028/CoffeeCup028.usd",
            scale=(0.75, 0.75, 0.75)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,0.56344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    cup002: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/cup002",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Cup/Cup002/Cup002.usd",
            scale=(0.75, 0.75, 0.75)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,0.56344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    cup012: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/cup012",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Cup/Cup012/Cup012.usd",
            scale=(0.75, 0.75, 0.75)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,0.56344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    cup030: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/cup030",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/objects/Cup/Cup030/Cup030.usd",
            scale=(0.75, 0.75, 0.75)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15,0.56344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )

    mug: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/mug",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/mug/mug.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.02, 0.15, 0.54344),
            rot=(0.5, 0.0, 0.0, 0.5),  # 180° around z-axis (xyzw)
        ),
    )
    pitcher_base: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/pitcher_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/pitcher_base/pitcher_base.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(0.02, 0.15, 0.62844),
            pos=(0.04075, 0.06091, 0.59478),
            rot=(0.5, 0.0, 0.0, 1),
        ),
    )

    #object_B


    plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/plate/plate.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.12, -0.05, 0.56344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    plate_scale0_8: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/plate_scale0_8",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/plate/plate_0.8.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.12 , -0.05, 0.56344),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    plate_scale1_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/plate_scale1_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd()
            + "/Assets/cosmos_assets/1_arrange_tableware/plate/plate_1.2.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(0.12, -0.05, 0.56344),
            pos=(0.1493, -0.06069, 0.56771),
            rot=(0, 0.0, 0.0, 1),
        ),
    )


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
    
    # scene USD path (可以通过 --scene_usd 参数覆盖)
    path_scene: str = os.getcwd() + "/Assets/scenes/RSR/0127_Real/RSR_Scene02.usd"