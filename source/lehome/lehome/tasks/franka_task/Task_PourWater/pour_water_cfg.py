from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG
from ...base.base_env_cfg import BaseEnvCfg
import os


@configclass
class PourWaterEnvCfg(BaseEnvCfg):
    """Environment configuration inheriting from base LW_Loft environment."""

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="Off"
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=1, render=render_cfg, use_fabric=False
    )

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
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome/Assets/cosmos_assets/1_arrange_tableware/bowl1/Bowl016.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.61, -2.3, 0.545),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.5, -0.3, 1.0),
            rot=(0.17106, -0.6861, 0.6861, -0.17106),
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
        update_period=1 / 30.0,  # 30FPS
    )

