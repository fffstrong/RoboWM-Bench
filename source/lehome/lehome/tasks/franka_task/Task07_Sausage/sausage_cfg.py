from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import ViewerCfg
from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG, SO101_KINFE_CFG
from ...base.base_env_cfg import BaseEnvCfg

import os


@configclass
class SausageEnvCfg(BaseEnvCfg):
    """Environment configuration inheriting from base LW_Loft environment."""

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="Off"
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 80, render_interval=1, render=render_cfg, use_fabric=False
    )

    from lehome.assets.robots.franka import FRANKA_KINFE_CFG
    robot: ArticulationCfg = FRANKA_KINFE_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_KINFE_CFG.init_state.replace(
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
    path_scene: str = os.getcwd() + "/Assets/scenes/LW_Loft/LW_Loft_cut_no_sausage.usd"
    # path_scene: str = os.getcwd() + "/Assets/kitchen_with_orange/360_room.usd"
    viewer = ViewerCfg(eye=(3.5, -4.7, 1.7), lookat=(3.9, 1.2, -1))