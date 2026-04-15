from __future__ import annotations


import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG
from ...base.base_env_cfg import BaseEnvCfg
from isaaclab.sim import SimulationCfg
import os


@configclass
class BurgerEnvCfg(BaseEnvCfg):
    """Environment configuration inheriting from base LW_Loft environment."""

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="FXAA"
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 80, render_interval=1, render=render_cfg)

    # robot
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

    burger_beef: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Burger/burger_beef",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/objects/burger/Assets/Burger_Beef_Patties001/Burger_Beef_Patties001_Def.usd"
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            # pos=(2.86, -1.8666, 0.63848),
            pos=(2.912925958633423, -2.0175609588623047, 0.660826563835144),
            # pos=(1.65, -2.5, 0.580),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    burger_board: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_board",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/objects/burger/Assets/Burger_ChoppingBlock/Burger_ChoppingBlock.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(2.86, -1.7366, 0.63848),
            pos=(2.912925958633423, -1.88175609588623047, 0.660826563835144),
            # pos=(1.7, -2.3, 0.575),
            # pos=(1.65, -2.3, 0.615),
            rot=(0, 0.0, 0.0, 1),
        ),
    )
    burger_plate: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/objects/burger/Assets/Burger_Plate/Burger_Plate.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(2.83, -1.6666, 0.64848),
            pos=(2.882925958633423, -1.81175609588623047, 0.670826563835144),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    burger_bread2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Burger/burger_bread2",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/objects/burger/Assets/Burger_Bread002/Burger_Bread002.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(1.65, -2.3, 0.615),
            # pos=(2.83, -1.6666, 0.65848),
            pos=(2.882925958633423, -1.81175609588623047, 0.680826563835144),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    burger_cheese: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/Burger/burger_cheese",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/feng/lehome_1/Assets/objects/burger/Assets/Burger_Cheese001/Burger_Cheese001_Def.usd"
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            # pos=(2.83, -1.6666, 0.66848),
            pos=(2.882925958633423, -1.81175609588623047, 0.690826563835144),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # path_scene: str = os.getcwd() + "/Assets/scenes/LW_Loft/LW_Loft_burger.usd"
    # use_random: bool = False
