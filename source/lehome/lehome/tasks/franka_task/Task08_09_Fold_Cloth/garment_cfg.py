from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG


@configclass
class GarmentEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0  # [N]
    action_space = 12
    observation_space = 12
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 80,
        render_interval=decimation,
        render=render_cfg,
        use_fabric=False,
    )
    # garment_name (str): Garment name in the format "Type_Length_Seen/Unseen_Index",
    # e.g., "Top_Long_Unseen_0", "Top_Short_Seen_1",
    garment_name: str = "Top_Long_Seen_0"
    garment_version: str = "Release"  # "Release" or "Holdout"
    garment_cfg_base_path: str = "Assets/benchmark/cloth/Challenge_Garment"
    particle_cfg_path: str = (
        "/source/lehome/lehome/tasks/franka_task/Task08_09_Fold_Cloth/config_file/particle_garment_cfg.yaml"
    )
    # random seed
    use_random_seed: bool = True
    random_seed: int = 42

    from lehome.assets.robots.franka import FRANKA_CFG
    left_robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=FRANKA_CFG.init_state.replace(
            # pos=(2.24213, -1.7366, 0.62748),
            pos=(3.41213, -1.7366, 0.62748),
            rot=(0, 0.0, 0, 1)
            # rot=(1, 0.0, 0, 0),#wxyz
        ),  
    )
    from lehome.assets.robots.franka import FRANKA_CFG
    right_robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=FRANKA_CFG.init_state.replace(
            pos=(2.12213, -1.7366, 0.62748),
            # pos=(3.41213, -1.7366, 0.62748),
            # rot=(0, 0.0, 0, 1)
            rot=(1, 0.0, 0, 0),#wxyz
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
    # top_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/top_camera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(3.41213-0.08763833, -1.7366-0.74637488, 0.62748+0.62755654),
    #         rot=(-0.47446683, 0.83054371, 0.2426624, -0.16184353),
    #         # rot=(-0.2857871, 0.676204, -0.6378321, 0.2329035),
    #         # pos=(2.6, -3.0, 1.8),
    #         # rot=[0.43642, -0.71327, -0.51472, 0.18932],
    #         convention="ros",
    #     ),  # wxyz
    #     data_types=["rgb", "depth", "instance_id_segmentation_fast"],
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=12.87,  # 修改为图像中的焦距
    #     #     focus_distance=400.0,
    #     #     horizontal_aperture=20.955,  # 修改为图像中的水平光圈
    #     #     clipping_range=(0.1, 50.0),  # 修改为图像中的裁剪范围
    #     #     lock_camera=True,
    #     # ),
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
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
    
