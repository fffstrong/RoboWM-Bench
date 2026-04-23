from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from lehome.assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from scipy.spatial.transform import Rotation
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.controllers import DifferentialIKControllerCfg


@configclass
class DrawerEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 60
    action_scale = 0.2
    action_space = 7
    observation_space = 9
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 80, render_interval=decimation, render=render_cfg
    )

    ik_controller = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(2.1, -2.27, 0.625),  # 2.7, -2.27, 0.625
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/Robot/Robot/.*",
        filter_prim_paths_expr=["/World/Object/*"],
        update_period=0.0,
        history_length=1,
        debug_vis=False
    )

    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.8844, -2.1901, 1.11),
            rot=[-0.50947, 0.85333, 0.08005, -0.07658],  # -x, w, z, -y
            convention="ros",
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.24,
            focus_distance=1.0,
            horizontal_aperture=5.568,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 100.0),
        ),
        width=1280,
        height=720,
    )
    top_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera2",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.5062, -2.35, 1.1323),
            rot=[-0.43772, 0.89896, 0.00862, -0.01417],  # -x, w, z, -y
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.24,
            focus_distance=1.0,
            horizontal_aperture=5.568,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 100.0),
        ),
        width=1280,
        height=720,
    )

    behind_pos = (-2.335, 3.36957, 0.91474)
    behind_quat_wxyz = (-0.20326, -0.17309, 0.62481, 0.73372)

    front_pos = (-0.80109, 3.14487, 0.86252)
    front_quat_wxyz = (0.27657, 0.24828, 0.62018, 0.69083)

    spawn_cfg = sim_utils.PinholeCameraCfg(
        focal_length=40,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=38.11,
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )

    behind_camera = TiledCameraCfg(
        prim_path="/World/camera_behindhaha",
        offset=TiledCameraCfg.OffsetCfg(
            pos=behind_pos,
            rot=behind_quat_wxyz,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=spawn_cfg,
        width=640,
        height=480,
        update_period=0.0,
    )

    front_camera = TiledCameraCfg(
        prim_path="/World/camera_fronthaha",
        offset=TiledCameraCfg.OffsetCfg(
            pos=front_pos,
            rot=front_quat_wxyz,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=spawn_cfg,
        width=640,
        height=480,
        update_period=0.0,
    )

    r_drawer_z = Rotation.from_euler('z', -90, degrees=True)
    r_drawer = r_drawer_z
    q_drawer_xyzw = r_drawer.as_quat()
    q_drawer_wxyz = [q_drawer_xyzw[3], q_drawer_xyzw[0], q_drawer_xyzw[1], q_drawer_xyzw[2]]

    drawer: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/Drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/human_assets/storge_bin/StorageBin022.usd",
            scale=(0.85, 0.85, 0.85),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.6643827, -1.94882, 0.68),
            rot=q_drawer_wxyz,
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                ".*": 0.0,
            },
        ),
        actuators={},
        soft_joint_pos_limit_factor=1.0,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(3.5, -4.7, 1.7), lookat=(3.9, 1.2, -1))
