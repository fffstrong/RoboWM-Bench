from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

from isaaclab.sensors import TiledCameraCfg
from lehome.assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.controllers import DifferentialIKControllerCfg


@configclass
class TablewareEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5
    episode_length_s = 60
    action_scale = 0.2  # [N]

    action_space = 14
    observation_space = 9
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 80, render_interval=decimation, render=render_cfg
    )

    ik_controller_left = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )
    ik_controller_right = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )

    left_arm: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/Robot/LeftArm",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(2.1, -2.27, 0.625),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    right_arm: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/Robot/RightArm",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(2.8, -2.27, 0.625),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    contact_sensor_left = ContactSensorCfg(
        prim_path="/World/Robot/LeftArm/.*",
        filter_prim_paths_expr=["/World/Object/Spatula"],
        update_period=0.0, history_length=1, debug_vis=False
    )
    contact_sensor_right = ContactSensorCfg(
        prim_path="/World/Robot/RightArm/.*",
        filter_prim_paths_expr=["/World/Object/Pan"],
        update_period=0.0, history_length=1, debug_vis=False
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
            pos=(2.5062, -2.4, 1.1323),
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

    r_spatula_z = Rotation.from_euler('z', 180, degrees=True)
    r_spatula = r_spatula_z
    q_spatula_xyzw = r_spatula.as_quat()
    q_spatula_wxyz = [q_spatula_xyzw[3], q_spatula_xyzw[0], q_spatula_xyzw[1], q_spatula_xyzw[2]]

    spatula: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/Spatula",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/human_assets/spatula/spatula.usd",
            scale=(0.16, 0.16, 0.31),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.3063827, -1.98082, 0.64),
            rot=q_spatula_wxyz,
        ),
    )

    r_pan_z = Rotation.from_euler('z', 0, degrees=True)
    r_pan = r_pan_z
    q_pan_xyzw = r_pan.as_quat()
    q_pan_wxyz = [q_pan_xyzw[3], q_pan_xyzw[0], q_pan_xyzw[1], q_pan_xyzw[2]]

    pan: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/Pan",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/human_assets/pan/pan.usd",
            scale=(0.26, 0.26, 0.26),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.00002),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5843827, -1.96482, 0.64),
            rot=q_pan_wxyz,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
