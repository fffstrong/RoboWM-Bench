from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
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

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
            pos=(2.1, -2.27, 0.625),  # 2.7, -2.27, 0.625
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/Robot/Robot/.*",
        filter_prim_paths_expr=["/World/Rubbish/*"],
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
            pos=(2.5062, -2.2062, 1.1323),
            rot=[-0.35558, 0.9345, 0000.00732, -0.01488],  # -x, w, z, -y
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

    button: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/button",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/human_assets/button/button.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5793827, -1.90282, 0.64),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

# scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
