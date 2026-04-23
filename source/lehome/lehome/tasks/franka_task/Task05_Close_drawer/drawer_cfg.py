from __future__ import annotations
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from lehome.assets.robots.franka import FRANKA_CFG


# -----------------------------------------------------------------------------
# Asset path configuration
# -----------------------------------------------------------------------------
def _resolve_assets_root() -> str:
    """Resolve the assets root directory without hard-coded absolute paths."""
    env_root = os.environ.get("LEHOME_ASSETS_ROOT")
    if env_root:
        return str(Path(env_root).expanduser())

    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "Assets"
        if candidate.is_dir():
            return str(candidate)

    return str(Path("Assets").resolve())


ASSETS_ROOT = _resolve_assets_root()
BENCHMARK_OBJECT_ROOT = os.path.join(ASSETS_ROOT, "benchmark", "object")


def _bench_object_usd(object_name: str) -> str:
    """Return USD path for a benchmark object under the assets root."""
    return os.path.join(BENCHMARK_OBJECT_ROOT, object_name, f"{object_name}.usd")


@configclass
class DrawerEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka close-drawer task."""

    # Environment
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0
    action_space = 6
    observation_space = 6
    state_space = 0

    # Simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg
    )

    # Robot
    robot: ArticulationCfg = FRANKA_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=FRANKA_CFG.init_state.replace(
            pos=(3.41213, -1.7366, 0.63748),
            rot=(0, 0.0, 0, 1),
        ),
    )

    # Sensors
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/Robot/Robot/.*",
        filter_prim_paths_expr=["/World/Object/*"],
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.41213 - 0.0863833, -1.7366 - 0.83637488, 0.62748 + 0.77755654),
            rot=(-0.47446683, 0.83054371, 0.2426624, -0.16184353),
            convention="ros",
        ),
        data_types=["rgb", "depth", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.4139,
            focus_distance=400.0,
            horizontal_aperture=2.7700,
            clipping_range=(0.1, 50.0),
            lock_camera=True,
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


    # Drawer (articulation)
    drawer: ArticulationCfg = ArticulationCfg(
        prim_path="/World/Object/drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(BENCHMARK_OBJECT_ROOT, "StorageBin022", "StorageBin022.usd"),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.6201299999999996, -1.8640999999999996, 0.90748), 
            # rot=(0.17365, 0.0, 0.0, -0.98481),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
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

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(3.5, -4.7, 1.7), lookat=(3.9, 1.2, -1))
