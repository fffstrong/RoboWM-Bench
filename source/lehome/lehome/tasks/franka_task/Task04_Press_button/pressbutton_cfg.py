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
# - Defaults avoid hard-coded absolute paths. The assets root is resolved from
#   the repository layout (searching upwards for an `Assets/` directory).
# - For portability, you can override the root via environment variable
#   `LEHOME_ASSETS_ROOT` (absolute or relative).
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
class PressButtonEnvCfg(DirectRLEnvCfg):
    """Configuration for the Franka button press task."""

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
        dt=1 / 80, render_interval=decimation, render=render_cfg
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
        # Match all robot links and report contacts against task objects.
        prim_path="/World/Robot/Robot/.*",
        filter_prim_paths_expr=["/World/Object/*"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(3.41213 - 0.14382973, -1.7366 - 0.77309364, 0.62748 + 0.7135533),
            # rot=(-0.47446683, 0.83054371, 0.2426624, -0.16184353),
            rot=(-0.50096300, 0.82463646, 0.23517914, -0.11705365),
            convention="ros",
        ),
        data_types=["rgb", "depth", "instance_id_segmentation_fast"],
        colorize_instance_id_segmentation=False,
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
    top_camera2: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/top_camera2",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.3, -2.6, 1.2),
            rot=[0.43642, -0.71327, -0.51472, 0.18932],
            convention="ros",
        ),
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

    # Task object
    Button: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/button",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_bench_object_usd("button")
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(2.86, -1.79989, 0.62848),
            pos=(2.902822256088257, -1.7413281202316284, 0.6475169062614441),
            rot=(1, 0.0, 0.0, 0), 
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(3.5, -4.7, 1.7), lookat=(3.9, 1.2, -1))
