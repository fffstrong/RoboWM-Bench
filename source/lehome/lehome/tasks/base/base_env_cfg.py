from __future__ import annotations
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import os


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    """Base environment configuration for LW_Loft scene."""

    # env
    decimation = 2
    episode_length_s = 100
    action_scale = 1.0  # [N]
    action_space = 12
    observation_space = 12
    state_space = 0

    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(
        rendering_mode="quality", antialiasing_mode="FXAA"
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=render_cfg
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )
    # viewer
    viewer: ViewerCfg = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))

    path_scene: str = os.getcwd() + "/Assets/scenes/LW_Loft/LW_Loft.usd"
