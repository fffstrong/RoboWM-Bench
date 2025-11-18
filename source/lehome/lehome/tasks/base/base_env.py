from __future__ import annotations
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from .base_env_cfg import BaseEnvCfg


class BaseEnv(DirectRLEnv):
    """Base environment class for LW_Loft scene."""

    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale

    def _setup_scene(self):
        """Setup the base scene with LW_Loft and robot."""

        # Setup LW_Loft scene
        loft = sim_utils.UsdFileCfg(usd_path=self.cfg.path_scene)
        loft.func(
            "/World/Scene",
            loft,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=5000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-physics step processing."""
        self.actions = self.action_scale * actions.clone()
        # step counter increments once per control step

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        pass

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        pass

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        pass

