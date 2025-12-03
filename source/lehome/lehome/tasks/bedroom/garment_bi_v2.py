from __future__ import annotations
import torch
from dataclasses import MISSING
from typing import Any, Dict, List

from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera
from pxr import Usd, UsdShade, Sdf
import random
import omni.kit.commands
from isaacsim.core.utils.prims import is_prim_path_valid

from .garment_bi_cfg_v2 import GarmentEnvCfg
from lehome.utils.success_checker_v2 import (
    success_checker_top_long_sleeve_fold,
    success_checker_top_short_sleeve_fold,
    success_checker_short_pant_fold,
    success_checker_long_pant_fold,
)
from lehome.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
from lehome.devices.action_process import preprocess_device_action
from lehome.assets.object.Garment import GarmentObject
from lehome.tasks.bedroom.garment_assets import GARMENT_ASSETS, VALID_GARMENT_TYPES
from omegaconf import OmegaConf
import numpy as np
import os


class GarmentEnv(DirectRLEnv):
    cfg: GarmentEnvCfg  

    def __init__(self, cfg: GarmentEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.action_scale = self.cfg.action_scale

        # Validate garment type
        if self.cfg.garment_type not in VALID_GARMENT_TYPES:
            raise ValueError(
                f"Invalid garment_type: {self.cfg.garment_type}. "
                f"Valid types: {VALID_GARMENT_TYPES}"
            )

        # Get asset list for this garment type
        self.garment_assets = GARMENT_ASSETS[self.cfg.garment_type]
        if len(self.garment_assets) == 0:
            raise ValueError(
                f"No assets available for garment_type: {self.cfg.garment_type}"
            )

        # Current garment index (for teleop mode, fixed; for eval mode, random on reset)
        self.current_garment_index = None
        self.object = None  # Will be created in _setup_scene
        self.garment_config = OmegaConf.load(
            "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
        )

        super().__init__(cfg, render_mode, **kwargs)
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos
        
    def _select_garment_asset(self):
        """Select garment asset based on garment_index config.
        If garment_index is None (eval mode), randomly select from available assets.
        If garment_index is set (teleop mode), use the specified index.
        """
        if self.cfg.garment_index is not None:
            # Teleop mode: use specified index
            if self.cfg.garment_index < 0 or self.cfg.garment_index >= len(
                self.garment_assets
            ):
                raise ValueError(
                    f"garment_index {self.cfg.garment_index} out of range [0, {len(self.garment_assets)-1}]"
                )
            self.current_garment_index = self.cfg.garment_index
        else:
            # Eval mode: random selection (will be reselected on each reset)
            self.current_garment_index = random.randint(0, len(self.garment_assets) - 1)

        self.current_garment_path = self.garment_assets[self.current_garment_index]

    def _setup_scene(self):
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        cfg = sim_utils.UsdFileCfg(usd_path=f"{KITCHEN_WITH_ORANGE_USD_PATH}")
        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Select initial garment asset
        self._select_garment_asset()

        # Create garment object with selected asset
        self._create_garment_object()

        # add articulation to scene
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_garment_object(self):
        """Create a new GarmentObject with the currently selected asset."""
        # Delete existing garment object if it exists
        if self.object is not None:
            self._delete_garment_object()

        # Create new garment object
        self.object = GarmentObject(
            prim_path="/World/Object/Cloth",
            usd_path=self.current_garment_path,
            visual_usd_path=None,
            config=self.garment_config,
        )

        # Store texture and light configs
        self.texture_cfg = self.object.config.objects.get("texture_randomization", {})
        self.light_cfg = self.object.config.objects.get("light_randomization", {})

    def _delete_garment_object(self):
        """Delete the current garment object from the stage."""
        if self.object is None:
            return

        try:
            # Delete the prim and all its children
            prim_path = self.object.usd_prim_path
            if is_prim_path_valid(prim_path):
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
        except Exception as e:
            print(f"[Warn] Failed to delete garment object: {e}")

        self.object = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        left_camera_rgb = self.left_camera.data.output["rgb"]
        right_camera_rgb = self.right_camera.data.output["rgb"]
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        success = self._check_success()
        if success:
            total_reward = 1
        else:
            total_reward = 0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _check_success(self) -> bool:
        """Check success based on garment type."""
        garment_type = self.cfg.garment_type
        if garment_type == "top-long-sleeve":
            return success_checker_top_long_sleeve_fold(self.object)
        elif garment_type == "top-short-sleeve":
            return success_checker_top_short_sleeve_fold(self.object)
        elif garment_type == "short-pant":
            return success_checker_short_pant_fold(self.object)
        elif garment_type == "long-pant":
            return success_checker_long_pant_fold(self.object)
        else:
            raise ValueError(f"Unknown garment_type: {garment_type}")

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = self._check_success()
        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]
        self.left_arm.write_joint_position_to_sim(
            left_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.right_arm.write_joint_position_to_sim(
            right_joint_pos, joint_ids=None, env_ids=env_ids
        )

        # In eval mode (garment_index is None), randomly select a new garment on each reset
        if self.cfg.garment_index is None:
            old_index = self.current_garment_index
            self._select_garment_asset()
            # If the selected garment is different, recreate the GarmentObject
            if old_index != self.current_garment_index:
                self._create_garment_object()

        # Reset the garment object
        if self.object is not None:
            self.object.reset()

        # Apply randomization if enabled in config
        if self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()

        if self.light_cfg.get("enable", False):
            self._randomize_light()

    def _randomize_table038_texture(self):
        """Randomize Table038 texture based on config."""
        if not self.texture_cfg.get("enable", False):
            return

        folder = self.texture_cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(self.texture_cfg.get("min_id", 1))
        max_id = int(self.texture_cfg.get("max_id", 1))
        shader_path = self.texture_cfg.get("prim_path", "")

        if not folder or not os.path.exists(folder):
            print(f"[Reset][Warn] Texture folder not found: {folder}")
            return
        if not shader_path:
            print("[Reset][Warn] No prim_path provided for texture randomization")
            return

        stage = self.scene.stage
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            print(f"[Reset][Warn] Shader prim not found at {shader_path}")
            return

        shader = UsdShade.Shader(shader_prim)
        idx = random.randint(min_id, max_id)
        tex_path = os.path.join(folder, f"{idx}.png")

        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            print("[Reset][Warn] No texture input found on shader")
            return

        tex_input.Set(Sdf.AssetPath(tex_path))
        # print(f"[Reset] Texture randomized -> {tex_path}")

    def _randomize_light(self):
        """Randomize DomeLight attributes based on config."""
        if not self.light_cfg.get("enable", False):
            return

        prim_path = self.light_cfg.get("prim_path", "/World/Light")
        intensity_range = self.light_cfg.get("intensity_range", [800, 2000])
        color_range = self.light_cfg.get("color_range", [0.0, 1.0])

        stage = self.scene.stage
        light_prim = stage.GetPrimAtPath(prim_path)
        if not light_prim.IsValid():
            print(f"[Reset][Warn] Light prim not found at {prim_path}")
            return

        intensity = random.uniform(*intensity_range)
        color = tuple[float, float, float](
            random.uniform(color_range[0], color_range[1]) for _ in range(3)
        )

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)

        # print(f"[Reset] Light randomized -> intensity={intensity:.1f}, color={color}")

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()

    def get_all_pose(self):
        return self.object.get_all_pose()

    def set_all_pose(self, pose):
        self.object.set_all_pose(pose)
