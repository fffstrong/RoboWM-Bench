from __future__ import annotations
from typing import Any
from collections.abc import Sequence
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from pxr import UsdShade, Sdf
import omni.kit.commands
from isaacsim.core.utils.prims import is_prim_path_valid
from .towel_cfg import TowelEnvCfg
from lehome.assets.scenes.byobu_table import BYOBU_TABLE_USD_PATH
from lehome.devices.action_process import preprocess_device_action
from lehome.assets.object.Garment import GarmentObject
from isaaclab.controllers import DifferentialIKController
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


class TowelEnv(DirectRLEnv):
    cfg: TowelEnvCfg

    def __init__(self, cfg: TowelEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.action_scale = self.cfg.action_scale
        self.object_A = None  # Will be created in _setup_scene
        self.garment_config = OmegaConf.load("Assets/human_assets/Towel/Towel.json")
        self.particle_config = OmegaConf.load(
            "source/lehome/lehome/tasks/human_task/Task10_Franka_Tableware_Towel/config_file/particle_garment_cfg.yaml"
        )
        if cfg.use_random_seed:
            # Use random seed (no fixed seed)
            self.garment_rng = np.random.RandomState()
        else:
            # Use fixed seed from config
            self.garment_rng = np.random.RandomState(cfg.random_seed)
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos

        self.ik_controller = DifferentialIKController(
            self.cfg.ik_controller, num_envs=self.scene.num_envs, device=self.device
        )
        self.ee_frame_name = "panda_hand"
        try:
            self.ee_idx = self.robot.find_bodies(self.ee_frame_name)[0][0]
        except (IndexError, RuntimeError):
            self.ee_idx = self.robot.num_bodies - 1
            print(f"[Warn] {self.ee_frame_name} not found, using last body.")

        cfg.viewer = cfg.viewer.replace(
            eye=(0, -1.2, 1.3),
            lookat=(0, 6.4, -2.8),
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.top_camera2 = TiledCamera(self.cfg.top_camera2)

        cfg = sim_utils.UsdFileCfg(usd_path=f"{BYOBU_TABLE_USD_PATH}")
        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Create garment object with selected asset
        self._create_garment_object()
        self.initialize_obs()

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["top_camera2"] = self.top_camera2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=900, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.joint_num = 9

    def _create_garment_object(self):
        """
        Create a new GarmentObject with the currently selected asset.
        """
        if self.object_A is not None:
            self._delete_garment_object()

        # Generate prim_path based on garment_name, default to "Cloth" if not specified
        garment_name = getattr(self.cfg, "c", None)
        if garment_name and garment_name.strip():
            prim_name = garment_name.strip()
        else:
            prim_name = "Cloth"

        prim_path = f"/World/Object/{prim_name}"

        try:
            if is_prim_path_valid(prim_path):
                logger.debug(
                    f"[GarmentEnv] Prim path {prim_path} still exists, deleting before creation"
                )
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                if hasattr(self, "sim") and self.sim is not None:
                    for _ in range(5):
                        self.sim.step(render=False)
                if is_prim_path_valid(prim_path):
                    logger.warning(
                        f"[GarmentEnv] WARNING: Prim path {prim_path} still exists after deletion attempt!"
                    )
                else:
                    logger.debug(
                        f"[GarmentEnv] Prim path {prim_path} successfully deleted"
                    )
        except Exception as e:
            logger.debug(
                f"[GarmentEnv] Could not delete existing prim (may not exist): {e}"
            )

        # Create new garment object
        try:
            print(f"[GarmentEnv] Creating GarmentObject at prim_path: {prim_path}")
            self.object_A = GarmentObject(
                prim_path=prim_path,
                particle_config=self.particle_config,
                garment_config=self.garment_config,
                rng=self.garment_rng,
            )
            logger.info("[GarmentEnv] GarmentObject created successfully")
        except Exception as e:
            logger.error(f"[GarmentEnv] Failed to create GarmentObject: {e}")
            raise RuntimeError(f"Failed to create GarmentObject: {e}") from e

        # Validate created object
        self._validate_created_object()

        self.texture_cfg = self.particle_config.objects.get("texture_randomization", {})
        self.light_cfg = self.particle_config.objects.get("light_randomization", {})
        logger.debug(
            f"[GarmentEnv] Loaded texture_cfg: {bool(self.texture_cfg)}, light_cfg: {bool(self.light_cfg)}"
        )

    def _validate_created_object(self):
        """
        Validate that the GarmentObject was created successfully and has required attributes.

        Raises:
            RuntimeError: If object validation fails
        """
        logger.debug("[GarmentEnv] Validating created GarmentObject...")

        if self.object_A is None:
            raise RuntimeError("GarmentObject creation returned None")

        required_attrs = [
            "usd_prim_path",
            "mesh_prim_path",
            "particle_system_path",
            "particle_material_path",
        ]

        for attr in required_attrs:
            if not hasattr(self.object_A, attr):
                raise RuntimeError(f"GarmentObject missing required attribute: {attr}")

            attr_value = getattr(self.object_A, attr)
            if attr_value is None:
                raise RuntimeError(f"GarmentObject attribute {attr} is None")

        prim_paths_to_check = [
            ("usd_prim_path", self.object_A.usd_prim_path),
            ("mesh_prim_path", self.object_A.mesh_prim_path),
        ]

        for path_name, path_value in prim_paths_to_check:
            if not is_prim_path_valid(path_value):
                logger.warning(
                    f"[GarmentEnv] Prim path {path_name} '{path_value}' is not valid in stage. "
                    "This may be expected if the prim hasn't been added yet."
                )
            else:
                logger.debug(
                    f"[GarmentEnv] Prim path {path_name} '{path_value}' is valid"
                )

        logger.debug("[GarmentEnv] GarmentObject validation passed")

    def _delete_garment_object(self):
        """Delete the current garment object from the stage.

        This method ensures complete cleanup of the garment object, including:
        - USD prim deletion
        - Particle system cleanup
        - All child prims removal
        """
        if self.object_A is None:
            return

        try:
            # Try to get prim_path from object first, then fallback to garment_name-based path
            if hasattr(self.object_A, "usd_prim_path") and self.object_A.usd_prim_path:
                prim_path = self.object_A.usd_prim_path
            else:
                # Fallback: generate prim_path based on garment_name, same logic as creation
                garment_name = getattr(self.cfg, "garment_name", None)
                if garment_name and garment_name.strip():
                    prim_name = garment_name.strip()
                else:
                    prim_name = "Cloth"
                prim_path = f"/World/Object/{prim_name}"

            if hasattr(self.object_A, "particle_system_path"):
                particle_path = self.object_A.particle_system_path
                try:
                    if is_prim_path_valid(particle_path):
                        omni.kit.commands.execute("DeletePrims", paths=[particle_path])
                        logger.debug(
                            f"[GarmentEnv] Deleted particle system at {particle_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[GarmentEnv] Failed to delete particle system: {e}"
                    )

            if is_prim_path_valid(prim_path):
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                logger.debug(f"[GarmentEnv] Deleted garment prim at {prim_path}")
            else:
                logger.warning(
                    f"[GarmentEnv] Prim path {prim_path} is not valid, skipping deletion"
                )

        except Exception as e:
            logger.warning(f"[GarmentEnv] Failed to delete garment object: {e}")
            import traceback

            traceback.print_exc()

        self.object_A = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7]
        ee_pos_curr = ee_pose_w[:, 0:3]
        ee_quat_curr = ee_pose_w[:, 3:7]

        ik_command = self.actions[:, :6]

        self.ik_controller.set_command(
            ik_command, ee_pos=ee_pos_curr, ee_quat=ee_quat_curr
        )

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx, :, :]
        joint_pos_curr = self.robot.data.joint_pos

        joint_pos_target = self.ik_controller.compute(
            ee_pos_curr, ee_quat_curr, jacobian, joint_pos_curr
        )

        gripper_pos_target = self.actions[:, 6].view(-1, 1).repeat(1, 2)

        if joint_pos_target.shape[1] == 7:
            full_joint_target = torch.cat([joint_pos_target, gripper_pos_target], dim=1)
        else:
            full_joint_target = joint_pos_target
            full_joint_target[:, 7:9] = gripper_pos_target

        self.robot.set_joint_position_target(full_joint_target)

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(self.joint_num)], dim=-1
        )
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": top_camera_depth.cpu().detach().numpy().copy(),
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Calculate distance-based reward for garment folding task."""
        if self.object_A is None or not hasattr(self.object_A, "_cloth_prim_view"):
            return 0.0

        result = self._evaluate_towel_success()
        if not isinstance(result, dict):
            return getattr(self, "_last_computed_reward", 0.0)

        success = result.get("success", False)
        details = result.get("details", {})

        if success:
            self._last_computed_reward = 1.0
            return 1.0

        num_conditions = len(details)
        if num_conditions == 0:
            return 0.0

        primary_rewards = []
        secondary_rewards = []

        for cond_key, cond_info in details.items():
            value = cond_info.get("value", 0.0)
            threshold = cond_info.get("threshold", 0.0)
            passed = cond_info.get("passed", False)
            description = cond_info.get("description", "")
            is_less_than = "<=" in description

            if passed:
                condition_reward = 1.0
            else:
                if is_less_than:
                    if threshold > 0:
                        excess_ratio = max(0.0, (value - threshold) / threshold)
                        condition_reward = np.exp(-3.0 * excess_ratio)
                    else:
                        condition_reward = 0.0
                else:
                    if threshold > 0:
                        ratio = value / threshold
                        condition_reward = max(0.0, 1.0 - np.exp(-1.5 * (1.0 - ratio)))
                    else:
                        condition_reward = 0.0

            if is_less_than:
                primary_rewards.append(condition_reward)
            else:
                secondary_rewards.append(condition_reward)

        num_primary = len(primary_rewards)
        num_secondary = len(secondary_rewards)

        if num_primary > 0:
            avg_primary = sum(primary_rewards) / num_primary
            min_primary = min(primary_rewards) if primary_rewards else 0.0
            primary_score = (avg_primary**0.7) * (min_primary**0.3)
        else:
            primary_score = 1.0

        if num_secondary > 0:
            avg_secondary = sum(secondary_rewards) / num_secondary
            secondary_score = avg_secondary
        else:
            secondary_score = 1.0

        final_reward = (0.8 * primary_score + 0.2 * secondary_score) * 0.9
        self._last_computed_reward = float(final_reward)
        return float(final_reward)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _evaluate_towel_success(self) -> dict:
        """Check the distances between checkpoints 0-3, 1-4, and 2-5."""
        if self.object_A is None or not hasattr(self.object_A, "_cloth_prim_view"):
            return {"success": False, "details": {}}

        check_points = self.object_A.check_points
        success_distance = self.object_A.success_distance

        # Ensure checkpoints are correctly positioned from the config
        if not check_points or len(check_points) < 6:
            return {"success": False, "details": {}}

        # Get the world coordinates of all current particles
        try:
            _, mesh_points, _, _ = self.object_A.get_current_mesh_points()
            positions = mesh_points * 100.0  # (N, 3), scale up to centimeter system
        except Exception:
            positions = self.object_A._get_points_pose().detach().cpu().numpy() * 100.0

        # Get matching particle coordinates
        p0 = positions[check_points[0]]
        p1 = positions[check_points[1]]
        p2 = positions[check_points[2]]
        p3 = positions[check_points[3]]
        p4 = positions[check_points[4]]
        p5 = positions[check_points[5]]

        # Calculate distances of corresponding matching points
        dist_0_3 = np.linalg.norm(p0 - p3)
        dist_1_4 = np.linalg.norm(p1 - p4)
        dist_2_5 = np.linalg.norm(p2 - p5)

        # Read the success thresholds for these 3 pairs from Towel.json, assign a default value (like 15) if missing
        t_0_3 = success_distance[0] if len(success_distance) > 0 else 15
        t_1_4 = success_distance[1] if len(success_distance) > 1 else 15
        t_2_5 = success_distance[2] if len(success_distance) > 2 else 15

        cond_0_3 = bool(dist_0_3 <= t_0_3)
        cond_1_4 = bool(dist_1_4 <= t_1_4)
        cond_2_5 = bool(dist_2_5 <= t_2_5)

        success = cond_0_3 and cond_1_4 and cond_2_5

        # Provide details to _get_rewards to compute dense rewards
        details = {
            "cond_0_3": {
                "value": float(dist_0_3),
                "threshold": float(t_0_3),
                "passed": cond_0_3,
                "description": "Distance 0-3 <=",
            },
            "cond_1_4": {
                "value": float(dist_1_4),
                "threshold": float(t_1_4),
                "passed": cond_1_4,
                "description": "Distance 1-4 <=",
            },
            "cond_2_5": {
                "value": float(dist_2_5),
                "threshold": float(t_2_5),
                "passed": cond_2_5,
                "description": "Distance 2-5 <=",
            },
        }

        return {"success": success, "details": details}

    def _check_success(self) -> bool:
        """Check success based on Towel type."""
        result = self._evaluate_towel_success()
        return result.get("success", False)

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        result = self._evaluate_towel_success()

        if result.get("details"):
            for key, cond_info in result.get("details", {}).items():
                status = "✓" if cond_info.get("passed", False) else "✗"
                # logger.info(f"  {cond_info.get('description', '')} [{cond_info.get('value', 0):.2f}/{cond_info.get('threshold', 0):.2f}] -> {status}")

            success = result.get("success", False)
            # logger.info(f"[Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}")
        else:
            success = False

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

        self.ik_controller.reset(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(
            joint_pos, joint_ids=None, env_ids=env_ids
        )

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()

        self.robot_reset_state = np.array(
            robot_root_state.cpu().detach(), dtype=np.float32
        )

        # Reset the garment object
        if self.object_A is not None:
            self.object_A.reset()

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
        self.object_A.initialize()

    def get_all_pose(self):
        return self.object_A.get_all_pose()

    def set_all_pose(self, pose):
        self.object_A.set_all_pose(pose)

    def switch_garment(self, garment_name: str, garment_version: str = None):
        """Switch to a different garment without recreating the environment.

        This method allows reusing the same environment instance for different garments,
        which is much faster than closing and recreating the environment.

        Args:
            garment_name: Name of the garment to switch to (e.g., "Top_Long_Seen_0")
            garment_version: Version of the garment ("Release" or "Holdout"),
                            defaults to current cfg.garment_version
        """
        logger.info(
            f"[GarmentEnv] Switching garment to: {garment_name} (version: {garment_version})"
        )

        if self.object_A is not None:
            self._delete_garment_object()
            logger.info("[GarmentEnv] Old garment object deleted")

        if garment_version is None:
            garment_version = self.cfg.garment_version

        # Update config
        self.cfg.garment_name = garment_name
        self.cfg.garment_version = garment_version

        # Reload garment configuration
        self.garment_config = self.garment_loader.load_garment_config(
            garment_name, garment_version
        )
        logger.debug(f"[GarmentEnv] Garment config reloaded for {garment_name}")

        # solve particle ditorition
        logger.debug(
            f"[GarmentEnv] Running physics steps to clean up old particle system..."
        )
        cleanup_steps = 20

        if hasattr(self, "sim") and self.sim is not None:
            for i in range(cleanup_steps):
                try:
                    self.sim.step(render=False)
                    # Log progress every 5 steps
                    if (i + 1) % 5 == 0:
                        logger.debug(
                            f"[GarmentEnv] Cleanup progress: {i+1}/{cleanup_steps}"
                        )
                except Exception as e:
                    logger.warning(f"[GarmentEnv] Error during cleanup step {i+1}: {e}")
                    # Continue with next step
                    continue
            logger.debug(f"[GarmentEnv] Cleanup physics steps completed")
        else:
            logger.warning(f"[GarmentEnv] sim not available, skipping cleanup steps")

        # create new garment object
        self._create_garment_object()
        logger.debug(f"[GarmentEnv] New garment object created for {garment_name}")
        logger.debug(
            f"[GarmentEnv] Running initial physics steps to register prim in stage..."
        )
        initial_steps = 5
        if hasattr(self, "sim") and self.sim is not None:
            for i in range(initial_steps):
                try:
                    self.sim.step(render=False)
                except Exception as e:
                    logger.warning(f"[GarmentEnv] Error during initial step {i+1}: {e}")
            logger.debug(f"[GarmentEnv] Initial physics steps completed")
        else:
            logger.warning(f"[GarmentEnv] sim not available, skipping initial steps")
        if hasattr(self, "render"):
            try:
                self.render()
                logger.debug(f"[GarmentEnv] Render called after initial physics steps")
            except Exception as e:
                logger.warning(
                    f"[GarmentEnv] Error during render after initial steps: {e}"
                )

        try:
            self.initialize_obs()
            logger.debug(
                f"[GarmentEnv] Observation system initialized for {garment_name}"
            )
            if hasattr(self, "render"):
                try:
                    self.render()
                    logger.debug(
                        f"[GarmentEnv] Render called after observation initialization"
                    )
                except Exception as e:
                    logger.debug(
                        f"[GarmentEnv] Error during render after observation init: {e}"
                    )
        except Exception as e:
            logger.warning(
                f"[GarmentEnv] Failed to initialize observations (may be expected): {e}"
            )

    def cleanup(self):
        """Cleanup method (defensive programming).

        Note: When environments are fully closed and recreated (as in eval.py),
        this cleanup is not strictly necessary since _create_garment_object()
        already handles checking and cleaning up existing prims when creating
        a new environment. However, this method is kept as a safety measure
        for cases where the same environment instance might be reused.
        """
        logger.debug("[GarmentEnv] Starting cleanup...")

        # Delete garment object if it exists
        if self.object_A is not None:
            self._delete_garment_object()
            logger.debug("[GarmentEnv] Garment object cleaned up")

        # Clear references
        self.object_A = None
        # Note: Don't clear garment_config and particle_config as they might be needed
        # if the environment is reset rather than recreated

        logger.debug("[GarmentEnv] Cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            if hasattr(self, "object") and self.object_A is not None:
                self.cleanup()
        except Exception:
            # Ignore errors during destruction
            pass
