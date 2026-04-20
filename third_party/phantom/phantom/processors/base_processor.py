import os
import json
import logging
import numpy as np
import shutil
import errno
from typing import Tuple
from pathlib import Path
from omegaconf import DictConfig

from phantom.utils.data_utils import get_parent_folder_of_package
from phantom.utils.image_utils import get_intrinsics_from_json, get_transformation_matrix_from_extrinsics
from phantom.processors.paths import Paths, PathsConfig

logger = logging.getLogger(__name__)


class BaseProcessor: 
    def __init__(self, cfg: DictConfig): 
        # Store configuration for potential future use
        self.cfg = cfg
        
        # Apply configuration to instance attributes
        self._apply_config(cfg)
        
        # Validate configuration
        self._validate_config(cfg)
        
        # Set up paths and data folders
        self._setup_paths_and_folders(cfg)
        
        # Initialize camera parameters
        self._init_camera_parameters()

    def _apply_config(self, cfg: DictConfig) -> None:
        """Apply configuration to instance attributes."""
        # Basic attributes
        self.input_resolution = cfg.input_resolution
        self.output_resolution = cfg.output_resolution
        self.project_folder = get_parent_folder_of_package("phantom")
        self.debug = cfg.debug
        self.n_processes = cfg.n_processes
        self.verbose = cfg.verbose
        self.skip_existing = cfg.skip_existing
        self.robot = cfg.robot
        self.gripper = cfg.gripper
        self.square = cfg.square
        self.epic = cfg.epic
        self.bimanual_setup = cfg.bimanual_setup
        self.target_hand = cfg.target_hand
        self.constrained_hand = cfg.constrained_hand
        self.depth_for_overlay = cfg.depth_for_overlay
        self.render = cfg.render
        self.debug_cameras = getattr(cfg, 'debug_cameras', [])
        
        # Apply bimanual setup logic
        if self.bimanual_setup != "single_arm":
            self.target_hand = "both"

    def _validate_config(self, cfg: DictConfig) -> None:
        """Validate critical configuration parameters."""
        if cfg.input_resolution <= 0 or cfg.output_resolution <= 0:
            raise ValueError(f"Resolutions must be positive: input={cfg.input_resolution}, output={cfg.output_resolution}")
        
        if not os.path.exists(cfg.data_root_dir):
            raise FileNotFoundError(f"Data root directory not found: {cfg.data_root_dir}")
        
        if not os.path.exists(cfg.camera_intrinsics):
            raise FileNotFoundError(f"Camera intrinsics file not found: {cfg.camera_intrinsics}")

    def _setup_paths_and_folders(self, cfg: DictConfig) -> None:
        """Set up paths configuration and create necessary directories."""
        # Set up paths configuration
        self.paths_config = PathsConfig()
        self.paths_config.config['data_root'] = cfg.data_root_dir
        self.paths_config.config['processed_root'] = cfg.processed_data_root_dir

        self.data_folder = os.path.join(cfg.data_root_dir, cfg.demo_name)
        self.processed_data_folder = os.path.join(cfg.processed_data_root_dir, cfg.demo_name)

        # Validate that data folder exists
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")

        os.makedirs(self.processed_data_folder, exist_ok=True)

        # Get all folders in data_folder
        try:
            all_data_folders = [d1 for d1 in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, d1))]
            self.all_data_folders = sorted(all_data_folders, key=lambda x: int(x))
            self.all_data_folders_idx = {x: idx for idx, x in enumerate(self.all_data_folders)}
        except OSError as e:
            if e.errno == errno.EACCES:
                raise PermissionError(f"Permission denied accessing data folder: {self.data_folder}")
            elif e.errno == errno.ENOENT:
                raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
            else:
                raise RuntimeError(f"OS error accessing data folder {self.data_folder}: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid folder name format in {self.data_folder}. Folders should be numbered: {e}")

    def _init_camera_parameters(self) -> None:
        """Initialize camera intrinsics and extrinsics."""
        # Get camera intrinsics and extrinsics
        self.intrinsics_dict, self.intrinsics_matrix = self.get_intrinsics(self.cfg.camera_intrinsics)

        # Use camera_extrinsics from config if available, otherwise determine from bimanual_setup
        if hasattr(self.cfg, 'camera_extrinsics') and self.cfg.camera_extrinsics:
            camera_extrinsics_path = self.cfg.camera_extrinsics
        else:
            camera_extrinsics_path = self._get_camera_extrinsics_path()
            
        self.T_cam2robot, self.extrinsics = self.get_extrinsics(camera_extrinsics_path)

    def _get_camera_extrinsics_path(self) -> str:
        """Get the appropriate camera extrinsics path based on bimanual setup."""
        if self.bimanual_setup == "shoulders":
            return "camera/camera_extrinsics_ego_bimanual_shoulders.json"
        elif self.bimanual_setup == "single_arm":
            return "camera/camera_extrinsics.json"
        else:
            raise ValueError(f"Invalid bimanual setup: {self.bimanual_setup}. Must be 'single_arm' or 'shoulders'.")
    
    def get_paths(self, data_path: str) -> Paths:
        """
        Get all file paths for a demo.
        
        Args:
            data_path: Path to the demo data
            
        Returns:
            Paths object containing all file paths
        """
        paths = Paths(
            data_path=Path(data_path),
            robot_name=self.robot
        )
        paths.ensure_directories_exist()
        return paths
    
    def get_save_folder(self, data_sub_folder: str) -> str:
        data_sub_folder_fullpath = os.path.join(self.data_folder, str(data_sub_folder))
        save_folder = os.path.join(self.processed_data_folder, str(data_sub_folder))
        # Check existing dirs using os.scandir
        with os.scandir(self.processed_data_folder) as it:
            existing_dirs = {entry.name for entry in it if entry.is_dir()}
        if str(data_sub_folder) not in existing_dirs:
            shutil.copytree(data_sub_folder_fullpath, save_folder)
        return save_folder
    
    def process_one_demo(self, data_sub_folder: str): 
        raise NotImplementedError

    def get_intrinsics(self, intrinsics_path: str) -> Tuple[dict, np.ndarray]:
        intrinsics_matrix, intrinsics_dict = get_intrinsics_from_json(intrinsics_path)
        if self.square:
            intrinsics_dict, intrinsics_matrix = self.update_intrinsics_for_square_image(self.input_resolution,
                                                                                        intrinsics_dict, 
                                                                                        intrinsics_matrix)
        return intrinsics_dict, intrinsics_matrix
    
    def get_extrinsics(self, extrinsics_path: str) -> Tuple[np.ndarray, dict]:
        """Load and process camera extrinsics from JSON file.
        
        Args:
            extrinsics_path: Path to the extrinsics JSON file
            
        Returns:
            Tuple of (transformation_matrix, extrinsics_dict)
            
        Raises:
            FileNotFoundError: If extrinsics file doesn't exist
            json.JSONDecodeError: If extrinsics file is invalid JSON
            ValueError: If extrinsics data is invalid
        """
        if not os.path.exists(extrinsics_path):
            raise FileNotFoundError(f"Camera extrinsics file not found: {extrinsics_path}")
            
        try:
            with open(extrinsics_path, "r") as f:
                camera_extrinsics = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in extrinsics file {extrinsics_path}: {str(e)}")
        
        try:
            T_cam2robot = get_transformation_matrix_from_extrinsics(camera_extrinsics)
        except Exception as e:
            raise ValueError(f"Failed to process extrinsics data from {extrinsics_path}: {str(e)}")
            
        return T_cam2robot, camera_extrinsics

    @staticmethod
    def update_intrinsics_for_square_image(img_h: int, intrinsics_dict: dict, 
                                           intrinsics_matrix: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        Adjusts camera intrinsic parameters for a square image by modifying the principal point offset.

        Args:
            img_h (int): Height of the image (assumed to be square).
            intrinsics_dict (dict): Dictionary of intrinsic parameters.
            intrinsics_matrix (np.ndarray): Intrinsic matrix.

        Returns:
            Tuple[dict, np.ndarray]: Updated intrinsic parameters and matrix.
        """
        img_w = img_h * 16 // 9
        offset = (img_w - img_h) // 2
        intrinsics_dict["cx"] -= offset
        intrinsics_matrix[0, 2] -= offset
        return intrinsics_dict, intrinsics_matrix
