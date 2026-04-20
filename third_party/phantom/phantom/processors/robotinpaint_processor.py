"""
Robot Inpainting Processor Module

This module uses MuJoCo to render robot models and overlay them onto human demonstration videos.

Processing Pipeline:
1. Load smoothed robot trajectories from previous processing stages
2. Initialize MuJoCo robot simulation with calibrated camera parameters
3. For each frame:
   - Move simulated robot to target pose from human demonstration
   - Render robot from calibrated camera viewpoint
   - Apply depth-based occlusion handling (Optional)
   - Create robot overlay on human demonstration video
4. Generate training data with robot state annotations
5. Save robot-inpainted videos and training data
"""

import os 
import pdb
import numpy as np
import cv2
from tqdm import tqdm
import mediapy as media
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

from phantom.processors.phantom_data import TrainingData, TrainingDataSequence, HandSequence
from phantom.processors.base_processor import BaseProcessor
from phantom.twin_bimanual_robot import TwinBimanualRobot, MujocoCameraParams
from phantom.twin_robot import TwinRobot
from phantom.processors.paths import Paths


logger = logging.getLogger(__name__)

@dataclass
class RobotState:
    """
    Container for robot state data including pose and gripper configuration.
    
    Attributes:
        pos: 3D position coordinates in world frame
        ori_xyzw: Quaternion orientation in XYZW format (scalar-last)
        gripper_pos: Gripper opening distance or action value
    """
    pos: np.ndarray
    ori_xyzw: np.ndarray
    gripper_pos: float

class RobotInpaintProcessor(BaseProcessor):  
    """
    Uses mujoco to overlay robot on human inpainted images.
    """
    # Processing constants for quality control and output formatting
    TRACKING_ERROR_THRESHOLD = 0.05  # Maximum tracking error in meters
    DEFAULT_FPS = 15                 # Standard frame rate for output videos
    DEFAULT_CODEC = "ffv1"          # Lossless codec for high-quality output

    def __init__(self, args: Any) -> None:
        """
        Initialize the robot inpainting processor with simulation parameters.
        
        Args:
            args: Command line arguments containing robot configuration,
                 camera parameters, and processing options
        """
        super().__init__(args)
        self.use_depth = self.depth_for_overlay
        self._initialize_robot()

    def _initialize_robot(self) -> None:
        """
        Initialize the twin robot simulation with calibrated camera parameters.
        """
        # Generate MuJoCo camera parameters from real-world calibration
        camera_params = self._get_mujoco_camera_params()  
        img_w, img_h = self._get_image_dimensions()
        
        # Initialize appropriate robot configuration
        if self.bimanual_setup == "single_arm":
            self.twin_robot = TwinRobot(
                self.robot, 
                self.gripper,
                camera_params,
                camera_height=img_h, 
                camera_width=img_w,
                render=self.render, 
                n_steps_short=3,    
                n_steps_long=75,    
                debug_cameras=self.debug_cameras,
                square=self.square,
            )
        else:
            self.twin_robot = TwinBimanualRobot(
                self.robot, 
                self.gripper, 
                self.bimanual_setup,
                camera_params,
                camera_height=img_h, 
                camera_width=img_w,
                render=self.render, 
                n_steps_short=10, 
                n_steps_long=75,
                debug_cameras=self.debug_cameras,
                epic=self.epic,
                joint_controller=False,  # Use operational-space control
            )

    def __del__(self):
        """Clean up robot simulation resources."""
        if hasattr(self, 'twin_robot'):
            self.twin_robot.close()

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process a single demonstration to create robot-inpainted visualization.
        
        Args:
            data_sub_folder: Path to demonstration data folder containing
                           smoothed trajectories and original video data
        """
        save_folder = self.get_save_folder(data_sub_folder)
        if self._should_skip_processing(save_folder):
            return
        paths = self.get_paths(save_folder)
        
        # Reinitialize robot simulation for each demo to ensure clean state
        self.__del__()
        self._initialize_robot()
        
        # Load and prepare demonstration data
        data = self._load_data(paths)
        images = self._load_images(paths, data["union_indices"])
        gripper_actions, gripper_widths = self._process_gripper_widths(paths, data)

        # Process all frames to generate robot overlays and training data
        sequence, img_overlay, img_birdview = self._process_frames(images, data, gripper_actions, gripper_widths)
        
        # Save comprehensive results
        self._save_results(paths, sequence, img_overlay, img_birdview)

    def _process_frames(self, images: Dict[str, np.ndarray], data: Dict[str, np.ndarray],
                       gripper_actions: Dict[str, np.ndarray], gripper_widths: Dict[str, np.ndarray]) -> Tuple[TrainingDataSequence, List[np.ndarray], Optional[List[np.ndarray]]]:
        """
        Process each frame to generate robot overlays and training data.
        
        Args:
            images: Dictionary containing human demonstration images and masks
            data: Robot trajectory data (positions and orientations)
            gripper_actions: Processed gripper action commands
            gripper_widths: Gripper opening distances
            
        Returns:
            Tuple containing:
                - TrainingDataSequence with robot state annotations
                - List of robot overlay images
                - Optional list of bird's eye view images (if debug cameras enabled)
        """
        sequence = TrainingDataSequence()
        img_overlay = []
        img_birdview = None
        if "birdview" in self.debug_cameras:
            img_birdview = []

        for idx in tqdm(range(len(images['human_imgs'])), desc="Processing frames"):
            # Extract robot states for current frame
            left_state = self._get_robot_state(
                data['ee_pts_left'][idx], 
                data['ee_oris_left'][idx], 
                gripper_widths['left'][idx]
            )
            right_state = self._get_robot_state(
                data['ee_pts_right'][idx], 
                data['ee_oris_right'][idx], 
                gripper_widths['right'][idx]
            )

            # Process individual frame with robot simulation
            frame_results = self._process_single_frame(
                images, left_state, right_state, idx
            )
            
            # Handle failed processing (tracking errors, simulation issues)
            if frame_results is None:
                print(f"sdfsdfsTracking error too large at frame {idx}, skipping")
                sequence.add_frame(TrainingData.create_empty_frame(
                    frame_idx=idx,
                ))
                img_overlay.append(np.zeros_like(images['human_imgs'][idx]))
                if "birdview" in self.debug_cameras:
                    img_birdview.append(np.zeros_like(images['human_imgs'][idx]))
            else:
                # Create comprehensive training data annotation
                sequence.add_frame(TrainingData(
                    frame_idx=idx,
                    valid=True,
                    action_pos_left=left_state.pos,
                    action_orixyzw_left=left_state.ori_xyzw,
                    action_pos_right=right_state.pos,
                    action_orixyzw_right=right_state.ori_xyzw,
                    action_gripper_left=gripper_actions['left'][idx],
                    action_gripper_right=gripper_actions['right'][idx],
                    gripper_width_left=gripper_widths['left'][idx],
                    gripper_width_right=gripper_widths['right'][idx],
                ))
                img_overlay.append(frame_results['rgb_robot_overlay'])
                if "birdview" in self.debug_cameras:
                    img_birdview.append(frame_results['birdview_img'])
        return sequence, img_overlay, img_birdview

    
    def _process_single_frame(self, images: Dict[str, np.ndarray],
                            left_state: RobotState,
                            right_state: RobotState,
                            idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single frame to generate robot overlay and validate tracking.
        
        Args:
            images: Dictionary containing human images and segmentation data
            left_state: Target state for left robot arm
            right_state: Target state for right robot arm
            idx: Frame index for initialization and logging
            
        Returns:
            Dictionary containing rendered robot overlay and debug camera views,
            or None if tracking error exceeds threshold
        """
        # Prepare robot target state based on configuration
        if self.bimanual_setup == "single_arm":
            if self.target_hand == "left":
                target_state = {
                    "pos": left_state.pos,
                    "ori_xyzw": left_state.ori_xyzw,
                    "gripper_pos": left_state.gripper_pos,
                }
            else:
                target_state = {
                    "pos": right_state.pos,
                    "ori_xyzw": right_state.ori_xyzw,
                    "gripper_pos": right_state.gripper_pos,
                }
        else:
            # Bimanual configuration requires coordinated control
            target_state = {
                "pos": [right_state.pos, left_state.pos],
                "ori_xyzw": [right_state.ori_xyzw, left_state.ori_xyzw],
                "gripper_pos": [right_state.gripper_pos, left_state.gripper_pos],
            }

        # Move robot to target state and get simulation results
        robot_results = self.twin_robot.move_to_target_state(
            target_state, init=(idx == 0)  # Initialize on first frame
        )

        # Validate tracking accuracy to ensure quality
        if self.bimanual_setup == "single_arm":
            if robot_results['pos_err'] > self.TRACKING_ERROR_THRESHOLD:
                print(f"Tracking error too large at frame {idx}, skipping", robot_results['pos_err'])
                logger.warning(f"Tracking error too large at frame {idx}, skipping")
                return None
        else:        
            if robot_results['left_pos_err'] > self.TRACKING_ERROR_THRESHOLD or robot_results['right_pos_err'] > self.TRACKING_ERROR_THRESHOLD:
                logger.warning(f"Tracking error too large at frame {idx}, skipping")
                return None

        # Generate robot overlay using appropriate method
        if self.use_depth:
            rgb_robot_overlay = self._process_robot_overlay_with_depth(
                images['human_imgs'][idx],
                images['human_masks'][idx],
                images['imgs_depth'][idx],
                robot_results
            )
        else:
            rgb_robot_overlay = self._process_robot_overlay(
                images['human_imgs'][idx], robot_results
            )

        # Prepare output with main overlay and debug camera views
        output = {
            'rgb_robot_overlay': rgb_robot_overlay,
        }

        # Add debug camera views if requested
        for cam in self.debug_cameras:
            output[f"{cam}_img"] = (robot_results[f"{cam}_img"] * 255).astype(np.uint8)

        return output

    def _should_skip_processing(self, save_folder: str) -> bool:
        """
        Check if processing should be skipped due to existing output files.
        
        Args:
            save_folder: Directory where output files would be saved
            
        Returns:
            True if processing should be skipped, False otherwise
        """
        if self.skip_existing:
            try:
                with os.scandir(save_folder) as it:
                    existing_files = {entry.name for entry in it if entry.is_file()}
                if str("video_overlay"+f"_{self.robot}_{self.bimanual_setup}.mkv") in existing_files:
                    print(f"Skipping existing demo {save_folder}")
                    return True
            except FileNotFoundError:
                return False
        return False

    def _load_data(self, paths: Paths) -> Dict[str, np.ndarray]:
        """
        Load robot trajectory data from smoothed action files.
        
        Args:
            paths: Paths object containing file locations
            
        Returns:
            Dictionary containing robot trajectory data and frame indices
        """
        if self.bimanual_setup == "single_arm":
            # Get paths based on target hand for single-arm operation
            smoothed_base = getattr(paths, f"smoothed_actions_{self.target_hand}")
            actions_base = getattr(paths, f"actions_{self.target_hand}")
            smoothed_actions_path = str(smoothed_base).replace(".npz", f"_{self.bimanual_setup}.npz")
            actions_path = str(actions_base).replace(".npz", f"_{self.bimanual_setup}.npz")
            
            # Load actual trajectory data for target hand
            ee_pts = np.load(smoothed_actions_path)["ee_pts"]
            ee_oris = np.load(smoothed_actions_path)["ee_oris"]
            
            # Create dummy data for non-target hand 
            dummy_pts = np.zeros((len(ee_pts), 3))
            dummy_oris = np.eye(3)[None, :, :].repeat(len(ee_oris), axis=0)
            
            # Create data dictionary with target hand data and dummy data for other hand
            other_hand = "right" if self.target_hand == "left" else "left"
            return {
                f'ee_pts_{self.target_hand}': ee_pts,
                f'ee_oris_{self.target_hand}': ee_oris,
                f'ee_pts_{other_hand}': dummy_pts,
                f'ee_oris_{other_hand}': dummy_oris,
                'union_indices': np.load(actions_path, allow_pickle=True)["union_indices"]
            }

        # Load bimanual trajectory data
        smoothed_actions_left_path = str(paths.smoothed_actions_left).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        smoothed_actions_right_path = str(paths.smoothed_actions_right).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        actions_left_path = str(paths.actions_left).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        return {
            'ee_pts_left': np.load(smoothed_actions_left_path)["ee_pts"],
            'ee_oris_left': np.load(smoothed_actions_left_path)["ee_oris"],
            'ee_pts_right': np.load(smoothed_actions_right_path)["ee_pts"],
            'ee_oris_right': np.load(smoothed_actions_right_path)["ee_oris"],
            'union_indices': np.load(actions_left_path, allow_pickle=True)["union_indices"]
        }

    def _load_images(self, paths: Paths, union_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Load and index human demonstration images and associated data.
        
        Args:
            paths: Paths object containing image file locations
            union_indices: Frame indices to extract from full video sequences
            
        Returns:
            Dictionary containing indexed human images, masks, and depth data
        """
        return {
            'human_masks': np.load(paths.masks_arm)[union_indices],
            'human_imgs': np.array(media.read_video(paths.video_human_inpaint))[union_indices],
            'imgs_depth': np.load(paths.depth)[union_indices] if self.use_depth else None
        }
    
    def _process_gripper_widths(self, paths: Paths, data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Process gripper distance data into robot action commands.
        
        Args:
            paths: Paths object containing smoothed action file locations
            data: Dictionary containing trajectory data and frame indices
            
        Returns:
            Tuple containing:
                - Dictionary of gripper action commands for each hand
                - Dictionary of gripper width values for each hand
        """
        if self.bimanual_setup == "single_arm":
            # Get the appropriate smoothed actions path based on target hand
            base_path = getattr(paths, f"smoothed_actions_{self.target_hand}")
            smoothed_actions_path = str(base_path).replace(".npz", f"_{self.bimanual_setup}.npz")
            
            # Compute gripper actions and widths from smoothed data
            actions, widths = self._compute_gripper_actions(
                np.load(smoothed_actions_path)["ee_widths"]
            )
            
            # Create return dictionaries with actions for target hand, zeros for the other
            num_indices = len(data['union_indices'])
            other_hand = "right" if self.target_hand == "left" else "left"
            
            return (
                {self.target_hand: actions, other_hand: np.zeros(num_indices)},
                {self.target_hand: widths, other_hand: np.zeros(num_indices)}
            )
        
        # Process bimanual gripper data
        smoothed_actions_left_path = str(paths.smoothed_actions_left).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        smoothed_actions_right_path = str(paths.smoothed_actions_right).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        left_actions, left_widths = self._compute_gripper_actions(
            np.load(smoothed_actions_left_path)["ee_widths"]
        )
        right_actions, right_widths = self._compute_gripper_actions(
            np.load(smoothed_actions_right_path)["ee_widths"]
        )
        return {'left': left_actions, 'right': right_actions}, {'left': left_widths, 'right': right_widths}


    def _compute_gripper_actions(self, list_gripper_dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert continuous gripper distances to discrete robot gripper actions.
        Args:
            list_gripper_dist: Array of gripper distances throughout trajectory
            
        Returns:
            Tuple containing:
                - Gripper action commands (0 for grasp, distance for open)
                - Processed gripper width values
        """
        try:
            # Analyze gripper distance range and determine grasp threshold
            min_val, max_val = np.min(list_gripper_dist), np.max(list_gripper_dist)
            thresh = min_val + 0.2 * (max_val - min_val)  # 20% above minimum
            
            # Classify gripper states: 0 = closed/grasping, 1 = open
            gripper_state = np.array([0 if dist < thresh else 1 for dist in list_gripper_dist])
            
            # Find range of grasping action
            min_idx_pos = np.where(gripper_state == 0)[0][0]
            max_idx_pos = np.where(gripper_state == 0)[0][-1]

            # Generate gripper action commands
            list_gripper_actions = []
            for idx in range(len(list_gripper_dist)):
                if min_idx_pos <= idx <= max_idx_pos:   
                    # During grasping phase: use grasp command (0) and limit distance
                    list_gripper_actions.append(0)
                    list_gripper_dist[idx] = np.min([list_gripper_dist[idx], thresh])
                else:
                    # Outside grasping phase: use distance as action command
                    list_gripper_actions.append(list_gripper_dist[idx])
        except:
            # Fallback: use distances directly if processing fails
            list_gripper_actions = list_gripper_dist.tolist()  
        
        return np.array(list_gripper_actions), list_gripper_dist
    
    def _get_robot_state(self, ee_pt: np.ndarray, ori_matrix: np.ndarray, gripper_dist: float) -> RobotState:
        """
        Convert trajectory data to robot state representation.
        
        Args:
            ee_pt: End-effector position in 3D space
            ori_matrix: 3x3 rotation matrix for end-effector orientation  
            gripper_dist: Gripper opening distance
            
        Returns:
            RobotState object containing pose and gripper information
        """
        # Convert rotation matrix to quaternion (XYZW format for robot control)
        ori_xyzw = Rotation.from_matrix(ori_matrix).as_quat(scalar_first=False)
        robot_state = RobotState(pos=ee_pt, ori_xyzw=ori_xyzw, gripper_pos=gripper_dist)
        return robot_state
    
    def _process_robot_overlay(self, img: np.ndarray, robot_results: Dict[str, Any]) -> np.ndarray:
        """
        Create robot overlay on human image using segmentation masks.
        
        Args:
            img: Original human demonstration image
            robot_results: Dictionary containing robot rendering results
            
        Returns:
            Image with robot overlay applied
        """
        # Extract robot rendering and segmentation data
        rgb_img_sim = (robot_results['rgb_img'] * 255).astype(np.uint8)
        H, W = rgb_img_sim.shape[:2]
        
        # Resize robot rendering and masks to match output resolution
        if self.square:
            rgb_img_sim = cv2.resize(rgb_img_sim, (self.output_resolution, self.output_resolution))
            robot_mask = cv2.resize(robot_results['robot_mask'], (self.output_resolution, self.output_resolution))
            robot_mask[robot_mask > 0] = 1
            gripper_mask = cv2.resize(robot_results['gripper_mask'], (self.output_resolution, self.output_resolution))
            gripper_mask[gripper_mask > 0] = 1
        else:
            rgb_img_sim = cv2.resize(rgb_img_sim, (int(W/H*self.output_resolution), self.output_resolution))
            robot_mask = cv2.resize(robot_results['robot_mask'], (int(W/H*self.output_resolution), self.output_resolution))
            robot_mask[robot_mask > 0] = 1
            gripper_mask = cv2.resize(robot_results['gripper_mask'], (int(W/H*self.output_resolution), self.output_resolution))
            gripper_mask[gripper_mask > 0] = 1
        
        # Create overlay by compositing robot over human image
        img_robot_overlay = img.copy()
        overlay_mask = (robot_mask == 1) | (gripper_mask == 1)
        img_robot_overlay[overlay_mask] = rgb_img_sim[overlay_mask]
        
        return img_robot_overlay

    def _process_robot_overlay_with_depth(self, img: np.ndarray, hand_mask: np.ndarray, 
                                    img_depth: np.ndarray, robot_results: Dict[str, Any]) -> np.ndarray:
        """
        Create depth-aware robot overlay with realistic occlusion handling.
        
        Args:
            img: Original human demonstration image
            hand_mask: Segmentation mask of human hand regions
            img_depth: Depth image corresponding to the demonstration
            robot_results: Dictionary containing robot rendering and depth results
            
        Returns:
            Image with depth-aware robot overlay applied
        """
        # Extract robot rendering and depth data
        robot_mask = robot_results['robot_mask']
        gripper_mask = robot_results['gripper_mask']
        rgb_img_sim = robot_results['rgb_img']
        depth_img_sim = np.squeeze(robot_results['depth_img'])
        H, W = rgb_img_sim.shape[:2]

        # Create masked depth images for occlusion analysis
        depth_sim_masked = self._create_masked_depth(depth_img_sim, robot_mask, gripper_mask)
        depth_masked = self._create_masked_depth(img_depth, robot_mask, gripper_mask)
        
        # Process hand mask for improved occlusion handling
        hand_mask = self._dilate_mask(hand_mask.astype(np.uint8))
        
        # Create overlay mask using depth-based occlusion
        img_robot_overlay = img.copy()
        overlay_mask = self._create_overlay_mask(
            robot_mask, gripper_mask, depth_masked, depth_sim_masked, hand_mask
        )

        # Convert and resize robot rendering
        rgb_img_sim = (rgb_img_sim * 255).astype(np.uint8)
        
        if self.square:
            resize_shape = (self.output_resolution, self.output_resolution)
        else:
            resize_shape = (int(W/H*self.output_resolution), self.output_resolution)

        # Apply final overlay with depth-aware occlusion
        rgb_img_sim = cv2.resize(rgb_img_sim, resize_shape)
        overlay_mask = cv2.resize(overlay_mask.astype(np.uint8), resize_shape)
        overlay_mask[overlay_mask > 0] = 1
        overlay_mask = overlay_mask.astype(bool)
        
        img_robot_overlay[overlay_mask] = rgb_img_sim[overlay_mask]
        
        return img_robot_overlay
    
    def _create_masked_depth(self, depth_img: np.ndarray, robot_mask: np.ndarray, 
                            gripper_mask: np.ndarray) -> np.ndarray:
        """
        Create depth image masked to robot regions for occlusion analysis.
        
        Args:
            depth_img: Input depth image
            robot_mask: Binary mask indicating robot regions
            gripper_mask: Binary mask indicating gripper regions
            
        Returns:
            Depth image with values only in robot/gripper regions
        """
        masked_img = np.zeros_like(depth_img)
        mask = (robot_mask == 1) | (gripper_mask == 1)
        masked_img[mask] = depth_img[mask]
        return masked_img

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological dilation to expand mask boundaries.
        
        Args:
            mask: Binary mask to dilate
            
        Returns:
            Dilated binary mask
        """
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    def _create_overlay_mask(self, robot_mask: np.ndarray, gripper_mask: np.ndarray,
                            depth_masked: np.ndarray, depth_sim_masked: np.ndarray,
                            hand_mask: np.ndarray) -> np.ndarray:
        """
        Create sophisticated overlay mask using depth-based occlusion reasoning.
        
        Args:
            robot_mask: Binary mask for robot body regions
            gripper_mask: Binary mask for robot gripper regions
            depth_masked: Real depth image masked to robot regions
            depth_sim_masked: Simulated robot depth masked to robot regions
            hand_mask: Binary mask for human hand regions
            
        Returns:
            Binary mask indicating where robot overlay should be applied
        """
        # Start with basic robot visibility mask
        overlay_mask = (robot_mask == 1) | (gripper_mask == 1)
        
        # Apply depth-based occlusion: hide robot when it's behind real objects
        # and not in hand regions (where occlusion handling is more complex)
        overlay_mask[(depth_masked < depth_sim_masked) & (hand_mask == 0)] = 0
        
        return overlay_mask

    def _save_results(self, paths: Paths, sequence: TrainingDataSequence, img_overlay: List[np.ndarray], 
                     img_birdview: Optional[List[np.ndarray]] = None) -> None:
        """
        Save comprehensive robot inpainting results to disk.
        
        Args:
            paths: Paths object containing output file locations
            sequence: Training data sequence with robot state annotations
            img_overlay: List of robot overlay images
            img_birdview: Optional list of bird's eye view images for analysis
        """
        # Create output directory
        os.makedirs(paths.inpaint_processor, exist_ok=True)

        if len(img_overlay) == 0:
            print("No robot inpainted images, skipping")
            return
        
        # Save main robot-inpainted video
        video_path = str(paths.video_overlay).split(".mkv")[0] + f"_{self.robot}_{self.bimanual_setup}.mkv"
        self._save_video(video_path, img_overlay)

        # Save bird's eye view video for analysis and debugging
        if img_birdview is not None:
            birdview_path = str(paths.video_birdview).split(".mkv")[0] + f"_{self.robot}_{self.bimanual_setup}.mkv"
            self._save_video(birdview_path, np.array(img_birdview))

        # Save comprehensive training data with robot state annotations
        training_data_path = str(paths.training_data).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        sequence.save(training_data_path)

    def _save_video(self, path: str, frames: List[np.ndarray]) -> None:
        """
        Save video with consistent encoding parameters.
        
        Args:
            path: Output video file path
            frames: List of video frames to save
        """
        media.write_video(
            path, 
            frames, 
            fps=self.DEFAULT_FPS, 
            codec=self.DEFAULT_CODEC
        )

    def _get_mujoco_camera_params(self) -> MujocoCameraParams:
        """
        Generate MuJoCo camera parameters from real-world camera calibration.
        
        Returns:
            MujocoCameraParams object with calibrated camera settings
        """
        # Extract real-world camera extrinsics and convert to MuJoCo format
        extrinsics = self.extrinsics[0]
        camera_ori_wxyz = self._convert_real_camera_ori_to_mujoco(
            np.array(extrinsics["camera_base_ori"])
        )

        # Calculate image dimensions and camera intrinsics
        img_w, img_h = self._get_image_dimensions()
        offset = self._calculate_image_offset(img_w, img_h)
        fx, fy, cx, cy = self._get_camera_intrinsics(offset)
        sensor_width, sensor_height = self._calculate_sensor_size(img_w, img_h, fx, fy)

        # Select appropriate camera name based on dataset
        if self.epic:
            camera_name = "zed"
        else:
            camera_name = "frontview"
            
        return MujocoCameraParams(
            name=camera_name,
            pos=extrinsics["camera_base_pos"],
            ori_wxyz=camera_ori_wxyz,
            fov=self.intrinsics_dict["v_fov"],
            resolution=(img_h, img_w),
            sensorsize=np.array([sensor_width, sensor_height]),
            principalpixel=np.array([img_w/2-cx, cy-img_h/2]),
            focalpixel=np.array([fx, fy])
        )
    
    def _get_image_dimensions(self) -> Tuple[int, int]:
        """
        Calculate image dimensions based on input resolution configuration.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        # Epic
        if self.input_resolution == 256:
            img_w = 456
        # 1280x720 video
        elif self.input_resolution == 720:
            img_w = 1280  # 720 * 16/9 = 1280
        # Phantom paper
        elif self.input_resolution == 1080:
            img_w = self.input_resolution * 16 // 9
        img_h = self.input_resolution
        return img_w, img_h
    
    def _calculate_image_offset(self, img_w: int, img_h: int) -> int:
        """
        Calculate horizontal image offset for square aspect ratio processing.
        
        Args:
            img_w: Image width in pixels
            img_h: Image height in pixels
            
        Returns:
            Horizontal offset in pixels
        """
        if self.square:
            offset = (img_w - img_h) // 2
        else:
            offset = 0
        return offset
    
    def _get_camera_intrinsics(self, offset: int) -> Tuple[float, float, float, float]:
        """
        Extract camera intrinsic parameters with offset correction.
        
        Args:
            offset: Horizontal offset for principal point adjustment
            
        Returns:
            Tuple of (fx, fy, cx, cy) camera intrinsic parameters
        """
        return self.intrinsics_dict["fx"], self.intrinsics_dict["fy"], self.intrinsics_dict["cx"]+offset, self.intrinsics_dict["cy"]

    def _calculate_sensor_size(self, img_w: int, img_h: int, fx: float, fy: float) -> Tuple[float, float]:
        """
        Calculate physical sensor dimensions from image resolution and focal length.
        
        Args:
            img_w: Image width in pixels
            img_h: Image height in pixels
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            
        Returns:
            Tuple of (sensor_width, sensor_height) in meters
        """
        sensor_width = img_w / fy / 1000
        sensor_height = img_h / fx / 1000
        return sensor_width, sensor_height
    
    @staticmethod
    def _convert_real_camera_ori_to_mujoco(camera_ori_matrix: np.ndarray) -> np.ndarray:
        """
        Convert real-world camera orientation to MuJoCo coordinate system.
        
        Args:
            camera_ori_matrix: 3x3 rotation matrix in real-world coordinates
            
        Returns:
            Quaternion in WXYZ format for MuJoCo
        """
        # Apply coordinate system transformation (flip Y and Z axes)
        camera_ori_matrix[:, [1, 2]] = -camera_ori_matrix[:, [1, 2]]
        
        # Convert to quaternion in MuJoCo's WXYZ format
        r = Rotation.from_matrix(camera_ori_matrix)
        camera_ori_wxyz = r.as_quat(scalar_first=True)
        return camera_ori_wxyz
 

