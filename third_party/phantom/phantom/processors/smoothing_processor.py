"""
Trajectory Smoothing Processor Module

This module does trajectory smoothing for end-effector positions, orientations, and gripper states
extracted from human demonstrations.

Processing Pipeline:
1. Load processed action data from previous pipeline stages
2. Apply Gaussian Process smoothing to 3D position trajectories
3. Apply SLERP-based smoothing to rotation matrix trajectories
4. Apply Gaussian Process smoothing to gripper distance trajectories
5. Save smoothed trajectories for robot execution
"""

import os
from typing import Optional
import argparse
import numpy as np
import logging
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # type: ignore
from scipy.spatial.transform import Rotation, Slerp

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.paths import Paths

logger = logging.getLogger(__name__)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a centered Gaussian kernel for local smoothing operations.
    
    Args:
        size: Size of the kernel (should be odd for proper centering)
        sigma: Standard deviation of the Gaussian distribution
        
    Returns:
        Normalized Gaussian kernel array
        
    Raises:
        ValueError: If size is not positive
    """
    if size <= 0:
        raise ValueError("Kernel size must be positive")
    
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()

class SmoothingProcessor(BaseProcessor): 
    """
    This processor takes raw trajectory data extracted from human demonstrations
    and applies smoothing techniques to create executable robot trajectories. 

    Attributes:
        bimanual_setup (str): Configuration mode ("single_arm" or bimanual type)
        target_hand (str): Target hand for single-arm processing ("left" or "right")
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the smoothing processor with configuration parameters.
        
        Args:
            args: Command line arguments containing smoothing configuration
                 including bimanual setup and target hand specification
        """
        super().__init__(args)

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process and smooth trajectories for a single demonstration.
        
        Args:
            data_sub_folder: Path to demonstration data folder containing
                           processed action trajectories from previous stages
        """
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)

        # Handle single-arm processing mode
        if self.bimanual_setup == "single_arm":
            self._process_single_arm_demo(paths)
        else:
            self._process_bimanual_demo(paths)

    def _process_single_arm_demo(self, paths: Paths) -> None:
        """
        Process single-arm demonstration data.
        
        Args:
            paths: Paths object containing file locations
        """
        # Load action data for target hand
        actions_path = self._get_actions_path(paths)
        actions = np.load(actions_path, allow_pickle=True)
        
        # Apply smoothing to each trajectory component
        smoothed_ee_pts = self.gaussian_process_smoothing(actions["ee_pts"])
        
        # Apply rotation smoothing with configuration-specific parameters
        if self.constrained_hand:
            smoothed_ee_oris = self.gaussian_slerp_smoothing(
                actions["ee_oris"], sigma=10.0, kernel_size=41
            )
        else:
            smoothed_ee_oris = self.gaussian_slerp_smoothing(
                actions["ee_oris"], sigma=10.0
            )

        smoothed_ee_widths = self.gaussian_process_smoothing(actions["ee_widths"])

        # Save results based on target hand
        if self.target_hand == "left":
            self._save_results(paths, smoothed_ee_pts_left=smoothed_ee_pts,
                             smoothed_ee_oris_left=smoothed_ee_oris,
                             smoothed_ee_widths_left=smoothed_ee_widths)
        else:
            self._save_results(paths, smoothed_ee_pts_right=smoothed_ee_pts,
                             smoothed_ee_oris_right=smoothed_ee_oris,
                             smoothed_ee_widths_right=smoothed_ee_widths)

    def _process_bimanual_demo(self, paths: Paths) -> None:
        """
        Process bimanual demonstration data.
        
        Args:
            paths: Paths object containing file locations
        """
        # Load data for both hands
        actions_left_path = str(paths.actions_left).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        actions_right_path = str(paths.actions_right).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
        actions_left = np.load(actions_left_path, allow_pickle=True)
        actions_right = np.load(actions_right_path, allow_pickle=True)

        # Apply position smoothing using Gaussian Process regression
        smoothed_ee_pts_left = self.gaussian_process_smoothing(actions_left["ee_pts"])
        smoothed_ee_pts_right = self.gaussian_process_smoothing(actions_right["ee_pts"])

        # Apply rotation smoothing using SLERP with optimized parameters for bimanual coordination
        smoothed_ee_oris_left = self.gaussian_slerp_smoothing(
            actions_left["ee_oris"], sigma=10.0, kernel_size=21
        )
        smoothed_ee_oris_right = self.gaussian_slerp_smoothing(
            actions_right["ee_oris"], sigma=10.0, kernel_size=21
        )

        # Apply gripper distance smoothing
        smoothed_ee_widths_left = self.gaussian_process_smoothing(actions_left["ee_widths"])
        smoothed_ee_widths_right = self.gaussian_process_smoothing(actions_right["ee_widths"])

        # Save all smoothed trajectories
        self._save_results(paths, smoothed_ee_pts_left, smoothed_ee_oris_left, smoothed_ee_widths_left, 
                           smoothed_ee_pts_right, smoothed_ee_oris_right, smoothed_ee_widths_right)

    def _get_actions_path(self, paths: Paths) -> str:
        """
        Get the appropriate actions file path based on target hand.
        
        Args:
            paths: Paths object containing file locations
            
        Returns:
            Path to the actions file for the target hand
        """
        if self.target_hand == "left":
            base_path = str(paths.actions_left)
        else:
            base_path = str(paths.actions_right)
        return base_path.split(".npz")[0] + f"_{self.bimanual_setup}.npz"

    def _save_results(self, paths: Paths, smoothed_ee_pts_left: Optional[np.ndarray] = None, 
                      smoothed_ee_oris_left: Optional[np.ndarray] = None, 
                      smoothed_ee_widths_left: Optional[np.ndarray] = None, 
                      smoothed_ee_pts_right: Optional[np.ndarray] = None, 
                      smoothed_ee_oris_right: Optional[np.ndarray] = None, 
                      smoothed_ee_widths_right: Optional[np.ndarray] = None) -> None:
        """
        Save smoothed trajectory results to disk.
        
        Args:
            paths: Paths object containing output file locations
            smoothed_ee_pts_left: Smoothed left hand position trajectory
            smoothed_ee_oris_left: Smoothed left hand orientation trajectory
            smoothed_ee_widths_left: Smoothed left hand gripper trajectory
            smoothed_ee_pts_right: Smoothed right hand position trajectory
            smoothed_ee_oris_right: Smoothed right hand orientation trajectory
            smoothed_ee_widths_right: Smoothed right hand gripper trajectory
        """
        # Create output directory
        os.makedirs(paths.smoothing_processor, exist_ok=True)
        
        # Save left hand trajectories if provided
        if smoothed_ee_pts_left is not None:
            smoothed_actions_left_path = str(paths.smoothed_actions_left).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
            np.savez(smoothed_actions_left_path, 
                    ee_pts=smoothed_ee_pts_left, 
                    ee_oris=smoothed_ee_oris_left, 
                    ee_widths=smoothed_ee_widths_left)
        
        # Save right hand trajectories if provided
        if smoothed_ee_pts_right is not None:
            smoothed_actions_right_path = str(paths.smoothed_actions_right).split(".npz")[0] + f"_{self.bimanual_setup}.npz"
            np.savez(smoothed_actions_right_path, 
                    ee_pts=smoothed_ee_pts_right, 
                    ee_oris=smoothed_ee_oris_right, 
                    ee_widths=smoothed_ee_widths_right)

    @staticmethod
    def gaussian_slerp_smoothing(rot_mats: np.ndarray, sigma: float = 2, kernel_size: int = 9) -> np.ndarray:
        """
        Apply Gaussian-weighted SLERP smoothing to rotation matrices.
        
        Args:
            rot_mats: Array of rotation matrices to smooth, shape (N, 3, 3)
            sigma: Standard deviation for Gaussian kernel
            kernel_size: Size of the smoothing kernel (should be odd)
            
        Returns:
            Array of smoothed rotation matrices, shape (N, 3, 3)
            
        Raises:
            ValueError: If kernel_size is not odd
        """
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd for proper centering")
            
        half_k = kernel_size // 2
        N = len(rot_mats)

        # Step 1: Convert rotation matrices to quaternions for interpolation
        quats = Rotation.from_matrix(rot_mats).as_quat()

        # Step 2: Apply hemisphere correction to ensure quaternion continuity
        quats_fixed = [quats[0]]
        for i in range(1, N):
            q = quats[i]
            # Choose quaternion hemisphere that minimizes distance to previous quaternion
            if np.dot(q, quats_fixed[-1]) < 0:
                q = -q
            quats_fixed.append(q)
        quats_fixed = np.array(quats_fixed)

        # Step 3: Prepare normalized Gaussian weights for local smoothing
        weights = gaussian_kernel(kernel_size, sigma)

        # Step 4: Apply weighted SLERP averaging for each time point
        smoothed_rots = []
        for i in range(N):
            # Define local neighborhood around current time point
            start = max(0, i - half_k)
            end = min(N, i + half_k + 1)

            # Extract local quaternions and corresponding weights
            local_quats = quats_fixed[start:end]
            local_weights = weights[half_k - (i - start): half_k + (end - i)]

            # Normalize weights for current neighborhood
            local_weights /= local_weights.sum()

            # Initialize weighted average with first quaternion
            q_avg = local_quats[0]
            r_avg = Rotation.from_quat(q_avg)

            # Iteratively apply weighted SLERP interpolation
            for j in range(1, len(local_quats)):
                r_next = Rotation.from_quat(local_quats[j])
                # Use SLERP with weight proportional to current quaternion's contribution
                r_avg = Slerp([0, 1], Rotation.concatenate([r_avg, r_next]))([local_weights[j] / (local_weights[:j+1].sum())])[0]

            smoothed_rots.append(r_avg.as_matrix())

        return np.stack(smoothed_rots)

    @staticmethod
    def gaussian_process_smoothing(pts: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian process smoothing to trajectory points.
        
        Args:
            pts: Trajectory points to smooth, shape (N,) for 1D or (N, D) for multi-dimensional
            
        Returns:
            Smoothed trajectory points with same shape as input
            
        Raises:
            ValueError: If pts is empty
        """
        if len(pts) == 0:
            raise ValueError("Cannot smooth empty trajectory")
            
        # Create time indices as features for GP regression
        time = np.arange(len(pts))[:, None]  # Time as a single feature
        
        # Configure GP kernel: RBF for smoothness + White noise for robustness
        kernel = RBF(length_scale=1) + WhiteKernel(noise_level=1)
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        # Handle 1D trajectory case
        if pts.ndim == 1:
            return gpr.fit(time, pts).predict(time)
        
        # Handle multi-dimensional trajectory case by processing each dimension independently
        return np.column_stack([gpr.fit(time, pts[:, i]).predict(time) for i in range(pts.shape[1])]) 