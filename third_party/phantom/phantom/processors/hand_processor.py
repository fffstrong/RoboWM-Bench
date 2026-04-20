"""
Hand Processor Module

This module converts detected hand bounding boxes into detailed 3D hand poses using 
state-of-the-art pose estimation models, with optional depth-based refinement for improved accuracy.

Processing Pipeline:
1. Load video frames and bounding box data from previous stage
2. Apply HaMeR pose estimation within detected bounding boxes
3. Filter poses based on edge proximity and quality metrics
4. Optionally refine 3D poses using depth data and segmentation
5. Generate hand mesh models and extract keypoint trajectories
6. Save processed hand sequences for downstream tasks

The module supports multiple processing modes:
- Hand2DProcessor: 2D pose estimation only (faster, camera-based)
- Hand3DProcessor: Full 3D processing with depth alignment (more accurate, if depth is available)

Output Data:
- HandSequence objects containing pose trajectories
- 2D keypoint positions in image coordinates
- 3D keypoint positions in camera coordinates
- Hand detection flags per frame
- Annotated visualization videos
"""

import glob
import os
import logging
from tqdm import tqdm
import numpy as np
import mediapy as media
import open3d as o3d  # type: ignore
from typing import Tuple, Optional, Dict, Any
import trimesh
from collections import defaultdict
import argparse

from phantom.utils.pcd_utils import get_visible_points, get_pcd_from_points, icp_registration, get_point_cloud_of_segmask, get_3D_points_from_pixels, remove_outliers, get_bbox_of_3d_points, trim_pcd_to_bbox, visualize_pcds
from phantom.utils.transform_utils import transform_pts
from phantom.processors.base_processor import BaseProcessor
from phantom.detectors.detector_hamer import DetectorHamer
from phantom.processors.phantom_data import HandSequence, HandFrame, hand_side_dict
from phantom.processors.paths import Paths
from phantom.processors.segmentation_processor import HandSegmentationProcessor

logger = logging.getLogger(__name__)

class HandBaseProcessor(BaseProcessor): 
    """
    Base class for hand pose processing using HaMeR detection and optional depth refinement.
    
    The processor operates on the output of BBoxProcessor, using detected hand bounding boxes
    to guide pose estimation. It supports both 2D and 3D processing modes, with the 3D mode
    providing enhanced accuracy through depth sensor integration.
    
    Processing Workflow:
    1. Load video frames and bounding box detection results
    2. For each frame with detected hands:
       - Apply HaMeR pose estimation within bounding box
       - Validate pose quality (edge proximity, confidence)
       - Optionally generate hand segmentation masks for depth refinement
       - Optionally apply depth-based pose refinement
    3. Generate temporal hand sequences with smooth trajectories
    4. Save processed results and visualization videos
    
    Attributes:
        process_hand_masks (bool): Whether to generate hand segmentation masks
        apply_depth_alignment (bool): Whether to use depth-based pose refinement
        detector_hamer (DetectorHamer): HaMeR pose estimation model
        hand_mask_processor: Segmentation processor for hand mask generation
        H (int): Video frame height
        W (int): Video frame width
        imgs_depth (np.ndarray): Depth images for 3D refinement
        left_masks (np.ndarray): Left hand segmentation masks
        right_masks (np.ndarray): Right hand segmentation masks
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the hand processor with configuration parameters.
        
        Args:
            args: Command line arguments containing processing configuration
                 including depth processing flags and model parameters
        """
        super().__init__(args)
        self.process_hand_masks: bool = False
        self._initialize_detectors()
        self.hand_mask_processor: Optional[HandSegmentationProcessor] = None
        self.apply_depth_alignment: bool = False

    def _initialize_detectors(self) -> None:
        """
        Initialize all required detection models.
        
        Sets up the HaMeR detector for hand pose estimation. 
        """
        self.detector_hamer = DetectorHamer()

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process a single demonstration video to extract hand poses and segmentation.
        
        Args:
            data_sub_folder: Path to the demonstration data folder containing
                           video files, bounding box data, and optional depth data
        """
        save_folder = self.get_save_folder(data_sub_folder)
   
        paths = self.get_paths(save_folder)
        
        # Load RGB video frames
        imgs_rgb = media.read_video(getattr(paths, f"video_left"))
        self.H, self.W, _ = imgs_rgb[0].shape

        # Load depth data if available (for 3D processing)
        if os.path.exists(paths.depth):
            self.imgs_depth = np.load(paths.depth)
        else:
            self.imgs_depth = np.zeros((len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1]))

        # Load hand segmentation masks if available
        if os.path.exists(paths.masks_hand_left) and os.path.exists(paths.masks_hand_right):
            self.left_masks = np.load(paths.masks_hand_left)
            self.right_masks = np.load(paths.masks_hand_right)
        else:
            self.left_masks = np.zeros((len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1]))
            self.right_masks = np.zeros((len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1]))

        # Load bounding box detection results from previous stage
        bbox_data = np.load(paths.bbox_data)
        left_hand_detected = bbox_data["left_hand_detected"]
        right_hand_detected = bbox_data["right_hand_detected"]
        left_bboxes = bbox_data["left_bboxes"]
        right_bboxes = bbox_data["right_bboxes"]

        # Validate data consistency
        assert len(left_hand_detected) == len(right_hand_detected)
        assert len(left_hand_detected) == len(imgs_rgb)

        # Process left and right hand sequences
        left_sequence = self._process_all_frames(imgs_rgb, left_bboxes, left_hand_detected, "left")
        right_sequence = self._process_all_frames(imgs_rgb, right_bboxes, right_hand_detected, "right")

        # Generate hand segmentation masks if enabled
        if self.process_hand_masks:
            self._get_hand_masks(data_sub_folder, left_sequence, right_sequence)
            self.left_masks = np.load(paths.masks_hand_left)
            self.right_masks = np.load(paths.masks_hand_right)
        
        # Apply depth-based pose refinement if enabled
        if self.apply_depth_alignment:
            left_sequence = self._process_all_frames_depth_alignment(imgs_rgb, left_hand_detected, "left", left_sequence)
            right_sequence = self._process_all_frames_depth_alignment(imgs_rgb, right_hand_detected, "right", right_sequence)

        # Save processed sequences and generate visualizations
        self._save_results(paths, left_sequence, right_sequence)

    def _process_all_frames(self, imgs_rgb: np.ndarray, bboxes: np.ndarray, 
                            hand_detections: np.ndarray, hand_side: str) -> HandSequence:
        """
        Process all frames in a video sequence to extract hand poses.
        
        This method iterates through all video frames, applying pose estimation
        where hands are detected and creating empty frames where they are not.
        It maintains temporal consistency and provides quality filtering.
        
        Args:
            imgs_rgb: RGB video frames, shape (num_frames, height, width, 3)
            bboxes: Hand bounding boxes per frame, shape (num_frames, 4)
            hand_detections: Boolean flags indicating valid detections per frame
            hand_side: "left" or "right" to specify which hand is being processed
            
        Returns:
            HandSequence object containing processed pose data for all frames
        """
        sequence = HandSequence()

        for img_idx in tqdm(range(len(imgs_rgb)), disable=False, leave=False):
            if not hand_detections[img_idx]:
                # Create empty frame for missing detections
                sequence.add_frame(HandFrame.create_empty_frame(
                    frame_idx=img_idx,
                    img_rgb=imgs_rgb[img_idx],
                ))
                continue

            # Process frame with detected hand
            frame_data = self._process_frame(img_idx, imgs_rgb[img_idx], bboxes[img_idx], 
                                            hand_side)
            sequence.add_frame(frame_data)

        return sequence
    
    def _process_frame(self, img_idx: int, img_rgb: np.ndarray, bbox: np.ndarray, 
                       hand_side: str, view: bool = False) -> HandFrame:
        """
        Process a single frame to extract hand pose and validate quality.
        
        This method applies HaMeR pose estimation within the detected bounding box
        and performs quality checks to ensure the pose is suitable for downstream
        processing. Poor quality poses (e.g., hands too close to image edges) are
        rejected to maintain data quality.
        
        Args:
            img_idx: Index of the current frame
            img_rgb: RGB image data for this frame
            bbox: Hand bounding box coordinates [x1, y1, x2, y2]
            hand_side: "left" or "right" specifying which hand is being processed
            view: Whether to display debug visualizations
            
        Returns:
            HandFrame object containing pose data or empty frame if quality is poor
        """
        try:
            # Apply HaMeR pose estimation within bounding box
            processed_data = self._process_image_with_hamer(img_rgb, bbox[None,...], hand_side, img_idx, view=view)
            
            kpts_3d = processed_data["kpts_3d"]
            offset_dist = 0.003 if hand_side == "left" else -0.003
            if len(kpts_3d) > 0:
                # 4 represents tip of thumb，8 represents tip of index finger
                thumb_tip = kpts_3d[4]
                index_tip = kpts_3d[8]
                dir_vec = index_tip - thumb_tip
                norm = np.linalg.norm(dir_vec)
                
                if norm > 1e-6:
                    unit_dir = dir_vec / norm
                    # Because the original keypoints are biased towards the direction from the thumb tip to the index finger tip, the correction needs to be made in the opposite direction (subtracting unit_dir)
                    # Correct the thumb keypoints (MCP, PIP, DIP, TIP -> indices 1, 2, 3, 4)
                    kpts_3d[1:5] -= offset_dist * unit_dir
                    # Correct the index finger keypoints (MCP, PIP, DIP, TIP -> indices 5, 6, 7, 8)
                    kpts_3d[5:9] -= offset_dist * unit_dir

            # Quality check: reject poses where keypoints are too close to image edges
            if self.are_kpts_too_close_to_margin(processed_data["kpts_2d"], self.W, self.H, margin=5, threshold=0.1):
                logger.error(f"Error processing frame {img_idx}: Edge hand")
                return HandFrame.create_empty_frame(
                    frame_idx=img_idx,
                    img_rgb=img_rgb,
                )
            
            # Create frame with validated pose data
            frame_data = HandFrame(
                frame_idx=img_idx,
                hand_detected=True,
                img_rgb=img_rgb,
                img_hamer=processed_data["img_hamer"],
                kpts_2d=processed_data["kpts_2d"],
                kpts_3d=processed_data["kpts_3d"],
            )

            return frame_data
            
        except Exception as e:
            logger.error(f"Error processing frame {img_idx}: {str(e)}")
            return HandFrame.create_empty_frame(
                frame_idx=img_idx,
                img_rgb=img_rgb,
            )

    def are_kpts_too_close_to_margin(self, kpts_2d: np.ndarray, img_width: int, img_height: int, 
                                   margin: int = 20, threshold: float = 0.5) -> bool:
        """
        Filter hand keypoints based on proximity to image edges.
        
        This quality check rejects hand poses where too many keypoints are near
        the image boundaries, which typically indicates partial occlusion or
        tracking errors that would lead to poor pose estimates.

        Args:
            kpts_2d: 2D keypoint positions, shape (N, 2) where N is number of keypoints
            img_width: Image width in pixels
            img_height: Image height in pixels
            margin: Distance from edge (in pixels) to consider "too close"
            threshold: Fraction of keypoints that triggers rejection (e.g., 0.5 = 50%)

        Returns:
            True if hand should be rejected due to edge proximity, False otherwise
        """
        x = kpts_2d[:, 0]
        y = kpts_2d[:, 1]

        # Create boolean mask for keypoints near any image edge
        near_edge = (
            (x < margin) |
            (y < margin) |
            (x > img_width - margin) |
            (y > img_height - margin)
        )

        frac_near_edge = np.mean(near_edge)  # Fraction of keypoints near edge
        return frac_near_edge > threshold

    def _save_results(self, paths: Paths, left_sequence: HandSequence, right_sequence: HandSequence) -> None:
        """
        Save processed hand sequences and generate visualization videos.
        
        Args:
            paths: Paths object containing output file locations
            left_sequence: Processed left hand pose sequence
            right_sequence: Processed right hand pose sequence
        """
        # Create output directory
        if not os.path.exists(getattr(paths, f"hand_processor")):
            os.makedirs(getattr(paths, f"hand_processor"))

        # Save hand sequence data in compressed format
        left_sequence.save(getattr(paths, f"hand_data_left"))
        right_sequence.save(getattr(paths, f"hand_data_right"))

        # Save RGB frames for reference
        media.write_video(getattr(paths, f"video_rgb_imgs"), left_sequence.imgs_rgb, fps=10, codec="ffv1")

        # Load additional visualization components
        imgs_bbox = media.read_video(getattr(paths, f"video_bboxes"))

        # Load segmentation visualization if available
        if os.path.exists(getattr(paths, f"video_sam_arm")):
            imgs_sam = media.read_video(getattr(paths, f"video_sam_arm"))
        else:
            imgs_sam = np.zeros((len(left_sequence.imgs_rgb), left_sequence.imgs_rgb[0].shape[0], left_sequence.imgs_rgb[0].shape[1], 3))

        # Create comprehensive annotation video showing all processing stages
        annot_imgs = []
        for idx in range(len(left_sequence.imgs_rgb)):
            img_hamer_left = left_sequence.imgs_hamer[idx]
            img_hamer_right = right_sequence.imgs_hamer[idx]
            img_bbox = imgs_bbox[idx]
            img_sam = imgs_sam[idx]
            
            # Combine visualizations in 2x2 grid: [bbox, sam] on top, [left_hand, right_hand] on bottom
            annot_img = np.vstack((np.hstack((img_bbox, img_sam)), np.hstack((img_hamer_left, img_hamer_right)))).astype(np.uint8)
            annot_imgs.append(annot_img)
            
        # Save comprehensive visualization video
        media.write_video(getattr(paths, f"video_annot"), np.array(annot_imgs), fps=10, codec="h264") # mp4

    def _create_hand_mesh(self, hamer_out: Dict[str, Any]) -> trimesh.Trimesh:
        """
        Create a 3D triangle mesh from HaMeR pose estimation output.
        
        Args:
            hamer_out: HaMeR output dictionary containing vertex positions
            
        Returns:
            Trimesh object representing the hand mesh
        """
        return trimesh.Trimesh(hamer_out["verts"].copy(), self.detector_hamer.faces_left.copy(), process=False)
    
    def _get_hand_masks(self, data_sub_folder: str, hamer_data_left: HandSequence, hamer_data_right: HandSequence) -> None:
        """
        Generate hand segmentation masks using processed pose data.
        
        This method integrates with the segmentation processor to generate
        detailed hand masks that can be used for depth-based pose refinement.
        
        Args:
            data_sub_folder: Path to demonstration data folder
            hamer_data_left: Processed left hand sequence for guidance
            hamer_data_right: Processed right hand sequence for guidance
        """
        hamer_data = {
            "left": hamer_data_left,
            "right": hamer_data_right
        }
        self.hand_mask_processor.process_one_demo(data_sub_folder, hamer_data)

    @staticmethod
    def _get_visible_pts_from_hamer(detector_hamer: DetectorHamer, hamer_out: Dict[str, Any], mesh: trimesh.Trimesh,
                                img_depth: np.ndarray, cam_intrinsics: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify visible hand vertices and their corresponding depth points.
        
        Args:
            detector_hamer: HaMeR detector instance for coordinate projections
            hamer_out: HaMeR output containing pose estimates and camera parameters
            mesh: 3D hand mesh generated from HaMeR output
            img_depth: Depth image corresponding to the RGB frame
            cam_intrinsics: Camera intrinsic parameters for 3D projection
            
        Returns:
            Tuple of (visible_points_3d, visible_hamer_vertices):
                - visible_points_3d: 3D points from depth image at visible mesh locations
                - visible_hamer_vertices: Corresponding vertices from the HaMeR mesh
        """
        # Perform ray-casting to identify visible mesh vertices
        visible_hamer_vertices, _ = get_visible_points(mesh, origin=np.array([0,0,0]))

        # Project 3D vertices to 2D image coordinates
        visible_points_2d = detector_hamer.project_3d_kpt_to_2d(
            (visible_hamer_vertices-hamer_out["T_cam_pred"].cpu().numpy()).astype(np.float32), 
            hamer_out["img_w"], hamer_out["img_h"], hamer_out["scaled_focal_length"], 
            hamer_out["camera_center"], hamer_out["T_cam_pred"])

        # Filter out points that fall outside the depth image boundaries
        original_visible_points_2d = visible_points_2d.copy()

        # Create valid region mask (note: depth indexing is [y, x])
        valid_mask = ((original_visible_points_2d[:, 0] < img_depth.shape[1]) & 
                     (original_visible_points_2d[:, 1] < img_depth.shape[0]))

        visible_points_2d = visible_points_2d[valid_mask]
        visible_hamer_vertices = visible_hamer_vertices[valid_mask]

        # Convert 2D depth pixels to 3D points using camera intrinsics
        visible_points_3d = get_3D_points_from_pixels(visible_points_2d, img_depth, cam_intrinsics)

        return visible_points_3d, visible_hamer_vertices

    @staticmethod
    def _get_transformation_estimate(visible_points_3d: np.ndarray, 
                                    visible_hamer_vertices: np.ndarray, 
                                    pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        Estimate transformation to align HaMeR mesh with observed point cloud.
        
        This method uses Iterative Closest Point (ICP) registration to find the
        optimal transformation that aligns the visible parts of the predicted
        hand mesh with the point cloud extracted from depth and segmentation data.
        
        Args:
            visible_points_3d: 3D points from depth image at mesh locations
            visible_hamer_vertices: Corresponding vertices from HaMeR mesh
            pcd: Point cloud from segmentation and depth data
            
        Returns:
            Tuple of (transformation_matrix, aligned_mesh_pointcloud):
                - transformation_matrix: 4x4 transformation to align mesh with depth
                - aligned_mesh_pointcloud: Transformed mesh point cloud after alignment
        """
        # Get initial transformation estimate using median translation
        T_0 = HandBaseProcessor._get_initial_transformation_estimate(visible_points_3d, visible_hamer_vertices)
        
        # Create point cloud from visible mesh vertices
        visible_hamer_pcd = get_pcd_from_points(visible_hamer_vertices, colors=np.ones_like(visible_hamer_vertices) * [0, 1, 0])
        
        try: 
            # Apply ICP registration for fine alignment
            aligned_hamer_pcd, T = icp_registration(visible_hamer_pcd, pcd, voxel_size=0.005, init_transform=T_0)
        except Exception as e:
            logger.error(f"ICP registration failed: {e}")
            return T_0, visible_hamer_pcd
            
        return T, aligned_hamer_pcd
    
    @staticmethod
    def _get_initial_transformation_estimate(visible_points_3d: np.ndarray, 
                                            visible_hamer_vertices: np.ndarray) -> np.ndarray:
        """
        Compute initial transformation estimate for mesh-to-depth alignment.
        
        This method provides a coarse alignment between the HaMeR prediction and
        the depth-based point cloud using median translation. It assumes that
        orientation is approximately correct and only translation correction is needed.
        
        Args:
            visible_points_3d: 3D points from depth image
            visible_hamer_vertices: Corresponding HaMeR mesh vertices
            
        Returns:
            4x4 transformation matrix with estimated translation
        """
        # Calculate median translation between corresponding point sets
        translation = np.nanmedian(visible_points_3d - visible_hamer_vertices, axis=0)
        
        # Create transformation matrix (identity rotation + translation)
        T_0 = np.eye(4)
        if not np.isnan(translation).any():
            T_0[:3, 3] = translation
            
        return T_0


class Hand2DProcessor(HandBaseProcessor): 
    """
    2D hand pose processor optimized for speed and RGB-only operation.
    
    This processor focuses on extracting 2D hand poses and basic 3D estimates
    without depth sensor integration. It's designed for applications where
    depth sensors are not available.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize 2D hand processor with RGB-only configuration.
        
        Args:
            args: Command line arguments for processor configuration
        """
        super().__init__(args)

    def _process_image_with_hamer(self, img_rgb: np.ndarray, bboxes: np.ndarray, hand_side: str, 
                                  img_idx: int, view: bool = False) -> Dict[str, Any]:
        """
        Process RGB image with HaMeR for 2D pose estimation.
        
        Args:
            img_rgb: RGB image to process
            bboxes: Hand bounding boxes for pose estimation guidance
            hand_side: "left" or "right" specifying which hand to process
            img_idx: Frame index for debugging and logging
            view: Whether to display debug visualizations
            
        Returns:
            Dictionary containing:
                - img_hamer: Annotated image with pose visualization
                - kpts_3d: Estimated 3D keypoints
                - kpts_2d: 2D keypoint projections in image coordinates
                
        Raises:
            ValueError: If no valid hand pose is detected in the image
        """
        # Configure HaMeR for target hand side
        is_right = np.array([hand_side_dict[str(hand_side)]*True]*len(bboxes))
        
        # Apply HaMeR pose estimation
        hamer_out = self.detector_hamer.detect_hand_keypoints(
            img_rgb, 
            hand_side=hand_side, 
            bboxes=bboxes, 
            is_right=is_right, 
            camera_params=self.intrinsics_dict, 
            visualize=False
        )
        
        if hamer_out is None or not hamer_out.get("success", False):  
            raise ValueError("No hand detected in image")

        return {
            "img_hamer": hamer_out["annotated_img"][:,:,::-1],  # Convert BGR to RGB
            "kpts_3d": hamer_out["kpts_3d"],
            "kpts_2d": hamer_out['kpts_2d']
        }
    
class Hand3DProcessor(HandBaseProcessor): 
    """
    3D hand pose processor with depth-based refinement capabilities.
    
    This processor provides more accurate 3D hand poses by combining HaMeR
    estimation with depth sensor data and hand segmentation. It uses point cloud
    registration techniques to refine the initial pose estimates, resulting in
    poses that are better aligned with the physical environment.
    
    Processing Enhancements:
    - Mesh generation from HaMeR output for visibility analysis
    - Hand segmentation using SAM2 for accurate depth extraction
    - ICP-based alignment between predicted mesh and observed point cloud
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize 3D hand processor with depth refinement capabilities.
        
        Args:
            args: Command line arguments containing depth processing configuration
        """
        super().__init__(args)
        self.args = args
        
        # Storage for HaMeR outputs needed for depth alignment
        self.hamer_out_dict: Dict[str, Dict[int, Dict[str, Any]]] = {
            "left": defaultdict(dict),
            "right": defaultdict(dict)
        }
        
        # Enable advanced processing features
        self.process_hand_masks = True
        self.apply_depth_alignment = True
        self.hand_mask_processor = HandSegmentationProcessor(self.args)

    def _process_image_with_hamer(self, img_rgb: np.ndarray, bboxes: np.ndarray, hand_side: str, 
                                  img_idx: int, view: bool = False) -> Dict[str, Any]:
        """
        Process RGB image with HaMeR optimized for subsequent depth refinement.
        
        This method applies HaMeR pose estimation configured for 3D processing,
        storing intermediate results needed for later depth-based refinement.
        
        Args:
            img_rgb: RGB image to process
            bboxes: Hand bounding boxes for pose estimation guidance
            hand_side: "left" or "right" specifying which hand to process
            img_idx: Frame index for result storage and debugging
            view: Whether to display debug visualizations
            
        Returns:
            Dictionary containing pose estimation results
            
        Raises:
            ValueError: If no valid hand pose is detected in the image
        """
        # Configure HaMeR for target hand side
        is_right = np.array([hand_side_dict[str(hand_side)]*True]*len(bboxes))
        
        # Apply HaMeR with 2D keypoint focus (3D refinement happens later)
        hamer_out = self.detector_hamer.detect_hand_keypoints(
            img_rgb, 
            hand_side=hand_side, 
            bboxes=bboxes, 
            is_right=is_right, 
            kpts_2d_only=True,  # Initial processing focuses on 2D
            camera_params=self.intrinsics_dict
        )
        
        if hamer_out is None or not hamer_out.get("success", False):  
            raise ValueError("No hand detected in image")
        
        # Store HaMeR output for later depth alignment processing
        self.hamer_out_dict[hand_side][img_idx] = hamer_out

        return {
            "img_hamer": hamer_out["annotated_img"][:,:,::-1],  # Convert BGR to RGB
            "kpts_3d": hamer_out["kpts_3d"],
            "kpts_2d": hamer_out['kpts_2d']
        }
    
    def _process_all_frames_depth_alignment(self, imgs_rgb: np.ndarray, hand_detections: np.ndarray, 
                                    hand_side: str, sequence: Optional[HandSequence] = None) -> HandSequence:
        """
        Apply depth-based refinement to all frames in the sequence.
        
        This method performs the depth alignment stage of processing, using
        segmentation masks and depth data to refine the initial HaMeR pose
        estimates for improved 3D accuracy.
        
        Args:
            imgs_rgb: RGB video frames for reference
            hand_detections: Boolean flags indicating frames with valid detections
            hand_side: "left" or "right" specifying which hand to process
            sequence: HandSequence containing initial pose estimates to refine
            
        Returns:
            HandSequence with refined 3D poses aligned to depth data
        """
        for img_idx in tqdm(range(len(imgs_rgb)), disable=False, leave=False):
            if not hand_detections[img_idx]:
                continue

            # Apply depth-based refinement to this frame
            frame_data = sequence.get_frame(img_idx)
            frame_data.kpts_3d = self._depth_alignment(img_idx, hand_side, imgs_rgb[img_idx])
            sequence.modify_frame(img_idx, frame_data)

        return sequence
    
    def _depth_alignment(self, img_idx: int, hand_side: str, img_rgb: np.ndarray) -> np.ndarray:
        """
        Perform depth-based pose refinement for a single frame.
        
        Algorithm Steps:
        1. Extract depth image and segmentation mask for the frame
        2. Obtain 3D hand mesh from HaMeR output
        3. Create point cloud from segmented depth region
        4. Identify visible mesh vertices through ray casting
        5. Apply ICP registration between mesh and point cloud
        6. Transform original keypoints using computed alignment
        
        Args:
            img_idx: Index of the frame to process
            hand_side: "left" or "right" specifying which hand to process
            img_rgb: RGB image for reference (used in point cloud generation)
            
        Returns:
            Refined 3D keypoint positions aligned with depth data
        """
        # Load frame-specific data
        img_depth = self.imgs_depth[img_idx]
        mask = self.left_masks[img_idx] if hand_side == "left" else self.right_masks[img_idx]
        hamer_out = self.hamer_out_dict[hand_side][img_idx]
        
        # Create 3D hand mesh from HaMeR pose estimate
        mesh = self._create_hand_mesh(hamer_out)

        # Generate point cloud from depth image within segmented hand region
        pcd = get_point_cloud_of_segmask(mask, img_depth, img_rgb, self.intrinsics_dict, visualize=False)

        # Identify visible mesh vertices and corresponding depth points
        visible_points_3d, visible_hamer_vertices = self._get_visible_pts_from_hamer(
            self.detector_hamer, 
            hamer_out, 
            mesh, 
            img_depth, 
            self.intrinsics_dict
        )
        
        # Compute optimal transformation using ICP registration
        T, _ = self._get_transformation_estimate(visible_points_3d, visible_hamer_vertices, pcd)

        # Apply transformation to refine original keypoint positions
        kpts_3d = transform_pts(hamer_out["kpts_3d"], T)

        return kpts_3d