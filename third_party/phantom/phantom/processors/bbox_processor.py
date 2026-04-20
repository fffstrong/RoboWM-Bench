"""
Bounding Box Processor Module

This module provides video processing capabilities for detecting and tracking hand bounding boxes
in demonstration videos. It serves as the first stage in the hand processing pipeline, providing
spatial localization data for downstream pose estimation and segmentation tasks.

Key Features:
- Multiple hand detection methods (DINO, EPIC-KITCHENS integration)
- Bimanual hand tracking with left/right classification
- Temporal consistency through outlier filtering and interpolation
- Spatial constraint validation (edge detection, center positioning)
- Visualization and annotation generation

Processing Pipeline:
1. Video loading and validation
2. Frame-by-frame hand detection using configured detectors
3. Bounding box classification (left/right) based on spatial positioning
4. Temporal filtering to remove outliers and large jumps
5. Gap interpolation for smooth trajectories
6. Edge distance calculation for quality assessment
7. Result visualization and storage

The processor supports multiple detection backends:
- DINO-based detection for general hand detection
- EPIC-KITCHENS pre-computed detections
- Configurable confidence thresholds and spatial constraints

Output Data:
- Hand detection flags per frame (boolean arrays)
- Bounding box coordinates [x1, y1, x2, y2] per frame
- Bounding box centers [x, y] per frame
- Distance metrics to image edges
- Annotated visualization videos
"""

import os
import pickle
import logging
import numpy as np
import mediapy as media
import cv2
import itertools
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any, Dict
from typing_extensions import Literal
import numpy.typing as npt
from omegaconf import DictConfig

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.paths import Paths
from phantom.processors.phantom_data import hand_side_dict

from phantom.utils.bbox_utils import get_bbox_center, get_bbox_center_min_dist_to_edge

logger = logging.getLogger(__name__)

# Type aliases for better readability
DetectionResults = Dict[str, npt.NDArray]
BBoxArray = npt.NDArray[np.float32]  # [x1, y1, x2, y2]
CenterArray = npt.NDArray[np.float32]  # [x, y]
DetectionFlagArray = npt.NDArray[np.bool_]
HandSide = Literal["left", "right"]

class BBoxProcessor(BaseProcessor):
    # Detection configuration constants
    HAND_SIDE_MARGIN = 50  # Pixel margin for hand side classification tolerance
    OVERLAP_THRESHOLD = 0.3  # Threshold for considering bboxes as overlapping
    MAX_INTERPOLATION_GAP = 10  # Maximum frames to interpolate over
    MAX_SPATIAL_JUMP = 200.0  # Maximum allowed pixel jump between detections
    MAX_JUMP_LOOKAHEAD = 10  # Maximum consecutive distant points to filter
    DINO_CONFIDENCE_THRESH = 0.2  # Default confidence threshold
    
    # Visualization constants
    LEFT_HAND_COLOR = (0, 0, 255)  # BGR format - Red for left hand
    RIGHT_HAND_COLOR = (0, 255, 0)  # BGR format - Green for right hand
    BBOX_THICKNESS = 2  # Thickness of bounding box lines
    
    """
    Bounding box detection and tracking processor for hand localization in videos.
    
    This processor serves as the foundation of the hand processing pipeline by detecting
    and tracking hand bounding boxes across video frames. It handles both single-arm
    and bimanual setups.
    
    The processor employs multiple strategies for reliable detection:
    - Primary detection using DINO or pre-computed EPIC data
    - Spatial reasoning for left/right hand classification
    - Temporal filtering to maintain trajectory consistency
    - Gap interpolation for handling missing detections
    - Quality assessment through edge distance metrics
    
    Attributes:
        H (int): Video frame height (set during processing)
        W (int): Video frame width (set during processing)
        center (int): Horizontal center of the frame for left/right classification
        margin (int): Pixel margin for hand side classification tolerance
        confidence_threshold (float): Minimum confidence for valid detections
        dino_detector: DINO-based hand detector (if not using EPIC data)
        filtered_hand_detection_data (dict): Processed EPIC detection data
        sorted_keys (list): Sorted frame indices for EPIC data processing
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the bounding box processor with configuration parameters.
        
        Args:
            cfg: Hydra configuration object containing processing configuration
                 including confidence thresholds, target hands, and dataset type
        """
        super().__init__(cfg)    
        # Image dimensions (set when processing video)
        self.H: int = 0
        self.W: int = 0

        # Initialize detection backend based on dataset type
        if not self.epic:
            from phantom.detectors.detector_dino import DetectorDino
            self.dino_detector: DetectorDino = DetectorDino("IDEA-Research/grounding-dino-base")
        else:
            self.dino_detector: Optional[DetectorDino] = None
            
        # EPIC-specific attributes
        self.filtered_hand_detection_data: Dict[str, List[Any]] = {}
        self.sorted_keys: List[str] = []

    # ============================================================================
    # COMMON/SHARED METHODS (Used by both Phantom and EPIC modes)
    # ============================================================================

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process a single demonstration video to extract hand bounding boxes.
        
        Args:
            data_sub_folder: Path to the demonstration data folder containing the video
                           and any pre-computed hand detection data.

        The method performs the following steps:
        1. Loads and validates input video and detection data
        2. Processes each frame to detect and classify hand positions
        3. Applies post-processing filters for temporal consistency
        4. Generates quality metrics and visualizations
        5. Saves all results in standardized format

        Raises:
            FileNotFoundError: If required input files (video, detection data) are not found
            ValueError: If video frames or hand detection data are invalid
        """
        # Setup and validation
        save_folder = self.get_save_folder(data_sub_folder)

        paths = self.get_paths(save_folder)

        # Load and validate input data
        imgs_rgb = self._load_video(paths)

        # Process frames based on dataset type
        if self.epic:
            self._load_epic_hand_data(paths)
            detection_results = self._process_epic_frames(imgs_rgb)
        else:
            detection_results = self._process_frames(imgs_rgb)

        # Post-process results for temporal consistency
        processed_results = self._post_process_detections(detection_results)
        
        # Generate visualization for quality assessment
        visualization_results = self._generate_visualization(imgs_rgb, processed_results)
        
        # Save all results to disk
        self._save_results(paths, processed_results, visualization_results)


    def _load_video(self, paths: Paths) -> np.ndarray:
        """
        Load and validate video data from the specified path.
        
        Args:
            paths: Paths object containing video file locations
            
        Returns:
            RGB video frames as array
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video is empty or corrupted
        """
        if not os.path.exists(paths.video_left):
            raise FileNotFoundError(f"Video file not found: {paths.video_left}")
        
        imgs_rgb = media.read_video(getattr(paths, f"video_left"))
        if len(imgs_rgb) == 0:
            raise ValueError("Empty video file")
        
        # Store video dimensions for coordinate calculations
        self.H, self.W, _ = imgs_rgb[0].shape
        self.center: int = self.W // 2  # Center line for left/right classification
        return imgs_rgb

    # ============================================================================
    # PHANTOM-SPECIFIC METHODS (DINO Detection)
    # ============================================================================
    def _process_frames(self, imgs_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process RGB frames using DINO detector for hand detection and classification.
        
        This method handles the core detection pipeline for non-EPIC datasets,
        using DINO for hand detection and implementing spatial reasoning for
        left/right classification.
        
        Args:
            imgs_rgb: Array of RGB images with shape (num_frames, height, width, 3)
            
        Returns:
            Dictionary containing:
                - left/right_hand_detected: Boolean arrays indicating hand detection per frame
                - left/right_bboxes: Bounding box coordinates [x1,y1,x2,y2] per frame
                - left/right_bboxes_ctr: Bounding box centers [x,y] per frame
        """
        num_frames = len(imgs_rgb)
        
        detection_arrays = self._initialize_detection_arrays(num_frames)

        for idx in range(num_frames):
            try:
                # Run DINO detection on current frame
                bboxes, scores = self.dino_detector.get_bboxes(imgs_rgb[idx], "a hand", threshold=self.DINO_CONFIDENCE_THRESH, visualize=False)
                if len(bboxes) == 0:
                    continue

                bboxes = np.array(bboxes)
                scores = np.array(scores)
                
                # Process detections for current frame
                self._process_frame_detections(idx, bboxes, scores, detection_arrays)
            except Exception as e:
                logger.warning(f"Frame {idx} processing failed: {str(e)}")
                continue

        return {
            'left_hand_detected': detection_arrays['left_hand_detected'],
            'right_hand_detected': detection_arrays['right_hand_detected'],
            'left_bboxes': detection_arrays['left_bboxes'],
            'right_bboxes': detection_arrays['right_bboxes'],
            'left_bboxes_ctr': detection_arrays['left_bboxes_ctr'],
            'right_bboxes_ctr': detection_arrays['right_bboxes_ctr'],
        }

    def _initialize_detection_arrays(self, num_frames: int) -> Dict[str, npt.NDArray]:
        """
        Initialize arrays for storing detection results.
        
        Args:
            num_frames: Number of frames in the video
            
        Returns:
            Dictionary containing pre-allocated arrays for left/right hand detections,
            bounding boxes, centers, and detection flags
        """
        return {
            'left_bboxes': np.zeros((num_frames, 4)),
            'right_bboxes': np.zeros((num_frames, 4)),
            'left_bboxes_ctr': np.zeros((num_frames, 2)),
            'right_bboxes_ctr': np.zeros((num_frames, 2)),
            'left_hand_detected': np.zeros(num_frames, dtype=bool),
            'right_hand_detected': np.zeros(num_frames, dtype=bool)
        }

    def _process_frame_detections(self, idx: int, bboxes: npt.NDArray, scores: npt.NDArray, 
                                 detection_arrays: Dict[str, npt.NDArray]) -> None:
        """
        Process detections for a single frame.

        Args:
            idx: Frame index
            bboxes: Array of detected bounding boxes
            scores: Array of detection confidence scores
            detection_arrays: Dictionary to store detection results
        """
        if len(bboxes) == 0:
            return

        if self.target_hand in ["left", "right"]:
            # Single-arm mode: select the bounding box with the highest score
            best_idx = np.argmax(scores)
            best_bbox = bboxes[best_idx]
            best_bbox_ctr = get_bbox_center(best_bbox)
   
            if self.target_hand == "left":
                detection_arrays['left_bboxes'][idx] = best_bbox
                detection_arrays['left_bboxes_ctr'][idx] = best_bbox_ctr
                detection_arrays['left_hand_detected'][idx] = True
            elif self.target_hand == "right":
                detection_arrays['right_bboxes'][idx] = best_bbox
                detection_arrays['right_bboxes_ctr'][idx] = best_bbox_ctr
                detection_arrays['right_hand_detected'][idx] = True
        else:
            # Bimanual mode: get top 2 scoring boxes
            top_indices = np.argsort(scores)[::-1][:2]
            for i in top_indices:
                bbox = bboxes[i]
                bbox_ctr = get_bbox_center(bbox)

                # Spatial reasoning: left side of image = left hand, right side = right hand
                if bbox_ctr[0] <= self.center:
                    # Assign to left hand (if not already assigned)
                    if not detection_arrays['left_hand_detected'][idx]:
                        detection_arrays['left_bboxes'][idx] = bbox
                        detection_arrays['left_bboxes_ctr'][idx] = bbox_ctr
                        detection_arrays['left_hand_detected'][idx] = True
                else:
                    # Assign to right hand (if not already assigned)
                    if not detection_arrays['right_hand_detected'][idx]:
                        detection_arrays['right_bboxes'][idx] = bbox
                        detection_arrays['right_bboxes_ctr'][idx] = bbox_ctr
                        detection_arrays['right_hand_detected'][idx] = True


    # ============================================================================
    # EPIC-SPECIFIC METHODS (EPIC Dataset Processing)
    # ============================================================================

    def _validate_epic_data_structure(self, epic_data: List[Any]) -> bool:
        """Validate EPIC data structure before processing."""
        if not epic_data:
            return False
        
        # Check if first item has required attributes
        try:
            first_item = epic_data[0]
            if not hasattr(first_item, 'side') or not hasattr(first_item, 'bbox'):
                logging.warning("EPIC data missing required attributes: 'side' or 'bbox'")
                return False
            
            # Check if bbox has required attributes
            bbox = first_item.bbox
            required_attrs = ['left', 'right', 'top', 'bottom']
            if not all(hasattr(bbox, attr) for attr in required_attrs):
                logging.warning("EPIC bbox missing required attributes: left, right, top, bottom")
                return False
                
            return True
        except Exception as e:
            logging.warning(f"Error validating EPIC data structure: {str(e)}")
            return False

    def _load_epic_hand_data(self, paths: Paths) -> Dict[str, Any]:
        """
        Load and validate pre-computed hand detection data from EPIC-KITCHENS dataset.
        
        EPIC-KITCHENS provides pre-computed hand detection annotations that we can
        use directly instead of running our own detection. This method filters and
        sorts the data for efficient frame-by-frame processing.
        
        Args:
            paths: Paths object containing detection data file location
            
        Returns:
            Dictionary of filtered and sorted hand detection data
            
        Raises:
            FileNotFoundError: If detection data file doesn't exist
        """
        if not os.path.exists(paths.hand_detection_data):
            raise FileNotFoundError(f"Hand detection data not found: {paths.hand_detection_data}")
        
        with open(paths.hand_detection_data, 'rb') as f:
            hand_detection_data = dict(pickle.load(f))
        
        # Filter out detection objects without valid side information
        filtered_data = {
            key: [obj for obj in obj_list if hasattr(obj, 'side')]
            for key, obj_list in hand_detection_data.items()
        }
        
        # Sort by frame index for sequential processing
        self.filtered_hand_detection_data = dict(sorted(filtered_data.items(), key=lambda x: int(x[0])))
        self.sorted_keys = sorted(self.filtered_hand_detection_data.keys(), key=lambda k: int(k))
        
        return self.filtered_hand_detection_data

    def _process_epic_frames(self, imgs_rgb: npt.NDArray[np.uint8]) -> DetectionResults:
        """
        Process frames using pre-computed EPIC-KITCHENS hand detection data.
        
        This method processes EPIC-KITCHENS dataset videos using their provided
        hand detection annotations, converting them to our standard format while
        applying spatial validation constraints.
        
        Args:
            imgs_rgb: Array of RGB images for dimension reference
            
        Returns:
            Dictionary containing detection results in the same format as _process_frames
        """
        num_frames = len(imgs_rgb)
        
        detection_arrays = self._initialize_detection_arrays(num_frames)

        # Process each frame using EPIC detection data
        for idx in range(num_frames):
            try:
                epic_data = self.filtered_hand_detection_data[self.sorted_keys[idx]]
                
                if len(epic_data) == 0:
                    continue
                
                # Process frame detections
                self._process_epic_frame_detections(idx, epic_data, detection_arrays)
            except KeyError:
                logger.warning(f"Missing EPIC data for frame {idx}")
                continue
            except Exception as e:
                logger.warning(f"EPIC frame {idx} processing failed: {str(e)}")
                continue

        return {
            'left_hand_detected': detection_arrays['left_hand_detected'],
            'right_hand_detected': detection_arrays['right_hand_detected'],
            'left_bboxes': detection_arrays['left_bboxes'],
            'right_bboxes': detection_arrays['right_bboxes'],
            'left_bboxes_ctr': detection_arrays['left_bboxes_ctr'],
            'right_bboxes_ctr': detection_arrays['right_bboxes_ctr']
        }

    def _process_epic_frame_detections(self, idx: int, epic_data: List[Any], 
                                      detection_arrays: Dict[str, npt.NDArray]) -> None:
        """Process EPIC detections for a single frame."""
        # Process left and right hands separately
        left_detected, left_bbox, left_bbox_ctr = self._process_epic_hand_detection(epic_data, "left")
        right_detected, right_bbox, right_bbox_ctr = self._process_epic_hand_detection(epic_data, "right")
        
        # Store results in pre-allocated arrays
        detection_arrays['left_hand_detected'][idx] = left_detected
        detection_arrays['right_hand_detected'][idx] = right_detected
        if left_detected:
            detection_arrays['left_bboxes'][idx] = left_bbox
            detection_arrays['left_bboxes_ctr'][idx] = left_bbox_ctr
        if right_detected:
            detection_arrays['right_bboxes'][idx] = right_bbox
            detection_arrays['right_bboxes_ctr'][idx] = right_bbox_ctr

        # Quality check: If hands appear crossed (left hand on right side), 
        # mark both as invalid to avoid confusion
        if left_detected and right_detected:
            self._validate_hand_positions(idx, left_bbox_ctr, right_bbox_ctr, detection_arrays)

    def _validate_hand_positions(self, idx: int, left_bbox_ctr: npt.NDArray, right_bbox_ctr: npt.NDArray,
                                detection_arrays: Dict[str, npt.NDArray]) -> None:
        """Validate that hands are on correct sides of the image."""
        if left_bbox_ctr[0] > right_bbox_ctr[0]:
            # Left hand appears to be on the right side - mark both as invalid
            detection_arrays['left_hand_detected'][idx] = False
            detection_arrays['right_hand_detected'][idx] = False
    
    def _process_epic_hand_detection(self, 
                            epic_data: List[Any], 
                            hand_side: HandSide) -> Tuple[bool, BBoxArray, CenterArray]:
        """
        Process EPIC hand detection data for a single frame and hand side.
        
        This method extracts and validates hand detection data from EPIC annotations,
        converting normalized coordinates to pixel coordinates and applying spatial
        validation constraints.

        Args:
            epic_data: List of detection objects for the current frame
            hand_side: Either "left" or "right" specifying which hand to process

        Returns:
            Tuple of (is_detected: bool, bbox: ndarray, bbox_center: ndarray)
        """
        if hand_side not in hand_side_dict:
            raise ValueError(f"Invalid hand side: {hand_side}")

        # Default empty result for failed detections
        empty_result = (False, np.array([0, 0, 0, 0]), np.array([0, 0]))
        
        try:
            # Filter and validate detection data
            hand_data = self._filter_epic_hand_data(epic_data, hand_side)
            if not hand_data:
                return empty_result
            
            # Validate data structure
            if not self._validate_epic_data_structure(hand_data):
                return empty_result

            # Extract and process bounding box
            bbox, bbox_center = self._extract_epic_bbox(hand_data[0])
            
            # Validate bounding box coordinates
            if not self._validate_bbox_coordinates(hand_data[0].bbox, hand_side):
                return empty_result

            # Apply spatial validation
            is_valid = self._validate_spatial_position(bbox_center, hand_side)
            return (is_valid, bbox, bbox_center) if is_valid else empty_result

        except Exception as e:
            logging.warning(f"Unexpected error processing {hand_side} hand detection: {str(e)}")
            return empty_result

    def _filter_epic_hand_data(self, epic_data: List[Any], hand_side: HandSide) -> List[Any]:
        """Filter EPIC detection data for the specified hand side."""
        return [data for data in epic_data if data.side.value == hand_side_dict[hand_side]]

    def _extract_epic_bbox(self, hand_data: Any) -> Tuple[BBoxArray, CenterArray]:
        """Extract bounding box and center from EPIC hand detection data."""
        bbox_cls = hand_data.bbox
        
        # Convert normalized coordinates to pixel coordinates
        bbox = np.array([
            bbox_cls.left * self.W,
            bbox_cls.top * self.H,
            bbox_cls.right * self.W,
            bbox_cls.bottom * self.H
        ])
        
        # Calculate center point for spatial validation
        bbox_center = np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ]).astype(np.int32)
        
        return bbox, bbox_center

    def _validate_spatial_position(self, bbox_center: CenterArray, hand_side: HandSide) -> bool:
        """Validate that hand center is on correct side of image."""
        if hand_side == "left":
            return bbox_center[0] <= (self.center + self.HAND_SIDE_MARGIN)
        else:  # right
            return bbox_center[0] >= (self.center - self.HAND_SIDE_MARGIN)
    
    def _validate_bbox_coordinates(self, bbox_cls: Any, hand_side: HandSide) -> bool:
        """Validate bounding box coordinates are within valid range [0,1]."""
        if not (0 <= bbox_cls.left <= 1 and 0 <= bbox_cls.right <= 1 and
                0 <= bbox_cls.top <= 1 and 0 <= bbox_cls.bottom <= 1):
            logging.warning(f"Invalid bbox coordinates detected for {hand_side} hand: "
                            f"left={bbox_cls.left:.3f}, right={bbox_cls.right:.3f}, "
                            f"top={bbox_cls.top:.3f}, bottom={bbox_cls.bottom:.3f}")
            return False
        return True


    # ============================================================================
    # UTILITY/HELPER METHODS (General utilities and post-processing)
    # ============================================================================


    def _post_process_detections(self, detection_results: DetectionResults) -> DetectionResults:
        """
        Apply post-processing to improve detection temporal consistency.
        
        This method applies several filters and enhancements to the raw detection
        results to improve their quality and temporal coherence:
        1. Filter out large spatial jumps that indicate tracking errors
        2. Interpolate short gaps in detection sequences
        3. Calculate quality metrics (distance to image edges)
        
        Args:
            detection_results: Raw detection results from frame processing
            
        Returns:
            Enhanced detection results with improved temporal consistency
        """
        # Filter out large jumps for both hands
        left_results = self._filter_large_jumps(
            detection_results['left_hand_detected'],
            detection_results['left_bboxes'],
            detection_results['left_bboxes_ctr'],
            max_jump=self.MAX_SPATIAL_JUMP,
            lookahead=self.MAX_JUMP_LOOKAHEAD
        )
        right_results = self._filter_large_jumps(
            detection_results['right_hand_detected'],
            detection_results['right_bboxes'],
            detection_results['right_bboxes_ctr'],
            max_jump=self.MAX_SPATIAL_JUMP,
            lookahead=self.MAX_JUMP_LOOKAHEAD
        )

        # Interpolate missing detections for smooth trajectories
        left_results = self._interpolate_detections(*left_results, max_gap=self.MAX_INTERPOLATION_GAP)
        right_results = self._interpolate_detections(*right_results, max_gap=self.MAX_INTERPOLATION_GAP)

        # Calculate quality metrics: minimum distance from bbox center to image edges
        left_bbox_min_dist = get_bbox_center_min_dist_to_edge(left_results[1], self.W, self.H)
        right_bbox_min_dist = get_bbox_center_min_dist_to_edge(right_results[1], self.W, self.H)

        return {
            'left_hand_detected': left_results[0],
            'right_hand_detected': right_results[0],
            'left_bboxes': left_results[1],
            'right_bboxes': right_results[1],
            'left_bboxes_ctr': left_results[2],
            'right_bboxes_ctr': right_results[2],
            'left_bbox_min_dist_to_edge': left_bbox_min_dist,
            'right_bbox_min_dist_to_edge': right_bbox_min_dist
        }

    def _generate_visualization(self, imgs_rgb: np.ndarray, results: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Generate visualization of detection results for quality assessment.
        
        Creates annotated frames showing detected bounding boxes for visual
        inspection of detection quality and temporal consistency.
        
        Args:
            imgs_rgb: Original RGB video frames
            results: Processed detection results
            
        Returns:
            List of annotated images with bounding boxes drawn
        """
        list_img_annot = []
        for idx in range(len(imgs_rgb)):
            left_bbox = None
            right_bbox = None
            
            # Prepare bounding boxes for visualization
            if results['left_hand_detected'][idx] or results['right_hand_detected'][idx]:
                left_bbox = results['left_bboxes'][idx] if results['left_hand_detected'][idx] else None
                right_bbox = results['right_bboxes'][idx] if results['right_hand_detected'][idx] else None
                
            # Generate annotated image
            img_annot = self.visualize_detections(imgs_rgb[idx], left_bbox, right_bbox, show_image=False)
            list_img_annot.append(img_annot)
        return list_img_annot

    def _save_results(self, paths: Paths, results: DetectionResults, visualization_results: List[npt.NDArray[np.uint8]]) -> None:
        """
        Save all processed results to disk in standardized format.
        
        Args:
            paths: Paths object containing output file locations
            results: Processed detection results
            visualization_results: Generated visualization frames
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(paths.bbox_processor):
            os.makedirs(paths.bbox_processor)

        # Save detection data in compressed NumPy format
        np.savez(paths.bbox_data, **results)
        
        # Save visualization video with lossless compression
        media.write_video(paths.video_bboxes, visualization_results, fps=15, codec="ffv1")

    def _interpolate_detections(self, detected: DetectionFlagArray,
                               bboxes: BBoxArray,
                               centers: CenterArray,
                               max_gap: int = 10) -> Tuple[DetectionFlagArray, BBoxArray, CenterArray]:
        """
        Interpolate bounding boxes and detection status for short gaps in tracking.
        
        This method fills in missing detections using linear interpolation when the
        gap is small enough to reasonably assume continuous hand motion. This helps
        create smoother trajectories for downstream processing.
        
        Args:
            detected: Boolean array of detection status per frame
            bboxes: Array of bounding boxes [N, 4] format [x1, y1, x2, y2]
            centers: Array of bbox centers [N, 2] format [x, y]
            max_gap: Maximum gap size (in frames) to interpolate over
            
        Returns:
            Tuple of (interpolated detection status, interpolated bboxes, interpolated centers)
        """
        detected = detected.copy()
        bboxes = bboxes.copy()
        centers = centers.copy()
        
        # Handle single-frame gaps first (most common case)
        for i in range(1, len(detected) - 1):
            if not detected[i] and detected[i-1] and detected[i+1]:
                # Get valid bboxes/centers before and after gap
                start_bbox = bboxes[i-1]
                end_bbox = bboxes[i+1]
                start_center = centers[i-1]
                end_center = centers[i+1]
                
                # Linear interpolation with t = 0.5 for single frame
                interpolated_bbox = 0.5 * (start_bbox + end_bbox)
                interpolated_center = 0.5 * (start_center + end_center)
                
                # Validate interpolated values are reasonable
                if self._is_valid_bbox(interpolated_bbox) and self._is_valid_center(interpolated_center):
                    bboxes[i] = interpolated_bbox
                    centers[i] = interpolated_center
                    detected[i] = True
        
        # Handle multi-frame gaps
        non_detect_start = None
        for i in range(1, len(detected) - 1):
            # Start of non-detection sequence
            if detected[i-1] and not detected[i]:
                non_detect_start = i
            # End of non-detection sequence
            elif non_detect_start is not None and not detected[i] and detected[i+1]:
                non_detect_end = i
                gap_size = non_detect_end - non_detect_start + 1

                # Only interpolate if gap is small enough and has valid detections on both sides
                if gap_size <= max_gap:
                    # Get valid bboxes/centers before and after gap
                    start_bbox = bboxes[non_detect_start - 1]
                    end_bbox = bboxes[non_detect_end + 1]
                    start_center = centers[non_detect_start - 1]
                    end_center = centers[non_detect_end + 1]
                    
                    # Generate interpolation steps
                    steps = gap_size + 1
                    for j in range(gap_size):
                        t = (j + 1) / steps  # Interpolation factor

                        # Linear interpolation of bbox coordinates
                        bboxes[non_detect_start + j] = (1 - t) * start_bbox + t * end_bbox
                        
                        # Linear interpolation of center coordinates
                        centers[non_detect_start + j] = (1 - t) * start_center + t * end_center
                        
                        # Mark as detected
                        detected[non_detect_start + j] = True

                non_detect_start = None
                
        return detected, bboxes, centers

    def _is_valid_bbox(self, bbox: BBoxArray) -> bool:
        """Validate that bbox coordinates are reasonable."""
        if bbox is None or len(bbox) != 4:
            return False
        # Check for reasonable bounds (not negative, not too large)
        return (bbox >= 0).all() and (bbox[:2] < bbox[2:]).all() and bbox.max() < max(self.W, self.H) * 2

    def _is_valid_center(self, center: CenterArray) -> bool:
        """Validate that center coordinates are reasonable."""
        if center is None or len(center) != 2:
            return False
        # Check for reasonable bounds
        return (center >= 0).all() and center[0] < self.W * 2 and center[1] < self.H * 2

    def visualize_detections(self, img: npt.NDArray[np.uint8],
                           left_bbox: Optional[npt.NDArray[np.float32]] = None,
                           right_bbox: Optional[npt.NDArray[np.float32]] = None,
                           show_image: bool = True) -> npt.NDArray[np.uint8]:
        """
        Visualize hand detections by drawing bounding boxes on the image.
        
        This method creates annotated images showing detected hand locations with
        color-coded bounding boxes (red for left hand, green for right hand).

        Args:
            img: Input RGB image to annotate
            left_bbox: Left hand bounding box [x1, y1, x2, y2] or None if not detected
            right_bbox: Right hand bounding box [x1, y1, x2, y2] or None if not detected
            show_image: Whether to display the image using cv2.imshow

        Returns:
            The annotated image
        """
        # Work directly with the input image (assumed to be in BGR format)
        img_bgr = img
        
        # Draw left hand bounding box in red
        if left_bbox is not None and not np.array_equal(left_bbox, np.array([0, 0, 0, 0])):
            cv2.rectangle(
                img_bgr, 
                (int(left_bbox[0]), int(left_bbox[1])), 
                (int(left_bbox[2]), int(left_bbox[3])), 
                self.LEFT_HAND_COLOR,
                self.BBOX_THICKNESS
            )
            
        # Draw right hand bounding box in green
        if right_bbox is not None and not np.array_equal(right_bbox, np.array([0, 0, 0, 0])):
            cv2.rectangle(
                img_bgr, 
                (int(right_bbox[0]), int(right_bbox[1])), 
                (int(right_bbox[2]), int(right_bbox[3])), 
                self.RIGHT_HAND_COLOR,
                self.BBOX_THICKNESS
            )
            
        # Optionally display the image for debugging
        if show_image:
            cv2.imshow("Hand Detections", img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return img_bgr

    @staticmethod
    def _filter_large_jumps(detected: DetectionFlagArray,
                           bboxes: BBoxArray,
                           centers: CenterArray,
                           max_jump: float = 200.0,
                           lookahead: int = 10) -> Tuple[DetectionFlagArray, BBoxArray, CenterArray]:
        """
        Filter out small groups of detections that are spatially inconsistent with the trajectory.
        
        This method identifies and removes isolated detections that are far from the
        expected trajectory, which usually indicate false positives or tracking errors.
        It helps maintain temporal consistency in hand tracking.
        
        Args:
            detected: Boolean array of detection status per frame
            bboxes: Array of bounding boxes [N, 4] format [x1, y1, x2, y2]
            centers: Array of bbox centers [N, 2] format [x, y]
            max_jump: Maximum allowed distance (in pixels) between consecutive detections
            lookahead: Maximum number of consecutive distant points to filter as a group
            
        Returns:
            Tuple of (filtered detection status, filtered bboxes, filtered centers)
        """
        detected = detected.copy()
        bboxes = bboxes.copy()
        centers = centers.copy()
        
        # Templates for clearing invalid detections
        empty_bbox = np.array([0, 0, 0, 0])
        empty_center = np.array([0, 0])
        
        i = 0
        while i < len(detected):
            # Find next detected point to compare against
            next_valid = i + 1

            if next_valid >= len(detected):
                break
                
            # Calculate spatial distance to next detection
            dist = np.linalg.norm(centers[next_valid] - centers[i])
            
            if dist > max_jump:
                # Large jump detected - check if it's part of a small group of outliers
                distant_points = []
                ref_center = centers[i]  # Use current point as reference
                
                # Look ahead to find consecutive distant points
                for j in range(next_valid, len(detected)):
                    curr_dist = np.linalg.norm(centers[j] - ref_center)
                    if curr_dist > max_jump:
                        distant_points.append(j)
                    else:
                        break
                
                # If we found a small group of distant points, filter them out
                if len(distant_points) > 0 and len(distant_points) <= lookahead:
                    for idx in distant_points:
                        detected[idx] = False
                        bboxes[idx] = empty_bbox
                        centers[idx] = empty_center
                        logging.warning(f"Filtered out frame {idx} as part of small distant group")
            
            i = next_valid
        
        return detected, bboxes, centers
    




    


    

