import logging
import os
import shutil
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed  # type: ignore
import hydra
from omegaconf import DictConfig
import fnmatch

from phantom.processors.base_processor import BaseProcessor

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")


class ProcessingMode(Enum):
    """Enumeration of valid processing modes."""
    BBOX = "bbox"
    HAND2D = "hand2d"
    HAND3D = "hand3d"
    HAND_SEGMENTATION = "hand_segmentation"
    ARM_SEGMENTATION = "arm_segmentation"
    ACTION = "action"
    SMOOTHING = "smoothing"
    HAND_INPAINT = "hand_inpaint"
    ROBOT_INPAINT = "robot_inpaint"
    ALL = "all"


PROCESSING_ORDER = [
    "bbox",
    "hand2d",
    "arm_segmentation",
    "hand_segmentation",
    "hand3d",
    "action",
    "smoothing",
    "hand_inpaint",
    "robot_inpaint",
]


PROCESSING_ORDER_EPIC = [
    "bbox",
    "hand2d",
    "arm_segmentation",
    "action",
    "smoothing",
    "hand_inpaint",
    "robot_inpaint",
]


def process_one_demo(data_sub_folder: str, cfg: DictConfig, processor_classes: dict) -> None:
    # Choose processing order based on epic flag
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER
    # Handle both string and list modes
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)

    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        try:
            processor.process_one_demo(data_sub_folder)
        except Exception as e:
            print(f"Error in {mode} processing: {e}")
            if cfg.debug:
                raise


def _extract_modes(mode_config, processing_order: List[str]) -> List[str]:
    """Extract and validate processing modes from config."""
    if isinstance(mode_config, str):
        if ',' in mode_config:
            modes = [m.strip() for m in mode_config.split(',')]
        else:
            modes = [mode_config]
    else:
        modes = list(mode_config) if isinstance(mode_config, (list, tuple)) else [mode_config]
    
    selected_modes = []
    for mode in modes:
        if mode == "all":
            selected_modes.extend(processing_order)
        elif mode in processing_order:
            selected_modes.append(mode)
    
    return selected_modes


def process_all_demos(cfg: DictConfig, processor_classes: dict) -> None:
    # Choose processing order based on epic flag
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER

    # Handle both string and list modes
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)

    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        for data_sub_folder in tqdm(all_data_folders):
            try:
                processor.process_one_demo(data_sub_folder)
            except Exception as e:
                print(f"Error in {mode} processing: {e}")
                if cfg.debug:
                    raise


def process_all_demos_parallel(cfg: DictConfig, processor_classes: dict) -> None:
    # Choose processing order based on epic flag
    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER

    # Handle both string and list modes
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)

    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    for mode in selected_modes:
        print(f"----------------- {mode.upper()} PROCESSOR -----------------")
        processor_cls = processor_classes[mode]
        processor = processor_cls(cfg)
        Parallel(n_jobs=cfg.n_processes)(
            delayed(processor.process_one_demo)(data_sub_folder)
            for data_sub_folder in all_data_folders
        )


def find_hand_dataset_videos(data_folder: str, task_name: Optional[str] = None,
                            video_patterns: Optional[List[str]] = None,
                            frame_idx: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Find videos in hand_dataset structure.
    Structure: data_folder/[task]/[frame_idx]/[episode_id]_[model_name]_rgb.mp4

    Returns:
        List of (task_name, frame_idx, video_filename) tuples
    """
    videos = []

    # Determine which task folders to process
    if task_name:
        task_folders = [task_name]
    else:
        task_folders = [d for d in os.listdir(data_folder)
                        if os.path.isdir(os.path.join(data_folder, d))]

    for task in task_folders:
        task_path = os.path.join(data_folder, task)
        if not os.path.isdir(task_path):
            print(f"Warning: Task folder not found: {task_path}")
            continue
        
        # Determine which frame_idx subfolders to process
        if frame_idx is not None:
            idx_folders = [str(frame_idx)]
        else:
            idx_folders = sorted(
                [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))],
                key=lambda x: int(x) if x.isdigit() else x
            )
            
        for idx in idx_folders:
            idx_path = os.path.join(task_path, idx)
            if not os.path.isdir(idx_path):
                print(f"Warning: frame_idx folder not found: {idx_path}")
                continue

        # Find all mp4 files in the task folder
        mp4_files = sorted([f for f in os.listdir(idx_path) if f.endswith('_rgb.mp4')])

        # Filter by video patterns if provided
        if video_patterns:
            filtered_files = []
            for pattern in video_patterns:
                filtered_files.extend([f for f in mp4_files if fnmatch.fnmatch(f, pattern)])
            mp4_files = filtered_files

        for mp4_file in mp4_files:
            videos.append((task, idx, mp4_file))

    return videos


def setup_hand_dataset_temp_structure(source_video_path: str, temp_data_folder: str) -> None:
    """
    Create temporary directory structure for hand_dataset videos.
    The processors expect: data_folder/[0,1,2...]/video_left.mp4 or similar.
    For hand_dataset, we create: temp_data_folder/0/video_left.mp4 (symlink or copy)
    """
    os.makedirs(temp_data_folder, exist_ok=True)

    # Create symlink or copy the video as video_left.mp4 (hand videos are single view)
    video_target = os.path.join(temp_data_folder, "video_L.mp4")
    if os.path.exists(video_target):
        os.remove(video_target)

    # Use symlink to avoid duplicating large video files
    try:
        os.symlink(os.path.abspath(source_video_path), video_target)
    except (OSError, NotImplementedError):
        # If symlink fails, copy the file
        shutil.copy2(source_video_path, video_target)
    
    source_depth_path = source_video_path.replace("_rgb.mp4", "_depth.npy")
    
    # The target filename must follow the standard naming convention depth.npy so that paths.depth can read it correctly.
    depth_target = os.path.join(temp_data_folder, "depth.npy")
    if os.path.exists(depth_target):
        os.remove(depth_target)

    if os.path.exists(source_depth_path):
        try:
            os.symlink(os.path.abspath(source_depth_path), depth_target)
        except (OSError, NotImplementedError):
            shutil.copy2(source_depth_path, depth_target)
    else:
        # If some videos have only RGB and no depth, give a warning
        print(f"Warning: Depth file not found at {source_depth_path}, Hand3DProcessor may skip depth alignment for this video.")


def process_hand_dataset_task(cfg: DictConfig, processor_classes: dict, task_name: Optional[str] = None,
                            video_patterns: Optional[List[str]] = None,
                            frame_idx: Optional[int] = None) -> None:
    """
    Process hand_dataset format videos.

    Args:
        cfg: Configuration
        processor_classes: Processor class mapping
        task_name: Task name (e.g., 'pick', 'push')
        video_patterns: List of video name patterns to process
    """
    data_folder = cfg.data_root_dir + '/hand_dataset'
    processed_root = cfg.processed_data_root_dir

    # Find all matching videos
    videos = find_hand_dataset_videos(data_folder, task_name, video_patterns, frame_idx)

    if not videos:
        print(f"No videos found for task '{task_name}'")
        return

    print(f"Found {len(videos)} video(s) for task '{task_name}'")

    processing_order = PROCESSING_ORDER_EPIC if cfg.epic else PROCESSING_ORDER
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            selected_modes = []
            for mode in cfg.mode.split(','):
                mode = mode.strip()
                if mode == "all":
                    selected_modes.extend(processing_order)
                elif mode in processing_order:
                    selected_modes.append(mode)
        else:
            selected_modes = [m for m in processing_order if m in cfg.mode or "all" in cfg.mode]
    else:
        # For list of modes, use the order provided by user
        selected_modes = []
        for mode in cfg.mode:
            if mode == "all":
                selected_modes.extend(processing_order)
            elif mode in processing_order:
                selected_modes.append(mode)
    print(f"Processing modes: {selected_modes}")

    # Create temporary folder for all videos of this task
    temp_base = os.path.join('../data/raw', f"{task_name}_temp")
    os.makedirs(temp_base, exist_ok=True)
    temp_base_processed = os.path.join(processed_root, f"{task_name}_temp")

    try:
        for task, idx, mp4_file in tqdm(videos, desc=f"Processing task '{task_name}'"):
            # Extract video name without extension and model name
            # Format: [episode_id]_[model_name]_rgb.mp4
            # Output folder: [episode_id]_[model_name]
            video_basename = mp4_file.replace("_rgb.mp4", "")

            # Create output folder structure
            output_folder = os.path.join(processed_root, 'hand_dataset', task, idx, video_basename)
            os.makedirs(output_folder, exist_ok=True)

            # Setup temporary processing folder (RAW Input)
            temp_data_folder = os.path.join(temp_base, "0")
            print(f"Setting up temporary folder: {temp_data_folder}")
            if os.path.exists(temp_data_folder):
                shutil.rmtree(temp_data_folder)

            temp_processed_folder_0 = os.path.join(temp_base_processed, "0")
            if os.path.exists(temp_processed_folder_0):
                print(f"Cleaning up stale processed data: {temp_processed_folder_0}")
                shutil.rmtree(temp_processed_folder_0)

            source_video = os.path.join(data_folder, task, idx, mp4_file)
            if not os.path.exists(source_video):
                print(f"Warning: Video not found: {source_video}")
                continue
            print(f"Processing video: {source_video}")
            setup_hand_dataset_temp_structure(source_video, temp_data_folder)

            # Create temporary config for this video
            temp_cfg = cfg.copy()
            temp_cfg.demo_name = f"{task_name}_temp"
            temp_cfg.data_root_dir = processed_root if task_name in cfg.data_root_dir else cfg.data_root_dir

            # Process each mode
            for mode in selected_modes:
                print(f"  [{task}/{idx}/{video_basename}] - {mode.upper()}")
                processor_cls = processor_classes[mode]
                processor = processor_cls(temp_cfg)

                try:
                    processor.process_one_demo("0")
                except Exception as e:
                    print(f"  Error in {mode} processing for {mp4_file}: {e}")
                    if cfg.debug:
                        raise

            # Move output files from temp to final location
            temp_processed_output = os.path.join(temp_base_processed, "0")
            
            # Fallback check: if nothing in processed temp, maybe it did write to raw (unlikely but safe)
            if not os.path.exists(temp_processed_output):
                print(f"Warning: Could not find output in {temp_processed_output}, checking raw temp...")
                temp_output_source = temp_data_folder
            else:
                temp_output_source = temp_processed_output

            # Copy all output files/folders except the video
            for item in os.listdir(temp_output_source):
                if item not in ["video_L.mp4", "video_R.mp4"]:
                    src = os.path.join(temp_output_source, item)
                    dst = os.path.join(output_folder, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

            print(f"  Output saved to: {output_folder}")
    finally:
        # Cleanup temporary RAW folder
        if os.path.exists(temp_base):
            shutil.rmtree(temp_base, ignore_errors=True)

        # Cleanup temporary PROCESSED folder (pick_temp)
        if os.path.exists(temp_base_processed):
            print(f"Cleaning up temporary artifact: {temp_base_processed}")
            shutil.rmtree(temp_base_processed, ignore_errors=True)


def get_processor_classes(cfg: DictConfig) -> dict:
    """Initialize the processor classes"""
    from phantom.processors.bbox_processor import BBoxProcessor
    from phantom.processors.segmentation_processor import HandSegmentationProcessor, ArmSegmentationProcessor
    from phantom.processors.hand_processor import Hand2DProcessor, Hand3DProcessor
    from phantom.processors.action_processor import ActionProcessor
    from phantom.processors.smoothing_processor import SmoothingProcessor
    from phantom.processors.robotinpaint_processor import RobotInpaintProcessor
    from phantom.processors.handinpaint_processor import HandInpaintProcessor

    return {
        "bbox": BBoxProcessor,
        "hand2d": Hand2DProcessor,
        "hand3d": Hand3DProcessor,
        "hand_segmentation": HandSegmentationProcessor,
        "arm_segmentation": ArmSegmentationProcessor,
        "action": ActionProcessor,
        "smoothing": SmoothingProcessor,
        "robot_inpaint": RobotInpaintProcessor,
        "hand_inpaint": HandInpaintProcessor,
    }


def validate_mode(cfg: DictConfig) -> None:
    """
    Validate that the mode parameter contains only valid processing modes.

    Args:
        cfg: Configuration object containing mode parameter

    Raises:
        ValueError: If mode contains invalid options
    """
    if isinstance(cfg.mode, str):
        # Handle comma-separated string format
        if ',' in cfg.mode:
            modes = [mode.strip() for mode in cfg.mode.split(',')]
        else:
            modes = [cfg.mode]
    else:
        modes = cfg.mode

    # Get valid modes from enum
    valid_modes = {mode.value for mode in ProcessingMode}
    invalid_modes = [mode for mode in modes if mode not in valid_modes]

    if invalid_modes:
        valid_mode_list = [mode.value for mode in ProcessingMode]
        raise ValueError(
            f"Invalid mode(s): {invalid_modes}. "
            f"Valid modes are: {valid_mode_list}"
        )


def main(cfg: DictConfig):
    # Validate mode parameter
    validate_mode(cfg)

    # Get processor classes
    processor_classes = get_processor_classes(cfg)

    # Check if processing hand_dataset format
    is_hand_dataset = hasattr(cfg, 'is_hand_dataset') and cfg.is_hand_dataset

    if is_hand_dataset:
        # Process hand_dataset format
        task_name = getattr(cfg, 'task_name', None)
        video_patterns = getattr(cfg, 'video_patterns', None)
        frame_idx = getattr(cfg, 'frame_idx', None)
        if isinstance(video_patterns, str):
            video_patterns = [video_patterns]

        process_hand_dataset_task(cfg, processor_classes, task_name, video_patterns, frame_idx)
    else:
        # Process standard format
        if cfg.n_processes > 1:
            process_all_demos_parallel(cfg, processor_classes)
        elif cfg.demo_num is not None:
            process_one_demo(cfg.demo_num, cfg, processor_classes)
        else:
            process_all_demos(cfg, processor_classes)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def hydra_main(cfg: DictConfig):
    """
    Main entry point using Hydra configuration.

    Example usage:
    - Process all demos with bbox: python process_data.py mode=bbox
    - Process single demo: python process_data.py mode=bbox demo_num=0
    - Use EPIC dataset: python process_data.py dataset=epic mode=bbox
    - Parallel processing: python process_data.py mode=bbox n_processes=4
    - Process multiple modes sequentially: python process_data.py mode=bbox,hand3d
    - Process with custom order: python process_data.py mode=hand3d,bbox,action
    - Process with bracket notation (use quotes): python process_data.py "mode=[bbox,hand3d]"
    """
    main(cfg)


if __name__ == "__main__":
    hydra_main()
