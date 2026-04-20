"""
Hand Inpainting Processor Module

This module removes human hands from demonstration videos using the E2FGVI model. 

Paper:
Towards An End-to-End Framework for Flow-Guided Video Inpainting
https://github.com/MCG-NKU/E2FGVI.git

Processing Pipeline:
1. Load pre-trained E2FGVI model and initialize GPU processing
2. Read input video frames and corresponding hand segmentation masks
3. Process frames in batches with neighboring temporal context
4. Apply mask-guided inpainting to remove hand regions
5. Verify complete processing and handle any missed frames
6. Save final hand-free video for robot learning applications
"""

import cv2
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torch
import mediapy as media
import logging
import gc
from typing import List, Tuple, Optional, Any, Union

from phantom.processors.base_processor import BaseProcessor
from phantom.utils.data_utils import get_parent_folder_of_package
from E2FGVI.model.e2fgvi_hq import InpaintGenerator  # type: ignore
from E2FGVI.core.utils import to_tensors  # type: ignore

DEFAULT_CHECKPOINT = 'E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth'

logger = logging.getLogger(__name__)

class HandInpaintProcessor(BaseProcessor): 
    """
    Hand inpainting processor for removing human hands from demonstration videos.
    
    Attributes:
        model: E2FGVI neural network model for video inpainting
        device: GPU/CPU device for model execution
        ref_length (int): Spacing between reference frames for temporal consistency
        num_ref (int): Number of reference frames to use (-1 for automatic)
        neighbor_stride (int): Spacing between neighboring frames in temporal context
        batch_size (int): Number of frame groups to process simultaneously
        scale_factor (int): Resolution scaling factor for processing optimization
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize the hand inpainting processor with E2FGVI model and parameters.
        
        Args:
            args: Command line arguments containing processing configuration
                 including scale factor and other inpainting parameters
        """
        super().__init__(args)
        
        # Load pre-trained E2FGVI model
        root_dir = get_parent_folder_of_package("E2FGVI")
        checkpoint_path = Path(root_dir, DEFAULT_CHECKPOINT)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and load the inpainting model
        self.model = InpaintGenerator().to(self.device)
        data = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(data)
        self.model.eval()

        # Configure temporal processing parameters
        self.ref_length: int = 20        # Spacing between reference frames
        self.num_ref: int = -1           # Number of reference frames (-1 = automatic)
        self.neighbor_stride: int = 5    # Stride for neighboring frame selection

        # Configure batch processing parameters for memory optimization
        self.batch_size: int = 10        # Number of frame groups per batch
        self.scale_factor: int = getattr(args, 'scale_factor', 2)  # Resolution scaling

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache and trigger garbage collection."""
        torch.cuda.empty_cache()
        gc.collect()

    def process_one_demo(self, data_sub_folder: str) -> None:
        """
        Process a single demonstration video to remove hand regions.
        
        Args:
            data_sub_folder: Path to demonstration data folder containing
                           input video and hand segmentation masks
        """
        save_folder = self.get_save_folder(data_sub_folder)
        paths = self.get_paths(save_folder)
        if not os.path.exists(paths.inpaint_processor):
            os.makedirs(paths.inpaint_processor)

        self._process_frames(paths)
    
    def _process_frames(self, paths: Any) -> None:
        """
        Process all video frames to remove hand regions using E2FGVI inpainting.
        
        Args:
            paths: Paths object containing input video and mask file locations
        """
        # Load and prepare video frames
        frames = self._load_and_prepare_frames(paths)
        video_length = len(frames)
        logger.info(f"Processing {video_length} frames")
        
        # Initialize tracking arrays for processed frames
        comp_frames: List[Optional[np.ndarray]] = [None] * video_length
        processed_frame_mask: List[bool] = [False] * video_length
        
        # Process frames in batches with temporal overlap for consistency
        self._process_frames_in_batches(frames, paths, comp_frames, processed_frame_mask)
        
        # Handle any missed frames
        self._process_missed_frames(frames, paths, comp_frames, processed_frame_mask)
        
        # Final verification and save
        self._verify_and_save_results(comp_frames, paths)

    def _load_and_prepare_frames(self, paths: Any) -> List[Image.Image]:
        """Load video frames and prepare them for processing."""
        frames = self.read_frame_from_videos(paths.video_rgb_imgs)
        
        # Calculate output dimensions based on configuration
        h, w = frames[0].height, frames[0].width
        
        if self.epic:
            size = (w, h)
        else:
            if self.square:
                output_resolution = np.array([self.output_resolution, self.output_resolution])
            else:
                output_resolution = np.array([int(w/h*self.output_resolution), self.output_resolution])
            output_resolution = output_resolution.astype(np.int32)
            size = output_resolution
            frames, size = self.resize_frames(frames, size)
            
        return frames

    def _process_frames_in_batches(self, frames: List[Image.Image], paths: Any, 
                                 comp_frames: List[Optional[np.ndarray]], 
                                 processed_frame_mask: List[bool]) -> None:
        """Process frames in batches with temporal overlap."""
        video_length = len(frames)
        h, w = frames[0].height, frames[0].width
        
        for batch_start in tqdm(range(0, video_length, self.batch_size * self.neighbor_stride), 
                               desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size * self.neighbor_stride + self.neighbor_stride, video_length)
            
            # Prepare batch data
            batch_data = self._prepare_batch_data(frames, paths, batch_start, batch_end, h, w)
            
            # Process frames within batch
            self._process_batch_frames(frames, batch_data, batch_start, batch_end, 
                                     comp_frames, processed_frame_mask, h, w)
            
            # Clean up batch memory
            del batch_data['batch_imgs'], batch_data['batch_masks']
            self._clear_gpu_memory()

    def _prepare_batch_data(self, frames: List[Image.Image], paths: Any, 
                          batch_start: int, batch_end: int, h: int, w: int) -> dict:
        """Prepare batch data including frames, masks, and binary masks."""
        batch_frames = frames[batch_start:batch_end]
        batch_imgs = to_tensors()(batch_frames).unsqueeze(0).to(self.device) * 2 - 1
        
        batch_masks = self.read_mask(paths.masks_arm, (w, h))[batch_start:batch_end]
        batch_masks = to_tensors()(batch_masks).unsqueeze(0).to(self.device)
        
        binary_masks = self._create_binary_masks(paths.masks_arm, batch_start, batch_end, w, h)
        
        return {
            'batch_imgs': batch_imgs,
            'batch_masks': batch_masks,
            'binary_masks': binary_masks
        }

    def _create_binary_masks(self, mask_path: str, batch_start: int, batch_end: int, 
                           w: int, h: int) -> List[np.ndarray]:
        """Create binary masks for the batch."""
        masks = self.read_mask(mask_path, (w, h))[batch_start:batch_end]
        binary_masks = []
        
        for mask in masks:
            mask_array = np.array(mask)
            binary_mask = np.expand_dims((mask_array != 0).astype(np.uint8), 2)
            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask = np.expand_dims(binary_mask, 2)
            binary_masks.append(binary_mask)
            
        return binary_masks

    def _process_batch_frames(self, frames: List[Image.Image], batch_data: dict, 
                            batch_start: int, batch_end: int, 
                            comp_frames: List[Optional[np.ndarray]], 
                            processed_frame_mask: List[bool], h: int, w: int) -> None:
        """Process individual frames within a batch."""
        stride = max(1, self.neighbor_stride if batch_start + self.batch_size * self.neighbor_stride < len(frames) else 1)
        
        for frame_idx in range(batch_start, batch_end, stride):
            neighbor_ids = self._get_neighbor_ids(frame_idx, batch_start, batch_end)
            ref_ids = self.get_ref_index(frame_idx, neighbor_ids, batch_end)
            
            if not neighbor_ids:
                continue
                
            # Convert to batch-relative indices
            batch_neighbor_ids = [i - batch_start for i in neighbor_ids]
            batch_ref_ids = [i - batch_start for i in ref_ids if batch_start <= i < batch_end]
            
            # Process frame with temporal context
            self._process_single_frame(frames, batch_data, neighbor_ids, batch_neighbor_ids, 
                                     batch_ref_ids, comp_frames, processed_frame_mask, h, w)
            
            self._clear_gpu_memory()

    def _get_neighbor_ids(self, frame_idx: int, batch_start: int, batch_end: int) -> List[int]:
        """Get neighboring frame indices for temporal context."""
        return list(range(
            max(batch_start, frame_idx - self.neighbor_stride), 
            min(batch_end, frame_idx + self.neighbor_stride + 1)
        ))

    def _process_single_frame(self, frames: List[Image.Image], batch_data: dict, 
                            neighbor_ids: List[int], batch_neighbor_ids: List[int], 
                            batch_ref_ids: List[int], comp_frames: List[Optional[np.ndarray]], 
                            processed_frame_mask: List[bool], h: int, w: int) -> None:
        """Process a single frame with its temporal context."""
        batch_start = neighbor_ids[0] - batch_neighbor_ids[0]
        
        # Select relevant frames and masks
        selected_imgs = batch_data['batch_imgs'][:, batch_neighbor_ids + batch_ref_ids, :, :, :]
        selected_masks = batch_data['batch_masks'][:, batch_neighbor_ids + batch_ref_ids, :, :]
        
        with torch.no_grad():
            # Apply masks and generate inpainted frames
            masked_imgs = selected_imgs * (1 - selected_masks)
            masked_imgs = self._pad_images(masked_imgs, h, w)
            
            pred_imgs, _ = self.model(masked_imgs, len(batch_neighbor_ids))
            pred_imgs = (pred_imgs[:, :, :h, :w] + 1) / 2
            pred_imgs = (pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            
            # Composite with original background
            for i, idx in enumerate(neighbor_ids):
                binary_mask = batch_data['binary_masks'][idx - batch_start]
                original_frame = np.array(frames[idx])
                
                inpainted_frame = (pred_imgs[i] * binary_mask + 
                                 original_frame * (1 - binary_mask))
                
                # Average with previous results if frame was already processed
                if comp_frames[idx] is None:
                    comp_frames[idx] = inpainted_frame
                else:
                    comp_frames[idx] = ((comp_frames[idx].astype(np.float32) + 
                                       inpainted_frame.astype(np.float32)) / 2).astype(np.uint8)
                processed_frame_mask[idx] = True

    def _process_missed_frames(self, frames: List[Image.Image], paths: Any, 
                             comp_frames: List[Optional[np.ndarray]], 
                             processed_frame_mask: List[bool]) -> None:
        """Process any frames that were missed during batch processing."""
        unprocessed_frames = [i for i, processed in enumerate(processed_frame_mask) if not processed]
        
        if not unprocessed_frames:
            return
            
        logger.warning(f"Found {len(unprocessed_frames)} unprocessed frames at indices: {unprocessed_frames}")
        
        # Determine processing context for missed frames
        start_idx, end_idx = self._get_missed_frame_context(unprocessed_frames, processed_frame_mask, len(frames))
        
        logger.info(f"Processing missed frames from {start_idx} to {end_idx}")
        self._process_missed_frame_sequence(frames, paths, unprocessed_frames, 
                                          start_idx, end_idx, comp_frames, processed_frame_mask)

    def _get_missed_frame_context(self, unprocessed_frames: List[int], 
                                processed_frame_mask: List[bool], video_length: int) -> Tuple[int, int]:
        """Get the context range for processing missed frames."""
        last_processed_idx = max([i for i, processed in enumerate(processed_frame_mask[:unprocessed_frames[0]]) 
                                if processed], default=-1)
        if last_processed_idx == -1:
            last_processed_idx = 0
        
        next_processed_idx = min([i for i, processed in enumerate(processed_frame_mask[unprocessed_frames[-1]:], 
                                 start=unprocessed_frames[-1]) if processed], default=video_length)
        
        start_idx = max(0, last_processed_idx - self.neighbor_stride)
        end_idx = min(video_length, next_processed_idx + self.neighbor_stride)
        
        return start_idx, end_idx

    def _process_missed_frame_sequence(self, frames: List[Image.Image], paths: Any, 
                                     unprocessed_frames: List[int], start_idx: int, end_idx: int,
                                     comp_frames: List[Optional[np.ndarray]], 
                                     processed_frame_mask: List[bool]) -> None:
        """Process the sequence containing missed frames."""
        h, w = frames[0].height, frames[0].width
        
        # Prepare sequence data
        batch_frames = frames[start_idx:end_idx]
        batch_imgs = to_tensors()(batch_frames).unsqueeze(0).to(self.device) * 2 - 1
        
        batch_masks = self.read_mask(paths.masks_arm, (w, h))[start_idx:end_idx]
        batch_masks = to_tensors()(batch_masks).unsqueeze(0).to(self.device)
        
        binary_masks = self._create_binary_masks(paths.masks_arm, start_idx, end_idx, w, h)
        
        # Process each missed frame
        for idx in tqdm(unprocessed_frames, desc="Processing missed frames"):
            self._process_missed_single_frame(frames, batch_imgs, batch_masks, binary_masks,
                                           idx, start_idx, end_idx, comp_frames, processed_frame_mask, h, w)
        
        del batch_imgs, batch_masks
        self._clear_gpu_memory()

    def _process_missed_single_frame(self, frames: List[Image.Image], batch_imgs: torch.Tensor,
                                   batch_masks: torch.Tensor, binary_masks: List[np.ndarray],
                                   frame_idx: int, start_idx: int, end_idx: int,
                                   comp_frames: List[Optional[np.ndarray]], 
                                   processed_frame_mask: List[bool], h: int, w: int) -> None:
        """Process a single missed frame."""
        relative_start = frame_idx - start_idx
        neighbor_ids = list(range(
            max(0, relative_start - self.neighbor_stride),
            min(end_idx - start_idx, relative_start + self.neighbor_stride + 1)
        ))
        ref_ids = self.get_ref_index(relative_start, neighbor_ids, end_idx - start_idx)
        
        with torch.no_grad():
            selected_imgs = batch_imgs[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = batch_masks[:, neighbor_ids + ref_ids, :, :]
            
            masked_imgs = selected_imgs * (1 - selected_masks)
            masked_imgs = self._pad_images(masked_imgs, h, w)
            
            pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
            pred_imgs = (pred_imgs[:, :, :h, :w] + 1) / 2
            pred_imgs = (pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            
            relative_idx = frame_idx - start_idx - neighbor_ids[0]
            binary_mask = binary_masks[frame_idx - start_idx]
            original_frame = np.array(frames[frame_idx])
            
            inpainted_frame = (pred_imgs[relative_idx] * binary_mask + 
                             original_frame * (1 - binary_mask))
            comp_frames[frame_idx] = inpainted_frame
            processed_frame_mask[frame_idx] = True

    def _verify_and_save_results(self, comp_frames: List[Optional[np.ndarray]], paths: Any) -> None:
        """Verify all frames were processed and save the final video."""
        missing_frames = [i for i, frame in enumerate(comp_frames) 
                         if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0)]
        
        if missing_frames:
            raise RuntimeError(f"Still found unprocessed frames after cleanup: {missing_frames}")
            
        logger.info("Successfully processed all frames")
        
        # Save final inpainted video
        media.write_video(paths.video_human_inpaint, comp_frames, fps=15, codec="ffv1")

    def get_ref_index(self, f: int, neighbor_ids: List[int], length: int) -> List[int]:
        """
        Select reference frame indices for temporal consistency.
        
        Args:
            f: Current frame index
            neighbor_ids: List of neighboring frame indices
            length: Total length of the sequence
            
        Returns:
            List of reference frame indices for temporal consistency
        """
        if self.num_ref == -1:
            # Automatic reference selection: every ref_length frames not in neighbors
            ref_index = [
                i for i in range(0, length, self.ref_length)
                if i not in neighbor_ids
            ]
        else:
            # Limited reference selection: specific number around current frame
            ref_index = []
            for i in range(max(0, f - self.ref_length * (self.num_ref // 2)),
                          min(length, f + self.ref_length * (self.num_ref // 2)) + 1,
                          self.ref_length):
                if i not in neighbor_ids and len(ref_index) < self.num_ref:
                    ref_index.append(i)
        return ref_index

    @staticmethod
    def read_mask(mask_path: str, size: Tuple[int, int]) -> List[Image.Image]:
        """
        Load and process hand segmentation masks for inpainting guidance.
        
        Args:
            mask_path: Path to mask file containing hand segmentation data
            size: Target size (width, height) for mask resizing
            
        Returns:
            List of processed PIL Images containing binary hand masks
        """
        masks = []
        frames_media = np.load(mask_path, allow_pickle=True)
        frames = [frame for frame in frames_media]
        
        for mask_frame in frames:
            # Convert to PIL Image and resize
            mask_img = Image.fromarray(mask_frame)
            mask_img = mask_img.resize(size, Image.NEAREST)
            mask_array = np.array(mask_img.convert('L'))
            
            # Create binary mask
            binary_mask = np.array(mask_array > 0).astype(np.uint8)
            
            # Apply morphological dilation to expand mask boundaries
            # This helps ensure complete coverage of hand regions
            dilated_mask = cv2.dilate(binary_mask,
                                    cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                                    iterations=4)
            masks.append(Image.fromarray(dilated_mask * 255))
        return masks

    @staticmethod
    def read_frame_from_videos(video_path: str) -> List[Image.Image]:
        """
        Load video frames and convert to PIL Images.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of PIL Images containing video frames
        """
        return [Image.fromarray(frame) for frame in media.read_video(video_path)]

    @staticmethod
    def resize_frames(frames: List[Image.Image], size: Optional[Tuple[int, int]] = None) -> Tuple[List[Image.Image], Tuple[int, int]]:
        """
        Resize video frames to target resolution.
        
        Args:
            frames: List of PIL Images to resize
            size: Target size (width, height), or None to keep original
            
        Returns:
            Tuple containing resized frames and final size
        """
        return ([f.resize(size) for f in frames], size)

    @staticmethod
    def _pad_images(img_tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Pad image tensor to meet model input requirements.
        
        Args:
            img_tensor: Input image tensor to pad
            h: Original height
            w: Original width
            
        Returns:
            Padded image tensor suitable for model input
        """
        # Model requires specific dimension multiples
        mod_size_h, mod_size_w = 60, 108
        
        # Calculate required padding
        h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
        
        # Apply reflection padding to avoid boundary artifacts
        img_tensor = torch.cat([img_tensor, torch.flip(img_tensor, [3])], 3)[:, :, :, :h + h_pad, :]
        return torch.cat([img_tensor, torch.flip(img_tensor, [4])], 4)[:, :, :, :, :w + w_pad]

