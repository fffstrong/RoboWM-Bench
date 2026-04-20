"""
Wrapper around SAM2 for object segmentation
"""
import numpy as np
import pdb
import os 
import logging
import requests
from typing import Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cv2
from PIL import Image
import torch
from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
from sam2.build_sam import build_sam2_video_predictor  # type: ignore

logger = logging.getLogger(__name__)

def download_sam2_ckpt(ckpt_path: str) -> None:
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    save_path = Path(ckpt_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"File downloaded successfully and saved to {save_path}")
    else:
        logger.info(f"Failed to download the file. Status code: {response.status_code}")

class DetectorSam2:
    """
    A detector that uses the SAM2 model for object segmentation in images and videos.
    """
    def __init__(self):
        checkpoint = "../submodules/sam2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        
        if not os.path.exists(checkpoint):
            download_sam2_ckpt(checkpoint)
        self.device = "cuda"
        
        self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)
    
    def segment_video(self, video_dir: Path, bbox: np.ndarray, points: np.ndarray, 
                      indices: int, reverse: bool=False, output_bboxes: Optional[np.ndarray]=None):
        """
        Segment an object across video frames using SAM2's video tracking capabilities.
        
        Parameters:
            video_dir: Directory containing video frames as image files
            bbox: Bounding box coordinates [x0, y0, x1, y1] for the object to track
            points: Point(s) on the object to track
            start_idx: Frame index to start tracking from
            
        Returns:
            video_segments: Dictionary mapping frame indices to segmentation masks
            list_annotated_imgs: Array of frames with the segmented object masked out
        """
        frame_names = os.listdir(video_dir)
        frame_names = sorted(frame_names)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.video_predictor.init_state(video_path=str(video_dir))
            self.video_predictor.reset_state(state)

            for point, idx in zip(points, indices):
                try: 
                    if bbox is None or np.all(bbox) == 0:
                        self.video_predictor.add_new_points_or_box(
                            state,
                            frame_idx=int(idx),
                            obj_id=0,
                            points=np.array(point),
                            labels=np.ones(len(point)),
                        )
                    else:
                        self.video_predictor.add_new_points_or_box(
                            state,
                            frame_idx=int(idx),
                            obj_id=0,
                            box=np.array(bbox),
                            points=np.array(point),
                            labels=np.ones(len(point)),
                        )
                except Exception as e:
                    print("Error in adding new points or box:", e)
                    pdb.set_trace()
 
            video_segments = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(state, reverse=reverse):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        frame_indices = list(video_segments.keys())
        frame_indices.sort()
        list_annotated_imgs = {}
        for out_frame_idx in frame_indices:
            img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
            img_arr = np.array(img)
            mask = video_segments[out_frame_idx][0]
            if output_bboxes is not None:
                # Crop the mask to the bounding box
                output_bbox = output_bboxes[out_frame_idx].astype(np.int32)
                if output_bbox.sum() > 0:
                    bbox_mask = np.zeros_like(mask)
                    bbox_mask = self._crop_mask_to_bbox(mask, output_bbox)
                    mask = mask * bbox_mask
            img_arr[mask[0]] = (0, 0, 0)
            list_annotated_imgs[out_frame_idx] = img_arr

        if output_bboxes is not None:
            for out_frame_idx in frame_indices:
                output_bbox = output_bboxes[out_frame_idx].astype(np.int32)
                mask = video_segments[out_frame_idx][0]
                mask_ori = mask.copy()
                if output_bbox.sum() > 0:
                    bbox_mask = np.zeros_like(mask)
                    bbox_mask = self._crop_mask_to_bbox(mask, output_bbox)
                    mask = mask * bbox_mask
                    video_segments[out_frame_idx] = {
                        0: mask
                    }
    
        # Fix gpu memory leak
        torch.cuda.empty_cache()

        return video_segments, list_annotated_imgs
    
    def _crop_mask_to_bbox(self, mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Crop a mask to a bounding box.
        """
        margin = 20
        bbox = bbox.astype(np.int32)
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - margin)
        x1 = min(mask.shape[2], x1 + margin)
        y0 = max(0, y0 - margin)
        y1 = min(mask.shape[1], y1 + margin)
        bbox_mask = np.zeros_like(mask)
        bbox_mask[:, y0:y1, x0:x1] = 1
        return bbox_mask

    def segment_video_from_mask(self, video_dir: str, mask: np.ndarray, frame_idx: int, reverse=False):
        """
        Propagate a segmentation mask through video frames (forward or backward).
        
        Parameters:
            video_dir: Directory containing video frames
            mask: Initial segmentation mask to propagate
            frame_idx: Frame index where the mask is defined
            reverse: If True, propagate backward in time; if False, propagate forward
            
        Returns:
            frame_indices: List of frame indices where masks were generated
            video_segments: Dictionary mapping frame indices to segmentation masks
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.video_predictor.init_state(video_path=video_dir)
            self.video_predictor.reset_state(state)

            self.video_predictor.add_new_mask(state, frame_idx, 0, mask)

            video_segments = {}
            mask_prob = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(state, reverse=reverse):
                mask_prob[out_frame_idx] = torch.mean(torch.sigmoid(out_mask_logits))
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        frame_indices = list(video_segments.keys())
        frame_indices.sort()
        return frame_indices, video_segments

    @staticmethod
    def show_mask(mask: np.ndarray, ax: Axes, random_color: bool=False, borders: bool = True) -> None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)


    @staticmethod
    def show_masks(image: np.ndarray, masks: np.ndarray, scores: np.ndarray, point_coords: Optional[np.ndarray]=None, 
                   box_coords: Optional[np.ndarray]=None, input_labels: Optional[np.ndarray]=None, borders: bool=True) -> None:
        n_masks = len(masks)
        fig, axs = plt.subplots(1, n_masks, figsize=(10*n_masks, 10))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            axs[i].imshow(image)
            DetectorSam2.show_mask(mask, axs[i], borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                DetectorSam2.show_points(point_coords, input_labels, axs[i])
            if box_coords is not None:
                DetectorSam2.show_box(box_coords, axs[i])
            if len(scores) > 1:
                axs[i].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            axs[i].axis('off')
        plt.show()

    @staticmethod
    def show_box(box: np.ndarray, ax: Axes) -> None:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


    @staticmethod
    def show_points(coords: np.ndarray, labels: np.ndarray, ax: Axes, marker_size: int=375) -> None:
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)   
