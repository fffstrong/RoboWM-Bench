"""
Wrapper around DINO-V2 for object detection
"""
from typing import Sequence, Tuple, Optional
import numpy as np
from transformers import pipeline  # type: ignore
from PIL import Image
import cv2
import logging

from phantom.utils.image_utils import DetectionResult

logger = logging.getLogger(__name__)

class DetectorDino:
    def __init__(self, detector_id: str):
        self.detector = pipeline(
            model=detector_id,
            task="zero-shot-object-detection",
            device="cuda",
            batch_size=4,
        )

    def get_bboxes(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
                   visualize: bool = False, pause_visualization: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect objects in a frame and return their bounding boxes and confidence scores.
        
        Args:
            frame: Input image as numpy array in RGB format
            object_name: Target object category to detect
            threshold: Confidence threshold for detection (0.0-1.0)
            visualize: If True, displays detection results visually
            pause_visualization: If True, waits for key press when visualizing
            
        Returns:
            Tuple of (bounding_boxes, confidence_scores) as numpy arrays
            Empty arrays if no objects detected
        """
        img_pil = Image.fromarray(frame)
        labels = [f"{object_name}."]
        results = self.detector(img_pil, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]
        if not results:
            return np.array([]), np.array([])
        bboxes = np.array([np.array(result.box.xyxy) for result in results])
        scores = np.array([result.score for result in results])

        if visualize:
            img_rgb = frame.copy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            for bbox, score in zip(bboxes, scores):
                cv2.rectangle(
                    img_bgr,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(img_bgr,
                            f"{score:.4f}",
                            (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)
            cv2.imshow("Detection", img_bgr)
            if pause_visualization:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        return bboxes, scores


    def get_best_bbox(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
               visualize: bool = False, pause_visualization: bool = True) -> Optional[np.ndarray]:
        bboxes, scores = self.get_bboxes(frame, object_name, threshold)
        if len(bboxes) == 0:
            return None
        best_idx = np.array(scores).argmax()
        best_bbox, best_score = bboxes[best_idx], scores[best_idx]

        if visualize:
            img_rgb = frame.copy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                img_bgr,
                (best_bbox[0], best_bbox[1]),
                (best_bbox[2], best_bbox[3]),
                (0, 255, 0),
                2,
            )
            cv2.putText(img_bgr,
                    f"{best_score:.4f}",
                    (int(best_bbox[0]), int(best_bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
            cv2.imshow("Detection", img_bgr)
            if pause_visualization:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        return best_bbox
    