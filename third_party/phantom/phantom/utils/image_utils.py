import json
import numpy as np
import cv2
import os
import mediapy as media
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )

def get_transformation_matrix_from_extrinsics(camera_extrinsics: List[Dict]) -> np.ndarray:
    """Get homogeneous transformation matrix from camera extrinsics."""
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)
    return T_cam2robot


def get_intrinsics_from_json(json_path: str) -> Tuple[np.ndarray, dict]:
    with open(json_path, "r") as f:
        camera_intrinsics = json.load(f)

    # Get camera matrix 
    fx = camera_intrinsics["left"]["fx"]
    fy = camera_intrinsics["left"]["fy"]
    cx = camera_intrinsics["left"]["cx"]
    cy = camera_intrinsics["left"]["cy"]
    v_fov = camera_intrinsics["left"]["v_fov"]
    intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    intrinsics_dict = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "v_fov": v_fov,
    }

    return intrinsics_matrix, intrinsics_dict

def resize_binary_image(image: np.ndarray, new_size: int) -> np.ndarray:
    max_value = np.max(image)

    # Resize the image
    resized_image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

    if max_value == 1:
        _, binary_image = cv2.threshold(resized_image, 0.5, 1, cv2.THRESH_BINARY)
    else:
        _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)

    return binary_image


def convert_video_to_images(video_path: str, save_folder: str, square=False, reverse=False):
    """Save each frame of video as an image in save_folder."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    imgs = np.array(media.read_video(str(video_path)))
    n_imgs = len(imgs)
    if reverse:
        imgs = imgs[::-1]
    for idx in range(n_imgs):
        img = imgs[idx]
        if square:
            delta = (img.shape[1] - img.shape[0]) // 2
            img = img[:, delta:-delta, :]
        media.write_image(f"{save_folder}/{idx:05d}.jpg", img)


