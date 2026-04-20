"""
Wrapper around HaMeR for hand pose estimation
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import cv2
import torch
from hamer.utils import recursive_to  # type: ignore
import matplotlib.pyplot as plt

from hamer.models import HAMER, DEFAULT_CHECKPOINT  # type: ignore
import sys
import os
# Add the phantom-hamer directory to Python path for vitpose_model import
hamer_path = os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'phantom-hamer')
if hamer_path not in sys.path:
    sys.path.insert(0, hamer_path)
from vitpose_model import ViTPoseModel  # type: ignore
from hamer.datasets.vitdet_dataset import ViTDetDataset  # type: ignore
from hamer.utils.renderer import cam_crop_to_full  # type: ignore
from hamer.utils.geometry import perspective_projection  # type: ignore
from hamer.configs import get_config  # type: ignore
from yacs.config import CfgNode as CN  # type: ignore

from phantom.utils.data_utils import get_parent_folder_of_package

logger = logging.getLogger(__name__)

THUMB_VERTEX = 756
INDEX_FINGER_VERTEX = 350

class DetectorHamer:
    """
    Detector using the HaMeR model for 3D hand pose estimation.
    
    The detection pipeline consists of:
    - Initial hand detection using general object detectors
    - Hand type classification (left/right) using ViTPose
    - 3D pose estimation using HaMeR
    - MANO parameters estimation for mesh reconstruction
    
    Dependencies:
    - HaMeR model for 3D pose estimation
    - ViTPose for keypoint detection
    - DINO and Detectron2 for initial hand detection
    """
    def __init__(self):
        root_dir = get_parent_folder_of_package("hamer")
        checkpoint_path = Path(root_dir, DEFAULT_CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rescale_factor = 2.0 # Factor for padding the box
        self.batch_size = 1 # Batch size for inference

        self.model, self.model_cfg = self.load_hamer_model(checkpoint_path, root_dir)
        self.model.to(self.device)
        self.model.eval()

        root_dir = "../submodules/phantom-hamer/"
        vit_dir = os.path.join(root_dir, "third-party/ViTPose/")
        self.cpm = ViTPoseModel(device=self.device, root_dir=root_dir, vit_dir=vit_dir)

        self.faces_right = self.model.mano.faces
        self.faces_left = self.faces_right[:,[0,2,1]]

    def detect_hand_keypoints(self, 
                              img: np.ndarray,
                              hand_side: str,
                              visualize: bool=False, 
                              visualize_3d: bool=False, 
                              pause_visualization: bool=True, 
                              bboxes: Optional[np.ndarray]=None,
                              is_right: Optional[np.ndarray]=None,
                              kpts_2d_only: Optional[bool]=False,
                              camera_params: Optional[dict]=None) -> Optional[dict]:
        """
        Detect hand keypoints in the input image.
        
        The method performs the following steps:
        1. Detect hand bounding boxes using object detectors
        2. Optionally refine boxes using ViTPose to determine hand type (left/right)
        3. Run HaMeR model to estimate 3D hand pose
        4. Project 3D keypoints back to 2D for visualization
        
        Args:
            img: Input RGB image as numpy array
            hand_side: Target hand side to detect (left or right)
            visualize: If True, displays detection results in a window
            visualize_3d: If True, shows 3D visualization of keypoints and mesh
            pause_visualization: If True, waits for key press when visualizing
            bboxes: Bounding boxes of the hands
            is_right: Whether the hand is right
            kpts_2d_only: If True, only cares about 2D keypoints, i.e., use default 
            focal length in HaMeR instead of real camera intrinsics
            camera_params: Optional camera intrinsics (fx, fy, cx, cy)
            
        Returns:
            Dictionary containing:
                - annotated_img: Image with keypoints drawn
                - success: Whether detection was successful (21 keypoints found)
                - kpts_3d: 3D keypoints in camera space
                - kpts_2d: 2D keypoints projected onto image
                - verts: 3D mesh vertices
                - T_cam_pred: Camera transformation matrix
                - Various camera parameters and MANO pose parameters
        """
        if not kpts_2d_only:
            scaled_focal_length, camera_center = self.get_image_params(img, camera_params)
        else:
            scaled_focal_length, camera_center = self.get_image_params(img, camera_params=None)


        dataset = ViTDetDataset(self.model_cfg, img, bboxes, is_right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        list_2d_kpts, list_3d_kpts, list_verts = [], [], []
        T_cam_pred_all: list[torch.Tensor] = []
        list_global_orient = []
        kpts_2d_hamer = None
        for batch in dataloader:
            batch = recursive_to(batch, "cuda")
            with torch.no_grad():
                out = self.model(batch)

            batch_T_cam_pred_all = DetectorHamer.get_all_T_cam_pred(batch, out, scaled_focal_length)

            for idx in range(len(batch_T_cam_pred_all)):
                kpts_3d = out["pred_keypoints_3d"][idx].detach().cpu().numpy()  # [21, 3]
                verts = out["pred_vertices"][idx].detach().cpu().numpy()  # [778, 3]
                is_right = batch["right"][idx].cpu().numpy()
                global_orient = out["pred_mano_params"]["global_orient"][idx].detach().cpu().numpy()
                hand_pose = out["pred_mano_params"]["hand_pose"][idx].detach().cpu().numpy()
                list_global_orient.append(global_orient)

                if hand_side == "left":
                    kpts_3d, verts = DetectorHamer.convert_right_hand_keypoints_to_left_hand(kpts_3d, verts)

                T_cam_pred = batch_T_cam_pred_all[idx]

                img_w, img_h = batch["img_size"][idx].float()

                kpts_2d_hamer = DetectorHamer.project_3d_kpt_to_2d(kpts_3d, img_w, img_h, scaled_focal_length, 
                                                            camera_center, T_cam_pred)

                # Keep T_cam_pred as tensor
                list_2d_kpts.append(kpts_2d_hamer)
                list_3d_kpts.append(kpts_3d + T_cam_pred.cpu().numpy())
                list_verts.append(verts + T_cam_pred.cpu().numpy())

            T_cam_pred_all += batch_T_cam_pred_all

        annotated_img = DetectorHamer.visualize_2d_kpt_on_img(
            kpts_2d=list_2d_kpts[0],
            img=img,
        )

        if visualize:
            if bboxes is not None:
                cv2.rectangle(annotated_img, (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), (0, 255, 0), 2)
            cv2.imshow("Annotated Image", annotated_img)
            cv2.waitKey(0 if pause_visualization else 1)

        if visualize_3d:
            DetectorHamer.visualize_keypoints_3d(annotated_img, list_3d_kpts[0], list_verts[0])


        return {
            "annotated_img": annotated_img,
            "success": len(list_2d_kpts[0]) == 21,
            "kpts_3d": list_3d_kpts[0],
            "kpts_2d": np.rint(list_2d_kpts[0]).astype(np.int32),
            "verts": list_verts[0],
            "T_cam_pred": T_cam_pred_all[0],
            "scaled_focal_length": scaled_focal_length,
            "camera_center": camera_center,
            "img_w": img_w,
            "img_h": img_h,
            "global_orient": list_global_orient[0],
            "hand_pose": hand_pose,
        }
    
    def get_image_params(self, img: np.ndarray, camera_params: Optional[dict]) -> Tuple[float, torch.Tensor]:
        """
        Get the scaled focal length and camera center.
        """
        img_w = img.shape[1]
        img_h = img.shape[0]
        if camera_params is not None:
            scaled_focal_length = camera_params["fx"]
            cx = camera_params["cx"]
            cy = camera_params["cy"]
            camera_center = torch.tensor([img_w-cx, img_h-cy])
        else:
            scaled_focal_length = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE 
                                   * max(img_w, img_h))
            camera_center = torch.tensor([img_w, img_h], dtype=torch.float).reshape(1, 2) / 2.0
        return scaled_focal_length, camera_center
    
    @staticmethod
    def convert_right_hand_keypoints_to_left_hand(kpts, verts):
        """
        Convert right hand keypoints/vertices to left hand by mirroring across the Y-Z plane.
        
        This is done by flipping the X coordinates of both keypoints and vertices.
        The MANO model internally uses right hand, so this conversion is needed
        when processing left hands.
        
        Args:
            kpts: 3D keypoints [21, 3]
            verts: 3D mesh vertices [778, 3]
            
        Returns:
            Transformed keypoints and vertices
        """
        kpts[:,0] = -kpts[:,0]
        verts[:,0] = -verts[:,0]
        return kpts, verts

    @staticmethod
    def visualize_keypoints_3d(annotated_img: np.ndarray, kpts_3d: np.ndarray, verts: np.ndarray) -> None:
        nfingers = len(kpts_3d) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([kpts_3d[0], kpts_3d[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
        finger_colors_bgr = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        finger_colors_rgb = [(color[2], color[1], color[0]) for color in finger_colors_bgr]
        fig, axs = plt.subplots(1,2, figsize=(20, 10))
        axs[0] = fig.add_subplot(111, projection='3d')
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors_rgb[finger_idx]
                axs[0].plot(
                    [finger_pts[i][0], finger_pts[i + 1][0]],
                    [finger_pts[i][1], finger_pts[i + 1][1]],
                    [finger_pts[i][2], finger_pts[i + 1][2]],
                    color=np.array(color)/255.0,
                )
        axs[0].scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2])
        axs[0].scatter(verts[:, 0], verts[:, 1], verts[:, 2])
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axs[1].imshow(annotated_img_rgb)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(annotated_img_rgb)

        plt.show()

    @staticmethod
    def get_all_T_cam_pred(batch: dict, out: dict, scaled_focal_length: float) -> torch.Tensor:
        """
        Get the camera transformation matrix
        """
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        # NOTE: FOR HaMeR, they are using the img_size as (W, H)
        W_H_shapes = batch["img_size"].float() 

        multiplier = 2 * batch["right"] - 1
        T_cam_pred_all = cam_crop_to_full(
            pred_cam, box_center, box_size, W_H_shapes, scaled_focal_length
        )

        return T_cam_pred_all

    @staticmethod
    def visualize_2d_kpt_on_img(kpts_2d: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Plot 2D hand keypoints on the image with finger connections.
        
        Each finger is drawn with a different color:
        - Thumb: Green
        - Index: Blue
        - Middle: Red
        - Ring: Magenta
        - Pinky: Cyan
        
        Args:
            kpts_2d: 2D keypoints as integers [21, 2]
            img: Input RGB image
            
        Returns:
            Image with keypoints and connections drawn (BGR format)
        """
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pts = kpts_2d.astype(np.int32)
        nfingers = len(pts) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([pts[0], pts[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
        finger_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        thickness = 5 if img_bgr.shape[0] > 1000 else 2
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors[finger_idx]
                cv2.line(
                    img_bgr,
                    tuple(finger_pts[i]),
                    tuple(finger_pts[i + 1]),
                    color,
                    thickness=thickness,
                )

        cv2.line(img_bgr, [1787, 1522], [1656,1400], (255,0,0), thickness=thickness)

        for pt in pts:
            cv2.circle(img_bgr, (pt[0], pt[1]), radius=thickness, color=(0,0,0), thickness=thickness-1)

        return img_bgr
    

    @staticmethod
    def project_3d_kpt_to_2d(kpts_3d: torch.Tensor, img_w: int, img_h: int, scaled_focal_length: float,
                                camera_center: torch.Tensor, T_cam: Optional[torch.Tensor] = None,) -> np.ndarray:
        """
        Project 3D keypoints to 2D image coordinates using perspective projection.
        """
        batch_size = 1

        rotation = torch.eye(3).unsqueeze(0)
        assert T_cam is not None

        T_cam = T_cam.cpu()
        kpts_3d = torch.tensor(kpts_3d).cpu()

        T_cam = T_cam.clone().cuda()
        kpts_3d = kpts_3d.clone().cuda()
        rotation = rotation.cuda()

        scaled_focal_length_full = torch.tensor([scaled_focal_length, scaled_focal_length]).reshape(1, 2)

        # IMPORTANT: The perspective_projection function assumes T_cam has not been added to kpts_3d already!
        kpts_2d = perspective_projection(
            kpts_3d.reshape(batch_size, -1, 3),
            rotation=rotation.repeat(batch_size, 1, 1),
            translation=T_cam.reshape(batch_size, -1),
            focal_length=scaled_focal_length_full.repeat(batch_size, 1),
            camera_center=camera_center.repeat(batch_size, 1),
            ).reshape(batch_size, -1, 2)
        kpts_2d = kpts_2d[0].cpu().numpy()

        return np.rint(kpts_2d).astype(np.int32)

    @staticmethod
    def annotate_bboxes_on_img(img: np.ndarray, debug_bboxes: dict) -> np.ndarray:
        """
        Annotate bounding boxes on the image.

        :param img: Input image (numpy array)
        :param debug_bboxes: Dictionary containing different sets of bounding boxes and optional scores
        :return: Annotated image
        """
        color_dict = {
            "dino_bboxes": (0, 255, 0),
            "det_bboxes": (0, 0, 255),
            "refined_bboxes": (255, 0, 0),
            "filtered_bboxes": (255, 255, 0),
        }
        corner_dict = {
            "dino_bboxes": "top_left",
            "det_bboxes": "top_right",
            "refined_bboxes": "bottom_left",
            "filtered_bboxes": "bottom_right",
        }
        
        def draw_bbox_and_label(bbox, label, color, label_pos, include_label=True):
            """ Helper function to draw the bounding box and add label """
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            if include_label:
                cv2.putText(
                    img, label, label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
                )

        label_pos_dict = {
            "top_left": lambda bbox: (int(bbox[0]), int(bbox[1]) - 10),
            "bottom_right": lambda bbox: (int(bbox[2]) - 150, int(bbox[3]) - 10),
            "top_right": lambda bbox: (int(bbox[2]) - 150, int(bbox[1]) - 10),
            "bottom_left": lambda bbox: (int(bbox[0]), int(bbox[3]) - 10),
        }

        for key, value in debug_bboxes.items():
            # Unpack bboxes and scores
            if key in ["dino_bboxes", "det_bboxes"]:
                bboxes, scores = value
            else:
                bboxes = value
                scores = [None] * len(bboxes)  

            color = color_dict.get(key, (0, 0, 0)) 
            label_pos_fn = label_pos_dict[corner_dict.get(key, "top_left")]

            # Draw each bounding box and its label
            for idx, bbox in enumerate(bboxes):
                score_text = f" {scores[idx]:.3f}" if scores[idx] is not None else ""
                label = key.split("_")[0] + score_text

                # Draw bounding box and label on the image
                label_pos = label_pos_fn(bbox)
                if key in ["dino_bboxes", "det_bboxes"] or idx == 0:
                    draw_bbox_and_label(bbox, label, color, label_pos)
        return img


    @staticmethod
    def load_hamer_model(checkpoint_path: str, root_dir: Optional[str] = None) -> Tuple[HAMER, CN]:
        """
        Load the HaMeR model from the checkpoint path.
        """
        model_cfg_path = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
        model_cfg = get_config(model_cfg_path, update_cachedir=True)
        # update model and params path
        if root_dir:
            model_cfg.defrost()
            model_cfg.MANO.DATA_DIR = os.path.join(root_dir, model_cfg.MANO.DATA_DIR)
            model_cfg.MANO.MODEL_PATH = os.path.join(root_dir, model_cfg.MANO.MODEL_PATH.replace("./", ""))
            model_cfg.MANO.MEAN_PARAMS = os.path.join(root_dir, model_cfg.MANO.MEAN_PARAMS.replace("./", ""))
            model_cfg.freeze()

        # Override some config values, to crop bbox correctly
        if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
            model_cfg.defrost()
            assert (
                model_cfg.MODEL.IMAGE_SIZE == 256
            ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()

        # Update config to be compatible with demo
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.freeze()

        model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
        return model, model_cfg
