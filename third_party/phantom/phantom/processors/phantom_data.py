from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np

hand_side_dict = {
    'left': 0,
    'right': 1,
}

class LazyLoadingMixin:
    """Mixin to provide lazy loading functionality for cached properties."""
    
    def _invalidate_cache(self) -> None:
        """Invalidate all cached properties. Override in subclasses."""
        pass
    
    def _get_cached_property(self, cache_attr: str, compute_func: Callable[[], Any]) -> Any:
        """Generic lazy loading for cached properties."""
        if getattr(self, cache_attr) is None:
            setattr(self, cache_attr, compute_func())
        return getattr(self, cache_attr)

@dataclass
class TrainingData:
    """Container for processing results"""
    frame_idx: int
    valid: bool
    action_pos_left: np.ndarray
    action_orixyzw_left: np.ndarray
    action_pos_right: np.ndarray
    action_orixyzw_right: np.ndarray
    action_gripper_left: np.ndarray
    action_gripper_right: np.ndarray
    gripper_width_left: np.ndarray
    gripper_width_right: np.ndarray

    @classmethod
    def create_empty_frame(cls, frame_idx: int) -> 'TrainingData':
        """Create a frame with no hand detection"""
        return cls(
            frame_idx=frame_idx,
            valid=False,
            action_pos_left=np.zeros((3,)),
            action_orixyzw_left=np.zeros((4,)),
            action_pos_right=np.zeros((3,)),
            action_orixyzw_right=np.zeros((4,)),
            action_gripper_left=0,
            action_gripper_right=0,
            gripper_width_left=0,
            gripper_width_right=0,
        )

class TrainingDataSequence(LazyLoadingMixin):
    """Container for a sequence of training data"""
    def __init__(self):
        self.frames: List[TrainingData] = []
        self.metadata: Dict = {}

        self._frame_indices: Optional[np.ndarray] = None
        self._valid: Optional[np.ndarray] = None
        self._action_pos_left: Optional[np.ndarray] = None
        self._action_orixyzw_left: Optional[np.ndarray] = None
        self._action_pos_right: Optional[np.ndarray] = None
        self._action_orixyzw_right: Optional[np.ndarray] = None
        self._action_gripper_left: Optional[np.ndarray] = None
        self._action_gripper_right: Optional[np.ndarray] = None
        self._gripper_width_left: Optional[np.ndarray] = None
        self._gripper_width_right: Optional[np.ndarray] = None
    
    def add_frame(self, frame: TrainingData) -> None:
        """Add a frame to the sequence and invalidate cached properties."""
        self.frames.append(frame)
        self._invalidate_cache()

    def save(self, path: str) -> None:
        """Save the sequence to disk in both frame-wise and sequence-wise formats"""
        
        sequence_data = {
            'frame_indices': self.frame_indices,
            'valid': self.valid,
            'action_pos_left': self.action_pos_left,
            'action_orixyzw_left': self.action_orixyzw_left,
            'action_pos_right': self.action_pos_right,
            'action_orixyzw_right': self.action_orixyzw_right,
            'action_gripper_left': self.action_gripper_left,
            'action_gripper_right': self.action_gripper_right,
            'gripper_width_left': self.gripper_width_left,
            'gripper_width_right': self.gripper_width_right,
        }
        
        np.savez_compressed(
            path,
            **sequence_data
        )

    @property
    def frame_indices(self) -> np.ndarray:
        """Lazy loading of all frame indices"""
        return self._get_cached_property(
            '_frame_indices',
            lambda: np.arange(len(self.frames))
        )
    
    @property
    def valid(self) -> np.ndarray:
        """Lazy loading of all valid flags"""
        return self._get_cached_property(
            '_valid',
            lambda: np.stack([f.valid for f in self.frames])
        )
    
    @property
    def action_pos_left(self) -> np.ndarray:
        """Lazy loading of all action positions"""
        return self._get_cached_property(
            '_action_pos_left',
            lambda: np.stack([f.action_pos_left for f in self.frames])
        )
    
    @property
    def action_orixyzw_left(self) -> np.ndarray:
        """Lazy loading of all action orientations"""
        return self._get_cached_property(
            '_action_orixyzw_left',
            lambda: np.stack([f.action_orixyzw_left for f in self.frames])
        )
    
    @property
    def action_pos_right(self) -> np.ndarray:
        """Lazy loading of all action positions"""
        return self._get_cached_property(
            '_action_pos_right',
            lambda: np.stack([f.action_pos_right for f in self.frames])
        )
    
    @property
    def action_orixyzw_right(self) -> np.ndarray:
        """Lazy loading of all action orientations"""
        return self._get_cached_property(
            '_action_orixyzw_right',
            lambda: np.stack([f.action_orixyzw_right for f in self.frames])
        )
    
    @property
    def action_gripper_left(self) -> np.ndarray:
        """Lazy loading of all action gripper distances"""
        return self._get_cached_property(
            '_action_gripper_left',
            lambda: np.stack([f.action_gripper_left for f in self.frames])
        )
    
    @property
    def action_gripper_right(self) -> np.ndarray:
        """Lazy loading of all action gripper distances"""
        return self._get_cached_property(
            '_action_gripper_right',
            lambda: np.stack([f.action_gripper_right for f in self.frames])
        )
    
    @property
    def gripper_width_left(self) -> np.ndarray:
        """Lazy loading of all gripper widths"""
        return self._get_cached_property(
            '_gripper_width_left',
            lambda: np.stack([f.gripper_width_left for f in self.frames])
        )
    
    @property
    def gripper_width_right(self) -> np.ndarray:
        """Lazy loading of all gripper widths"""
        return self._get_cached_property(
            '_gripper_width_right',
            lambda: np.stack([f.gripper_width_right for f in self.frames])
        )
    
    def _invalidate_cache(self):
        """Invalidate all cached properties."""
        self._frame_indices = None
        self._valid = None
        self._action_pos_left = None
        self._action_orixyzw_left = None
        self._action_pos_right = None
        self._action_orixyzw_right = None
        self._action_gripper_left = None
        self._action_gripper_right = None
        self._gripper_width_left = None
        self._gripper_width_right = None

    @classmethod
    def load(cls, path: str) -> 'TrainingDataSequence':
        """Load a sequence from disk"""
        data = np.load(path, allow_pickle=True)
        sequence = cls()

        sequence._frame_indices = data['frame_indices']
        sequence._valid = data['valid']
        sequence._action_pos_left = data['action_pos_left']
        sequence._action_orixyzw_left = data['action_orixyzw_left']
        sequence._action_pos_right = data['action_pos_right']
        sequence._action_orixyzw_right = data['action_orixyzw_right']
        sequence._action_gripper_left = data['action_gripper_left']
        sequence._action_gripper_right = data['action_gripper_right']
        sequence._gripper_width_left = data['gripper_width_left']
        sequence._gripper_width_right = data['gripper_width_right']

        return sequence

@dataclass
class HandFrame:
    """Data structure for a single frame of hand data"""
    frame_idx: int
    hand_detected: bool
    img_rgb: np.ndarray
    img_hamer: np.ndarray
    kpts_2d: np.ndarray  # shape: (N, 2)
    kpts_3d: np.ndarray  # shape: (N, 3)

    @classmethod
    def create_empty_frame(cls, frame_idx: int, img_rgb: np.ndarray) -> 'HandFrame':
        """Create a frame with no hand detection"""
        return cls(
            frame_idx=frame_idx,
            hand_detected=False,
            img_rgb=img_rgb,
            img_hamer=np.zeros_like(img_rgb),
            kpts_2d=np.zeros((21, 2)),
            kpts_3d=np.zeros((21, 3)),
        )

class HandSequence(LazyLoadingMixin):
    """Container for a sequence of hand data"""
    def __init__(self):
        self.frames: List[HandFrame] = []
        self.metadata: Dict = {}

        self._frame_indices: Optional[np.ndarray] = None
        self._hand_detected: Optional[np.ndarray] = None
        self._img_rgb: Optional[np.ndarray] = None
        self._img_hamer: Optional[np.ndarray] = None
        self._kpts_2d: Optional[np.ndarray] = None
        self._kpts_3d: Optional[np.ndarray] = None
    
    def add_frame(self, frame: HandFrame) -> None:
        """Add a frame to the sequence and invalidate cached properties."""
        self.frames.append(frame)
        self._invalidate_cache()
    
    def get_frame(self, frame_idx: int) -> HandFrame:
        """Get a frame by index."""
        return self.frames[frame_idx]

    def modify_frame(self, frame_idx: int, frame: HandFrame) -> None:
        """Modify a frame at the given index and invalidate cached properties."""
        self.frames[frame_idx] = frame
        self._invalidate_cache()

    def save(self, path: str) -> None:
        """Save the sequence to disk in both frame-wise and sequence-wise formats"""
        sequence_data = {
            'hand_detected': self.hand_detected,
            'kpts_2d': self.kpts_2d,
            'kpts_3d': self.kpts_3d,
            'frame_indices': self.frame_indices,
        }
        
        np.savez_compressed(
            path,
            **sequence_data
        )

    @property
    def frame_indices(self) -> np.ndarray:
        """Lazy loading of all frame indices"""
        return self._get_cached_property(
            '_frame_indices',
            lambda: np.arange(len(self.frames))
        )

    @property
    def hand_detected(self) -> np.ndarray:
        """Lazy loading of all hand detection flags"""
        return self._get_cached_property(
            '_hand_detected',
            lambda: np.stack([f.hand_detected for f in self.frames])
        )
    
    @property
    def imgs_rgb(self) -> np.ndarray:
        """Lazy loading of all RGB images"""
        return self._get_cached_property(
            '_img_rgb',
            lambda: np.stack([f.img_rgb for f in self.frames])
        )
    
    @property
    def imgs_hamer(self) -> np.ndarray:
        """Lazy loading of all HAMER images"""
        return self._get_cached_property(
            '_img_hamer',
            lambda: np.stack([f.img_hamer for f in self.frames])
        )
    
    @property
    def kpts_2d(self) -> np.ndarray:
        """Lazy loading of all 2D keypoints"""
        return self._get_cached_property(
            '_kpts_2d',
            lambda: np.stack([f.kpts_2d for f in self.frames])
        )
    
    @property
    def kpts_3d(self) -> np.ndarray:
        """Lazy loading of all 3D keypoints"""
        return self._get_cached_property(
            '_kpts_3d',
            lambda: np.stack([f.kpts_3d for f in self.frames])
        )
    
    @classmethod
    def load(cls, path: str) -> 'HandSequence':
        """Load a sequence from disk"""
        data = np.load(path, allow_pickle=True)
        sequence = cls()
        
        # Load pre-computed sequence-wise data
        sequence._frame_indices = data['frame_indices']
        sequence._hand_detected = data['hand_detected']
        sequence._kpts_2d = data['kpts_2d']
        sequence._kpts_3d = data['kpts_3d']
    
        return sequence

    def _invalidate_cache(self):
        """Invalidate all cached properties."""
        self._frame_indices = None
        self._hand_detected = None
        self._img_rgb = None
        self._img_hamer = None
        self._kpts_2d = None
        self._kpts_3d = None