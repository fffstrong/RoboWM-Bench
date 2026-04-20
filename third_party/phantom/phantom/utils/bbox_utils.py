import numpy as np
import numpy.typing as npt

def get_bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Calculate center point of bounding box."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def get_bbox_area(bbox: np.ndarray) -> float:
    """Get the area of a bounding box."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_overlap_score(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """ Get the overlap area between two boxes divided by the area of the smaller box """
    area1 = get_bbox_area(bbox1)
    area2 = get_bbox_area(bbox2)
    overlap_area = get_overlap_area(bbox1, bbox2)
    return overlap_area / min(area1, area2)

def get_overlap_area(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """ Get the overlap area between two boxes """
    return max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])) * max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))

def get_bbox_center_min_dist_to_edge(bboxes: npt.NDArray[np.float32], W: int, H: int) -> npt.NDArray[np.float32]:
    """
    Get the minimum distance of the bbox center to the edge of the image.
    """
    center_min_dist_to_edge_list = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        min_dist_to_edge = min(center[0], center[1], W - center[0], H - center[1])
        center_min_dist_to_edge_list.append(min_dist_to_edge)
    return np.array(center_min_dist_to_edge_list)



