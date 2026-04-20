import re
import os
import numpy as np
import pandas as pd
from pathlib import Path

def get_finger_poses_from_pkl(path: Path) -> dict:
    """Get human finger poses from pkl file."""
    finger_poses = pd.read_pickle(path)
    thumb_poses = np.vstack(finger_poses["thumb"])
    index_poses = np.vstack(finger_poses["index"])
    hand_ee_poses = np.vstack(finger_poses["hand_ee"])
    skeleton_poses = np.stack(finger_poses["skeleton"], axis=0)
    hand_poses = np.stack(finger_poses["hand_pose"], axis=0)
    all_global_orient = np.vstack(finger_poses["global_orient"])
    data = {
        "thumb": thumb_poses,
        "index": index_poses,
        "hand_ee": hand_ee_poses,
        "skeleton": skeleton_poses,
        "hand_pose": hand_poses,
        "global_orient": all_global_orient
    }
    return data

def get_parent_folder_of_package(package_name: str) -> str:
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = package.__file__
    if package_path is None:
        raise ValueError(f"Package {package_name} does not have a valid __file__ attribute")
    package_path = os.path.abspath(package_path)

    # Get the parent directory of the package directory
    return os.path.dirname(os.path.dirname(package_path))

