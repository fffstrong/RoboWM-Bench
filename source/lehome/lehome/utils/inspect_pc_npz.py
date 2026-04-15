import argparse
from pathlib import Path
import numpy as np
import sys


def load_npz_pointcloud(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(str(path))
    # support key='pointcloud' or directly the first array
    if "pointcloud" in data:
        arr = data["pointcloud"]
    else:
        # take the first array
        keys = list(data.keys())
        if len(keys) == 0:
            raise ValueError("npz file does not contain any arrays")
        arr = data[keys[0]]
    return arr


def to_xyz_rgb(arr: np.ndarray):
    # support (N,3) , (N,6) , or (N,>=3)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("pointcloud array should be (N,3+) shape")
    xyz = arr[:, :3].astype(np.float64)
    rgb = None
    if arr.shape[1] >= 6:
        rgb = arr[:, 3:6].astype(np.float64)
        # normalize to [0,1]
        if rgb.max() > 1.0:
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    return xyz, rgb


def vis_with_open3d(xyz, rgb):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])


def vis_with_matplotlib(xyz, rgb, max_points=20000):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    N = xyz.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if rgb is not None:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1)
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="gray", s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def find_first_npz_in_dir(d: Path):
    for p in d.rglob("*.npz"):
        return p
    return None


def main():
    p = argparse.ArgumentParser(description="Visualize saved pointcloud .npz")
    p.add_argument(
        "--path",
        nargs="?",
        default="/home/glzn/project/lehome/Datasets/record/259/pointclouds/episode_005/frame_1.npz",
        help="path to .npz file or directory",
    )
    args = p.parse_args()
    if args.path is None:
        # default to find in project under Datasets/record
        base = Path("Datasets/record")
        if not base.exists():
            print(
                "No path specified, and default Datasets/record does not exist. Please enter a .npz file or a directory containing .npz."
            )
            sys.exit(1)
        npz_path = find_first_npz_in_dir(base)
        if npz_path is None:
            print("No .npz file found under Datasets/record.")
            sys.exit(1)
    else:
        pth = Path(args.path)
        if pth.is_dir():
            npz_path = find_first_npz_in_dir(pth)
            if npz_path is None:
                print(f"No .npz file found under directory {pth}.")
                sys.exit(1)
        else:
            npz_path = pth

    print(f"Loading {npz_path}")
    arr = load_npz_pointcloud(npz_path)
    print("loaded shape:", arr.shape)
    xyz, rgb = to_xyz_rgb(arr)

    # try open3d first
    try:
        vis_with_open3d(xyz, rgb)
    except Exception as e:
        print("open3d visualization failed, fallback to matplotlib:", e)
        vis_with_matplotlib(xyz, rgb)


if __name__ == "__main__":
    main()
