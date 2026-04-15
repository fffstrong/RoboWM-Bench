import numpy as np
from scipy.spatial.transform import Rotation as R

# import plotly.graph_objs as go


# ==========================================
# 1. FPS Sampling
# ==========================================
def farthest_point_sampling_with_color(points, colors, n_samples):
    N, D = points.shape
    if N < n_samples:
        indices = np.random.choice(N, n_samples, replace=True)
        return points[indices], colors[indices]

    xyz = points
    centroids = np.zeros((n_samples,), dtype=int)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)

    return points[centroids], colors[centroids]


# ==========================================
# 2. Remove Outliers
# ==========================================
def remove_outliers_statistical(points, colors, nb_neighbors=20, std_ratio=2.0):
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return points, colors

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors)
    mean_dists = np.mean(dists, axis=1)

    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)

    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold

    return points[mask], colors[mask]


# ==========================================
# 3. Pointcloud Generation
# ==========================================
def generate_pointcloud_from_data(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_wxyz: np.ndarray,
    intrinsic_matrix: np.ndarray = None,
    num_points: int = 4096,
    use_fps: bool = True,
):
    """
    Inputs:
        rgb_image: (H, W, 3) np.uint8
        depth_image: (H, W) np.float32 (meters)
        cam_pos_w: (3,) camera world position
        cam_quat_wxyz: (4,) camera world quaternion in wxyz convention
        intrinsic_matrix: (3, 3) camera intrinsic matrix, optional
        num_points: number of final output points
        use_fps: whether to use FPS
    Outputs:
        points_with_color: (N, 6) [x, y, z, r, g, b]
    """

    img_h, img_w = depth_image.shape

    # Camera Intrinsics
    if intrinsic_matrix is not None:
        fx = float(intrinsic_matrix[0, 0])
        fy = float(intrinsic_matrix[1, 1])
        cx = float(intrinsic_matrix[0, 2])
        cy = float(intrinsic_matrix[1, 2])
    else:
        fx, fy = 482.0, 482.0
        cx, cy = img_w / 2.0, img_h / 2.0

    # 1. get valid depth index
    valid_mask = depth_image > 0
    v_idx, u_idx = np.nonzero(valid_mask)
    total_valid = len(v_idx)

    if total_valid == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # 2. pre-sampling to accelerate calculation
    pre_sample_target = max(8192, num_points * 2)

    if total_valid > pre_sample_target:
        sample_indices = np.random.choice(total_valid, pre_sample_target, replace=False)
        v_sample = v_idx[sample_indices]
        u_sample = u_idx[sample_indices]
    else:
        v_sample = v_idx
        u_sample = u_idx

    # 3. unproject depth to camera optical frame (OpenCV: z-forward, x-right, y-down)
    Z_sample = depth_image[v_sample, u_sample]
    X_sample = (u_sample - cx) * Z_sample / fx
    Y_sample = (v_sample - cy) * Z_sample / fy

    points_cam = np.stack([X_sample, Y_sample, Z_sample], axis=1)

    # 4. get color
    if rgb_image.shape[-1] == 4:
        colors_sample = rgb_image[v_sample, u_sample, :3]
    else:
        colors_sample = rgb_image[v_sample, u_sample]

    # 5. camera optical frame -> world frame
    # quat_w_ros: rotation from optical frame (x-right, y-down, z-forward) to world frame
    quat_xyzw = [cam_quat_wxyz[1], cam_quat_wxyz[2], cam_quat_wxyz[3], cam_quat_wxyz[0]]
    r_optical_to_world = R.from_quat(quat_xyzw).as_matrix()

    points_world = points_cam @ r_optical_to_world.T + cam_pos_w

    # 6. statistical filtering to remove noise
    points_world, colors_sample = remove_outliers_statistical(
        points_world, colors_sample, nb_neighbors=50, std_ratio=1.0
    )

    # 7. final sampling (FPS or random)
    if points_world.shape[0] > num_points:
        if use_fps:
            points_world, colors_sample = farthest_point_sampling_with_color(
                points_world, colors_sample, num_points
            )
        else:
            # simple random downsampling
            indices = np.random.choice(points_world.shape[0], num_points, replace=False)
            points_world = points_world[indices]
            colors_sample = colors_sample[indices]

    # merge to (N,6) -> [x,y,z,r,g,b], dtype float32, color normalized to [0,1]
    points_with_color = np.concatenate(
        [points_world.astype(np.float32), colors_sample.astype(np.float32)], axis=1
    )

    return points_with_color
