import numpy as np
from scipy.spatial.transform import Rotation as R


def _to_xyzw(q_wxyz):
    """Convert quaternion format from (w, x, y, z) to (x, y, z, w)."""
    q_wxyz = np.asarray(q_wxyz, dtype=float)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def _to_wxyz(q_xyzw):
    """Convert quaternion format from (x, y, z, w) to (w, x, y, z)."""
    q_xyzw = np.asarray(q_xyzw, dtype=float)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _as_range_tuple(val):
    """
    Normalize input to a (min, max) tuple:
    - Scalar a       -> (-a, a)
    - Tuple (a, b)   -> (a, b)
    """
    if np.isscalar(val):
        return (-float(val), float(val))
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (float(val[0]), float(val[1]))
    raise ValueError("Range must be a scalar or a 2-element tuple/list.")


def _sample_uniform(rng, r):
    """Sample uniformly within the given range tuple or scalar."""
    lo, hi = _as_range_tuple(r)
    return rng.uniform(lo, hi)


def _axis_to_unit_vec(axis):
    """Convert axis input ('x', 'y', 'z' or vector) to a normalized 3D vector."""
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "x":
            return np.array([1.0, 0.0, 0.0])
        if axis == "y":
            return np.array([0.0, 1.0, 0.0])
        if axis == "z":
            return np.array([0.0, 0.0, 1.0])
        raise ValueError("Axis string must be one of 'x', 'y', or 'z'.")
    v = np.asarray(axis, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Axis vector has zero norm.")
    return v / n


def randomize_pose(
    base_translation,
    base_quat_wxyz,
    trans_range=0.0,
    axis="z",
    deg_range=0.0,
    axis_space="world",
    rng=None,
):
    """
    Randomize a pose by applying translation and rotation noise.

    Returns:
        (translation, quat_wxyz)

    Args:
        base_translation: Base position (tx, ty, tz)
        base_quat_wxyz: Base quaternion (w, x, y, z)
        trans_range:
            1) Scalar a -> sample each axis from [-a, a]
            2) Tuple/list (ax, ay, az) -> sample each axis independently
            3) Dict {'x':(min,max), 'y':..., 'z':...} -> per-axis control
        deg_range:
            1) Scalar d -> sample from [-d, d] degrees
            2) Tuple (dmin, dmax) -> sample from [dmin, dmax] degrees
        axis: Rotation axis ('x', 'y', 'z', or a vector)
        axis_space: Rotation applied in 'world' or 'local' space
        rng: Optional numpy random generator (np.random.default_rng by default)
    """
    rng = rng or np.random.default_rng()

    # 1) Sample translation noise
    if isinstance(trans_range, dict):
        dx = _sample_uniform(rng, trans_range.get("x", 0.0))
        dy = _sample_uniform(rng, trans_range.get("y", 0.0))
        dz = _sample_uniform(rng, trans_range.get("z", 0.0))
    elif isinstance(trans_range, (list, tuple)) and len(trans_range) == 3:
        dx = _sample_uniform(rng, trans_range[0])
        dy = _sample_uniform(rng, trans_range[1])
        dz = _sample_uniform(rng, trans_range[2])
    else:
        # Scalar: same range for all axes
        dx = _sample_uniform(rng, trans_range)
        dy = _sample_uniform(rng, trans_range)
        dz = _sample_uniform(rng, trans_range)

    t_base = np.asarray(base_translation, dtype=float)
    t_new = t_base + np.array([dx, dy, dz])

    # 2) Sample rotation around the specified axis
    ddeg = _sample_uniform(rng, deg_range)
    axis_unit = _axis_to_unit_vec(axis)
    r_add = R.from_rotvec(np.deg2rad(ddeg) * axis_unit)

    # 3) Combine rotations (convert quaternion formats as needed)
    q_base_xyzw = _to_xyzw(base_quat_wxyz)
    r_base = R.from_quat(q_base_xyzw)

    if axis_space == "world":
        # Apply additional rotation in world frame: R_total = R_add * R_base
        r_total = r_add * r_base
    elif axis_space == "local":
        # Apply additional rotation in local frame: R_total = R_base * R_add
        r_total = r_base * r_add
    else:
        raise ValueError("axis_space must be 'world' or 'local'.")

    q_new_xyzw = r_total.as_quat()
    q_new_wxyz = _to_wxyz(q_new_xyzw)

    # Normalize quaternion for numerical stability
    q_new_wxyz = q_new_wxyz / np.linalg.norm(q_new_wxyz)

    return t_new, q_new_wxyz
