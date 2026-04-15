"""
End-effector pose calculation utility functions (shared module).

Used by augment_ee_pose.py, teleop_record.py, and other scripts.
"""

import numpy as np


def mat_to_quat(rot_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw)."""
    m = rot_mat
    t = np.trace(m)
    if t > 0.0:
        r = np.sqrt(1.0 + t)
        w = 0.5 * r
        r = 0.5 / r
        x = (m[2, 1] - m[1, 2]) * r
        y = (m[0, 2] - m[2, 0]) * r
        z = (m[1, 0] - m[0, 1]) * r
    else:
        i = np.argmax([m[0, 0], m[1, 1], m[2, 2]])
        if i == 0:
            r = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            x = 0.5 * r
            r = 0.5 / r
            y = (m[0, 1] + m[1, 0]) * r
            z = (m[0, 2] + m[2, 0]) * r
            w = (m[2, 1] - m[1, 2]) * r
        elif i == 1:
            r = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2])
            y = 0.5 * r
            r = 0.5 / r
            x = (m[0, 1] + m[1, 0]) * r
            z = (m[1, 2] + m[2, 1]) * r
            w = (m[0, 2] - m[2, 0]) * r
        else:
            r = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2])
            z = 0.5 * r
            r = 0.5 / r
            x = (m[0, 2] + m[2, 0]) * r
            y = (m[1, 2] + m[2, 1]) * r
            w = (m[1, 0] - m[0, 1]) * r
    quat = np.array([x, y, z, w], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (qx, qy, qz, qw) to 3x3 rotation matrix."""
    qx, qy, qz, qw = quat
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to [x, y, z, w] format."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] format."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def compute_ee_pose_single_arm(solver, arm_joints: np.ndarray, state_unit: str) -> np.ndarray:
    """
    Compute end-effector pose (pos + quat + gripper) for a single arm, input 6D joint angles.
    
    Args:
        solver: RobotKinematics instance
        arm_joints: 6D joint angles [5 arm joints + gripper]
        state_unit: 'rad' or 'deg'
    
    Returns:
        8D array [x, y, z, qx, qy, qz, qw, gripper]
    
    Raises:
        ValueError: If arm_joints is not 6D or state_unit is invalid
    """
    # Input validation
    if len(arm_joints) != 6:
        raise ValueError(f"Expected 6D joint angles [5 arm joints + gripper], got {len(arm_joints)}D")
    if state_unit not in ["rad", "deg"]:
        raise ValueError(f"Invalid state_unit '{state_unit}'. Must be 'rad' or 'deg'")
    
    # Separate arm joints (first 5) and gripper (last 1)
    # Note: Gripper is also an angle (range: -0.17~1.75 rad or -10~100 deg),
    #       but FK only uses the first 5 joints for pose calculation,
    #       so gripper value passes through unchanged and maintains input unit
    arm_only = arm_joints[:5]
    gripper = arm_joints[5:6]
    
    # Convert arm joint units for FK calculation
    if state_unit == "rad":
        arm_deg = np.rad2deg(arm_only)
    else:
        arm_deg = arm_only
    
    # Forward kinematics calculation (only uses first 5 arm joints)
    T = solver.forward_kinematics(arm_deg)
    pos = T[:3, 3].astype(np.float32)
    quat = mat_to_quat(T[:3, :3])  # Returns [qx, qy, qz, qw]
    
    # Concatenate pose components: position (3D) + quaternion (4D) + gripper (1D)
    return np.concatenate([pos, quat, gripper], axis=0)  # 8D


def _compute_joints_from_world_point_internal(
    solver,
    target_pos_world: np.ndarray,
    base_pos_world: np.ndarray,
    base_quat_world: np.ndarray,
    current_joints: np.ndarray,
    state_unit: str = "rad",
    target_quat_world: np.ndarray | None = None,
    gripper_angle: float = 0.0,
) -> dict:
    """
    Internal implementation: compute joint angles from world coordinates.
    
    Args:
        solver: RobotKinematics instance
        target_pos_world: Target position in world frame [x, y, z], meters
        base_pos_world: Base position in world frame [x, y, z], meters
        base_quat_world: Base orientation in world frame [w, x, y, z]
        current_joints: Current joint angles (6D) for IK initial guess
        state_unit: 'rad' or 'deg'
        target_quat_world: Optional target orientation [w, x, y, z]. 
                          If None, performs position-only IK with weak orientation constraint
        gripper_angle: Gripper angle in same unit as joints
    
    Returns:
        dict with keys:
            - "success": bool
            - "joints": np.ndarray (6D) or None
            - "error_msg": str
            - "target_pos_base": np.ndarray (3D) - target in base frame
    """
    import warnings
    
    try:
        # Convert quaternion format: [w,x,y,z] -> [x,y,z,w]
        base_quat_xyzw = quat_wxyz_to_xyzw(base_quat_world)
        
        # Build base frame transformation matrix
        T_world_base = np.eye(4)
        T_world_base[:3, 3] = base_pos_world
        T_world_base[:3, :3] = quat_to_mat(base_quat_xyzw)
        
        # Sanity check: base frame Z axis should point roughly upward
        z_axis = T_world_base[:3, 2]
        if z_axis[2] < 0.3:
            warnings.warn(
                f"Base frame Z axis [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}] "
                f"is not pointing upward (z component < 0.3). "
                f"Check base pose source or quaternion order.",
                UserWarning
            )
        
        # Build target transformation matrix
        T_world_target = np.eye(4)
        T_world_target[:3, 3] = target_pos_world
        
        # Set target orientation
        if target_quat_world is not None:
            # Use provided orientation (convert [w,x,y,z] -> [x,y,z,w])
            target_quat_xyzw = quat_wxyz_to_xyzw(target_quat_world)
            T_world_target[:3, :3] = quat_to_mat(target_quat_xyzw)
        else:
            # Position-only IK: use base orientation as reference
            # Note: orientation_weight will be set to 0.001 in IK solver
            T_world_target[:3, :3] = quat_to_mat(base_quat_xyzw)
        
        # Transform target to base frame
        T_base_world = np.linalg.inv(T_world_base)
        T_base_target = T_base_world @ T_world_target
        
        # Extract target pose in base frame
        target_pos_base = T_base_target[:3, 3]
        target_quat_base_xyzw = mat_to_quat(T_base_target[:3, :3])
        
        # Build 8D end-effector pose [pos(3) + quat(4) + gripper(1)]
        ee_pose_target = np.concatenate([
            target_pos_base,
            target_quat_base_xyzw,
            [gripper_angle]
        ])
        
        # Call IK solver
        # For position-only IK (target_quat_world=None), use weak orientation constraint
        orientation_weight = 0.01 if target_quat_world is None else 1.0
        joint_angles = compute_joints_from_ee_pose(
            solver, current_joints, ee_pose_target, state_unit, orientation_weight=orientation_weight
        )
        
        if joint_angles is not None:
            return {
                "success": True,
                "joints": joint_angles,
                "error_msg": "",
                "target_pos_base": target_pos_base,
            }
        else:
            return {
                "success": False,
                "joints": None,
                "error_msg": "IK solver failed to converge. Target may be out of workspace.",
                "target_pos_base": target_pos_base,
            }
        
    except Exception as e:
        return {
            "success": False,
            "joints": None,
            "error_msg": f"Exception during IK: {str(e)}",
            "target_pos_base": None,
        }


def compute_joints_from_world_point(
    solver,
    env,
    arm: str,
    target_pos_world: np.ndarray,
    current_joints: np.ndarray | None = None,
    state_unit: str = "rad",
    gripper_angle: float = 0.0,
) -> np.ndarray | None:
    """
    Compute joint angles to reach a world-space grasp point (simplified API).
    
    This is the recommended interface for typical grasping tasks.
    Base pose is automatically extracted from the environment.
    
    For 5DOF arms: Position is primary constraint, orientation is weakly constrained.
    
    Args:
        solver: RobotKinematics instance (initialized with URDF)
        env: Isaac Lab environment with robot arms (e.g., DirectRLEnv)
        arm: Which arm to use, "left" or "right"
        target_pos_world: Grasp target position in world frame [x, y, z], meters
        current_joints: Current joint angles (6D) for IK warm start.
                       If None, reads from env automatically.
        state_unit: Joint angle unit, 'rad' or 'deg' (default: 'rad')
        gripper_angle: Gripper angle in same unit as joints (default: 0.0)
    
    Returns:
        6D joint angles [5 arm joints + gripper] or None if IK fails
    
    Raises:
        ValueError: If arm is invalid or env doesn't have the specified arm
    
    Example:
        >>> from lehome.utils import RobotKinematics, compute_joints_from_world_point
        >>> solver = RobotKinematics("robot.urdf", "gripper_frame_link", joint_names)
        >>> target = np.array([1.5, -2.0, 0.6])  # Grasp point in world frame
        >>> joints = compute_joints_from_world_point(solver, env, "right", target)
        >>> if joints is not None:
        ...     env.step(joints)
    """
    # Explicit validation: check if env has the specified arm
    arm_attr = f"{arm}_arm"
    if not hasattr(env, arm_attr):
        raise ValueError(
            f"Invalid arm '{arm}'. Environment has no attribute '{arm_attr}'. "
            f"Expected 'left' or 'right'."
        )
    
    # Extract base pose from environment
    arm_obj = getattr(env, arm_attr)
    base_pos_world = arm_obj.data.root_pos_w[0].cpu().numpy()
    base_quat_world_xyzw = arm_obj.data.root_quat_w[0].cpu().numpy()
    
    # Convert quaternion: env uses [x,y,z,w], internal uses [w,x,y,z]
    base_quat_world_wxyz = quat_xyzw_to_wxyz(base_quat_world_xyzw)
    
    # Read current joints if not provided
    if current_joints is None:
        current_joints = arm_obj.data.joint_pos[0].cpu().numpy()
    
    # Call internal implementation (position-only IK when target_quat_world=None)
    result = _compute_joints_from_world_point_internal(
        solver=solver,
        target_pos_world=target_pos_world,
        base_pos_world=base_pos_world,
        base_quat_world=base_quat_world_wxyz,
        current_joints=current_joints,
        state_unit=state_unit,
        target_quat_world=None,  # Position-only IK
        gripper_angle=gripper_angle,
    )
    
    # Return simple format for backward compatibility
    return result["joints"]


def compute_joints_from_world_point_detailed(
    solver,
    target_pos_world: np.ndarray,
    base_pos_world: np.ndarray,
    base_quat_world: np.ndarray,
    current_joints: np.ndarray,
    state_unit: str = "rad",
    target_quat_world: np.ndarray | None = None,
    gripper_angle: float = 0.0,
) -> dict:
    """
    Compute joint angles from world coordinates (detailed API for advanced use).
    
    **Advanced use only**: For testing, debugging, or non-standard environments.
    Requires manual specification of base pose. For typical use cases, prefer
    the simpler `compute_joints_from_world_point` which extracts base from env.
    
    Args:
        solver: RobotKinematics instance
        target_pos_world: Target position in world frame [x, y, z], meters
        base_pos_world: Base position in world frame [x, y, z], meters
        base_quat_world: Base orientation in world frame [w, x, y, z]
        current_joints: Current joint angles (6D) for IK warm start
        state_unit: 'rad' or 'deg'
        target_quat_world: Optional target orientation [w, x, y, z].
                          If None, position-only IK (weak orientation constraint)
        gripper_angle: Gripper angle in same unit as joints
    
    Returns:
        dict with keys:
            - "success" (bool): Whether IK succeeded
            - "joints" (np.ndarray | None): 6D joint angles if successful
            - "error_msg" (str): Error description if failed
            - "target_pos_base" (np.ndarray): Target position in base frame (for debugging)
    
    Example:
        >>> result = compute_joints_from_world_point_detailed(
        ...     solver, target, base_pos, base_quat, current_joints
        ... )
        >>> if result["success"]:
        ...     joints = result["joints"]
        ...     print(f"Target in base frame: {result['target_pos_base']}")
    """
    return _compute_joints_from_world_point_internal(
        solver=solver,
        target_pos_world=target_pos_world,
        base_pos_world=base_pos_world,
        base_quat_world=base_quat_world,
        current_joints=current_joints,
        state_unit=state_unit,
        target_quat_world=target_quat_world,
        gripper_angle=gripper_angle,
    )


def compute_joints_from_ee_pose(
    solver, 
    current_joints: np.ndarray, 
    ee_pose: np.ndarray, 
    state_unit: str,
    orientation_weight: float = 1.0,
    joint_num: int = 9,
) -> np.ndarray | None:
    """
    Compute joint angles from end-effector pose via IK (inverse of compute_ee_pose_single_arm).
    
    Args:
        solver: RobotKinematics instance
        current_joints: 6D current joint angles (as initial guess for IK)
        ee_pose: 8D end-effector pose [x,y,z,qx,qy,qz,qw,gripper]
        state_unit: 'rad' or 'deg'
        orientation_weight: Weight for orientation constraint (default: 1.0)
                           - 1.0: position and orientation equally important
                           - 0.01: position 100x more important (for position-only tasks)
    
    Returns:
        6D joint angles, or None if IK fails
    """
    try:
        # Extract pose components
        pos = ee_pose[:3]
        quat = ee_pose[3:7]  # [qx, qy, qz, qw]
        gripper = ee_pose[7]
        
        # Construct 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(quat)
        T[:3, 3] = pos
        # Unit conversion for IK (solver expects degrees)
        if state_unit == "rad":
            current_deg = np.rad2deg(current_joints)
        else:
            current_deg = current_joints

        # IK solving (first n joints)
        ik_result_deg = solver.inverse_kinematics(
            current_deg, T, position_weight=1.0, orientation_weight=orientation_weight
        )

        # Convert back to original unit
        if state_unit == "rad":
            ik_joints = np.deg2rad(ik_result_deg)
        else:
            ik_joints = ik_result_deg

        # Combine: first 5 from IK, gripper from ee_pose
        result = ik_joints.copy()
        result[joint_num-1] = gripper/2
        result[joint_num-2] = gripper/2
        return result[:joint_num]
        
    except Exception:
        return None
