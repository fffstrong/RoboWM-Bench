"""
Hand Model Module

This module provides hand modeling for action processors. It converts detected hand 
keypoints into kinematic models that can be used for robot control

Key Components:
- HandModel: Base class for unconstrained hand kinematic modeling
- PhysicallyConstrainedHandModel: Extended class with constrained joint and velocity limits
- Grasp point and orientation calculation for robot end-effector control

The hand model follows the MediaPipe hand landmark convention with 21 keypoints:
- Wrist (1 point)
- Thumb (4 points: MCP, PIP, DIP, TIP)
- Index finger (4 points: MCP, PIP, DIP, TIP)
- Middle finger (4 points: MCP, PIP, DIP, TIP)
- Ring finger (4 points: MCP, PIP, DIP, TIP)
- Pinky finger (4 points: MCP, PIP, DIP, TIP)

Coordinate System:
- All calculations performed in robot coordinate frame
- Grasp orientations aligned with robot end-effector conventions
- Joint rotations represented as rotation matrices and Euler angles
"""

from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np
import pdb
import torch
from scipy.spatial.transform import Rotation
import logging
import yaml
from pathlib import Path

from phantom.utils.transform_utils import * 
logger = logging.getLogger(__name__)


class HandModel:
    """
    Base class for hand kinematic modeling and trajectory analysis.
    
    This class provides a kinematic representation of a human hand using 21 keypoints
    from hand pose estimation. It calculates joint rotations, tracks hand motion over
    time, and computes grasp points and orientations suitable for robot control.
    
    The kinematic structure follows a tree topology with the wrist as the root,
    and each finger as a separate chain. Joint rotations are calculated relative
    to parent joints using vector alignment methods.
    
    Key Features:
    - 21-point hand keypoint processing
    - Joint rotation calculation using vector alignment
    - Grasp point computation from thumb-index / thumb-middle finger positioning
    - End-effector orientation calculation for robot control
    
    Attributes:
        robot_name (str): Name of the target robot for coordinate frame alignment
        kinematic_tree (List[Tuple[int, int]]): Parent-child relationships for hand joints
        joint_to_neighbors_mapping (Dict[int, Tuple[int, int, int]]): Mapping of joints to their neighbors
        vertex_positions (List[np.ndarray]): Time series of hand keypoint positions
        joint_rotations (List[List[np.ndarray]]): Time series of joint rotation matrices
        grasp_points (List[np.ndarray]): Time series of computed grasp points
        grasp_oris (List[np.ndarray]): Time series of grasp orientation matrices
        timestamps (List[float]): Time stamps for each frame
        num_joints (int): Total number of joints in the hand model
        joint_rotations_xyz (List[List[np.ndarray]]): Time series of Euler angle representations
    """
    def __init__(self, robot_name: str, config_path: Optional[str] = None) -> None:
        """
        Initialize the hand model with kinematic structure.

        Args:
            robot_name: Name of the target robot for coordinate alignment
        """
        self.robot_name: str = robot_name

        # Load target hand from config
        self.target_hand = self._load_target_hand(config_path)

        # Define the kinematic tree structure for hand joints
        # Format: (joint_index, parent_index) where -1 indicates root (wrist)
        self.kinematic_tree: List[Tuple[int, int]] = [
            (0, -1),    # wrist base (root of the kinematic tree)

            # Thumb chain (4 joints)
            (1, 0),     # thumb mcp 
            (2, 1),     # thumb pip 
            (3, 2),     # thumb dip 
            (4, 3),     # thumb tip

            # Index finger chain (4 joints)
            (5, 0),     # index mcp
            (6, 5),     # index pip
            (7, 6),     # index dip
            (8, 7),     # index tip

            # Middle finger chain (4 joints)
            (9, 0),     # middle mcp
            (10, 9),    # middle pip
            (11, 10),   # middle dip
            (12, 11),   # middle tip

            # Ring finger chain (4 joints)
            (13, 0),    # ring mcp
            (14, 13),   # ring pip
            (15, 14),   # ring dip
            (16, 15),   # ring tip

            # Pinky finger chain (4 joints)
            (17, 0),    # pinky mcp
            (18, 17),   # pinky pip
            (19, 18),   # pinky dip
            (20, 19),   # pinky tip
        ]

        # Mapping from joint index to (current_vertex, child_vertex, parent_vertex)
        # This defines the local coordinate system for each joint rotation calculation
        self.joint_to_neighbors_mapping: Dict[int, Tuple[int, int, int]] = {
            # Thumb joint mappings
            0: (0, 1, -1),  # wrist to thumb mcp (no parent)
            1: (1, 2, 0),   # thumb mcp to pip (parent: wrist)
            2: (2, 3, 1),   # thumb pip to dip (parent: thumb mcp)
            3: (3, 4, 2),   # thumb dip to tip (parent: thumb pip)
            
            # Index finger joint mappings
            4: (0, 5, -1),  # wrist to index mcp (no parent)
            5: (5, 6, 0),   # index mcp to pip (parent: wrist)
            6: (6, 7, 5),   # index pip to dip (parent: index mcp)
            7: (7, 8, 6),   # index dip to tip (parent: index pip)
            
            # Middle finger joint mappings
            8: (0, 9, -1),  # wrist to middle mcp (no parent)
            9: (9, 10, 0),  # middle mcp to pip (parent: wrist)
            10: (10, 11, 9), # middle pip to dip (parent: middle mcp)
            11: (11, 12, 10),# middle dip to tip (parent: middle pip)
            
            # Ring finger joint mappings
            12: (0, 13, -1), # wrist to ring mcp (no parent)
            13: (13, 14, 0),# ring mcp to pip (parent: wrist)
            14: (14, 15, 13),# ring pip to dip (parent: ring mcp)
            15: (15, 16, 14),# ring dip to tip (parent: ring pip)
            
            # Pinky finger joint mappings
            16: (0, 17, -1), # wrist to pinky mcp (no parent)
            17: (17, 18, 0),# pinky mcp to pip (parent: wrist)
            18: (18, 19, 17),# pinky pip to dip (parent: pinky mcp)
            19: (19, 20, 18),# pinky dip to tip (parent: pinky pip)
        }
        
        self.num_joints: int = len(self.joint_to_neighbors_mapping)
        
        # Time series data storage
        self.vertex_positions: List[np.ndarray] = []    # List of (21, 3) arrays for each timestep
        self.joint_rotations: List[List[np.ndarray]] = []     # List of rotation matrices for each joint
        self.joint_rotations_xyz: List[List[np.ndarray]] = [] # List of Euler angle representations
        self.grasp_points: List[np.ndarray] = []        # List of computed grasp points (3D positions)
        self.grasp_oris: List[np.ndarray] = []          # List of grasp orientation matrices (3x3)
        self.timestamps: List[float] = []          # List of timestamps for temporal analysis

    def _load_target_hand(self, config_path: Optional[str] = None) -> str:
        """
        Load target_hand parameter from YAML config file.
        
        Args:
            config_path: Path to config file. If None, uses default location.
            
        Returns:
            "right" or "left" indicating hand orientation
        """
        if config_path is None:
            # Default to configs/default.yaml relative to this file
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        else:
            config_path = Path(config_path)
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                target_hand = config.get('target_hand', 'right')
                logger.info(f"Loaded target_hand from config: {target_hand}")
                return target_hand.lower()
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Defaulting to 'right'")
            return 'right'
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Defaulting to 'right'")
            return 'right'
    
    def calculate_joint_rotation(self, current_pos: np.ndarray, child_pos: np.ndarray, parent_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the rotation matrix for a single joint using vector alignment.
        
        This method computes the rotation that aligns the previous direction vector
        with the current direction vector. For root joints (no parent), it uses
        a default upward direction as the reference.
        
        Args:
            current_pos: 3D position of the current joint
            child_pos: 3D position of the child joint
            parent_pos: 3D position of the parent joint
            
        Returns:
            Tuple containing:
                - rotation_matrix: 3x3 rotation matrix
                - euler_angles: Rotation as XYZ Euler angles
        """
        # Calculate current direction vector (current -> child)
        current_dir = child_pos - current_pos
        current_norm = np.linalg.norm(current_dir)
        if current_norm < 1e-10:
            return np.eye(3), np.array([0,0,0])
        current_dir /= current_norm
        
        # Calculate previous direction vector (parent -> current, or default up)
        prev_dir = np.array([0.0, 0.0, 1.0]) if parent_pos is None else current_pos - parent_pos
        prev_norm = np.linalg.norm(prev_dir)
        if prev_norm < 1e-10:
            return np.eye(3), np.array([0,0,0])
        prev_dir /= prev_norm
        
        # Check if vectors are already aligned (no rotation needed)
        if np.abs((np.abs(np.dot(current_dir, prev_dir)) - 1)) < 1e-8:
            return np.eye(3), np.array([0,0,0])
        
        # Calculate rotation that aligns prev_dir with current_dir
        rotation, _ = Rotation.align_vectors([current_dir], [prev_dir])
        return rotation.as_matrix(), rotation.as_euler('xyz')
    
    def calculate_frame_rotations(self, vertices: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calculate rotation matrices for all joints in a single frame.
        
        This method processes all joints in the hand and computes their rotations
        based on the kinematic structure and current vertex positions.
        
        Args:
            vertices: Hand keypoints, shape (21, 3)
            
        Returns:
            Tuple containing:
                - rotation_matrices: List of 3x3 rotation matrices
                - euler_angles: List of XYZ Euler angle arrays
        """
        rotations, rotations_xyz = zip(*[
            self.calculate_joint_rotation(vertices[m[0]], vertices[m[1]],
                                           None if m[2] == -1 else vertices[m[2]])
            for m in self.joint_to_neighbors_mapping.values()
        ])
        return list(rotations), list(rotations_xyz)
    
    def calculate_angular_velocity(self, joint_idx: int, t1: int, t2: int) -> np.ndarray:
        """
        Calculate angular velocity for a specific joint between two time frames.
        
        Angular velocity is computed as the rotation vector difference divided
        by the time difference between frames.
        
        Args:
            joint_idx: Index of the joint
            t1: Index of the first time frame
            t2: Index of the second time frame
            
        Returns:
            Angular velocity vector (3,) in rad/s
        """
        dt = self.timestamps[t2] - self.timestamps[t1]
        if dt == 0:
            return np.zeros(3)
        
        # Get rotation matrices for the two time frames
        R1, R2 = self.joint_rotations[t1][joint_idx], self.joint_rotations[t2][joint_idx]
        
        # Calculate relative rotation and convert to angular velocity
        R_relative = Rotation.from_matrix(R2) * Rotation.from_matrix(R1).inv()
        return R_relative.as_rotvec() / dt

    def calculate_frame_angular_velocities(self, current_frame_idx: int) -> np.ndarray:
        """
        Calculate angular velocities for all joints at the current frame.
        
        This method computes the angular velocity vectors for all joints by
        comparing rotations with the previous frame. Returns zeros for the
        first frame since no previous frame exists.
        
        Args:
            current_frame_idx: Index of the current frame. Must be > 0.
            
        Returns:
            Array of angular velocity vectors (shape: num_joints x 3)
            Each row contains [wx, wy, wz] for one joint.
            Returns zeros if current_frame_idx == 0.
        """
        if current_frame_idx == 0:
            return np.zeros((self.num_joints, 3))
            
        prev_frame_idx = current_frame_idx - 1
        
        # Vectorized calculation for all joints
        velocities = np.array([
            self.calculate_angular_velocity(joint_idx, prev_frame_idx, current_frame_idx)
            for joint_idx in range(self.num_joints)
        ])
        
        return velocities
    
    def calculate_grasp_plane(self, vertices: np.ndarray) -> np.ndarray:
        """
        Calculate the plane that best fits through a set of hand vertices.
        
        This method uses Singular Value Decomposition (SVD) to find the plane.
        The plane is typically fitted through thumb and index finger points.
        
        Args:
            vertices: Set of 3D points to fit plane through, shape (N, 3)
            
        Returns:
            Plane coefficients [a, b, c, d] for ax + by + cz + d = 0
        """
        # Create augmented matrix with homogeneous coordinates for plane fitting
        A = np.c_[vertices[:, 0], vertices[:, 1], vertices[:, 2], np.ones(vertices.shape[0])]

        # Right-hand side is zeros for the plane equation ax + by + cz + d = 0
        b = np.zeros(vertices.shape[0])

        # Use SVD to solve the least squares problem
        U, S, Vt = np.linalg.svd(A)

        # Plane coefficients are in the last row of Vt (smallest singular value)
        plane_coeffs = Vt[-1, :]

        # Normalize coefficients for easier interpretation (unit normal vector)
        plane_coeffs = plane_coeffs / np.linalg.norm(plane_coeffs[:3])

        return plane_coeffs  # [a, b, c, d]
    
    def calculate_grasp_point(self, grasp_plane: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """
        Calculate the optimal grasp point for robot end-effector positioning.
        
        The grasp point is computed as the midpoint between projected thumb tip
        and index finger tip on the grasp plane. This provides a stable reference
        point for robot grasping operations.
        
        Args:
            grasp_plane: Plane coefficients [a, b, c, d]
            vertices: Hand keypoints, shape (21, 3)
            
        Returns:
            3D grasp point coordinates
        """
        # Project fingertips onto the grasp plane
        thumb_pt = project_point_to_plane(vertices[4], grasp_plane)
        index_pt = project_point_to_plane(vertices[8], grasp_plane)
        # thumb_pt = project_point_to_plane(vertices[3], grasp_plane)
        # index_pt = project_point_to_plane(vertices[7], grasp_plane)
        
        # Compute midpoint as the grasp reference
        hand_ee_pt = np.mean([thumb_pt, index_pt], axis=0)
        return hand_ee_pt

    def add_frame(self, vertices: np.ndarray, timestamp: float, hand_detected: bool = True) -> None:
        """
        Add a new frame of vertex positions and calculate corresponding data.
        
        This is the main method for processing hand data over time. It computes
        grasp points, orientations, and stores all relevant information for
        the current timestep.
        
        Args:
            vertices: Array of 21 3D vertex positions
            timestamp: Time of the frame in seconds
            hand_detected: Whether hand was successfully detected
        """
        if len(vertices) != 21:
            raise ValueError(f"Expected 21 vertices, got {len(vertices)}")
        
        # Handle frames without hand detection
        if not hand_detected: 
            self.vertex_positions.append(np.zeros((21, 3)))
            self.grasp_points.append(np.zeros(3))
            self.grasp_oris.append(np.eye(3))
            self.timestamps.append(timestamp)
            return
        
        # Extract key finger tip positions
        thumb_tip = vertices[4]
        index_tip = vertices[8]
        middle_tip = vertices[12]

        # Calculate grasp point as midpoint between thumb and index finger tips
        # control_point = (thumb_tip + middle_tip) / 2
        control_point = (thumb_tip + index_tip) / 2
        grasp_pt = control_point

        # Calculate gripper orientation from thumb-index finger configuration
        gripper_ori, _ = HandModel.get_gripper_orientation(thumb_tip, index_tip, vertices, target_hand=self.target_hand)
        
        # Apply 90-degree rotation to align with robot gripper convention
        rot_90_deg = Rotation.from_euler('Z', 90, degrees=True).as_matrix()
        grasp_ori = gripper_ori @ rot_90_deg

        # Store all frame data
        self.vertex_positions.append(vertices)
        self.grasp_points.append(grasp_pt)
        self.grasp_oris.append(grasp_ori)
        self.timestamps.append(timestamp)


    def get_joint_data(self, joint_idx: int) -> Dict[str, Union[List[float], List[np.ndarray]]]:
        """
        Get all trajectory data for a specific joint across all frames.
        
        Args:
            joint_idx: Index of the joint
            
        Returns:
            Dictionary containing joint trajectory data with keys:
                - 'timestamps': List of time stamps
                - 'rotations': List of rotation matrices for this joint
        """
        return {
            'timestamps': self.timestamps,
            'rotations': [frame[joint_idx] for frame in self.joint_rotations],
        }
    
    @staticmethod
    def get_parallel_plane(a: float, b: float, c: float, d: float, dist: float) -> Tuple[float, float, float, float]:
        """
        Calculate coefficients of a plane parallel to the given plane at specified distance.
        
        This utility method is useful for creating offset grasp planes that account
        for gripper thickness or provide clearance during grasping operations.
        
        Parameters:
            a, b, c, d: Coefficients of the original plane ax + by + cz + d = 0
            dist: Distance between planes (positive moves in normal direction)
        
        Returns:
            (a, b, c, d_new) coefficients of the parallel plane
        """
        # Calculate the magnitude of the normal vector
        normal_magnitude = np.sqrt(a**2 + b**2 + c**2)
        
        # Parallel plane has same normal direction, only d changes
        d_new = d - dist * normal_magnitude
        
        return (a, b, c, d_new)

    @staticmethod
    def get_gripper_orientation(thumb_tip: np.ndarray, index_tip: np.ndarray, vertices: np.ndarray, grasp_plane: Optional[np.ndarray] = None, target_hand: str = 'right') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robot gripper orientation matrix from hand keypoints and fingertip positions.
        This method calculates a coordinate frame suitable for robot gripper control
        based on the relative positions of thumb, index finger, and wrist. The resulting
        orientation matrix can be directly used for robot end-effector control.
        target_hand: "right" or "left" to determine X-axis orientation
        Args:
            thumb_tip: 3D position of thumb tip
            index_tip: 3D position of index finger tip
            vertices: All hand keypoints, shape (21, 3)
            grasp_plane: Plane coefficients [a,b,c,d]
        Returns:
            Tuple containing:
                - gripper_orientation: 3x3 rotation matrix
                - z_axis: Z-axis direction vector of the gripper frame
        """
        # Calculate gripper opening direction (thumb to index finger)
        gripper_direction = thumb_tip - index_tip

        # Calculate gripper reference point (midpoint of fingertips)
        midpoint = (thumb_tip + index_tip) / 2

        if grasp_plane is None:
            # Use palm geometry when no plane is provided
            palm_axis = vertices[5] - midpoint  # index MCP to midpoint
            x_axis = gripper_direction / max(np.linalg.norm(gripper_direction), 1e-10)
            z_axis = -palm_axis / max(np.linalg.norm(palm_axis), 1e-10)

            # Compute y-axis
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= max(np.linalg.norm(y_axis), 1e-10)

            # Ensure orthogonality by recalculating z_axis
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= max(np.linalg.norm(z_axis), 1e-10)

            # Check orientation consistency with palm direction
            if type(palm_axis) == torch.Tensor:
                palm_axis = palm_axis.cpu().numpy()
            if z_axis @ palm_axis > 0:
                x_axis, y_axis, z_axis = -x_axis, -y_axis, -z_axis

        else:
            # 1. Project Thumb DIP (idx 3) and Index DIP (idx 7) to the grasp plane
            thumb_dip_proj = project_point_to_plane(vertices[3], grasp_plane)
            index_dip_proj = project_point_to_plane(vertices[7], grasp_plane)

            # 2. Define X-axis: Line connecting Thumb DIP projection to Index DIP projection
            x_axis = thumb_dip_proj - index_dip_proj
            if target_hand == 'right':
                print("Using right hand convention: flipping X-axis")
                x_axis = -x_axis
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-10:
                x_axis = np.array([1.0, 0.0, 0.0])
            else:
                x_axis /= x_norm

            # 3. Define Y-axis: Grasp plane normal
            y_axis = grasp_plane[:3]
            y_axis /= max(np.linalg.norm(y_axis), 1e-10)

            # 4. Define Z-axis: Right-hand rule (Cross product of X and Y)
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= max(np.linalg.norm(z_axis), 1e-10)

            # 5. Re-orthogonalize Y-axis to ensure valid rotation matrix (Y = Z x X)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= max(np.linalg.norm(y_axis), 1e-10)

        # Construct orientation matrix
        gripper_ori = np.column_stack([x_axis, y_axis, z_axis])

        # Ensure proper handedness (right-handed coordinate system)
        if np.linalg.det(gripper_ori) < 0:
            x_axis = -x_axis  # Flip one axis to fix handedness
            gripper_ori = np.column_stack([x_axis, y_axis, z_axis])

        # Verify determinant for debugging
        det = np.linalg.det(gripper_ori)
        if det < 0.9:
            pdb.set_trace()

        return gripper_ori, z_axis


class PhysicallyConstrainedHandModel(HandModel):
    """
    Extended hand model with physical constraints and realistic joint limits.
    
    This class builds upon the base HandModel by adding realistic constraints
    that enforce physically plausible hand poses and motion. It includes:
    - Joint angle limits based on human hand anatomy
    - Angular velocity constraints for smooth motion
    - Pose reconstruction with constraint enforcement
    - Enhanced grasp point calculation with plane-based refinement

    Constrained hand model is used in Phantom
    
    Key Constraints:
    - Anatomically correct joint limits for each finger joint
    - Velocity limiting to prevent jerky motions
    - Iterative pose refinement with constraint satisfaction
    - More robust grasp plane calculation and orientation alignment
    
    Attributes:
        joint_limits (Dict[int, Tuple[float, ...]]): Joint angle limits for each joint in radians
        max_angular_velocity (float): Maximum allowed angular velocity in rad/s
    """
    def __init__(self, robot_name: str, config_path: Optional[str] = None) -> None:
        """
        Initialize the physically constrained hand model.
        
        Args:
            robot_name: Name of the target robot for coordinate alignment
            config_path: Path to the configuration file
        """
        super().__init__(robot_name, config_path)
        
        # Define joint rotation limits (in radians) for each joint
        # Format: (min_x, max_x, min_y, max_y, min_z, max_z) for XYZ Euler angles
        small_angle = np.pi/40  # Small constraint for fine motor control
        
        self.joint_limits: Dict[int, Tuple[float, float, float, float, float, float]] = {
            # Thumb joints - more flexible due to opposable nature
            0: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # wrist to thumb mcp
            1: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # thumb mcp to pip
            2: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # thumb pip to dip
            3: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # thumb dip to tip
            
            # Index finger joints - moderate constraints
            4: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # wrist to index mcp
            5: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # index mcp to pip
            6: (-small_angle, small_angle, -np.pi/8, np.pi/8, -small_angle, small_angle), # index pip to dip
            7: (-small_angle, small_angle, -np.pi/8, np.pi/8, -small_angle, small_angle), # index dip to tip
            
            # Middle finger joints - tighter constraints for stability
            8: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # wrist to middle mcp
            9: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # middle mcp to pip
            10: (-np.pi, np.pi, -np.pi, np.pi, -np.pi/4, np.pi/4), # middle pip to dip
            11: (-np.pi, np.pi, -np.pi, np.pi, -np.pi/4, np.pi/4), # middle dip to tip
            
            # Ring finger joints - similar to middle finger
            12: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # wrist to ring mcp
            13: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # ring mcp to pip
            14: (-np.pi, np.pi, -np.pi, np.pi, -np.pi/4, np.pi/4), # ring pip to dip
            15: (-np.pi, np.pi, -np.pi, np.pi, -np.pi/4, np.pi/4), # ring dip to tip
            
            # Pinky finger joints - most constrained due to size
            16: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # wrist to pinky mcp
            17: (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi), # pinky mcp to pip
            18: (-np.pi, np.pi, -np.pi, np.pi, -np.pi/4, np.pi/4), # pinky pip to dip
            19: (-np.pi, np.pi, -np.pi, np.pi, -np.pi/4, np.pi/4), # pinky dip to tip
        }
        
        # Maximum angular velocity constraint (2π rad/s = 360°/s)
        self.max_angular_velocity: float = np.pi * 2
    
    def reconstruct_vertices(self, input_vertices: np.ndarray, rotations: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct vertex positions from base vertex and constrained rotations.
        
        This method applies the kinematic chain to reconstruct hand vertex positions
        while respecting the calculated bone lengths from the input vertices.
        This ensures consistent hand proportions while applying constraints.
        
        Args:
            input_vertices: Original vertex positions, shape (21, 3)
            rotations: List of constrained rotation matrices
            
        Returns:
            Reconstructed vertex positions, shape (21, 3)
        """
        vertices = np.zeros((21, 3))
        vertices[0] = input_vertices[0]  # Wrist position remains fixed
        
        # Calculate bone lengths from original vertices to maintain proportions
        bone_lengths: Dict[Tuple[int, int], float] = {}
        min_bone_length = 1e-6  # Minimum length to avoid numerical issues
        
        # Extract bone lengths from the kinematic chain
        for current in range(self.num_joints):
            mapping = self.joint_to_neighbors_mapping[current]
            current_vertex = mapping[0]
            child_vertex = mapping[1]
            parent_vertex = mapping[2]
            
            # Calculate bone length for current->child connection
            if child_vertex != -2:
                length = np.linalg.norm(input_vertices[child_vertex] - input_vertices[current_vertex])
                bone_lengths[(current_vertex, child_vertex)] = max(length, min_bone_length)
        
        # Reconstruct positions following the kinematic chain
        for current in range(self.num_joints):
            mapping = self.joint_to_neighbors_mapping[current]
            current_vertex = mapping[0]
            child_vertex = mapping[1]
            parent_vertex = mapping[2]

            if child_vertex == -2:
                continue
            
            # Get positions and rotation for this joint
            parent_pos = vertices[parent_vertex]
            current_pos = vertices[current_vertex]
            rotation = rotations[current]
            
            # Determine reference direction for rotation application
            if parent_vertex == -1:
                # Root joints use upward direction as reference
                prev_dir = np.array([0, 0, 1])
            else:
                # Use direction from parent to current vertex
                prev_dir = vertices[current_vertex] - vertices[parent_vertex]
                prev_dir = prev_dir / np.linalg.norm(prev_dir)
            
            # Apply rotation to get new direction
            current_dir = rotation @ prev_dir
            
            # Position child vertex using calculated bone length
            bone_length = bone_lengths[(current_vertex, child_vertex)]
            vertices[child_vertex] = current_pos + current_dir * bone_length

        return vertices

    def constrain_rotation(self, rotation_matrix: np.ndarray, joint_idx: int) -> np.ndarray:
        """
        Apply joint angle constraints to a rotation matrix.
        
        This method converts the rotation to Euler angles, clips them to the
        joint limits, and converts back to a rotation matrix. This ensures
        all joint angles remain within anatomically realistic ranges.
        
        Args:
            rotation_matrix: 3x3 rotation matrix to constrain
            joint_idx: Index of the joint for limit lookup
            
        Returns:
            Constrained 3x3 rotation matrix
        """
        try:
            # Convert rotation matrix to Euler angles
            rot = Rotation.from_matrix(rotation_matrix)
            euler = rot.as_euler('xyz')
            
            # Get joint limits for this joint
            limits = self.joint_limits[joint_idx]
            
            # Clip Euler angles to the specified limits
            constrained_euler = np.clip(euler, 
                                      [limits[0], limits[2], limits[4]],  # min limits
                                      [limits[1], limits[3], limits[5]])  # max limits
            
            # Convert back to rotation matrix if any clipping occurred
            if not np.allclose(euler, constrained_euler):
                return Rotation.from_euler('xyz', constrained_euler).as_matrix()
            return rotation_matrix
            
        except ValueError:
            logger.error("Error constraining rotation")
            # Return identity matrix if rotation is invalid
            return np.eye(3)
        
    def constrain_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Apply angular velocity constraints to limit motion speed.
        
        This method ensures that joint angular velocities don't exceed the
        maximum allowed velocity, preventing jerky or unrealistic motions.
        
        Args:
            velocity: Angular velocity vector to constrain
            
        Returns:
            Constrained angular velocity vector
        """
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.max_angular_velocity:
            # Scale velocity to maximum while preserving direction
            return velocity * (self.max_angular_velocity / velocity_magnitude)
        return velocity

    def add_frame(self, vertices: np.ndarray, timestamp: float, finger_pts: Any) -> None:
        """
        Add a new frame with physical constraints applied.
        
        This method extends the base add_frame functionality by applying
        joint limits, velocity constraints, and enhanced grasp calculations.
        The result is a more realistic and stable hand model suitable for
        robot control applications.
        
        Args:
            vertices: Hand keypoints, shape (21, 3)
            timestamp: Time of the frame in seconds
            finger_pts: Additional finger point data (currently unused)
        """
        # Calculate initial rotations from raw vertex positions
        rotations, rotations_xyz = self.calculate_frame_rotations(vertices)

        # Apply joint angle constraints to all rotations
        constrained_rotations: List[np.ndarray] = []
        for joint_idx, rotation in enumerate(rotations):
            constrained_rot = self.constrain_rotation(rotation, joint_idx)
            constrained_rotations.append(constrained_rot)
        
        # Apply velocity constraints if this is not the first frame
        if len(self.timestamps) > 0:
            dt = timestamp - self.timestamps[-1]
            for joint_idx in range(self.num_joints):
                # Calculate angular velocity for this joint
                prev_rot = Rotation.from_matrix(self.joint_rotations[-1][joint_idx])
                curr_rot = Rotation.from_matrix(constrained_rotations[joint_idx])
                rel_rot = curr_rot * prev_rot.inv()
                velocity = rel_rot.as_rotvec() / dt

                # Apply velocity constraint if needed
                if np.linalg.norm(velocity) > self.max_angular_velocity:
                    # Constrain velocity and reconstruct rotation
                    constrained_velocity = self.constrain_velocity(velocity)
                    delta_rot = Rotation.from_rotvec(constrained_velocity * dt)
                    new_rot = delta_rot * prev_rot
                    constrained_rotations[joint_idx] = new_rot.as_matrix()

        # Reconstruct vertices with constrained rotations
        constrained_vertices = self.reconstruct_vertices(vertices, constrained_rotations)

        # Extract key points for grasp calculation
        thumb_tip = constrained_vertices[4]
        index_tip = constrained_vertices[8]

        # Calculate grasp plane using thumb and index finger regions
        grasp_plane = self.calculate_grasp_plane(constrained_vertices[3:9])

        # Organize fingers for direction analysis
        n_fingers = len(constrained_vertices) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([constrained_vertices[0], constrained_vertices[i:i + npts_per_finger]]) 
                       for i in range(1, n_fingers, npts_per_finger)]

        # Calculate finger direction vector for plane orientation
        dir_vec = list_fingers[1][1] - list_fingers[-1][1]  # index to pinky MCP
        dir_vec = dir_vec / np.linalg.norm(dir_vec)

        # Ensure consistent plane orientation (normal pointing away from palm)
        if np.dot(dir_vec, grasp_plane[:3]) > 0:
            grasp_plane = -grasp_plane

        # Create slightly offset plane for grasp point calculation
        shifted_grasp_plane = self.get_parallel_plane(*grasp_plane, 0.01)  # only pour water 0_human is 0.1, others are 0.01 pick cube is 0.04
        grasp_pt = self.calculate_grasp_point(shifted_grasp_plane, constrained_vertices)

        # Calculate gripper orientation using the grasp plane
        gripper_ori, _ = HandModel.get_gripper_orientation(thumb_tip, index_tip, constrained_vertices, grasp_plane, target_hand=self.target_hand)

        # Apply coordinate frame transformations for robot compatibility
        rot_90_deg = Rotation.from_euler('Z', 90, degrees=True).as_matrix()
        grasp_ori = gripper_ori @ rot_90_deg

        # Apply pitch adjustment
        # angle = np.pi/18 * 1.2  # -20 degrees
        angle = np.pi/18 * 0.0  # - np.pi/18 * 2.0 for right hand cook task, 0 for left hand cook task
        # grasp_ori = Rotation.from_rotvec(angle * np.array([0, 1, 0])).apply(grasp_ori)
        grasp_ori = Rotation.from_rotvec(angle * np.array([1, 0, 0])).apply(grasp_ori)

        # Offset grasp point along gripper Z-axis for clearance
        unit_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        transformed_vectors = unit_vectors @ grasp_ori.T
        grasp_pt = grasp_pt - transformed_vectors[2] * 0.03
        '''
        -0.04 for only pour water 0_human
        -0.05 for others
        -0.03 for pick banana cosmos and cook
        + 0.005 for fold towl
        '''
        # Store all frame data
        self.joint_rotations.append(constrained_rotations)
        self.joint_rotations_xyz.append(rotations_xyz)
        self.vertex_positions.append(constrained_vertices)
        self.grasp_points.append(grasp_pt)
        self.grasp_oris.append(grasp_ori)
        self.timestamps.append(timestamp)


def get_list_finger_pts_from_skeleton(skeleton_pts: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Organize hand skeleton points into finger-specific groups.

    This utility function takes the 21-point hand skeleton and organizes
    it into a dictionary with separate arrays for each finger. This makes
    it easier to perform finger-specific calculations and analysis.

    Args:
        skeleton_pts: Hand skeleton points, shape (21, 3)
            Points are ordered as: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)

    Returns:
        Dictionary with finger names as keys and point arrays as values:
            - "thumb": Wrist + 4 thumb points, shape (5, 3)
            - "index": Wrist + 4 index points, shape (5, 3) 
            - "middle": Wrist + 4 middle points, shape (5, 3)
            - "ring": Wrist + 4 ring points, shape (5, 3)
            - "pinky": Wrist + 4 pinky points, shape (5, 3)
    """
    n_fingers = len(skeleton_pts) - 1  # Exclude wrist point
    npts_per_finger = 4  # MCP, PIP, DIP, TIP for each finger

    # Create finger arrays by combining wrist with each finger's points
    list_fingers = [
        np.vstack([skeleton_pts[0], skeleton_pts[i : i + npts_per_finger]])
        for i in range(1, n_fingers, npts_per_finger)
    ]

    # Return organized finger dictionary
    return {
        "thumb": list_fingers[0],
        "index": list_fingers[1],
        "middle": list_fingers[2],
        "ring": list_fingers[3],
        "pinky": list_fingers[4]
    }