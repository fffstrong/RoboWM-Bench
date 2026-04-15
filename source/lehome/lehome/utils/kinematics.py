# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class RobotKinematics:
    """基于 Pinocchio 的前向/逆向运动学工具（用于 lehome 项目）。"""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        初始化 Pinocchio 运动学求解器。

        Args:
            urdf_path: 机器人 URDF 文件路径
            target_frame_name: 末端执行器 frame 名称
            joint_names: 参与求解的关节名列表；None 则使用模型中的所有可动关节
        """
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.joint_names = joint_names
        self.backend = "pinocchio"

        self._init_pinocchio()
    
    def _init_pinocchio(self):
        """Initialize Pinocchio-based kinematics solver."""
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "Pinocchio is required but not available. "
                "Please install it with: pip install pin"
            )
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "scipy is required for inverse kinematics with Pinocchio. "
                "Please install it with: pip install scipy"
            )
        
        # Load URDF model
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        
        # Get end-effector frame ID
        # Pinocchio getFrameId may return an out-of-range sentinel when the name is missing
        # (instead of raising), which would break oMf[fid]. Always validate fid < nframes.
        self.ee_frame_id = None
        try:
            _cand = self.model.getFrameId(self.target_frame_name)
            if _cand < self.model.nframes:
                self.ee_frame_id = _cand
        except Exception:
            pass

        if self.ee_frame_id is None:
            frame_found = False
            for i in range(self.model.nframes):
                frame_name = self.model.frames[i].name
                if (
                    frame_name == self.target_frame_name
                    or (
                        self.target_frame_name
                        and (
                            self.target_frame_name in frame_name
                            or frame_name in self.target_frame_name
                        )
                    )
                ):
                    self.ee_frame_id = i
                    frame_found = True
                    break
            if not frame_found:
                for alt in (
                    "panda_panda_hand",
                    "panda_link8",
                    "panda_hand",
                    "panda_end_effector",
                    "tool0",
                    "ee_link",
                    "gripper_frame_link",
                ):
                    try:
                        _cand = self.model.getFrameId(alt)
                        if _cand < self.model.nframes:
                            self.ee_frame_id = _cand
                            frame_found = True
                            break
                    except Exception:
                        continue
            if not frame_found:
                raise ValueError(
                    f"Frame '{self.target_frame_name}' not found in URDF. "
                    f"Available frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}"
                )
        
        # Set joint names
        if self.joint_names is None:
            # Get all revolute and prismatic joints
            self.joint_names = []
            for i in range(1, self.model.njoints):  # Skip root joint
                joint_name = self.model.names[i]
                joint_model = self.model.joints[i]
                if joint_model.nq > 0:  # Has configuration space
                    self.joint_names.append(joint_name)
        else:
            self.joint_names = self.joint_names
        
        # Get joint indices and configuration indices
        self.joint_indices = []
        self.joint_q_indices = []  # Indices in the configuration vector
        for joint_name in self.joint_names:
            try:
                joint_id = self.model.getJointId(joint_name)
                self.joint_indices.append(joint_id)
                joint_model = self.model.joints[joint_id]
                if joint_model.nq > 0:
                    self.joint_q_indices.append(joint_model.idx_q)
            except Exception:
                raise ValueError(f"Joint '{joint_name}' not found in URDF model")
        
        self.nq = len(self.joint_indices)
        
        # Store minimize function for IK
        self._minimize = minimize

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics (Pinocchio) for given joint configuration.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        return self._forward_kinematics_pinocchio(joint_pos_deg)
    
    def _forward_kinematics_pinocchio(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Forward kinematics using Pinocchio."""
        import pinocchio as pin
        
        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: self.nq])
        
        # Create full configuration vector (including fixed joints)
        q = pin.neutral(self.model)
        for i, q_idx in enumerate(self.joint_q_indices):
            q[q_idx] = joint_pos_rad[i]
        
        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector pose (4x4 homogeneous matrix)
        ee_pose = self.data.oMf[self.ee_frame_id]
        return ee_pose.homogeneous

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using Pinocchio + scipy.optimize.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        return self._inverse_kinematics_pinocchio(
            current_joint_pos, desired_ee_pose, position_weight, orientation_weight
        )
    
    def _inverse_kinematics_pinocchio(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float,
        orientation_weight: float,
    ) -> np.ndarray:
        """Inverse kinematics using Pinocchio with scipy.optimize."""
        import pinocchio as pin
        
        # Target pose as Pinocchio SE3
        target_pose = pin.SE3(desired_ee_pose[:3, :3], desired_ee_pose[:3, 3])
        
        # Initial guess (convert to radians)
        current_joint_rad = np.deg2rad(current_joint_pos[: self.nq])
        q0_controlled = np.array(current_joint_rad)
        
        # Optimization objective function
        def objective(q_controlled):
            # Reconstruct full configuration
            q = pin.neutral(self.model)
            for i, q_idx in enumerate(self.joint_q_indices):
                q[q_idx] = q_controlled[i]
            
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_pose = self.data.oMf[self.ee_frame_id]
            
            # Position error
            pos_error = target_pose.translation - current_pose.translation
            pos_cost = position_weight * np.sum(pos_error ** 2)
            
            # Orientation error (logarithmic map)
            if orientation_weight > 0:
                rot_error = pin.log3(target_pose.rotation.T @ current_pose.rotation)
                rot_cost = orientation_weight * np.sum(rot_error ** 2)
            else:
                rot_cost = 0.0
            
            return pos_cost + rot_cost
        
        # Joint limits (bounds)
        bounds = []
        for q_idx in self.joint_q_indices:
            # Get joint limits from model
            lower = self.model.lowerPositionLimit[q_idx]
            upper = self.model.upperPositionLimit[q_idx]
            # If limits are infinite, use reasonable defaults
            if not np.isfinite(lower):
                lower = -np.pi
            if not np.isfinite(upper):
                upper = np.pi
            bounds.append((lower, upper))
        
        # Solve IK using optimization
        result = self._minimize(
            objective,
            q0_controlled,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        # Convert result to degrees
        joint_pos_deg = np.rad2deg(result.x)
        
        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > self.nq:
            result_full = np.zeros_like(current_joint_pos)
            result_full[: self.nq] = joint_pos_deg
            result_full[self.nq :] = current_joint_pos[self.nq :]
            return result_full
        else:
            return joint_pos_deg
        
