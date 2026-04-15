# source/lehome/lehome/devices/hybrid/xlerobot_hybrid_controller.py

import weakref
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device
from ..lerobot import BiXlerobotLeader
from ..keyboard.xlerobot_keyboard import XlerobotKeyboard


class XlerobotHybridController(Device):
    """Xlerobot Hybrid Controller: Keyboard Control the Base and Head, BiSO101Leader Control the Arms"""

    def __init__(
        self,
        env,
        sensitivity: float = 1.0,
        left_arm_port: str = "/dev/ttyACM0",
        right_arm_port: str = "/dev/ttyACM1",
        recalibrate: bool = False,
    ):
        super().__init__(env)

        # Create a keyboard controller (only for chassis and head control)
        self.keyboard_controller = XlerobotKeyboard(env, sensitivity)

        # Create a dual-arm controller.
        self.bi_arm_controller = BiXlerobotLeader(
            env,
            left_port=left_arm_port,
            right_port=right_arm_port,
            recalibrate=recalibrate,
        )

        # Control mode flag.
        self.control_mode = "hybrid"  # "keyboard", "hybrid", "arms_only"

        # State flag.
        self.started = False
        self._reset_state = False

    def set_control_mode(self, mode: str):
        """Set the control mode."""
        assert mode in [
            "keyboard",
            "hybrid",
            "arms_only",
        ], f"Invalid control mode: {mode}"
        self.control_mode = mode
        print(f"Control mode switched to: {mode}")

    def get_device_state(self):
        """Get the device state."""
        if self.control_mode == "keyboard":
            state = self.keyboard_controller.get_device_state()
            return state
        elif self.control_mode == "arms_only":
            arms_state = self.bi_arm_controller.get_device_state()
            # Cache arms_action to avoid repeated calls.
            if not hasattr(self, "_cached_arms_action"):
                self._cached_arms_action = self.bi_arm_controller.input2action()

            full_state = np.zeros(17)  # Change to 17 dimensions.

            # Left arm control (index 3-8).
            if "left_arm" in arms_state and arms_state["left_arm"] is not None:
                left_arm_data = arms_state["left_arm"]
                if isinstance(left_arm_data, dict):
                    left_motor_limits = self._cached_arms_action.get(
                        "motor_limits", {}
                    ).get("left_arm", {})
                    left_processed = self._convert_arm_action(
                        left_arm_data, left_motor_limits
                    )
                    full_state[3:9] = left_processed

            # Right arm control (index 9-14).
            if "right_arm" in arms_state and arms_state["right_arm"] is not None:
                right_arm_data = arms_state["right_arm"]
                if isinstance(right_arm_data, dict):
                    right_motor_limits = self._cached_arms_action.get(
                        "motor_limits", {}
                    ).get("right_arm", {})
                    right_processed = self._convert_arm_action(
                        right_arm_data, right_motor_limits
                    )
                    full_state[9:15] = right_processed

            return full_state
        else:  # hybrid mode
            keyboard_state = self.keyboard_controller.get_device_state()
            arms_state = self.bi_arm_controller.get_device_state()

            # Cache arms_action to avoid repeated calls.
            if not hasattr(self, "_cached_arms_action"):
                self._cached_arms_action = self.bi_arm_controller.input2action()

            hybrid_state = np.zeros(17)  # Change to 17 dimensions.

            # Chassis control (index 0-2): use keyboard control.
            hybrid_state[0:3] = keyboard_state[0:3]

            # Left arm control (index 3-8): use dual-arm controller.
            if "left_arm" in arms_state and arms_state["left_arm"] is not None:
                left_arm_data = arms_state["left_arm"]
                if isinstance(left_arm_data, dict):
                    left_motor_limits = self._cached_arms_action.get(
                        "motor_limits", {}
                    ).get("left_arm", {})
                    left_processed = self._convert_arm_action(
                        left_arm_data, left_motor_limits
                    )
                    hybrid_state[3:9] = left_processed

            # Right arm control (index 9-14): use dual-arm controller.
            if "right_arm" in arms_state and arms_state["right_arm"] is not None:
                right_arm_data = arms_state["right_arm"]
                if isinstance(right_arm_data, dict):
                    right_motor_limits = self._cached_arms_action.get(
                        "motor_limits", {}
                    ).get("right_arm", {})
                    right_processed = self._convert_arm_action(
                        right_arm_data, right_motor_limits
                    )
                    hybrid_state[9:15] = right_processed

            # Head control (index 15-16): use keyboard control.
            hybrid_state[15:17] = keyboard_state[15:17]

            return hybrid_state

    def input2action(self):
        """Convert input to action."""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            return action
        elif self.control_mode == "arms_only":
            action = self.bi_arm_controller.input2action()
            self.started = action.get("started", False)
            # Update cache.
            self._cached_arms_action = action
            return action
        else:  # hybrid mode
            keyboard_action = self.keyboard_controller.input2action()
            arms_action = self.bi_arm_controller.input2action()

            # Set started state.
            self.started = arms_action.get("started", False)

            # Update cache.
            self._cached_arms_action = arms_action

            hybrid_action = {
                "reset": keyboard_action.get("reset", False)
                or arms_action.get("reset", False),
                "started": arms_action.get("started", False),
                "hybrid_controller": True,
                "keyboard": True,
                "bi_so101_leader": True,
                "joint_state": self.get_device_state(),
                "motor_limits": arms_action.get("motor_limits", {}),
            }

            return hybrid_action

    def advance(self):
        """Get the current action."""

        if not self.started:
            return None

        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def reset(self):
        """Reset the controller state."""
        self.keyboard_controller.reset()
        self.bi_arm_controller.reset()
        self.started = False
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Add a callback function."""
        # Keyboard callback for control mode switching
        if key == "F6":  # F6 key to switch control mode

            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arms_only")
                else:
                    self.set_control_mode("hybrid")

            self.keyboard_controller.add_callback(key, toggle_mode)
        else:
            # Other callbacks passed to the keyboard controller
            self.keyboard_controller.add_callback(key, func)

            # For B key, also pass to BiSO101Leader to start dual-arm control
            if key == "B":
                # Directly pass B key callback to BiSO101Leader
                self.bi_arm_controller.add_callback(key, func)
            # Other keys not passed to BiSO101Leader, to avoid unnecessary key processing

    def __str__(self) -> str:
        """Return the controller information."""
        msg = "Xlerobot Hybrid Controller\n"
        msg += f"\tCurrent control mode: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tKeyboard control: chassis movement(W/A/S/D) + head movement(Home/End/PageUp/PageDown)\n"
        msg += "\tDual-arm controller: arm and gripper control\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: switch control mode (hybrid <-> keyboard <-> arms_only)\n"
        msg += "\tB: start control\n"
        msg += "\tF5: reset environment\n"
        msg += "\tExit: Ctrl+C\n"
        return msg

    def _convert_arm_action(self, joint_state: dict, motor_limits: dict) -> np.ndarray:
        """Convert single arm action, using the same conversion logic as the original BiSO101Leader"""
        processed_action = np.zeros(6)

        # If there is no motor_limits, return zero action
        if not motor_limits:
            print("Warning: no motor_limits found, returning zero action")
            return processed_action

        # Use the same joint limits as the original code (angle)
        from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS

        # Joint name to index mapping
        joint_mapping = {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_roll": 4,
            "gripper": 5,
        }

        # Joint direction correction configuration (if a joint direction is opposite, set to -1)
        joint_direction_correction = {
            "shoulder_pan": 1,  # 1 represents normal direction, -1 represents opposite direction
            "shoulder_lift": -1,
            "elbow_flex": 1,
            "wrist_flex": 1,
            "wrist_roll": -1,
            "gripper": 1,
        }

        # Joint zero position offset correction (radians)
        # Positive value means the simulated joint needs to be offset in the positive direction, negative value means offset in the negative direction
        joint_zero_offset = {
            "shoulder_pan": 0.0,  # Shoulder rotation
            "shoulder_lift": 1.57,  # Shoulder lift: offset -90 degrees (-π/2)
            "elbow_flex": 1.57,  # Elbow flex: offset -90 degrees (-π/2)
            "wrist_flex": 0.0,  # Wrist flex
            "wrist_roll": 0.0,  # Wrist roll
            "gripper": 0.0,  # Gripper
        }

        for joint_name, index in joint_mapping.items():
            if joint_name in joint_state and joint_name in motor_limits:
                motor_limit_range = motor_limits[joint_name]
                joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]

                # Map motor range to joint range (angle)
                processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (
                    motor_limit_range[1] - motor_limit_range[0]
                ) * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                processed_radius = processed_degree / 180.0 * np.pi  # Convert to radians

                # Apply direction correction
                direction_correction = joint_direction_correction.get(joint_name, 1)
                processed_radius = processed_radius * direction_correction

                # Apply zero position offset correction
                zero_offset = joint_zero_offset.get(joint_name, 0.0)
                processed_action[index] = processed_radius + zero_offset

        return processed_action
