import weakref
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device


class XlerobotKeyboard(Device):
    """Xlerobot dedicated keyboard controller, supports mobile chassis and dual-arm control."""

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env)
        self.sensitivity = sensitivity
        
        # Obtaining the Omniverse interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        
        # Keyboard event subscription
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): 
                obj._on_keyboard_event(event, *args),
        )
        
        # Create keyboard bindings
        self._create_key_bindings()
        
        # Command buffer - modified to speed control
        self._base_velocity = np.zeros(3)  # 3 chassis joints speed: x, y, rotation
        self._left_arm_delta = np.zeros(5)     # 5 joints
        self._right_arm_delta = np.zeros(5)    # 5 joints
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta = np.zeros(2)         # pan, tilt
        
        # Flags and callbacks
        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}
        
        # Key state tracking
        self._pressed_keys = set()

    def _create_key_bindings(self):
        """Create keyboard bindings - modified to joint control"""
        self._key_bindings = {
            # Chassis joint control - modified to joint control
            "W": ("base_joint", 0, 1.0),   # Forward (root_x_axis_joint)
            "S": ("base_joint", 0, -1.0),  # Backward (root_x_axis_joint)
            "A": ("base_joint", 1, 1.0),   # Left (root_y_axis_joint)
            "D": ("base_joint", 1, -1.0),  # Right (root_y_axis_joint)
            "Q": ("base_joint", 2, 1.0),   # Left turn (root_z_rotation_joint)
            "E": ("base_joint", 2, -1.0),  # Right turn (root_z_rotation_joint)
            
            # Left arm control
            "R": ("left_arm", 0, 1.0),      # left arm joint 1
            "F": ("left_arm", 0, -1.0),
            "T": ("left_arm", 1, 1.0),      # left arm joint 2
            "G": ("left_arm", 1, -1.0),
            "Y": ("left_arm", 2, 1.0),      # left arm joint 3
            "H": ("left_arm", 2, -1.0),
            "U": ("left_arm", 3, 1.0),      # left arm joint 4
            "J": ("left_arm", 3, -1.0),
            "I": ("left_arm", 4, 1.0),      # left arm joint 5
            "K": ("left_arm", 4, -1.0),
            
            # Right arm control
            "NUMPAD_8": ("right_arm", 0, 1.0),   # right arm joint 1
            "NUMPAD_2": ("right_arm", 0, -1.0),
            "NUMPAD_4": ("right_arm", 1, 1.0),   # right arm joint 2
            "NUMPAD_6": ("right_arm", 1, -1.0),
            "NUMPAD_7": ("right_arm", 2, 1.0),   # right arm joint 3
            "NUMPAD_9": ("right_arm", 2, -1.0),
            "NUMPAD_1": ("right_arm", 3, 1.0),   # right arm joint 4
            "NUMPAD_3": ("right_arm", 3, -1.0),
            "NUMPAD_0": ("right_arm", 4, 1.0),   # right arm joint 5
            "NUMPAD_PERIOD": ("right_arm", 4, -1.0),
            
            # Gripper control
            "F1": ("left_gripper", 0, 1.0),   # left gripper open
            "F2": ("left_gripper", 0, -1.0),  # left gripper close
            "F3": ("right_gripper", 0, 1.0),   # right gripper open
            "F4": ("right_gripper", 0, -1.0),  # right gripper close
            
            # Head control
            "HOME": ("head", 0, 1.0),   # head left turn
            "END": ("head", 0, -1.0),   # head right turn
            "PAGE_UP": ("head", 1, 1.0),     # head up
            "PAGE_DOWN": ("head", 1, -1.0),  # head down
            
            # Control keys
            "B": ("control", "start", 1.0),   # Start control
            "N": ("control", "success", 1.0), # Task success
            "F5": ("control", "reset", 1.0),  # Reset environment
        }

    def get_device_state(self):
        """Get device state - using speed control"""
        action = np.zeros(17)  # 17 dimensions, including head control
        
        # Chassis speed control (3 joints)
        action[0] = self._base_velocity[0] * self.sensitivity  # W/S -> root_x_axis_joint speed
        action[1] = self._base_velocity[1] * self.sensitivity  # A/D -> root_y_axis_joint speed
        action[2] = self._base_velocity[2] * self.sensitivity  # Q/E -> root_z_rotation_joint speed
        
        # Left arm joint control (5 joints)
        action[3] = self._left_arm_delta[0] * self.sensitivity  # R/F -> left arm joint 1
        action[4] = self._left_arm_delta[1] * self.sensitivity  # T/G -> left arm joint 2
        action[5] = self._left_arm_delta[2] * self.sensitivity  # Y/H -> left arm joint 3
        action[6] = self._left_arm_delta[3] * self.sensitivity  # U/J -> left arm joint 4
        action[7] = self._left_arm_delta[4] * self.sensitivity  # I/K -> left arm joint 5
        
        # Left gripper control
        action[8] = self._left_gripper_delta * self.sensitivity  # F1/F2 -> left gripper
        
        # right arm joint control (5 joints)
        action[9] = self._right_arm_delta[0] * self.sensitivity   # NUMPAD_8/2 -> right arm joint 1
        action[10] = self._right_arm_delta[1] * self.sensitivity   # NUMPAD_4/6 -> right arm joint 2
        action[11] = self._right_arm_delta[2] * self.sensitivity   # NUMPAD_7/9 -> right arm joint 3
        action[12] = self._right_arm_delta[3] * self.sensitivity  # NUMPAD_1/3 -> right arm joint 4
        action[13] = self._right_arm_delta[4] * self.sensitivity  # NUMPAD_0/. -> right arm joint 5
        
        # right gripper control
        action[14] = self._right_gripper_delta * self.sensitivity  # F3/F4 -> right gripper
        
        # head control (2 joints)
        action[15] = self._head_delta[0] * self.sensitivity  # Home/End -> head horizontal
        action[16] = self._head_delta[1] * self.sensitivity  # PageUp/PageDown -> head vertical
        
        return action

    def input2action(self):
        """Convert input to action"""
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict['keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def advance(self):
        """Get current action - compatible with Device interface"""
        if not self.started:
            return None
            
        # return the tensor on the correct device
        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def reset(self):
        """Reset controller state"""
        self._base_velocity.fill(0)  # reset speed
        self._left_arm_delta.fill(0)
        self._right_arm_delta.fill(0)
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta.fill(0)
        self._pressed_keys.clear()

    def add_callback(self, key: str, func: Callable):
        """Add callback function"""
        self._additional_callbacks[key] = func

    def __del__(self):
        """Release keyboard interface"""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """Process keyboard event - modified to speed control"""
        try:
            # get key name
            if hasattr(event, 'input') and hasattr(event.input, 'name'):
                key_name = event.input.name
            elif hasattr(event, 'name'):
                key_name = event.name
            else:
                return
            
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]
                    
                    # process control keys
                    if control_type == "control":
                        if index == "start":
                            self.started = True
                            self._reset_state = False
                            print("Xlerobot control started!")
                        elif index == "success":
                            self.started = False
                            self._reset_state = True
                            if "N" in self._additional_callbacks:
                                self._additional_callbacks["N"]()
                        elif index == "reset":
                            self._reset_state = True
                            self.started = False
                            self.reset()
                            if "R" in self._additional_callbacks:
                                self._additional_callbacks["R"]()
                        return
                    
                    # Handling joint control keys
                    if control_type == "base_joint":
                        self._base_velocity[index] = value  # Set speed value
                    elif control_type == "left_arm":
                        self._left_arm_delta[index] = value
                    elif control_type == "right_arm":
                        self._right_arm_delta[index] = value
                    elif control_type == "left_gripper":
                        self._left_gripper_delta = value
                    elif control_type == "right_gripper":
                        self._right_gripper_delta = value
                    elif control_type == "head":
                        self._head_delta[index] = value
                    
                    self._pressed_keys.add(key_name)
                   
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]
                    
                    # Only handle the release of robot control keys, not the control keys themselves.
                    if control_type == "base_joint":
                        self._base_velocity[index] = 0.0  # Release speed set to 0
                    elif control_type == "left_arm":
                        self._left_arm_delta[index] = 0.0
                    elif control_type == "right_arm":
                        self._right_arm_delta[index] = 0.0
                    elif control_type == "left_gripper":
                        self._left_gripper_delta = 0.0
                    elif control_type == "right_gripper":
                        self._right_gripper_delta = 0.0
                    elif control_type == "head":
                        self._head_delta[index] = 0.0
                    
                    self._pressed_keys.discard(key_name)
                    
        except Exception as e:
            print(f"Keyboard event processing error: {e}")

    def __str__(self) -> str:
        """Return controller information"""
        msg = "Xlerobot keyboard controller (direct position control mode)\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += f"\tSensitivity: {self.sensitivity}\n"
        msg += f"\tControl state: {'Started' if self.started else 'Not started'}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tDirect position control:\n"
        msg += "\t  Forward/Backward: W/S (X-axis movement)\n"
        msg += "\t  Left/Right: A/D (Y-axis movement)\n"
        msg += "\t  Left/Right turn: Q/E (Z-axis rotation)\n"
        msg += "\tLeft arm control:\n"
        msg += "\t  Joint 1: R/F\n"
        msg += "\t  Joint 2: T/G\n"
        msg += "\t  Joint 3: Y/H\n"
        msg += "\t  Joint 4: U/J\n"
        msg += "\t  Joint 5: I/K\n"
        msg += "\tRight arm control (number keyboard):\n"
        msg += "\t  Joint 1: 8/2\n"
        msg += "\t  Joint 2: 4/6\n"
        msg += "\t  Joint 3: 7/9\n"
        msg += "\t  Joint 4: 1/3\n"
        msg += "\t  Joint 5: 0/.\n"
        msg += "\tGripper control:\n"
        msg += "\t  Left gripper: F1/F2 (open/close)\n"
        msg += "\t  Right gripper: F3/F4 (open/close)\n"
        msg += "\tHead control:\n"
        msg += "\t  Horizontal: Home/End (left/right)\n"
        msg += "\t  Vertical: PageUp/PageDown (up/down)\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart control: B\n"
        msg += "\tTask success: N\n"
        msg += "\tReset environment: F5\n"
        msg += f"\tThe key currently pressed: {list(self._pressed_keys)}"
        return msg