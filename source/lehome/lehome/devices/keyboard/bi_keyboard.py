import weakref
import numpy as np

from collections.abc import Callable
from pynput.keyboard import Listener

import carb
import omni

from ..device_base import Device


class BiKeyboard(Device):
    """A keyboard controller for sending SE(3) commands for bi-arm lerobot.

    Key bindings:
        Left Arm (字母键):
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Joint 1 (shoulder_pan)         T                 G
        Joint 2 (shoulder_lift)        Y                 H
        Joint 3 (elbow_flex)           U                 J
        Joint 4 (wrist_flex)           I                 K
        Joint 5 (wrist_roll)           O                 L
        Joint 6 (gripper)              Q                 A

        Right Arm (数字键):
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Joint 1 (shoulder_pan)         1                 7
        Joint 2 (shoulder_lift)        2                 8
        Joint 3 (elbow_flex)           3                 9
        Joint 4 (wrist_flex)           4                 0
        Joint 5 (wrist_roll)           5                 MINUS
        Joint 6 (gripper)              6                 EQUALS
        ============================== ================= =================
    """

    def __init__(self, env, sensitivity: float = 0.05):
        super().__init__(env)
        """Initialize the bi-keyboard layer.
        """
        # store inputs
        self.sensitivity = sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )
        # bindings for keyboard to command
        self._create_key_bindings()

        # command buffers for left and right arms
        self._left_delta_pos = np.zeros(6)
        self._right_delta_pos = np.zeros(6)

        # some flags and callbacks
        self.started = False
        self._reset_state = 0
        self._additional_callbacks = {}

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of bi-keyboard."""
        msg = "Bi-Keyboard Controller for SE(3).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft Arm (字母键):\n"
        msg += "\t  Joint 1 (shoulder_pan):  T/G\n"
        msg += "\t  Joint 2 (shoulder_lift): Y/H\n"
        msg += "\t  Joint 3 (elbow_flex):    U/J\n"
        msg += "\t  Joint 4 (wrist_flex):    I/K\n"
        msg += "\t  Joint 5 (wrist_roll):    O/L\n"
        msg += "\t  Joint 6 (gripper):       Q/A\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight Arm (数字键):\n"
        msg += "\t  Joint 1 (shoulder_pan):  1/7\n"
        msg += "\t  Joint 2 (shoulder_lift): 2/8\n"
        msg += "\t  Joint 3 (elbow_flex):    3/9\n"
        msg += "\t  Joint 4 (wrist_flex):    4/0\n"
        msg += "\t  Joint 5 (wrist_roll):    5/-\n"
        msg += "\t  Joint 6 (gripper):       6/=\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tStart Control: B\n"
        msg += "\tTask Failed and Reset: R\n"
        msg += "\tTask Success and Reset: N\n"
        msg += "\tControl+C: quit"
        return msg

    def on_press(self, key):
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            if key.char == "b":
                self.started = True
                self._reset_state = False
            elif key.char == "r":
                self.started = False
                self._reset_state = True
                self._additional_callbacks["R"]()
            elif key.char == "n":
                self.started = False
                self._reset_state = True
                self._additional_callbacks["N"]()
        except AttributeError:
            pass

    def get_device_state(self):
        return {
            "left_arm": self._left_delta_pos.copy(),
            "right_arm": self._right_delta_pos.copy(),
        }

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state["started"] = self.started
        if reset:
            self._reset_state = False
            return state
        state["joint_state"] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict["started"] = self.started
        ac_dict["bi_keyboard"] = True
        if reset:
            return ac_dict
        ac_dict["joint_state"] = state["joint_state"]
        return ac_dict

    def reset(self):
        self._left_delta_pos = np.zeros(6)
        self._right_delta_pos = np.zeros(6)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        # 安全获取按键名称
        try:
            if isinstance(event.input, str):
                key_name = event.input
            else:
                key_name = event.input.name
        except AttributeError:
            return True

        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in self._LEFT_KEY_MAPPING.keys():
                self._left_delta_pos += self._LEFT_KEY_MAPPING[key_name]
            elif key_name in self._RIGHT_KEY_MAPPING.keys():
                self._right_delta_pos += self._RIGHT_KEY_MAPPING[key_name]
        # remove the command when un-pressed
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in self._LEFT_KEY_MAPPING.keys():
                self._left_delta_pos -= self._LEFT_KEY_MAPPING[key_name]
            elif key_name in self._RIGHT_KEY_MAPPING.keys():
                self._right_delta_pos -= self._RIGHT_KEY_MAPPING[key_name]
        return True

    def _create_key_bindings(self):
        """Creates key bindings for left and right arms."""
        # Left arm (字母键)
        self._LEFT_KEY_MAPPING = {
            "T": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "Y": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "U": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "I": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.sensitivity,
            "O": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.sensitivity,
            "Q": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.sensitivity,
            "G": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "H": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "J": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "K": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.sensitivity,
            "L": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.sensitivity,
            "A": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.sensitivity,
        }

        # Right arm (数字键) - 主键盘数字键使用 KEY_X 格式
        self._RIGHT_KEY_MAPPING = {
            "KEY_1": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_2": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_3": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_4": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_5": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.sensitivity,
            "KEY_6": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.sensitivity,
            "KEY_7": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_8": np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_9": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.sensitivity,
            "KEY_0": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.sensitivity,
            # 减号和等号键，支持多种可能的格式
            "KEY_MINUS": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.sensitivity,
            "KEY_EQUALS": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            * self.sensitivity,
            "MINUS": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.sensitivity,
            "EQUALS": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.sensitivity,
        }
