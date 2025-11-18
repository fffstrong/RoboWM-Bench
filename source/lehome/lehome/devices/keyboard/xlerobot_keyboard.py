import weakref
import numpy as np
import torch
from collections.abc import Callable

import carb
import omni
from ..device_base import Device


class XlerobotKeyboard(Device):
    """Xlerobot专用键盘控制器，支持移动底盘和双臂控制"""

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env)
        self.sensitivity = sensitivity
        
        # 获取Omniverse接口
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        
        # 键盘事件订阅
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): 
                obj._on_keyboard_event(event, *args),
        )
        
        # 创建键盘绑定
        self._create_key_bindings()
        
        # 命令缓冲区 - 修改为速度控制
        self._base_velocity = np.zeros(3)  # 3个底盘关节的速度：x, y, rotation
        self._left_arm_delta = np.zeros(5)     # 5个关节
        self._right_arm_delta = np.zeros(5)    # 5个关节
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta = np.zeros(2)         # pan, tilt
        
        # 标志和回调
        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}
        
        # 按键状态跟踪
        self._pressed_keys = set()

    def _create_key_bindings(self):
        """创建键盘绑定 - 修改为关节控制"""
        self._key_bindings = {
            # 底盘关节控制 - 改为关节控制
            "W": ("base_joint", 0, 1.0),   # 前进 (root_x_axis_joint)
            "S": ("base_joint", 0, -1.0),  # 后退 (root_x_axis_joint)
            "A": ("base_joint", 1, 1.0),   # 左移 (root_y_axis_joint)
            "D": ("base_joint", 1, -1.0),  # 右移 (root_y_axis_joint)
            "Q": ("base_joint", 2, 1.0),   # 左转 (root_z_rotation_joint)
            "E": ("base_joint", 2, -1.0),  # 右转 (root_z_rotation_joint)
            
            # 左臂控制
            "R": ("left_arm", 0, 1.0),      # 左臂关节1
            "F": ("left_arm", 0, -1.0),
            "T": ("left_arm", 1, 1.0),      # 左臂关节2
            "G": ("left_arm", 1, -1.0),
            "Y": ("left_arm", 2, 1.0),      # 左臂关节3
            "H": ("left_arm", 2, -1.0),
            "U": ("left_arm", 3, 1.0),      # 左臂关节4
            "J": ("left_arm", 3, -1.0),
            "I": ("left_arm", 4, 1.0),      # 左臂关节5
            "K": ("left_arm", 4, -1.0),
            
            # 右臂控制
            "NUMPAD_8": ("right_arm", 0, 1.0),   # 右臂关节1
            "NUMPAD_2": ("right_arm", 0, -1.0),
            "NUMPAD_4": ("right_arm", 1, 1.0),   # 右臂关节2
            "NUMPAD_6": ("right_arm", 1, -1.0),
            "NUMPAD_7": ("right_arm", 2, 1.0),   # 右臂关节3
            "NUMPAD_9": ("right_arm", 2, -1.0),
            "NUMPAD_1": ("right_arm", 3, 1.0),   # 右臂关节4
            "NUMPAD_3": ("right_arm", 3, -1.0),
            "NUMPAD_0": ("right_arm", 4, 1.0),   # 右臂关节5
            "NUMPAD_PERIOD": ("right_arm", 4, -1.0),
            
            # 夹爪控制
            "F1": ("left_gripper", 0, 1.0),   # 左夹爪开
            "F2": ("left_gripper", 0, -1.0),  # 左夹爪关
            "F3": ("right_gripper", 0, 1.0),   # 右夹爪开
            "F4": ("right_gripper", 0, -1.0),  # 右夹爪关
            
            # 头部控制
            "HOME": ("head", 0, 1.0),   # 头部左转
            "END": ("head", 0, -1.0),   # 头部右转
            "PAGE_UP": ("head", 1, 1.0),     # 头部上仰
            "PAGE_DOWN": ("head", 1, -1.0),  # 头部下俯
            
            # 控制键
            "B": ("control", "start", 1.0),   # 启动控制
            "N": ("control", "success", 1.0), # 任务成功
            "F5": ("control", "reset", 1.0),  # 重置环境
        }

    def get_device_state(self):
        """获取设备状态 - 使用速度控制"""
        action = np.zeros(17)  # 17维，包含头部控制
        
        # 底盘速度控制 (3个关节)
        action[0] = self._base_velocity[0] * self.sensitivity  # W/S -> root_x_axis_joint 速度
        action[1] = self._base_velocity[1] * self.sensitivity  # A/D -> root_y_axis_joint 速度
        action[2] = self._base_velocity[2] * self.sensitivity  # Q/E -> root_z_rotation_joint 速度
        
        # 左臂关节控制 (5个关节)
        action[3] = self._left_arm_delta[0] * self.sensitivity  # R/F -> 左臂关节1
        action[4] = self._left_arm_delta[1] * self.sensitivity  # T/G -> 左臂关节2
        action[5] = self._left_arm_delta[2] * self.sensitivity  # Y/H -> 左臂关节3
        action[6] = self._left_arm_delta[3] * self.sensitivity  # U/J -> 左臂关节4
        action[7] = self._left_arm_delta[4] * self.sensitivity  # I/K -> 左臂关节5
        
        # 左夹爪控制
        action[8] = self._left_gripper_delta * self.sensitivity  # F1/F2 -> 左夹爪
        
        # 右臂关节控制 (5个关节)
        action[9] = self._right_arm_delta[0] * self.sensitivity   # 数字键盘8/2 -> 右臂关节1
        action[10] = self._right_arm_delta[1] * self.sensitivity   # 数字键盘4/6 -> 右臂关节2
        action[11] = self._right_arm_delta[2] * self.sensitivity   # 数字键盘7/9 -> 右臂关节3
        action[12] = self._right_arm_delta[3] * self.sensitivity  # 数字键盘1/3 -> 右臂关节4
        action[13] = self._right_arm_delta[4] * self.sensitivity  # 数字键盘0/. -> 右臂关节5
        
        # 右夹爪控制
        action[14] = self._right_gripper_delta * self.sensitivity  # F3/F4 -> 右夹爪
        
        # 头部控制 (2个关节)
        action[15] = self._head_delta[0] * self.sensitivity  # Home/End -> 头部水平
        action[16] = self._head_delta[1] * self.sensitivity  # PageUp/PageDown -> 头部垂直
        
        return action

    def input2action(self):
        """将输入转换为动作"""
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
        """获取当前动作 - 兼容Device接口"""
        if not self.started:
            return None
            
        # 返回正确的设备上的张量
        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)

    def reset(self):
        """重置控制器状态"""
        self._base_velocity.fill(0)  # 重置速度
        self._left_arm_delta.fill(0)
        self._right_arm_delta.fill(0)
        self._left_gripper_delta = 0.0
        self._right_gripper_delta = 0.0
        self._head_delta.fill(0)
        self._pressed_keys.clear()

    def add_callback(self, key: str, func: Callable):
        """添加回调函数"""
        self._additional_callbacks[key] = func

    def __del__(self):
        """释放键盘接口"""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args):
        """处理键盘事件 - 修改为速度控制"""
        try:
            # 获取按键名称
            if hasattr(event, 'input') and hasattr(event.input, 'name'):
                key_name = event.input.name
            elif hasattr(event, 'name'):
                key_name = event.name
            else:
                return
            
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key_name in self._key_bindings:
                    control_type, index, value = self._key_bindings[key_name]
                    
                    # 处理控制键
                    if control_type == "control":
                        if index == "start":
                            self.started = True
                            self._reset_state = False
                            print("Xlerobot控制已启动！")
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
                    
                    # 处理关节控制键
                    if control_type == "base_joint":
                        self._base_velocity[index] = value  # 设置速度值
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
                    
                    # 只处理机器人控制键的释放，不处理控制键
                    if control_type == "base_joint":
                        self._base_velocity[index] = 0.0  # 释放时速度设为0
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
            print(f"键盘事件处理错误: {e}")

    def __str__(self) -> str:
        """返回控制器信息"""
        msg = "Xlerobot键盘控制器 (直接位置控制模式)\n"
        msg += f"\t键盘名称: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += f"\t灵敏度: {self.sensitivity}\n"
        msg += f"\t控制状态: {'已启动' if self.started else '未启动'}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t直接位置控制:\n"
        msg += "\t  前进/后退: W/S (X轴移动)\n"
        msg += "\t  左移/右移: A/D (Y轴移动)\n"
        msg += "\t  左转/右转: Q/E (Z轴旋转)\n"
        msg += "\t左臂控制:\n"
        msg += "\t  关节1: R/F\n"
        msg += "\t  关节2: T/G\n"
        msg += "\t  关节3: Y/H\n"
        msg += "\t  关节4: U/J\n"
        msg += "\t  关节5: I/K\n"
        msg += "\t右臂控制 (数字键盘):\n"
        msg += "\t  关节1: 8/2\n"
        msg += "\t  关节2: 4/6\n"
        msg += "\t  关节3: 7/9\n"
        msg += "\t  关节4: 1/3\n"
        msg += "\t  关节5: 0/.\n"
        msg += "\t夹爪控制:\n"
        msg += "\t  左夹爪: F1/F2 (开/关)\n"
        msg += "\t  右夹爪: F3/F4 (开/关)\n"
        msg += "\t头部控制:\n"
        msg += "\t  水平: Home/End (左/右)\n"
        msg += "\t  垂直: PageUp/PageDown (上/下)\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t启动控制: B\n"
        msg += "\t任务成功: N\n"
        msg += "\t重置环境: F5\n"
        msg += "\t退出: Ctrl+C\n"
        msg += f"\t当前按下的键: {list(self._pressed_keys)}"
        return msg