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
    """Xlerobot混合控制器：键盘控制底盘和头部，BiSO101Leader控制双臂"""
    
    def __init__(self, env, sensitivity: float = 1.0, 
                 left_arm_port: str = '/dev/ttyACM0', 
                 right_arm_port: str = '/dev/ttyACM1', 
                 recalibrate: bool = False):
        super().__init__(env)
        
        # 创建键盘控制器（仅用于底盘和头部控制）
        self.keyboard_controller = XlerobotKeyboard(env, sensitivity)
        
        # 创建双臂控制器
        self.bi_arm_controller = BiXlerobotLeader(
            env, 
            left_port=left_arm_port, 
            right_port=right_arm_port, 
            recalibrate=recalibrate
        )
        
        # 控制模式标志
        self.control_mode = "hybrid"  # "keyboard", "hybrid", "arms_only"
        
        # 状态标志
        self.started = False
        self._reset_state = False
        
    def set_control_mode(self, mode: str):
        """设置控制模式"""
        assert mode in ["keyboard", "hybrid", "arms_only"], f"Invalid control mode: {mode}"
        self.control_mode = mode
        print(f"控制模式切换为: {mode}")
        
    def get_device_state(self):
        """获取设备状态"""
        if self.control_mode == "keyboard":
            state = self.keyboard_controller.get_device_state()
            return state
        elif self.control_mode == "arms_only":
            arms_state = self.bi_arm_controller.get_device_state()
            # 缓存arms_action，避免重复调用
            if not hasattr(self, '_cached_arms_action'):
                self._cached_arms_action = self.bi_arm_controller.input2action()
            
            full_state = np.zeros(17)  # 改为17维
            
            # 左臂控制（索引3-8）
            if 'left_arm' in arms_state and arms_state['left_arm'] is not None:
                left_arm_data = arms_state['left_arm']
                if isinstance(left_arm_data, dict):
                    left_motor_limits = self._cached_arms_action.get('motor_limits', {}).get('left_arm', {})
                    left_processed = self._convert_arm_action(left_arm_data, left_motor_limits)
                    full_state[3:9] = left_processed
            
            # 右臂控制（索引9-14）
            if 'right_arm' in arms_state and arms_state['right_arm'] is not None:
                right_arm_data = arms_state['right_arm']
                if isinstance(right_arm_data, dict):
                    right_motor_limits = self._cached_arms_action.get('motor_limits', {}).get('right_arm', {})
                    right_processed = self._convert_arm_action(right_arm_data, right_motor_limits)
                    full_state[9:15] = right_processed
                    
            return full_state
        else:  # hybrid mode
            keyboard_state = self.keyboard_controller.get_device_state()
            arms_state = self.bi_arm_controller.get_device_state()
            
            # 缓存arms_action，避免重复调用
            if not hasattr(self, '_cached_arms_action'):
                self._cached_arms_action = self.bi_arm_controller.input2action()
            
            hybrid_state = np.zeros(17)  # 改为17维
            
            # 底盘控制（索引0-2）：使用键盘控制
            hybrid_state[0:3] = keyboard_state[0:3]
            
            # 左臂控制（索引3-8）：使用双臂控制器
            if 'left_arm' in arms_state and arms_state['left_arm'] is not None:
                left_arm_data = arms_state['left_arm']
                if isinstance(left_arm_data, dict):
                    left_motor_limits = self._cached_arms_action.get('motor_limits', {}).get('left_arm', {})
                    left_processed = self._convert_arm_action(left_arm_data, left_motor_limits)
                    hybrid_state[3:9] = left_processed
            
            # 右臂控制（索引9-14）：使用双臂控制器
            if 'right_arm' in arms_state and arms_state['right_arm'] is not None:
                right_arm_data = arms_state['right_arm']
                if isinstance(right_arm_data, dict):
                    right_motor_limits = self._cached_arms_action.get('motor_limits', {}).get('right_arm', {})
                    right_processed = self._convert_arm_action(right_arm_data, right_motor_limits)
                    hybrid_state[9:15] = right_processed
            
            # 头部控制（索引15-16）：使用键盘控制
            hybrid_state[15:17] = keyboard_state[15:17]
            
            return hybrid_state

    def input2action(self):
        """将输入转换为动作"""
        if self.control_mode == "keyboard":
            action = self.keyboard_controller.input2action()
            self.started = action.get("started", False)
            return action
        elif self.control_mode == "arms_only":
            action = self.bi_arm_controller.input2action()
            self.started = action.get("started", False)
            # 更新缓存
            self._cached_arms_action = action
            return action
        else:  # hybrid mode
            keyboard_action = self.keyboard_controller.input2action()
            arms_action = self.bi_arm_controller.input2action()
            
            # 设置started状态
            self.started = arms_action.get("started", False)
            
            # 更新缓存
            self._cached_arms_action = arms_action
            
            hybrid_action = {
                "reset": keyboard_action.get("reset", False) or arms_action.get("reset", False),
                "started": arms_action.get("started", False),
                "hybrid_controller": True,
                "keyboard": True,
                "bi_so101_leader": True,
                "joint_state": self.get_device_state(),
                "motor_limits": arms_action.get("motor_limits", {}),
            }
            
            return hybrid_action
    
    def advance(self):
        """获取当前动作"""
        # 不要在这里调用input2action()，因为状态应该在按键事件中更新
        # self.input2action()  # 移除这行
        
        if not self.started:
            return None
            
        action = self.get_device_state()
        return torch.tensor(action, dtype=torch.float32, device=self.env.device)
    
    def reset(self):
        """重置控制器状态"""
        self.keyboard_controller.reset()
        self.bi_arm_controller.reset()
        self.started = False
        self._reset_state = False
    
    def add_callback(self, key: str, func: Callable):
        """添加回调函数"""
        # 键盘回调用于控制模式切换
        if key == "F6":  # F6键切换控制模式
            def toggle_mode():
                if self.control_mode == "hybrid":
                    self.set_control_mode("keyboard")
                elif self.control_mode == "keyboard":
                    self.set_control_mode("arms_only")
                else:
                    self.set_control_mode("hybrid")
            self.keyboard_controller.add_callback(key, toggle_mode)
        else:
            # 其他回调传递给键盘控制器
            self.keyboard_controller.add_callback(key, func)
            
            # 对于B键，也需要传递给BiSO101Leader以启动双臂控制
            if key == "B":
                # 直接传递B键回调给BiSO101Leader
                self.bi_arm_controller.add_callback(key, func)
            # 其他键不传递给BiSO101Leader，避免不必要的按键处理
    
    def __str__(self) -> str:
        """返回控制器信息"""
        msg = "Xlerobot混合控制器\n"
        msg += f"\t当前控制模式: {self.control_mode}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\t键盘控制: 底盘移动(W/A/S/D) + 头部运动(Home/End/PageUp/PageDown)\n"
        msg += "\t双臂控制器: 机械臂和夹爪控制\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tF6: 切换控制模式 (hybrid <-> keyboard <-> arms_only)\n"
        msg += "\tB: 启动控制\n"
        msg += "\tF5: 重置环境\n"
        msg += "\t退出: Ctrl+C\n"
        return msg

    def _convert_arm_action(self, joint_state: dict, motor_limits: dict) -> np.ndarray:
        """转换单臂动作，使用与原有BiSO101Leader相同的转换逻辑"""
        processed_action = np.zeros(6)
        
        # 如果没有motor_limits，返回零动作
        if not motor_limits:
            print("警告：没有找到motor_limits，返回零动作")
            return processed_action
        
        # 使用与原有代码相同的关节限制（角度）
        from lehome.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
        
        # 关节名称到索引的映射
        joint_mapping = {
            'shoulder_pan': 0,
            'shoulder_lift': 1,
            'elbow_flex': 2,
            'wrist_flex': 3,
            'wrist_roll': 4,
            'gripper': 5
        }
        
        # 关节方向校正配置（如果某个关节方向相反，设置为-1）
        joint_direction_correction = {
            'shoulder_pan': 1,    # 1表示正常方向，-1表示反向
            'shoulder_lift': -1,
            'elbow_flex': 1,
            'wrist_flex': 1,
            'wrist_roll': -1,
            'gripper': 1
        }
        
        # 关节零位偏移校正（弧度）
        # 正值表示仿真关节需要向正方向偏移，负值表示向负方向偏移
        joint_zero_offset = {
            'shoulder_pan': 0.0,      # 肩部旋转
            'shoulder_lift': 1.57,   # 肩部抬升：偏移-90度（-π/2）
            'elbow_flex': 1.57,      # 肘部弯曲：偏移-90度（-π/2）
            'wrist_flex': 0.0,        # 腕部弯曲
            'wrist_roll': 0.0,        # 腕部旋转
            'gripper': 0.0            # 夹爪
        }
        
        for joint_name, index in joint_mapping.items():
            if joint_name in joint_state and joint_name in motor_limits:
                motor_limit_range = motor_limits[joint_name]
                joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]
                
                # 将电机范围映射到关节范围（角度）
                processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                    * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                processed_radius = processed_degree / 180.0 * np.pi  # 转换为弧度
                
                # 应用方向校正
                direction_correction = joint_direction_correction.get(joint_name, 1)
                processed_radius = processed_radius * direction_correction
                
                # 应用零位偏移校正
                zero_offset = joint_zero_offset.get(joint_name, 0.0)
                processed_action[index] = processed_radius + zero_offset
        
        return processed_action