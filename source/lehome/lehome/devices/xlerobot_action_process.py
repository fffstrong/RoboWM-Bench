import torch
from typing import Any

import isaaclab.envs.mdp as mdp

# Xlerobot关节限制
XLEROBOT_JOINT_LIMITS = {
    # 移动底盘关节
    "root_x_axis_joint": (-20.0, 20.0),      # X轴平移
    # "root_y_axis_joint": (-20.0, 20.0),      # Y轴平移  
    "root_z_rotation_joint": (-3.14159, 3.14159),  # Z轴旋转
    
    # 左臂关节
    "Rotation": (-2.1, 2.1),                 # 肩部旋转
    "Pitch": (-0.1, 3.45),                   # 肩部抬升
    "Elbow": (-0.2, 3.14159),                # 肘部弯曲
    "Wrist_Pitch": (-1.8, 1.8),              # 腕部弯曲
    "Wrist_Roll": (-3.14159, 3.14159),       # 腕部旋转
    "Jaw": (-0.5, 0.5),                      # 左臂夹爪
    
    # 右臂关节
    "Rotation_2": (-2.1, 2.1),               # 右肩部旋转
    "Pitch_2": (-0.1, 3.45),                 # 右肩部抬升
    "Elbow_2": (-0.2, 3.14159),              # 右肘部弯曲
    "Wrist_Pitch_2": (-1.8, 1.8),            # 右腕部弯曲
    "Wrist_Roll_2": (-3.14159, 3.14159),     # 右腕部旋转
    "Jaw_2": (-0.5, 0.5),                    # 右臂夹爪
    
    # 头部关节
    "head_pan_joint": (-1.57, 1.57),         # 头部水平旋转
    "head_tilt_joint": (-0.76, 1.45),        # 头部垂直倾斜
}


def init_xlerobot_action_cfg(action_cfg, device):
    """初始化xlerobot的动作配置"""
    if device in ['keyboard']:
        # 键盘控制 - 移除底盘动作配置，使用直接位置控制
        # 注释掉或删除base_action配置
        # action_cfg.base_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["root_x_axis_joint", "root_z_rotation_joint"],
        #     scale=2.0,
        # )
        
        # 只保留手臂和头部的动作配置
        action_cfg.left_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=3.0,
        )
        action_cfg.left_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=2.0,
        )
        action_cfg.right_arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            scale=3.0,
        )
        action_cfg.right_gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw_2"],
            scale=2.0,
        )
        action_cfg.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint", "head_tilt_joint"],
            scale=2.0,
        )
        
        # 清空base_action，避免冲突
        action_cfg.base_action = None
        
    elif device in ['xlerobot_leader']:
        # 物理设备控制 - 使用绝对位置控制
        action_cfg.base_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["root_x_axis_joint", "root_z_rotation_joint"],
            scale=1.0,
        )
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw_2"],
            scale=1.0,
        )
        action_cfg.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_pan_joint", "head_tilt_joint"],
            scale=1.0,
        )
    elif device in ['xbox', 'gamepad']:
        # Xbox 控制 - 使用统一的动作空间
        action_cfg.unified_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "root_x_axis_joint", "root_z_rotation_joint",  # 底盘
                "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw",  # 左臂
                "Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2", "Jaw_2",  # 右臂
                "head_pan_joint", "head_tilt_joint"  # 头部
            ],
            scale=1.0,
        )
        # 清空其他动作配置
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    else:
        # 默认配置
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    
    return action_cfg


# 更新关节映射索引
xlerobot_joint_names_to_motor_ids = {
    # 旋转关节 (0)
    "root_z_rotation_joint": 0,
    
    # 左臂 (1-5)
    "Rotation": 1,
    "Pitch": 2,
    "Elbow": 3,
    "Wrist_Pitch": 4,
    "Wrist_Roll": 5,
    "Jaw": 6,  # 左夹爪
    
    # 右臂 (7-11)
    "Rotation_2": 7,
    "Pitch_2": 8,
    "Elbow_2": 9,
    "Wrist_Pitch_2": 10,
    "Wrist_Roll_2": 11,
    "Jaw_2": 12,  # 右夹爪
    
    # 头部 (13-14)
    "head_pan_joint": 13,
    "head_tilt_joint": 14,
}


def convert_action_from_xlerobot_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    """从xlerobot leader设备转换动作"""
    processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
    
    for joint_name, motor_id in xlerobot_joint_names_to_motor_ids.items():
        if joint_name in joint_state and joint_name in motor_limits:
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = XLEROBOT_JOINT_LIMITS[joint_name]
            
            # 将设备范围映射到关节范围
            processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
            
            # 转换为弧度（如果需要）
            if joint_name in ["root_x_axis_joint", "root_y_axis_joint"]:
                # 平移关节保持米制单位
                processed_action[:, motor_id] = processed_degree
            else:
                # 旋转关节转换为弧度
                processed_radius = processed_degree / 180.0 * torch.pi
                processed_action[:, motor_id] = processed_radius
    
    return processed_action


def preprocess_xlerobot_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    """预处理xlerobot设备动作"""
    if action.get('hybrid_controller') is not None:
        # 混合控制器动作处理
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xlerobot_leader') is not None:
        processed_action = convert_action_from_xlerobot_leader(action['joint_state'], action['motor_limits'], teleop_device)
    elif action.get('keyboard') is not None:
        # 键盘动作直接使用
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xbox') is not None:
        # Xbox 控制器动作处理 
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('bi_xlerobot_leader') is not None:
        # 双臂xlerobot leader设备 
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = convert_action_from_xlerobot_leader(action['joint_state'], action['motor_limits'], teleop_device)
    else:
        raise NotImplementedError("Only teleoperation with xlerobot_leader, bi_xlerobot_leader, keyboard, xbox, hybrid_controller is supported for xlerobot.")
    
    return processed_action


def get_xlerobot_action_space_size():
    """获取xlerobot动作空间大小"""
    return 15  # 1(旋转) + 5(左臂) + 1(左夹爪) + 5(右臂) + 1(右夹爪) + 2(头部)


def get_xlerobot_joint_names():
    """获取xlerobot所有关节名称"""
    return list(xlerobot_joint_names_to_motor_ids.keys())


def get_xlerobot_joint_limits():
    """获取xlerobot关节限制"""
    return XLEROBOT_JOINT_LIMITS.copy()