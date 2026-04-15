import torch
from typing import Any

import isaaclab.envs.mdp as mdp

# Xlerobot joint limit
XLEROBOT_JOINT_LIMITS = {
    # mobile base joint
    "root_x_axis_joint": (-20.0, 20.0),      # X-axis translation 
    # "root_y_axis_joint": (-20.0, 20.0),      # Y-axis translation  
    "root_z_rotation_joint": (-3.14159, 3.14159),  # Z-axis rotation
    
    # left arm joint
    "Rotation": (-2.1, 2.1),                 # shoulder rotation
    "Pitch": (-0.1, 3.45),                   # shoulder lift
    "Elbow": (-0.2, 3.14159),                # elbow flex
    "Wrist_Pitch": (-1.8, 1.8),              # wrist pitch
    "Wrist_Roll": (-3.14159, 3.14159),       # wrist roll
    "Jaw": (-0.5, 0.5),                      # left arm gripper
    
    # right arm joint
    "Rotation_2": (-2.1, 2.1),               # right shoulder rotation
    "Pitch_2": (-0.1, 3.45),                 # right shoulder lift
    "Elbow_2": (-0.2, 3.14159),              # right elbow flex
    "Wrist_Pitch_2": (-1.8, 1.8),            # right wrist pitch
    "Wrist_Roll_2": (-3.14159, 3.14159),     # right wrist roll
    "Jaw_2": (-0.5, 0.5),                    # right arm gripper
    
    # head joint
    "head_pan_joint": (-1.57, 1.57),         # head horizontal rotation
    "head_tilt_joint": (-0.76, 1.45),        # head vertical tilt
}


def init_xlerobot_action_cfg(action_cfg, device):
    """Initialize xlerobot action configuration"""
    if device in ['keyboard']:
        # keyboard control - remove base action configuration, use direct position control
        # comment out or delete base_action configuration
        # action_cfg.base_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["root_x_axis_joint", "root_z_rotation_joint"],
        #     scale=2.0,
        # )
        
        # only keep arm and head action configuration
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
        
        # clear base_action, avoid conflict
        action_cfg.base_action = None
        
    elif device in ['xlerobot_leader']:
        # physical device control - use absolute position control
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
        # Xbox control - use unified action space
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
        # clear other action configuration
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    else:
        # default configuration
        action_cfg.base_action = None
        action_cfg.left_arm_action = None
        action_cfg.left_gripper_action = None
        action_cfg.right_arm_action = None
        action_cfg.right_gripper_action = None
        action_cfg.head_action = None
    
    return action_cfg


# update joint mapping index
xlerobot_joint_names_to_motor_ids = {
    # rotation joint (0)
    "root_z_rotation_joint": 0,
    
    # left arm (1-5)
    "Rotation": 1,
    "Pitch": 2,
    "Elbow": 3,
    "Wrist_Pitch": 4,
    "Wrist_Roll": 5,
    "Jaw": 6,  # left gripper
    
    # right arm (7-11)
    "Rotation_2": 7,
    "Pitch_2": 8,
    "Elbow_2": 9,
    "Wrist_Pitch_2": 10,
    "Wrist_Roll_2": 11,
    "Jaw_2": 12,  # right gripper
    
    # head (13-14)
    "head_pan_joint": 13,
    "head_tilt_joint": 14,
}


def convert_action_from_xlerobot_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    """convert action from xlerobot leader device"""
    processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
    
    for joint_name, motor_id in xlerobot_joint_names_to_motor_ids.items():
        if joint_name in joint_state and joint_name in motor_limits:
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = XLEROBOT_JOINT_LIMITS[joint_name]
            
            # map device range to joint range
            processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
            
            # convert to radians (if needed)
            if joint_name in ["root_x_axis_joint", "root_y_axis_joint"]:
                # translation joint keep meter unit
                processed_action[:, motor_id] = processed_degree
            else:
                # rotation joint convert to radians
                processed_radius = processed_degree / 180.0 * torch.pi
                processed_action[:, motor_id] = processed_radius
    
    return processed_action


def preprocess_xlerobot_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    """preprocess xlerobot device action"""
    if action.get('hybrid_controller') is not None:
        # hybrid controller action processing
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xlerobot_leader') is not None:
        processed_action = convert_action_from_xlerobot_leader(action['joint_state'], action['motor_limits'], teleop_device)
    elif action.get('keyboard') is not None:
        # keyboard action directly use
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xbox') is not None:
        # Xbox controller action processing 
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('bi_xlerobot_leader') is not None:
        # dual arm xlerobot leader device 
        processed_action = torch.zeros(teleop_device.env.num_envs, 15, device=teleop_device.env.device)
        processed_action[:, :] = convert_action_from_xlerobot_leader(action['joint_state'], action['motor_limits'], teleop_device)
    else:
        raise NotImplementedError("Only teleoperation with xlerobot_leader, bi_xlerobot_leader, keyboard, xbox, hybrid_controller is supported for xlerobot.")
    
    return processed_action


def get_xlerobot_action_space_size():
    """get xlerobot action space size"""
    return 15  # 1(rotation) + 5(left arm) + 1(left gripper) + 5(right arm) + 1(right gripper) + 2(head)


def get_xlerobot_joint_names():
    """get xlerobot all joint names"""
    return list(xlerobot_joint_names_to_motor_ids.keys())


def get_xlerobot_joint_limits():
    """get xlerobot joint limits"""
    return XLEROBOT_JOINT_LIMITS.copy()