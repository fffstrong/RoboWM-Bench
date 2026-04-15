import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

import torch
@dataclass
class WorldState:
    """World State - Unified state representation"""
    # Robot state
    robot_joint_pos: torch.Tensor
    robot_joint_vel: torch.Tensor
    robot_ee_pose: torch.Tensor  # [x, y, z, qw, qx, qy, qz]
    sim_js_names: List[str]
    
    # Object state
    objects: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Sensor data
    rgb: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    point_cloud: Optional[torch.Tensor] = None
    
    # Additional information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def device(self):
        return self.robot_joint_pos.device
    
    def to(self, device):
        """Move all tensors to device"""
        self.robot_joint_pos = self.robot_joint_pos.to(device)
        self.robot_joint_vel = self.robot_joint_vel.to(device)
        self.robot_ee_pose = self.robot_ee_pose.to(device)
        self.objects = {k: v.to(device) for k, v in self.objects.items()}
        if self.rgb is not None:
            self.rgb = self.rgb.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.point_cloud is not None:
            self.point_cloud = self.point_cloud.to(device)
        return self
    
def get_world_from_env(env):
    """
    Get world object from environment
    Compatible with different versions of Isaac Lab
    
    Args:
        env: Isaac Lab environment
        
    Returns:
        world object (with stage attribute)
        
    Raises:
        RuntimeError: If unable to get world
    """
    # Method 1: env.scene._world (old version)
    if hasattr(env, 'scene'):
        scene = env.scene
        if hasattr(scene, '_world'):
            return scene._world
    
    # Method 2: env.scene is InteractiveScene itself (new version)
    if hasattr(env, 'scene') and hasattr(env.scene, 'stage'):
        return env.scene
    
    # Method 3: env.unwrapped.scene
    if hasattr(env, 'unwrapped'):
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'scene'):
            scene = unwrapped.scene
            if hasattr(scene, '_world'):
                return scene._world
            if hasattr(scene, 'stage'):
                return scene
    
    # Method 4: env itself
    if hasattr(env, 'stage'):
        return env
    
    raise RuntimeError(
        "Cannot get world from env. "
        "Please check your Isaac Lab version and environment setup."
    )


def get_robot_from_env(env, robot_name: str = "robot"):
    scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene
    
    if hasattr(scene, robot_name):
        return getattr(scene, robot_name)
    
    if hasattr(scene, '__getitem__'):
        try:
            return scene[robot_name]
        except (KeyError, TypeError):
            pass
    
    for key in scene.keys():
        asset = scene[key]
        if hasattr(asset, 'num_joints') and asset.num_joints > 0:
            return asset
    
    raise RuntimeError(f"No robot found in scene")

def create_world_state(env, robot=None):
    
    if robot is None:
        robot = get_robot_from_env(env)
    
    if robot.num_joints > 0:
        joint_pos = robot.data.joint_pos[0]
        joint_vel = robot.data.joint_vel[0]
    else:
        joint_pos = torch.zeros(0, device=robot.data.root_pos_w.device)
        joint_vel = torch.zeros(0, device=robot.data.root_pos_w.device)
    joint_pos_limits = robot.data.joint_pos_limits[0, :, :]
    lower, upper = joint_pos_limits[:, 0], joint_pos_limits[:, 1]
    joint_pos = torch.clamp(robot.data.joint_pos[0, :], min=lower, max=upper)
    sim_js_names = robot.joint_names
    
    ee_link_candidates = ["left_hand_link", "panda_hand", "tool0", "ee_link", "link_tcp"]
    ee_link = None
    for candidate in ee_link_candidates:
        if candidate in robot.data.body_names:
            ee_link = candidate
            break
    if ee_link is None:
        ee_link = robot.data.body_names[-1]  # fallback: 最后一个 body
    ee_idx = robot.data.body_names.index(ee_link)
    ee_state = robot.data.body_state_w[0, ee_idx]
    ee_pose = ee_state[:7]
    
    objects = {}
    scene = env.unwrapped.scene if hasattr(env, 'unwrapped') else env.scene
    for key in scene.keys():
        obj = scene[key]
        if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w') and key != "robot":
            obj_pos = obj.data.root_pos_w[0]
            obj_quat = obj.data.root_quat_w[0] if hasattr(obj.data, 'root_quat_w') else torch.tensor([1, 0, 0, 0], device=obj_pos.device)
            objects[key] = torch.cat([obj_pos, obj_quat])
    
    return WorldState(
        robot_joint_pos=joint_pos,
        robot_joint_vel=joint_vel,
        robot_ee_pose=ee_pose,
        sim_js_names=sim_js_names,
        objects=objects,
    )
