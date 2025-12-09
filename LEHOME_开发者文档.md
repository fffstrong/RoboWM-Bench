# Lehome 开发者文档

## 1. 项目概述

Lehome 是一个基于 NVIDIA IsaacLab 构建的家庭机器人仿真和学习框架。该项目专注于家庭场景中的各类操作任务，包括衣物折叠、食物制备、液体操作和表面清洁等。

### 1.1 技术栈

- **仿真引擎**: NVIDIA IsaacSim / IsaacLab
- **深度学习**: PyTorch
- **机器人学习**: LeRobot
- **Python 版本**: 3.11

### 1.2 项目结构

```
lehome/
├── Assets/                    # 资产文件
│   ├── Material/             # 材质文件（流体、纺织物等）
│   ├── objects/              # 物体模型（衣服、汉堡、毛巾等）
│   ├── robots/               # 机器人模型
│   └── scenes/               # 场景文件（厨房、客厅、卧室等）
├── Datasets/                 # 数据集
│   ├── record/               # 录制的数据
│   └── replay/               # 回放数据
├── scripts/                  # 工具脚本（见各脚本目录下的 README）
│   ├── eval/                 # 评估脚本
│   ├── teleoperation/        # 遥操作录制脚本
│   ├── tool/                 # 工具脚本（replay 等）
│   └── rl_*/                 # 强化学习训练脚本
└── source/lehome/lehome/     # 核心代码库 ⭐
    ├── __init__.py
    ├── tasks/                # 任务定义（核心）
    ├── assets/               # 资产配置
    ├── devices/              # 设备控制
    └── utils/                # 工具函数
```

## 2. 核心架构详解

### 2.1 整体设计理念

Lehome 基于 IsaacLab 的 **DirectRLEnv** 架构，采用以下设计模式：

1. **配置与实现分离**: 每个任务都包含 `*_cfg.py`（配置）和 `*.py`（实现）
2. **继承与扩展**: 基于 BaseEnv 基类进行任务扩展
3. **模块化组件**: Assets、Devices、Utils 独立模块化

### 2.2 目录结构详解

```
source/lehome/lehome/
├── __init__.py              # 包初始化，注册 Gym 环境
├── tasks/                   # ⭐ 任务定义（本文档重点）
│   ├── __init__.py         # 自动导入所有任务包
│   ├── base/               # 基础环境类
│   │   ├── base_env.py         # 基础环境实现
│   │   └── base_env_cfg.py     # 基础环境配置
│   ├── bedroom/            # 卧室任务（衣物操作）
│   ├── kitchen/            # 厨房任务（食物制备）
│   ├── livingroom/         # 客厅任务（液体操作）
│   └── washroom/           # 洗手间任务（清洁任务）
├── assets/                  # 资产配置
│   ├── object/             # 物体类（流体、衣物等）
│   ├── robots/             # 机器人配置
│   └── scenes/             # 场景配置
├── devices/                 # 设备控制
│   ├── action_process.py   # 动作预处理
│   ├── keyboard/           # 键盘控制
│   └── lerobot/            # LeRobot 设备（leader-follower）
└── utils/                   # 工具函数
    ├── success_checker.py  # 成功判定
    ├── record.py           # 数据记录
    └── ...                 # 其他工具
```

## 3. Task 创建完整指南

### 3.1 Task 的核心概念

在 Lehome 中，**Task** 是一个完整的交互式环境，定义了：
- 场景布局（机器人、物体、相机等）
- 观测空间（传感器数据）
- 动作空间（机器人控制）
- 奖励函数（强化学习）
- 成功条件（任务完成判定）
- 重置逻辑（环境初始化）

### 3.2 Task 创建流程

创建一个新的 Task 需要以下步骤：

#### 第一步：创建配置文件 `*_cfg.py`

配置文件定义了环境的所有静态参数。

**示例：`loft_water_cfg.py`**

```python
from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG
from ..base.base_env_cfg import BaseEnvCfg
import os

@configclass
class LoftWaterEnvCfg(BaseEnvCfg):
    """倒水任务配置类 - 继承自基础环境配置"""
    
    # 1. 仿真参数配置
    render_cfg: sim_utils.RenderCfg = sim_utils.RenderCfg(
        rendering_mode="quality",      # 渲染质量
        antialiasing_mode="Off"        # 抗锯齿
    )
    
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,                    # 物理步长: 120Hz
        render_interval=1,             # 渲染间隔
        render=render_cfg,
        use_fabric=False               # 是否使用 Fabric 加速
    )
    
    # 2. 机器人配置
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(3, 0.58, 0.76541),    # 初始位置
            rot=(0.707, 0.0, 0.0, -0.707),  # 初始旋转（四元数）
            joint_pos={                # 初始关节角度
                "shoulder_pan": -0.0363,
                "shoulder_lift": -1.7135,
                "elbow_flex": 1.4979,
                "wrist_flex": 1.0534,
                "wrist_roll": -0.085,
                "gripper": -0.01176,
            },
        ),
    )
    
    # 3. 相机配置
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],            # 数据类型：RGB 图像
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,        # 30 FPS
    )
    
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(...),
        data_types=["rgb", "depth"],   # RGB + 深度图
        ...
    )
    
    # 4. 刚体物体配置（碗）
    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd() + "/Assets/scenes/LW_Loft/Loft/Bowl016/Bowl016.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.7, 0.45, 0.83),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )
    
    # 5. 场景文件路径
    path_scene: str = os.getcwd() + "/Assets/scenes/LW_Loft/LW_Loft_water.usd"
```

**配置文件关键点说明**：

1. **继承 BaseEnvCfg**: 获得基础环境的默认配置
2. **使用 @configclass**: IsaacLab 的配置装饰器
3. **replace() 方法**: 复制并修改预定义配置（如机器人配置）
4. **路径约定**: 
   - `/World/Robot/*`: 机器人相关
   - `/World/Object/*`: 物体相关
   - `/World/Scene`: 场景根节点

#### 第二步：创建环境实现文件 `*.py`

环境实现文件包含所有的动态逻辑。

**示例：`loft_water.py`**

```python
from __future__ import annotations
import os
import torch
from typing import Any, Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from .loft_water_cfg import LoftWaterEnvCfg
from ..base.base_env import BaseEnv
from lehome.assets.object.fluid import FluidObject
from omegaconf import OmegaConf

class LoftWaterEnv(BaseEnv):
    """倒水任务环境 - 继承自基础环境"""
    
    cfg: LoftWaterEnvCfg
    
    def __init__(self, cfg: LoftWaterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.arm.data.joint_pos
    
    # ============ 核心方法 1: 场景设置 ============
    def _setup_scene(self):
        """
        设置场景中的所有组件
        
        执行时机：环境初始化时调用一次
        作用：创建并注册所有的资产（机器人、物体、传感器等）
        """
        # 1. 调用父类方法，加载基础场景（Loft 房间）
        super()._setup_scene()
        
        # 2. 创建机器人（关节体）
        self.arm = Articulation(self.cfg.robot)
        
        # 3. 创建相机传感器
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        
        # 4. 创建流体物体（特殊物体，使用粒子系统）
        self.object = FluidObject(
            env_id=0,
            env_origin=torch.zeros(1, 3),
            prim_path="/World/Object/fluid_items/fluid_items_1",
            usd_path=os.getcwd() + "/Assets/scenes/LW_Loft/water.usdc",
            config=OmegaConf.load(
                os.getcwd() + "/source/lehome/lehome/tasks/livingroom/config_file/fluid.yaml"
            ),
            use_container=True,  # 使用容器
        )
        
        # 5. 创建刚体物体（碗）
        self.bowl = RigidObject(self.cfg.bowl)
        
        # 6. 将所有资产注册到场景中（重要！）
        self.scene.articulations["robot"] = self.arm
        self.scene.rigid_objects["bowl"] = self.bowl
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera
    
    # ============ 核心方法 2: 动作预处理 ============
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        物理步进前的动作预处理
        
        执行时机：每次 step() 调用时，物理更新之前
        作用：对原始动作进行缩放和处理
        """
        self.actions = self.action_scale * actions.clone()
    
    # ============ 核心方法 3: 动作应用 ============
    def _apply_action(self) -> None:
        """
        将处理后的动作应用到机器人
        
        执行时机：_pre_physics_step 之后，物理更新之前
        作用：实际控制机器人运动
        """
        self.arm.set_joint_position_target(self.actions)
    
    # ============ 核心方法 4: 获取观测 ============
    def _get_observations(self) -> dict:
        """
        获取环境观测数据
        
        执行时机：每次 step() 调用后
        作用：返回传感器数据，用于策略学习
        
        返回格式：与 LeRobot 数据格式兼容
        """
        action = self.actions.squeeze(0)
        joint_pos = torch.cat(
            [self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        ).squeeze(0)
        
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu().detach().numpy().squeeze(),
            "observation.images.wrist_rgb": wrist_camera_rgb.cpu().detach().numpy().squeeze(),
            "observation.top_depth": top_camera_depth.cpu().detach().numpy(),
        }
        return observations
    
    # ============ 核心方法 5: 奖励计算 ============
    def _get_rewards(self) -> torch.Tensor:
        """
        计算奖励值
        
        执行时机：每次 step() 调用后
        作用：用于强化学习的奖励信号
        """
        # TODO: 实现具体的奖励函数
        total_reward = 0
        return total_reward
    
    # ============ 核心方法 6: 完成判定 ============
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        判断 episode 是否结束
        
        返回：(终止标志, 超时标志)
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
    
    def _get_successes(self) -> torch.Tensor:
        """
        判断任务是否成功
        
        作用：用于评估和统计
        """
        # TODO: 实现成功判定逻辑
        successes = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return successes
    
    # ============ 核心方法 7: 环境重置 ============
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        重置指定的环境
        
        执行时机：episode 结束时或手动重置
        作用：将环境恢复到初始状态（可加入随机化）
        """
        if env_ids is None:
            env_ids = self.arm._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # 1. 重置机器人关节位置
        joint_pos = self.arm.data.default_joint_pos[env_ids]
        self.arm.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        
        # 2. 重置流体物体
        self.object.reset()
        
        # 3. 重置碗的位置（加入随机扰动）
        bowl_pos = self.bowl.data.default_root_state[env_ids].clone()
        rand_vals = torch.empty(len(env_ids), 2, device=bowl_pos.device).uniform_(-0.05, 0.05)
        random_bowl_pos = bowl_pos.clone()
        random_bowl_pos[..., :2] += rand_vals      # x, y 位置随机化
        random_bowl_pos[..., 7:] = 0.0             # 速度归零
        self.bowl.write_root_state_to_sim(random_bowl_pos, env_ids=env_ids)
    
    # ============ 辅助方法：设备动作预处理 ============
    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        """
        预处理来自遥操作设备的动作
        
        作用：将不同设备（键盘、leader arm）的输入转换为统一格式
        """
        from lehome.devices.action_process import preprocess_device_action
        return preprocess_device_action(action, teleop_device)
    
    # ============ 辅助方法：观测初始化 ============
    def initialize_obs(self):
        """
        初始化观测相关的组件
        
        执行时机：环境创建后，第一次使用前
        作用：对特殊物体（如流体、布料）进行初始化
        """
        self.object.initialize()
        self.bowl.reset()
    
    # ============ 辅助方法：位姿获取与设置 ============
    def get_all_pose(self):
        """
        获取所有物体的位姿
        
        作用：用于数据记录和回放
        """
        poses = {}
        poses.update(self.object.get_all_pose())  # 流体物体的位姿
        
        # 获取碗的位姿
        bowl_root_state = self.bowl.data.root_state_w[0]
        bowl_pose = torch.cat([bowl_root_state[:3], bowl_root_state[3:7]]).cpu().numpy()
        poses.update({"bowl": bowl_pose})
        return poses
    
    def set_all_pose(self, pose, env_ids: Sequence[int] | None):
        """
        设置所有物体的位姿
        
        作用：用于数据回放
        """
        if env_ids is None:
            env_ids = self.bowl._ALL_INDICES
        
        self.object.set_all_pose(pose)
        
        if "bowl" in pose:
            bowl_pose = pose["bowl"]
            bowl_root_state = self.bowl.data.default_root_state[env_ids].clone()
            if isinstance(bowl_pose, np.ndarray):
                bowl_pose = torch.from_numpy(bowl_pose).float()
            if len(bowl_pose) >= 7:
                bowl_root_state[..., :3] = bowl_pose[:3]      # 位置
                bowl_root_state[..., 3:7] = bowl_pose[3:7]    # 四元数
            bowl_root_state[..., 7:] = 0.0                     # 速度归零
            self.bowl.write_root_state_to_sim(bowl_root_state, env_ids=env_ids)
```

**环境实现关键点说明**：

| 方法 | 执行时机 | 必须实现 | 作用 |
|------|----------|----------|------|
| `_setup_scene()` | 环境初始化 | ✅ | 创建并注册所有资产 |
| `_pre_physics_step()` | 每步之前 | ✅ | 动作预处理 |
| `_apply_action()` | 每步执行 | ✅ | 应用动作到机器人 |
| `_get_observations()` | 每步之后 | ✅ | 返回观测数据 |
| `_get_rewards()` | 每步之后 | ✅ | 计算奖励（强化学习） |
| `_get_dones()` | 每步之后 | ✅ | 判断是否结束 |
| `_reset_idx()` | Episode 结束 | ✅ | 重置环境 |
| `preprocess_device_action()` | 遥操作时 | ⚠️ | 处理设备输入 |
| `initialize_obs()` | 首次使用前 | ⚠️ | 初始化特殊组件 |
| `get_all_pose()` / `set_all_pose()` | 记录/回放 | ⚠️ | 位姿的保存和恢复 |

#### 第三步：注册任务到包

在任务目录的 `__init__.py` 中注册新任务：

```python
# source/lehome/lehome/tasks/livingroom/__init__.py
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

import lehome.tasks.livingroom.loft_water as loft_water_env

##
# Register Gym environments.
##

# 注册到 Gym
import gymnasium as gym
gym.register(
    id="Isaac-Loft-Water-v0",
    entry_point="lehome.tasks.livingroom.loft_water:LoftWaterEnv",
    kwargs={"env_cfg_entry_point": loft_water_env.LoftWaterEnvCfg},
)
```

### 3.3 不同类型 Task 的特点

Lehome 支持多种类型的任务，每种任务处理不同类型的物体：

#### 类型 1：单臂 + 刚体物体（Livingroom - 倒水）

**特点**：
- 单个机器人臂
- 包含流体物体（粒子系统）+ 刚体物体（碗）
- 需要特殊的流体配置文件

**关键组件**：
```python
# 机器人
self.arm = Articulation(self.cfg.robot)

# 流体物体
self.object = FluidObject(
    prim_path="/World/Object/fluid_items/fluid_items_1",
    usd_path="...",
    config=OmegaConf.load("fluid.yaml"),
    use_container=True
)

# 刚体物体
self.bowl = RigidObject(self.cfg.bowl)
```

#### 类型 2：双臂 + 布料物体（Bedroom - 折叠衣物）

**特点**：
- 两个机器人臂（左臂 + 右臂）
- 包含可变形物体（衣物，使用布料模拟）
- 支持纹理和光照随机化

**关键组件**：
```python
# 双臂机器人
self.left_arm = Articulation(self.cfg.left_robot)
self.right_arm = Articulation(self.cfg.right_robot)

# 双臂相机
self.left_camera = TiledCamera(self.cfg.left_wrist)
self.right_camera = TiledCamera(self.cfg.right_wrist)

# 布料物体（使用粒子布料系统）
self.object = GarmentObject(
    prim_path="/World/Object/Cloth",
    usd_path="...",
    config=OmegaConf.load("particle_garment_cfg.yaml")
)

# 动作应用（双臂）
def _apply_action(self):
    self.left_arm.set_joint_position_target(self.actions[:, :6])
    self.right_arm.set_joint_position_target(self.actions[:, 6:])
```

**随机化功能**：
```python
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)
    # ... 重置机器人和物体 ...
    
    # 纹理随机化
    if self.texture_cfg.get("enable", False):
        self._randomize_table038_texture()
    
    # 光照随机化
    if self.light_cfg.get("enable", False):
        self._randomize_light()
```

#### 类型 3：双臂 + 可变形物体（Kitchen - 汉堡制作）

**特点**：
- 两个机器人臂
- 包含多种物体类型：
  - 可变形物体：牛肉饼、奶酪（DeformableObject）
  - 刚体物体：盘子、面包、砧板（RigidObject）

**关键组件**：
```python
# 可变形物体
self.burger_beef = DeformableObject(self.cfg.burger_beef)
self.burger_cheese = DeformableObject(self.cfg.burger_cheese)

# 刚体物体
self.burger_board = RigidObject(self.cfg.burger_board)
self.burger_plate = RigidObject(self.cfg.burger_plate)
self.burger_bread2 = RigidObject(self.cfg.burger_bread2)

# 注册到场景（注意区分类型）
self.scene.articulations["left_arm"] = self.left_arm
self.scene.articulations["right_arm"] = self.right_arm
self.scene.deformable_objects["burger_beef"] = self.burger_beef
self.scene.deformable_objects["burger_cheese"] = self.burger_cheese
self.scene.rigid_objects["burger_board"] = self.burger_board
self.scene.rigid_objects["burger_plate"] = self.burger_plate
self.scene.rigid_objects["burger_bread2"] = self.burger_bread2
```

**成功判定**：
```python
def _get_success(self):
    beef_pos = self.burger_beef.data.root_pos_w
    plate_pos = self.burger_plate.data.root_pos_w
    success = success_checker_burger(beef_pos=beef_pos, plate_pos=plate_pos)
    # 转换为 tensor
    if isinstance(success, bool):
        success_tensor = torch.tensor(
            [success] * len(self.episode_length_buf), device=self.device
        )
    else:
        success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
    return success_tensor
```

#### 类型 4：单臂 + 混合物体（Washroom - 擦拭）

**特点**：
- 单臂机器人
- 包含布料物体（毛巾）+ 流体物体（水）

**关键组件**：
```python
# 布料物体（毛巾）
self.towel = GarmentObject(
    prim_path="/World/Objects/Towel",
    usd_path=os.getcwd() + "/Assets/objects/Towel/towel.usd",
    visual_usd_path=os.getcwd() + "/Assets/Material/Garment/linen_Blue.usd",
    config=OmegaConf.load("particle_towel_cfg.yaml"),
)

# 流体物体（水，不使用容器）
self.object = FluidObject(
    env_id=0,
    env_origin=torch.zeros(1, 3),
    prim_path="/World/Object/fluid_items/fluid_items_1",
    usd_path=os.getcwd() + "/Assets/scenes/LW_Loft/water.usdc",
    config=OmegaConf.load("fluid.yaml"),
    use_container=False  # 不使用容器！
)

# 初始化
def initialize_obs(self):
    self.object.initialize()
    self.towel.initialize()

# 重置
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)
    joint_pos = self.robot.data.default_joint_pos[env_ids]
    self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
    self.towel.reset()
    self.object.reset(soft=True)  # soft reset
```

### 3.4 物体类型总结

| 物体类型 | 类名 | 适用场景 | 特点 |
|---------|------|----------|------|
| **刚体** | `RigidObject` | 盘子、碗、砧板 | 不变形，有碰撞 |
| **关节体** | `Articulation` | 机器人 | 有关节，可控制 |
| **可变形体** | `DeformableObject` | 牛肉饼、奶酪 | 可以变形 |
| **布料** | `GarmentObject` | 衣物、毛巾 | 粒子布料模拟 |
| **流体** | `FluidObject` | 水、液体 | 粒子流体模拟 |

**注意事项**：
- `GarmentObject` 和 `FluidObject` 需要配置文件（YAML）
- 布料和流体物体需要调用 `initialize()` 初始化
- 可变形和刚体物体使用 `reset()` 初始化

## 4. Assets 模块详解

### 4.1 机器人配置（`assets/robots/`）

**文件**: `lerobot.py`

定义了机器人的配置参数：

```python
from pathlib import Path
from isaaclab.assets.articulation import ArticulationCfg
from lehome.utils.constant import ASSETS_ROOT

SO101_FOLLOWER_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "so101_follower_good.usd"

SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,       # 启用自碰撞
            solver_position_iteration_count=16,  # 求解器迭代次数
            solver_velocity_iteration_count=16,
            fix_root_link=True,                  # 固定基座
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.4, -2.3, 0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "sts3215-gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "sts3215-arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan", "shoulder_lift", "elbow_flex", 
                "wrist_flex", "wrist_roll"
            ],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
)
```

**关键配置项**：
- `usd_path`: USD 资产文件路径
- `rigid_props`: 刚体属性（重力、碰撞等）
- `articulation_props`: 关节体属性（自碰撞、求解器参数）
- `init_state`: 初始状态（位置、旋转、关节角度）
- `actuators`: 执行器配置（刚度、阻尼、力/速度限制）

### 4.2 特殊物体类（`assets/object/`）

#### 4.2.1 流体物体 `FluidObject`

**文件**: `fluid.py`

**核心功能**：
1. 粒子生成：从网格生成流体粒子
2. 物理模拟：使用 PhysX 粒子系统
3. 容器支持：可选容器（杯子）
4. 位姿管理：支持获取和设置粒子位置

**初始化流程**：
```python
self.object = FluidObject(
    env_id=0,
    env_origin=torch.zeros(1, 3),
    prim_path="/World/Object/fluid_items/fluid_items_1",
    usd_path=os.getcwd() + "/Assets/scenes/LW_Loft/water.usdc",
    config=OmegaConf.load("fluid.yaml"),
    use_container=True,  # 是否使用容器
)

# 初始化粒子
self.object.initialize()

# 重置
self.object.reset(soft=True)  # soft=True: 软重置，保留部分状态
```

**配置文件示例** (`fluid.yaml`):
```yaml
objects:
  common:
    position: [0.0, 0.0, 0.0]
  
  fluid_items:
    num_per_env: 1
    common:
      position: [2.9, 0.45, 0.91]
      orientation: [0, 0, 0, 1]
      scale: [1.0, 1.0, 1.0]
    
    fluid_items_1:
      physics:
        fluid_volumn: 1.0  # 流体体积倍数
        particle_system:
          particle_contact_offset: 0.025
          contact_offset: 0.025
          rest_offset: 0.0225
          fluid_rest_offset: 0.0135
          max_velocity: 2.5
          smoothing: false
          anisotropy: false
          isosurface: true
```

#### 4.2.2 布料物体 `GarmentObject`

**文件**: `Garment.py`

**核心功能**：
1. 粒子布料模拟：使用 PhysX 布料系统
2. 材质管理：支持视觉材质和物理材质
3. 自碰撞：支持布料自碰撞检测
4. 属性配置：拉伸/弯曲/剪切刚度等

**初始化流程**：
```python
self.object = GarmentObject(
    prim_path="/World/Object/Cloth",
    usd_path=os.getcwd() + "/Assets/objects/garment/Tops/.../obj.usd",
    visual_usd_path=os.getcwd() + "/Assets/Material/Garment/linen_Blue.usd",
    config=OmegaConf.load("particle_garment_cfg.yaml"),
)

# 初始化
self.object.initialize()

# 重置
self.object.reset()

# 获取/设置位姿
poses = self.object.get_all_pose()
self.object.set_all_pose(poses)
```

**配置文件关键参数**:
```yaml
objects:
  particle_system:
    particle_system_enabled: true
    enable_ccd: true  # 连续碰撞检测
    solver_position_iteration_count: 16
    global_self_collision_enabled: true  # 全局自碰撞
    contact_offset: 0.01
    particle_contact_offset: 0.01
  
  particle_material:
    friction: 0.5
    damping: 0.1
    drag: 0.0
    lift: 0.0
  
  garment_config:
    particle_mass: 0.01
    self_collision: true
    stretch_stiffness: 50000.0  # 拉伸刚度
    bend_stiffness: 500.0       # 弯曲刚度
    shear_stiffness: 500.0      # 剪切刚度
    spring_damping: 5.0
```

### 4.3 场景配置（`assets/scenes/`）

**文件**: `kitchen.py`, `loft.py`, `bedroom.py`

定义场景资产路径：

```python
# kitchen.py
from lehome.utils.constant import ASSETS_ROOT
from pathlib import Path

KITCHEN_WITH_ORANGE_USD_PATH = (
    Path(ASSETS_ROOT) / "scenes" / "kitchen_with_orange" / "scene.usd"
)
```

场景通常在 `_setup_scene()` 中加载：

```python
def _setup_scene(self):
    # 加载场景
    cfg = sim_utils.UsdFileCfg(usd_path=KITCHEN_WITH_ORANGE_USD_PATH)
    cfg.func("/World/Scene", cfg, translation=(0.0, 0.0, 0.0))
    
    # 添加光照
    light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
```

## 5. Devices 模块详解

### 5.1 设备类型

Lehome 支持多种遥操作设备：

| 设备类型 | 实现文件 | 用途 |
|---------|---------|------|
| `so101leader` | `lerobot/so101_leader.py` | SO101 Leader Arm 单臂 |
| `bi-so101leader` | `lerobot/bi_so101_leader.py` | SO101 Leader Arm 双臂 |
| `keyboard` | `keyboard/xlerobot_keyboard.py` | 键盘控制单臂 |
| `bi-keyboard` | `keyboard/bi_keyboard.py` | 键盘控制双臂 |

### 5.2 动作预处理（`action_process.py`）

**核心函数**: `preprocess_device_action()`

作用：将不同设备的输入统一转换为机器人动作。

**工作流程**：
```
设备输入 → preprocess_device_action() → 统一动作张量 → 环境应用
```

**单臂示例**（SO101 Leader）：
```python
def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    if action.get("so101_leader") is not None:
        # 1. 从 motor 范围映射到 joint 范围
        processed_action = convert_action_from_so101_leader(
            action["joint_state"],
            action["motor_limits"],
            teleop_device
        )
        return processed_action  # shape: (num_envs, 6)
```

**双臂示例**（双 SO101 Leader）：
```python
elif action.get("bi_so101_leader") is not None:
    processed_action = torch.zeros(
        teleop_device.env.num_envs, 12, device=teleop_device.env.device
    )
    # 处理左臂
    processed_action[:, :6] = convert_action_from_so101_leader(
        action["joint_state"]["left_arm"],
        action["motor_limits"]["left_arm"],
        teleop_device,
    )
    # 处理右臂
    processed_action[:, 6:] = convert_action_from_so101_leader(
        action["joint_state"]["right_arm"],
        action["motor_limits"]["right_arm"],
        teleop_device,
    )
    return processed_action  # shape: (num_envs, 12)
```

**键盘控制**（相对增量）：
```python
elif action.get("keyboard") is not None:
    # 获取当前关节位置
    current_joint_pos = teleop_device.env.robot.data.joint_pos[:, :6]
    
    # 获取相对增量
    relative_delta = action["joint_state"]
    if isinstance(relative_delta, np.ndarray):
        relative_delta = torch.tensor(
            relative_delta, device=teleop_device.env.device, dtype=torch.float32
        )
    
    # 相对增量 + 当前位置 = 目标位置
    processed_action = current_joint_pos + relative_delta
    return processed_action
```

### 5.3 坐标转换

**电机角度 → 关节角度**：

```python
def convert_action_from_so101_leader(
    joint_state: dict[str, float],
    motor_limits: dict[str, tuple[float, float]],
    teleop_device,
) -> torch.Tensor:
    """
    将 Leader Arm 的电机角度转换为 Follower Arm 的关节角度
    
    转换公式：
    normalized_value = (motor_angle - motor_min) / (motor_max - motor_min)
    joint_degree = normalized_value * (joint_max - joint_min) + joint_min
    joint_radian = joint_degree * π / 180
    """
    processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS  # 关节限位（度）
    
    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        
        # 归一化到 [0, 1]
        normalized = (joint_state[joint_name] - motor_limit_range[0]) / (
            motor_limit_range[1] - motor_limit_range[0]
        )
        
        # 映射到关节范围（度）
        processed_degree = normalized * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
        
        # 转换为弧度
        processed_radius = processed_degree / 180.0 * torch.pi
        
        processed_action[:, motor_id] = processed_radius
    
    return processed_action
```

## 6. Utils 模块详解

### 6.1 成功判定（`success_checker.py`）

用于判断任务是否完成。

**示例 1：折叠衣物**

```python
def success_checker_fold(garment_obj) -> bool:
    """
    判断衣物是否折叠成功
    
    原理：计算衣物的边界框（Bounding Box）
    - 折叠前：边界框较大
    - 折叠后：边界框缩小
    """
    particle_positions = garment_obj.get_particle_positions()  # (N, 3)
    
    # 计算边界框
    min_pos = particle_positions.min(axis=0)
    max_pos = particle_positions.max(axis=0)
    bbox_size = max_pos - min_pos
    
    # 阈值判断
    threshold_x = 0.3  # x 方向阈值
    threshold_y = 0.3  # y 方向阈值
    
    if bbox_size[0] < threshold_x and bbox_size[1] < threshold_y:
        return True
    return False
```

**示例 2：汉堡制作**

```python
def success_checker_burger(beef_pos: torch.Tensor, plate_pos: torch.Tensor) -> bool:
    """
    判断牛肉饼是否放到盘子上
    
    原理：计算牛肉饼中心与盘子中心的距离
    """
    distance = torch.norm(beef_pos - plate_pos, dim=-1)
    threshold = 0.1  # 10cm 以内算成功
    
    if distance < threshold:
        return True
    return False
```

### 6.2 数据记录（`record.py`）

用于记录演示数据，支持 LeRobot 格式。

**核心类**: `DataRecorder`

```python
from lehome.utils.record import DataRecorder

# 初始化记录器
recorder = DataRecorder(
    repo_id="lerobot/water_task",
    root="./Datasets/record",
    fps=30,
)

# 开始记录
recorder.start_recording()

# 记录一帧
recorder.record_frame(
    observation=obs,
    action=action,
    next_observation=next_obs,
    reward=reward,
    done=done,
    info=info,
)

# 结束记录
recorder.stop_recording()
```

### 6.3 深度图转点云（`depth_to_pointcloud.py`）

将深度图像转换为 3D 点云。

```python
from lehome.utils.depth_to_pointcloud import generate_pointcloud_from_data

points, colors = generate_pointcloud_from_data(
    rgb_image=rgb_img,      # (H, W, 3/4), 值域 [0, 1]
    depth_image=depth_img,  # (H, W), 单位：米
    num_points=2048,        # 采样点数
    use_fps=False,          # 是否使用最远点采样
)
# points: (N, 3), 点云坐标
# colors: (N, 3), 颜色 [0, 255]
```

**用途**：
- 可视化场景
- 点云策略输入
- 碰撞检测

### 6.4 常量定义（`constant.py`）

定义全局常量，如资产根目录：

```python
# constant.py
import os

ASSETS_ROOT = os.path.join(os.getcwd(), "Assets")
```

## 7. 完整开发流程示例

假设要创建一个新任务：**厨房 - 切菜任务（Kitchen - Cut Vegetable）**

### 步骤 1：准备资产

1. 将蔬菜 USD 文件放到 `Assets/objects/vegetable/carrot.usd`
2. 将刀具 USD 文件放到 `Assets/robots/so101_knife.usd`
3. 使用现有的厨房场景 `Assets/scenes/kitchen_with_orange/scene.usd`

### 步骤 2：创建配置文件

```python
# source/lehome/lehome/tasks/kitchen/loft_cut_veg_cfg.py
from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from lehome.assets.robots.lerobot import SO101_KINFE_CFG
from ..base.base_env_cfg import BaseEnvCfg
import os

@configclass
class LoftCutVegEnvCfg(BaseEnvCfg):
    """切菜任务配置"""
    
    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        use_fabric=False
    )
    
    # 机器人（带刀）
    robot: ArticulationCfg = SO101_KINFE_CFG.replace(
        prim_path="/World/Robot/Robot",
        init_state=SO101_KINFE_CFG.init_state.replace(
            pos=(2.7, -2.5, 0.5),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )
    
    # 蔬菜（刚体）
    carrot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/Carrot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd() + "/Assets/objects/vegetable/carrot.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.4, -2.76, 0.85),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )
    
    # 砧板
    board: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Object/Board",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.getcwd() + "/Assets/scenes/kitchen_with_orange/objects/board.usd"
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.4, -2.76, 0.8),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )
    
    # 相机
    wrist_camera: TiledCameraCfg = TiledCameraCfg(...)
    top_camera: TiledCameraCfg = TiledCameraCfg(...)
    
    # 场景路径
    path_scene: str = os.getcwd() + "/Assets/scenes/kitchen_with_orange/scene.usd"
```

### 步骤 3：创建环境实现

```python
# source/lehome/lehome/tasks/kitchen/loft_cut_veg.py
from __future__ import annotations
import torch
from typing import Any, Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from .loft_cut_veg_cfg import LoftCutVegEnvCfg
from ..base.base_env import BaseEnv

class LoftCutVegEnv(BaseEnv):
    """切菜任务环境"""
    
    cfg: LoftCutVegEnvCfg
    
    def __init__(self, cfg: LoftCutVegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.robot.data.joint_pos
    
    def _setup_scene(self):
        super()._setup_scene()
        
        # 创建资产
        self.robot = Articulation(self.cfg.robot)
        self.carrot = RigidObject(self.cfg.carrot)
        self.board = RigidObject(self.cfg.board)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        
        # 注册到场景
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["carrot"] = self.carrot
        self.scene.rigid_objects["board"] = self.board
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
    
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)
    
    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        joint_pos = torch.cat([self.joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1).squeeze(0)
        
        top_camera_rgb = self.top_camera.data.output["rgb"]
        wrist_camera_rgb = self.wrist_camera.data.output["rgb"]
        
        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu().detach().numpy().squeeze(),
            "observation.images.wrist_rgb": wrist_camera_rgb.cpu().detach().numpy().squeeze(),
        }
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        # TODO: 实现奖励函数（例如：切菜进度）
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out
    
    def _get_successes(self) -> torch.Tensor:
        # TODO: 判断是否切完（例如：胡萝卜分成多段）
        successes = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        return successes
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # 重置机器人
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_position_to_sim(joint_pos, joint_ids=None, env_ids=env_ids)
        
        # 重置胡萝卜位置（加入随机化）
        carrot_pos = self.carrot.data.default_root_state[env_ids].clone()
        rand_vals = torch.empty(len(env_ids), 2, device=carrot_pos.device).uniform_(-0.05, 0.05)
        carrot_pos[..., :2] += rand_vals
        carrot_pos[..., 7:] = 0.0
        self.carrot.write_root_state_to_sim(carrot_pos, env_ids=env_ids)
        
        # 重置砧板
        board_pos = self.board.data.default_root_state[env_ids].clone()
        self.board.write_root_state_to_sim(board_pos, env_ids=env_ids)
    
    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        from lehome.devices.action_process import preprocess_device_action
        return preprocess_device_action(action, teleop_device)
    
    def initialize_obs(self):
        pass  # 无需特殊初始化
    
    def get_all_pose(self):
        poses = {}
        carrot_state = self.carrot.data.root_state_w[0]
        poses["carrot"] = torch.cat([carrot_state[:3], carrot_state[3:7]]).cpu().numpy()
        board_state = self.board.data.root_state_w[0]
        poses["board"] = torch.cat([board_state[:3], board_state[3:7]]).cpu().numpy()
        return poses
    
    def set_all_pose(self, pose, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.carrot._ALL_INDICES
        
        if "carrot" in pose:
            carrot_pose = pose["carrot"]
            carrot_state = self.carrot.data.default_root_state[env_ids].clone()
            if isinstance(carrot_pose, np.ndarray):
                carrot_pose = torch.from_numpy(carrot_pose).float()
            carrot_state[..., :7] = carrot_pose[:7]
            carrot_state[..., 7:] = 0.0
            self.carrot.write_root_state_to_sim(carrot_state, env_ids=env_ids)
        
        if "board" in pose:
            # 类似处理 board
            pass
```

### 步骤 4：注册任务

```python
# source/lehome/lehome/tasks/kitchen/__init__.py
import lehome.tasks.kitchen.loft_cut_veg as loft_cut_veg_env

import gymnasium as gym
gym.register(
    id="Isaac-Loft-Cut-Veg-v0",
    entry_point="lehome.tasks.kitchen.loft_cut_veg:LoftCutVegEnv",
    kwargs={"env_cfg_entry_point": loft_cut_veg_env.LoftCutVegEnvCfg},
)
```

### 步骤 5：测试运行

```python
# test_cut_veg.py
import gymnasium as gym

# 创建环境
env = gym.make("Isaac-Loft-Cut-Veg-v0", num_envs=1)

# 重置
obs, info = env.reset()

# 运行
for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## 8. 最佳实践与注意事项

### 8.1 开发建议

1. **从简单开始**: 先从单臂 + 刚体物体开始，再逐步增加复杂度
2. **复用配置**: 使用 `.replace()` 复用现有机器人/相机配置
3. **模块化**: 将成功判定、奖励计算等逻辑独立成函数
4. **测试**: 创建简单的测试脚本验证环境能否正常运行

### 8.2 常见问题

#### 问题 1：物体掉落或穿模

**原因**: 碰撞设置不正确

**解决**:
```python
# 在配置中增加碰撞参数
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=10.0,  # 增加这个参数
)
```

#### 问题 2：布料/流体物体无法显示

**原因**: 未调用 `initialize()`

**解决**:
```python
def initialize_obs(self):
    self.garment.initialize()  # 必须调用！
    self.fluid.initialize()    # 必须调用！
```

#### 问题 3：相机图像全黑

**原因**: 相机位置或朝向不正确

**解决**:
- 检查 `offset` 中的 `pos` 和 `rot`
- 使用 IsaacSim 的可视化工具调试相机位置
- 确认 `convention="ros"` 坐标系正确

#### 问题 4：动作空间不匹配

**原因**: 配置的 `action_space` 与实际维度不符

**解决**:
```python
# 单臂 6 DOF
action_space = 6

# 双臂 12 DOF
action_space = 12

# 确保 _apply_action() 中的维度匹配
def _apply_action(self):
    self.left_arm.set_joint_position_target(self.actions[:, :6])
    self.right_arm.set_joint_position_target(self.actions[:, 6:])
```

### 8.3 性能优化

1. **减少渲染频率**: 设置 `render_interval > 1`
2. **使用 Fabric**: `use_fabric=True`（适用于简单场景）
3. **降低求解器迭代次数**: 根据任务复杂度调整 `solver_position_iteration_count`
4. **批量环境**: 增加 `num_envs` 以并行训练

### 8.4 调试技巧

1. **可视化边界框**:
```python
# 在 IsaacSim 中启用调试绘制
self.carrot.visualize_bounding_box(enable=True)
```

2. **打印关节状态**:
```python
print(f"Joint pos: {self.robot.data.joint_pos}")
print(f"Joint vel: {self.robot.data.joint_vel}")
```

3. **使用 IsaacSim GUI**:
   - 手动调整物体位置
   - 查看碰撞体
   - 调试粒子系统

## 9. 附录

### 9.1 常用坐标系

- **World 坐标系**: 场景全局坐标系
- **Robot Base 坐标系**: 机器人基座坐标系
- **Camera 坐标系**: 相机坐标系（ROS 约定：x前 y左 z上）

### 9.2 四元数约定

四元数格式: `(w, x, y, z)` 或 `(x, y, z, w)`

**IsaacLab 使用**: `(w, x, y, z)` （标量在前）

示例：
- 无旋转: `(1, 0, 0, 0)` 或 `(0, 0, 0, 1)`
- 绕 Z 轴旋转 90°: `(0.707, 0, 0, 0.707)`

### 9.3 相关资源

- **IsaacLab 文档**: https://isaac-sim.github.io/IsaacLab/
- **LeRobot 文档**: https://github.com/huggingface/lerobot
- **NVIDIA IsaacSim**: https://developer.nvidia.com/isaac-sim

### 9.4 项目脚本说明

详见各脚本目录下的 README 文件：

- `scripts/teleoperation/README_TELEOP_RECORD.md` - 遥操作和数据记录
- `scripts/tool/README_REPLAY.md` - 数据回放
- `scripts/eval/README_EVAL_IL.md` - 模仿学习评估

---

**文档版本**: v1.0
**最后更新**: 2025-12-09
**维护者**: Lehome 开发团队

