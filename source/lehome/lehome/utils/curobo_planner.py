"""
IsaacLab + CuRobo Motion Planner Integration
重构版本：学习自 isaaclab_mimic 的最佳实践
"""

import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Set
from pathlib import Path

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml


X7s_joint_names = [
        "base_x_joint", 
        "base_y_joint", 
        "base_yaw_joint", 
        "body_z_joint",     # joint1
        "body_y_joint",     # joint2
        "right_shoulder_y", # joint14
        "head_z_joint",     # joint3
        "left_shoulder_y",  # joint5
        "right_shoulder_x", # joint15
        "head_y_joint",     # joint4
        "left_shoulder_x",  # joint6
        "right_shoulder_z", # joint16
        "left_shoulder_z",  # joint7
        "right_elbow_y",    # joint17
        "left_elbow_y",     # joint8
        "right_elbow_x",    # joint18
        "left_elbow_x",     # joint9
        "right_wrist_y",    # joint19
        "left_wrist_y",     # joint10
        "right_wrist_z",    # joint20
        "left_wrist_z",     # joint11
        "right_gripper1",   # joint21
        "right_gripper2",   # joint22
        "left_gripper1",    # joint13
        "left_gripper2"     # joint12
        ]

Franka_action_names = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
        'panda_finger_joint1',
        'panda_finger_joint2',
    ]

X7s_robot_joint_names = [
    'base_x_joint', 'base_y_joint', 'base_yaw_joint', 'body_z_joint', 'body_y_joint', 
    'right_shoulder_y', 'head_z_joint', 'left_shoulder_y', 'right_shoulder_x', 
    'head_y_joint', 'left_shoulder_x', 'right_shoulder_z', 'left_shoulder_z', 
    'right_elbow_y', 'left_elbow_y', 'right_elbow_x', 'left_elbow_x', 
    'right_wrist_y', 'left_wrist_y', 'right_wrist_z', 'left_wrist_z',
    "right_gripper1",   # joint21
        "right_gripper2",   # joint22
        "left_gripper1",    # joint13
        "left_gripper2"
]

def map_cu2sim_joint_names(cu_js_names):
    """将 CuRobo 关节名映射到 sim 关节名。
    x7s: CuRobo 顺序与 sim 顺序不同，需要按 X7s_joint_names→X7s_robot_joint_names 映射。
    franka: CuRobo 名称与 sim 名称相同，直接返回。
    """
    sim_js_names = []
    for name in cu_js_names:
        if name in X7s_joint_names:
            idx = X7s_joint_names.index(name)
            sim_js_names.append(X7s_robot_joint_names[idx])
        else:
            # franka 或其他机器人：cu名称即sim名称
            sim_js_names.append(name)
    return sim_js_names


# =============================================================================
# 日志类
# =============================================================================
class PlannerLogger:
    """CuRobo Planner 日志类"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self._name = name
        self._level = level
        self._logger = None
    
    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self._name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(self._level)
        return self._logger
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class CuroboPlannerCfg:
    """CuRobo Planner 配置"""
    
    # 机器人配置
    robot_config_file: str = "x7s.yml"
    robot_prim_path: Optional[str] = None
    
    # 碰撞检测配置
    collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH
    collision_cache: Dict[str, int] = field(default_factory=lambda: {"obb": 1000, "mesh": 500})
    # collision_activation_distance: float = 0.025
    collision_activation_distance: float = 0.05  # 建议增加到 5cm 
    collision_safety_distance: float = 0.02
    
    # 规划配置
    interpolation_dt: float = 0.03
    num_trajopt_seeds: int = 12
    num_graph_seeds: int = 12
    max_planning_attempts: int = 10
    enable_graph: bool = True
    enable_graph_attempt: int = 4
    time_dilation_factor: float = 0.5
    
    # 世界配置
    env_prim_path: str = "/World/envs/env_0"
    world_ignore_substrings: List[str] = field(default_factory=lambda: [
        "/World/envs/env_0/Scene/floor_room",
        "/curobo",
        "Camera",
    ])
    static_obstacle_paths: List[str] = field(default_factory=list)
    
    # 其他配置
    debug: bool = False
    cuda_device: Optional[int] = None
    use_cuda_graph: bool = True


# =============================================================================
# 主 Planner 类
# =============================================================================
class IsaacLabCuroboPlanner:
    """
    Isaac Lab + CuRobo 运动规划器
    
    主要功能：
    - 从 USD stage 读取障碍物
    - 规划无碰撞路径
    - 支持动态世界更新
    - 设备隔离（CuRobo 始终在 CUDA 上运行）
    """
    
    def __init__(
        self, 
        robot_cfg: str | dict,
        world: Any,
        robot_prim_path: str, 
        env: Optional[Any] = None,
        config: Optional[CuroboPlannerCfg] = None,
    ):
        """
        初始化 CuRobo Planner
        
        Args:
            robot_cfg: 机器人配置文件名或配置字典
            world: Isaac Lab World (scene)
            robot_prim_path: 机器人 USD 路径，如 "/World/envs/env_0/Robot"
            env: Isaac Lab 环境实例
            config: 配置对象，如果为 None 则使用默认配置
        """
        # 配置
        self.config = config or CuroboPlannerCfg()
        if isinstance(robot_cfg, str):
            self.config.robot_config_file = robot_cfg
        self.config.robot_prim_path = robot_prim_path
        
        # 日志
        log_level = logging.DEBUG if self.config.debug else logging.INFO
        self.logger = PlannerLogger("CuroboPlanner", log_level)
        
        # 环境引用
        self.world = world
        self.env = env
        self.robot_prim_path = robot_prim_path
        
        # 设置 CuRobo 日志级别
        setup_curobo_logger("warn")
        
        self._init_tensor_args()
        
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(world.stage)
        
        self.robot_cfg_dict = self._load_robot_config(robot_cfg)
        
        dt_control = self._get_control_dt(env)
        self.config.interpolation_dt = dt_control
        
        self._init_motion_gen()
       
        self.cu_traj: Optional[JointState] = None
        self._cached_object_mappings: Optional[Dict[str, str]] = None
        self._expected_objects: Optional[Set[str]] = None
        
        
        # 支持的 CuRobo primitive 类型
        self.primitive_types = ["mesh", "cuboid", "sphere", "capsule", "cylinder"]
        
        self.logger.info(f"CuRobo Planner 初始化完成. DOF: {self.n_dof}, Joints: {self.joint_names}")
    
    # =========================================================================
    # 初始化方法
    # =========================================================================
    
    def _init_tensor_args(self):
        """初始化 tensor 参数，强制使用 CUDA"""
        if torch.cuda.is_available():
            idx = self.config.cuda_device if self.config.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self.logger.debug(f"CuRobo 使用 CUDA 设备 {idx}")
        else:
            self.tensor_args = TensorDeviceType()
            self.logger.warning("CUDA 不可用，CuRobo 使用 CPU - 可能影响性能")
    
    def _load_robot_config(self, robot_cfg: str | dict) -> dict:
        """加载机器人配置"""
        # root_path = Path(__file__).resolve().parent.parent
        root_path = Path("/home/feng/lehome_1/LW-BenchHub/lwautosim-lw_benchhub-success_task")
        configs_path = root_path / "configs" / "content" / "configs"
        assets_path = root_path / "configs" / "content" / "assets"
        
        if isinstance(robot_cfg, str):
            self.logger.info(f"加载机器人配置: {robot_cfg}")
            content_path = ContentPath(
                robot_config_root_path=str(configs_path / "robot"),
                robot_urdf_root_path=str(assets_path),
                robot_asset_root_path=str(assets_path),
                robot_config_file=robot_cfg,
            )
            robot_data = load_robot_yaml(content_path)
            robot_data["robot_cfg"]["kinematics"]["external_asset_path"] = str(assets_path)
            return robot_data
        else:
            self.logger.info("使用自定义机器人配置")
            return robot_cfg
    
    def _get_control_dt(self, env: Optional[Any]) -> float:
        """获取控制时间步"""
        if env is not None and hasattr(env, 'cfg'):
            try:
                physics_dt = env.cfg.sim.dt
                decimation = env.cfg.decimation
                dt_control = physics_dt * decimation
                self.logger.info(f"控制 dt: physics_dt={physics_dt}, decimation={decimation}, dt_control={dt_control}")
                return dt_control
            except AttributeError:
                pass
        
        self.logger.warning("无法获取 dt，使用默认值 0.01")
        return 0.01
    
    def _init_motion_gen(self):
        """初始化 MotionGen"""
        

        world_cfg = WorldConfig()

            
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg_dict,
            world_cfg,
            self.tensor_args,
            interpolation_dt=self.config.interpolation_dt,
            collision_checker_type=self.config.collision_checker_type,
            collision_cache=self.config.collision_cache,
            collision_activation_distance=self.config.collision_activation_distance,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            num_graph_seeds=self.config.num_graph_seeds,
            use_cuda_graph=self.config.use_cuda_graph,
            fixed_iters_trajopt=True,
            maximum_trajectory_dt=0.5,
            ik_opt_iters=500,
        )
        
        self.motion_gen = MotionGen(motion_gen_config)

        # 获取关节名称
        if hasattr(self.motion_gen, 'kinematics'):
            self.joint_names = self.motion_gen.kinematics.joint_names
        elif hasattr(self.motion_gen, 'robot_config'):
            self.joint_names = self.motion_gen.robot_config.kinematics.joint_names
        else:
            self.joint_names = self.robot_cfg_dict.get("robot_cfg", {}).get("kinematics", {}).get("cspace", {}).get("joint_names", [])
        
        self.n_dof = len(self.joint_names)
    
    # =========================================================================
    # 设备转换工具
    # =========================================================================
    
    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 tensor 转换到 CuRobo 设备"""
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)
    
    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将 tensor 转换回环境设备"""
        if self.env is not None:
            return tensor.to(device=self.env.device, dtype=tensor.dtype)
        return tensor
    
    # =========================================================================
    # 世界管理
    # =========================================================================
    
    def warmup(self):
        """预热 MotionGen"""
        self.logger.info("预热 MotionGen...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.logger.info("预热完成")
    
    def initialize_static_world(
        self, 
        only_paths: Optional[List[str]] = None,
        ignore_substrings: Optional[List[str]] = None,
    ):
        """
        初始化静态世界几何体（只调用一次）
        
        Args:
            only_paths: 只包含这些路径下的障碍物
            ignore_substrings: 忽略包含这些子字符串的路径
        """
        env_prim_path = self.config.env_prim_path
        
        # 默认忽略列表
        default_ignore = [
            "/World/envs/env_0/Scene/floor_room",
            "/curobo",
        ]
        ignore_list = ignore_substrings or default_ignore
        
        # 如果没有指定 only_paths，使用 env_prim_path
        search_paths = only_paths or [env_prim_path]
        
        self.logger.info(f"初始化静态世界: paths={search_paths}")
        self.logger.debug(f"忽略列表: {ignore_list}")
        
        try:
            obstacles = self.usd_helper.get_obstacles_from_stage(
                only_paths=search_paths,
                reference_prim_path=self.robot_prim_path,
                ignore_substring=[],
            ).get_collision_check_world()
            
            self.motion_gen.update_world(obstacles)
            
            # 记录障碍物数量
            mesh_count = len(obstacles.mesh) if obstacles.mesh else 0
            cuboid_count = len(obstacles.cuboid) if obstacles.cuboid else 0
            self.logger.info(f"静态世界初始化完成: {mesh_count} meshes, {cuboid_count} cuboids")
            
        except Exception as e:
            self.logger.error(f"静态世界初始化失败: {e}")
            raise
    
    def update_obstacles(
        self,
        only_paths: Optional[List[str]] = ["/World/envs/env_0/Scene/"],
        extra_ignore: Optional[List[str]] = None,
        debug: bool = False,
    ):
        """
        更新障碍物（兼容旧接口）
        
        Args:
            only_paths: 只包含这些路径下的障碍物（如果为 None，使用 env_prim_path）
            extra_ignore: 额外要忽略的路径子字符串
            debug: 是否打印调试信息
        """
        env_prim_path = self.config.env_prim_path
        
        # 构建忽略列表
        ignore_list = [
            "/World/envs/env_0/Scene/floor_room",
            "/curobo",
            "Camera",
        ]
        if extra_ignore:
            ignore_list.extend(extra_ignore)
        
        # 搜索路径
        search_paths = only_paths 
        
        if debug:
            self.logger.info(f"更新障碍物: paths={search_paths}, ignore={ignore_list}")
        
        try:
            obstacles = self.usd_helper.get_obstacles_from_stage(
                only_paths=search_paths,
                reference_prim_path=self.robot_prim_path,
                ignore_substring=[],
            ).get_collision_check_world()
            
            self.motion_gen.update_world(obstacles)
            
            if debug:
                mesh_count = len(obstacles.mesh) if obstacles.mesh else 0
                cuboid_count = len(obstacles.cuboid) if obstacles.cuboid else 0
                self.logger.info(f"障碍物更新完成: {mesh_count} meshes, {cuboid_count} cuboids")
                
        except Exception as e:
            self.logger.error(f"障碍物更新失败: {e}")
    
    def sync_dynamic_objects(self):
        """
        同步动态对象位姿（仅更新位姿，不重新加载几何体）
        
        这个方法用于高频更新，比如每帧同步物体位置
        """
        if self.env is None:
            self.logger.warning("env 为 None，无法同步动态对象")
            return
        
        # 获取对象映射
        object_mappings = self._get_object_mappings()
        
        if not object_mappings:
            return
        
        # 同步每个对象的位姿
        rigid_objects = self.env.scene.rigid_objects if hasattr(self.env.scene, 'rigid_objects') else {}
        updated_count = 0
        
        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue
            
            try:
                obj = rigid_objects[object_name]
                env_origin = self.env.scene.env_origins[0]  # env_id = 0
                current_pos = obj.data.root_pos_w[0] - env_origin
                current_quat = obj.data.root_quat_w[0]  # (w, x, y, z)
                
                # 创建 CuRobo Pose
                curobo_pose = Pose(
                    position=self._to_curobo_device(current_pos),
                    quaternion=self._to_curobo_device(current_quat),
                )
                
                # 更新碰撞检测器中的位姿
                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    object_path, curobo_pose, update_cpu_reference=True
                )
                updated_count += 1
                
            except Exception as e:
                self.logger.debug(f"同步对象 {object_name} 失败: {e}")
        
        if updated_count > 0:
            self.logger.debug(f"同步了 {updated_count} 个动态对象")
    
    def _get_object_mappings(self) -> Dict[str, str]:
        """获取对象映射（带缓存）"""
        if self._cached_object_mappings is not None:
            return self._cached_object_mappings
        
        if self.env is None:
            return {}
        
        # 计算映射
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects if hasattr(self.env.scene, 'rigid_objects') else {}
        
        mappings = {}
        env_prefix = f"{self.config.env_prim_path}/"
        world_object_paths = []
        
        # 收集世界模型中的所有对象路径
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type, None)
            if primitive_list:
                for primitive in primitive_list:
                    if primitive.name and env_prefix in str(primitive.name):
                        world_object_paths.append(str(primitive.name))
        
        # 匹配 Isaac Lab 对象名称到世界路径
        for object_name in rigid_objects.keys():
            for path in world_object_paths:
                if object_name.lower().replace("_", "") in path.lower().replace("_", ""):
                    mappings[object_name] = path
                    self.logger.debug(f"对象映射: {object_name} -> {path}")
                    break
        
        self._cached_object_mappings = mappings
        return mappings
    
    def invalidate_object_cache(self):
        """使对象映射缓存失效"""
        self._cached_object_mappings = None
    
    # =========================================================================
    # 规划方法
    # =========================================================================
    
    def plan(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_q: np.ndarray,
        current_qd: Optional[np.ndarray] = None,
        sim_ordered_cu_js_names: Optional[List[str]] = None,
        link_goal: Optional[dict] = None,
        position_only: bool = False,
    ) -> Optional[np.ndarray]:
        """
        规划轨迹
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标四元数 [qw, qx, qy, qz]
            current_q: 当前关节位置
            current_qd: 当前关节速度
            sim_ordered_cu_js_names: 仿真关节名称列表
            link_goal: 额外的链接目标
            
        Returns:
            轨迹 (T, dof) 或 None
        """
        if current_qd is None:
            current_qd = np.zeros_like(current_q)
        
        target_joint_names = self.joint_names
        if sim_ordered_cu_js_names is not None and len(sim_ordered_cu_js_names) != len(self.joint_names):
            target_joint_names = sim_ordered_cu_js_names
        dof_needed = len(target_joint_names)
        
        # 调整关节数量
        if len(current_q) < dof_needed:
            pad = np.zeros(dof_needed - len(current_q), dtype=current_q.dtype)
            current_q = np.concatenate([current_q, pad], axis=0)
            current_qd = np.concatenate([current_qd, np.zeros_like(pad)], axis=0)
        elif len(current_q) > dof_needed:
            current_q = current_q[:dof_needed]
            current_qd = current_qd[:dof_needed]
        
        # 构建目标 Pose
        goal = Pose(
            position=self.tensor_args.to_device(target_pos),
            quaternion=self.tensor_args.to_device(target_quat),
        )
        
        # 构建当前状态
        state = JointState(
            position=self.tensor_args.to_device(current_q),
            velocity=self.tensor_args.to_device(current_qd) * 0.0,
            acceleration=self.tensor_args.to_device(current_qd) * 0.0,
            jerk=self.tensor_args.to_device(current_qd) * 0.0,
            joint_names=sim_ordered_cu_js_names,
        )
        
        cu_js = state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # 规划配置
        pose_cost_metric = None
        if position_only:
            from curobo.rollout.cost.pose_cost import PoseCostMetric
            pose_cost_metric = PoseCostMetric(
                reach_partial_pose=True,
                reach_vec_weight=self.tensor_args.to_device(
                    np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
                ),
            )
        plan_config = MotionGenPlanConfig(
            enable_graph=self.config.enable_graph,
            enable_graph_attempt=self.config.enable_graph_attempt,
            max_attempts=self.config.max_planning_attempts,
            time_dilation_factor=self.config.time_dilation_factor,
            pose_cost_metric=pose_cost_metric,
        )
        
        # 执行规划
        result = self.motion_gen.plan_single(
            cu_js.unsqueeze(0),
            goal,
            plan_config.clone(),
            link_poses=link_goal,
        )
        
        if result.success.item():
            self.cu_traj = result.get_interpolated_plan()
            cmd_plan = self.cu_traj.get_ordered_joint_state(sim_ordered_cu_js_names)
            sim_js_names = map_cu2sim_joint_names(sim_ordered_cu_js_names)
            sim_traj = cmd_plan.clone()
            sim_traj.joint_names = sim_js_names
            
            self.logger.debug(f"规划成功: {len(sim_traj.position)} 个路点")
            return sim_traj.position.cpu().numpy()
        
        self.logger.warning(f"规划失败: {result.status}")

        # 调试：直接用 MotionGen 内部的 ik_solver（与 plan_single 完全一致）
        try:
            ik_solver = self.motion_gen.ik_solver

            # 1) 只测位置（identity quat）
            goal_pos_only = Pose(
                position=self.tensor_args.to_device(target_pos),
                quaternion=self.tensor_args.to_device(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            )
            res_pos = ik_solver.solve_single(goal_pos_only, retract_config=cu_js.position)
            print(f"[IK DEBUG] target_pos={np.round(target_pos, 3)}, target_quat={np.round(target_quat, 4)}")
            print(f"[IK DEBUG] 1) position-only: success={res_pos.success.item()}, err_pos={res_pos.position_error.item():.5f}m")

            # 2) 完整 pose
            res_full = ik_solver.solve_single(goal, retract_config=cu_js.position)
            print(f"[IK DEBUG] 2) full pose:     success={res_full.success.item()}, err_pos={res_full.position_error.item():.5f}m, err_rot={res_full.rotation_error.item():.5f}rad")

            if not res_pos.success.item():
                print(f"[IK DEBUG] → 位置本身不可达")
            elif res_pos.success.item() and not res_full.success.item():
                print(f"[IK DEBUG] → 位置可达，朝向不可达（关节限位/奇异点）")
            else:
                print(f"[IK DEBUG] → IK 成功但 MotionGen plan_single 流程仍失败（轨迹优化/碰撞问题）")
        except Exception as e:
            print(f"[IK DEBUG] 失败: {e}")

        # 碰撞体信息
        try:
            wc = self.motion_gen.world_model
            mesh_names  = [m.name for m in wc.mesh]   if wc.mesh   else []
            cube_names  = [c.name for c in wc.cuboid] if wc.cuboid else []
            sphere_names= [s.name for s in wc.sphere] if wc.sphere else []
            print(f"[COLLISION] world meshes={mesh_names}, cuboids={cube_names}, spheres={sphere_names}")
        except Exception as e:
            print(f"[COLLISION] 获取碰撞体失败: {e}")

        return None
    
    def get_ee_traj(self) -> tuple:
        """获取末端执行器轨迹"""
        if self.cu_traj is None:
            return None, None
        
        q_traj = self.cu_traj.position
        if not isinstance(q_traj, torch.Tensor):
            q_traj = torch.tensor(q_traj, device=self.tensor_args.device)
        
        kinematics = self.motion_gen.kinematics
        ee_state = kinematics.get_state(q_traj)
        
        return ee_state.ee_position, ee_state.ee_quaternion
    
    def reset(self):
        """重置 planner 状态"""
        self.cu_traj = None
        self.motion_gen.reset()