from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

"""Configuration for the SO101 Follower Robot."""

FRANKA_ASSET_PATH = Path(ISAAC_NUCLEUS_DIR) / "Robots" / "FrankaRobotics" / "FrankaPanda"/"franka.usd"


ACTION_NAMES = [
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

FR3_ACTION_NAMES = [
    'fr3_joint1', 
    'fr3_joint2', 
    'fr3_joint3', 
    'fr3_joint4', 
    'fr3_joint5', 
    'fr3_joint6', 
    'fr3_joint7', 
    'fr3_finger_joint1', 
    'fr3_finger_joint2',
]

FRANKA_CFG = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FRANKA_ASSET_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, 
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.02829,
                "panda_joint2": -0.2848,
                "panda_joint3": -0.00921,
                "panda_joint4": -2.1854,
                "panda_joint5": 0.0016,
                "panda_joint6": 1.8824,
                "panda_joint7": 0.7581,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=5200.0,
                velocity_limit_sim=2.175,
                stiffness=1100.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=720.0,
                velocity_limit_sim=2.61,
                stiffness=1000.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

FRANKA_USD_JOINT_LIMLITS = {
    "panda_joint1": (-110.0, 110.0),
    "panda_joint2": (-100.0, 100.0),
    "panda_joint3": (-100.0, 90.0),
    "panda_joint4": (-95.0, 95.0),
    "panda_joint5": (-160.0, 160.0),
    "panda_finger_joint1": (-10.0, 100.0),
    "panda_finger_joint2": (-10.0, 100.0),
}

FRANKA_JOINT_LIMITS = {
    "panda_joint1": {
        "radians": (-2.897, 2.897),  
        "degrees": (-165.89, 165.89),  
    },
    "panda_joint2": {
        "radians": (-1.763, 1.763),  
        "degrees": (-101.05, 101.05),  
    },
    "panda_joint3": {
        "radians": (-2.897, 2.897),  
        "degrees": (-165.89, 165.89),  
    },
    "panda_joint4": {
        "radians": (-3.072, -0.070),  
        "degrees": (-176.05, -4.01),  
    },
    "panda_joint5": {
        "radians": (-2.897, 2.897),  
        "degrees": (-165.89, 165.89),  
    },
    "panda_joint6": {
        "radians": (-0.018, 3.753),  
        "degrees": (-1.03, 215.10),  
    },
    "panda_joint7": {
        "radians": (-2.897, 2.897),  
        "degrees": (-165.89, 165.89),  
    },
    "panda_finger_joint1": {
        "radians": (0.000, 0.040),  
        "degrees": (0.00, 2.29),  
    },
    "panda_finger_joint2": {
        "radians": (0.000, 0.040),  
        "degrees": (0.00, 2.29),  
    },
}

FRANKA_MOTOR_LIMITS = {
    "panda_joint1": (-110.0, 110.0),
    "panda_joint2": (-100.0, 100.0),
    "panda_joint3": (-100.0, 90.0),
    "panda_joint4": (-95.0, 95.0),
    "panda_joint5": (-160.0, 160.0),
    "panda_finger_joint1": (-10.0, 100.0),
    "panda_finger_joint2": (-10.0, 100.0),
}

FR3_JOINT_LIMITS = {
    "fr3_joint1": {
        "radians": (-2.744, 2.744),  # 弧度
        "degrees": (-157.2, 157.2),  # 角度
    },
    "fr3_joint2": {
        "radians": (-1.784, 1.784),  # 弧度
        "degrees": (-102.2, 102.2),  # 角度
    },
    "fr3_joint3": {
        "radians": (-2.901, 2.901),  # 弧度
        "degrees": (-166.2, 166.2),  # 角度
    },
    "fr3_joint4": {
        "radians": (-3.042, -0.152),  # 弧度
        "degrees": (-174.3, -8.7),  # 角度
    },
    "fr3_joint5": {
        "radians": (-2.806, 2.806),  # 弧度
        "degrees": (-160.8, 160.8),  # 角度
    },
    "fr3_joint6": {
        "radians": (0.544, 4.517),  # 弧度
        "degrees": (31.2, 258.8),  # 角度
    },
    "fr3_joint7": {
        "radians": (-3.016, 3.016),  # 弧度
        "degrees": (-172.8, 172.8),  # 角度
    },
    "fr3_finger_joint1": {
        "radians": (0.000, 0.040),  # 弧度
        "degrees": (0.0, 2.29),  # 角度
    },
    "fr3_finger_joint2": {
        "radians": (0.000, 0.040),  # 弧度
        "degrees": (0.0, 2.29),  # 角度
    },
}

