import pandas as pd
import math
from typing import List, Tuple

# 定义关节顺序
JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# sim 的关节限制（度）
SIM_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# real 的关节限制（度）
REAL_LIMITS_DEG = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

# 构建 6 维与 12 维的限制表
REAL_LIMITS_6 = [REAL_LIMITS_DEG[j] for j in JOINT_ORDER]
REAL_LIMITS_12 = REAL_LIMITS_6 + REAL_LIMITS_6

SIM_LIMITS_6 = [SIM_LIMITS_DEG[j] for j in JOINT_ORDER]
SIM_LIMITS_12 = SIM_LIMITS_6 + SIM_LIMITS_6


def inverse_linear_map_to_sim(
    values_real: List[float],
    sim_limits: List[Tuple[float, float]],
    real_limits: List[Tuple[float, float]],
) -> List[float]:
    """逆线性映射：将 real 区间的值映射回 sim 区间，并将角度转换为弧度"""
    mapped = []
    for v, (sim_lo, sim_hi), (real_lo, real_hi) in zip(
        values_real, sim_limits, real_limits
    ):
        if real_hi == real_lo:
            mapped.append(math.radians(sim_lo))  # 避免除零，退化成 sim_lo，并转换为弧度
        else:
            ratio = (v - real_lo) / (real_hi - real_lo)
            mapped_val = sim_lo + ratio * (sim_hi - sim_lo)
            mapped.append(math.radians(mapped_val))  # 转换为弧度
    return mapped


def process_parquet_file(input_path: str, output_path: str):
    """
    处理 Parquet 文件，将 action 和 observation.state 列中的 real 数值映射到 sim 数值，并保存为新的 Parquet 文件。
    
    :param input_path: 输入 Parquet 文件路径
    :param output_path: 输出 Parquet 文件路径
    """
    df = pd.read_parquet(input_path)

    # 打印 DataFrame
    # print(df.head())
    # 映射 action 列
    if "action" in df.columns:
        df["action"] = df["action"].apply(
            lambda x: inverse_linear_map_to_sim(x, SIM_LIMITS_6, REAL_LIMITS_6)
        )

    # 映射 observation.state 列
    if "observation.state" in df.columns:
        df["observation.state"] = df["observation.state"].apply(
            lambda x: inverse_linear_map_to_sim(x, SIM_LIMITS_12, REAL_LIMITS_12)
        )

    # 保存到新的 Parquet 文件
    df.to_parquet(output_path)
    print(f"Processed file saved to {output_path}")


# 示例调用
input_parquet_path = "/home/feng/lehome_1/datasets/idm_last/data/chunk-000/real.parquet"
output_parquet_path = "/home/feng/lehome_1/datasets/idm_last/data/chunk-000/file-000.parquet"
process_parquet_file(input_parquet_path, output_parquet_path)