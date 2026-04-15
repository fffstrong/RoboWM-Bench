import math
import numpy as np
import torch
from lehome.utils.logger import get_logger
logger = get_logger(__name__)

def step_interval(interval=50):
    """工厂函数：创建一个可自定义步长的装饰器"""

    def decorator(func):
        call_count = 0

        def wrapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count % interval == 0:
                return func(*args, **kwargs)
            else:
                return False

        return wrapper

    return decorator


# def calculate_distance(point_a, point_b):
#     # calculate distance
#     point_a = np.array(point_a)
#     point_b = np.array(point_b)
#     return np.linalg.norm(point_a - point_b)


def get_object_particle_position(particle_object, index_list):
    position = (
        particle_object._cloth_prim_view.get_world_positions()
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        * 100
    )
    select_position = []
    for index in index_list:
        select_position.append(tuple(position[index]))
    return select_position

def get_cloth_object_particle_position(particle_object, index_list):
    try:
        _, mesh_points, _, _ = particle_object.get_current_mesh_points()
    except Exception as e1:
        try:
            logger.error(f"Error in get_object_particle_position: {e1}")
            mesh_points = (
                particle_object._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        except Exception as e2:
            logger.error(f"Error in get_object_particle_position: {e2}")
            return
    positions = (mesh_points[index_list] * 100).tolist()
    return positions

def get_fluid_position(fluid_object, sample_count: int = 100, global_coord: bool = False):
    """
    随机抽取流体粒子位置。

    Args:
        fluid_object: FluidObject 实例，需实现 get_particle_positions。
        sample_count: 抽取粒子数量，超过总数则返回全部。
        global_coord: 是否返回全局坐标。

    Returns:
        list[tuple]: 抽取到的粒子坐标列表。
    """
    positions, _, _ = fluid_object.get_particle_positions(
        visualize=False, global_coord=global_coord
    )
    if len(positions) == 0:
        return []

    sample_count = min(sample_count, len(positions))
    indices = np.random.choice(len(positions), size=sample_count, replace=False)
    return [tuple(positions[i]) for i in indices]

@step_interval(interval=10)
def success_checker_pick(
    rigid_object_a, ori_z,env_id: int = 0
):

    pos_a = rigid_object_a.data.root_pos_w[env_id].clone()     # (3,)

    a_z = pos_a[2].item()
    
    success = (
        abs(a_z-ori_z)>0.05
    )
    # print(f"[Pick] ori_z={ori_z}, a_z={a_z}")
    # print(f"[Pick] success={success}")
    return bool(success)

def success_checker_pick_once(
    rigid_object_a, ori_z,env_id: int = 0
):

    pos_a = rigid_object_a.data.root_pos_w[env_id].clone()     # (3,)

    a_z = pos_a[2].item()
    
    success = (
        abs(a_z-ori_z)>0.05
    )
    # print(f"[Pick] ori_z={ori_z}, a_z={a_z}")
    # print(f"[Pick] success={success}")
    return bool(success)

# @step_interval(interval=50)
def success_checker_pour(
    fluid_object,
    sample_count: int = 100,
    xy_tolerance: float = 0.08,
    z_tolerance: float = 0.12,
    success_ratio: float = 0.7,
):
    """
    检查流体是否“倒入”容器：随机取部分粒子，判断是否有足够比例落在容器附近。

    Args:
        fluid_object: FluidObject 实例，需要实现 get_particle_positions 和 container.get_world_pose。
        sample_count: 随机抽取粒子数。
        xy_tolerance: 粒子与容器中心在 xy 平面的容差（米）。
        success_ratio: 满足条件的粒子比例阈值。

    Returns:
        bool: True 表示达到成功条件。
    """
    particles, _, _ = fluid_object.get_particle_positions(visualize=False, global_coord=True)
    if len(particles) == 0:
        return False

    sample_count = min(sample_count, len(particles))
    indices = np.random.choice(len(particles), size=sample_count, replace=False)
    sampled = np.array([particles[i] for i in indices], dtype=np.float32)

    # 使用 loft_water_cfg 中 bowl 的初始化位置作为容器中心

    from lehome.tasks.livingroom.loft_water_cfg import LoftWaterEnvCfg

    cfg = LoftWaterEnvCfg()
    container_pos = np.array(cfg.bowl.init_state.pos, dtype=np.float32)
# container_pos: [2.91405, 0.87336, 0.83]

    diff = np.abs(sampled - container_pos)
    in_xy = (diff[:, 0] <= xy_tolerance) & (diff[:, 1] <= xy_tolerance)
    in_z = diff[:, 2] <= z_tolerance
    inside = in_xy & in_z

    ratio = inside.mean() if len(inside) > 0 else 0.0
    return bool(ratio >= success_ratio)
    
@step_interval(interval=50)
def success_checker_orangeinbowl(
    bowl_object_a, bowl_object_b
):
    position_a = bowl_object_a._position()
    position_b = bowl_object_b._position()
    a_x = position_a[0].item()
    a_y = position_a[1].item()
    a_z = position_a[2].item()
    b_x = position_b[0].item()
    b_y = position_b[1].item()
    b_z = position_b[2].item()
    print(calculate_distance(a_x,b_x))
    print(calculate_distance(a_y,b_y))
    print(calculate_distance(a_z,b_z))
    success = (
        calculate_distance(a_x, b_x) <= 0.05
        and calculate_distance(a_y, b_y) <= 0.05
        and calculate_distance(a_z, b_z) <= 0.05
    )
    return bool(success)

@step_interval(interval=5)
def success_checker_bowlinplate(
    rigid_object_a, rigid_object_b, env_id: int = 0
):
    pos_a = rigid_object_a.data.root_pos_w[env_id]      # (3,)
    pos_b = rigid_object_b.data.root_pos_w[env_id]
    a_x = pos_a[0].item()
    a_y = pos_a[1].item()
    b_x = pos_b[0].item()
    b_y = pos_b[1].item()

    success = (
        calculate_distance(a_x, b_x) <= 0.15
        and calculate_distance(a_y, b_y) <= 0.15
    )
    return bool(success)

# @step_interval(interval=50)
def success_checker_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_cloth_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)

@step_interval(interval=50)
def success_checker_fling(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)

    def xy_distance(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def z_distance(a, b):
        return abs(a[2] - b[2])

    success = (
        xy_distance(p[0], p[4]) > 18
        and z_distance(p[0], p[4]) < 2
        and xy_distance(p[1], p[5]) > 18
        and z_distance(p[1], p[5]) < 2
    )

    return bool(success)


@step_interval(interval=30)
def success_checker_burger(beef_pos, plate_pos):
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    # z distance
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # success condition: both xy < 0.05 and z < 0.1
    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2

def calculate_distance_2D(point_a, point_b):
    """计算二维坐标点的直线距离"""
    a = np.array(point_a, dtype=np.float32).reshape(-1)[:2]
    b = np.array(point_b, dtype=np.float32).reshape(-1)[:2]
    return float(np.linalg.norm(a - b))

def calculate_distance(point_a, point_b):
    # Calculate distance
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)

@step_interval(interval=50)
def success_checker_pickandplace(
    rigid_object_a, rigid_object_b, env_id: int = 0
):
    pos_a = rigid_object_a.data.root_pos_w[env_id]      # (3,)
    pos_b = rigid_object_b.data.root_pos_w[env_id]
    a_x = pos_a[0].item()
    a_y = pos_a[1].item()
    b_x = pos_b[0].item()
    b_y = pos_b[1].item()

    print(calculate_distance_2D((a_x, a_y), (b_x, b_y)))
    # print(calculate_distance(a_y,b_y))

    success = (
        calculate_distance_2D((a_x, a_y), (b_x, b_y))<= 0.05
    )
    return bool(success)

@step_interval(interval=5)
def success_checker_pull(
    rigid_object_a,ori_x, env_id: int = 0
):
    env_id = int(env_id)
    pos_a = rigid_object_a.data.root_pos_w[env_id] 
    a_x = pos_a[0].item()     # (3,)
    # print(f"Object A position: {pos_a} | Target ori_x: {ori_x} |  {abs(a_x-ori_x)}")
    

    success = (
        abs(a_x-ori_x)>0.03
    )
    return bool(success)

@step_interval(interval=30)
def success_checker_rubbish(food_rubbish01_pos, food_rubbish02_pos, food_rubbish03_pos, desktop_dustpan_pos, dustpan_size_x=0.15, dustpan_size_y=0.15):

    diff_xy_1 = food_rubbish01_pos[:, :2] - desktop_dustpan_pos[:, :2]
    dist_x_1 = torch.abs(diff_xy_1[:, 0])
    dist_y_1 = torch.abs(diff_xy_1[:, 1])
    in_dustpan_1 = (dist_x_1 <= dustpan_size_x) & (dist_y_1 <= dustpan_size_y)

    diff_xy_2 = food_rubbish02_pos[:, :2] - desktop_dustpan_pos[:, :2]
    dist_x_2 = torch.abs(diff_xy_2[:, 0])
    dist_y_2 = torch.abs(diff_xy_2[:, 1])
    in_dustpan_2 = (dist_x_2 <= dustpan_size_x) & (dist_y_2 <= dustpan_size_y)
    
    diff_xy_3 = food_rubbish03_pos[:, :2] - desktop_dustpan_pos[:, :2]
    dist_x_3 = torch.abs(diff_xy_3[:, 0])
    dist_y_3 = torch.abs(diff_xy_3[:, 1])
    in_dustpan_3 = (dist_x_3 <= dustpan_size_x) & (dist_y_3 <= dustpan_size_y)
    
    success_mask = in_dustpan_1 & in_dustpan_2 & in_dustpan_3
    success = success_mask.any().item()
    
    return bool(success)

@step_interval(interval=50)
def success_checker_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


def check_top_sleeve(p, success_distance):
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_2_3 = calculate_distance(p[2], p[3])
    dist_1_5 = calculate_distance(p[1], p[5])
    dist_0_1 = calculate_distance(p[0], p[1])
    dist_4_5 = calculate_distance(p[4], p[5])
    cond1 = dist_0_4 <= success_distance[0]
    cond2 = dist_2_3 <= success_distance[1]
    cond3 = dist_1_5 <= success_distance[2]
    cond4 = dist_0_1 >= success_distance[3]
    cond5 = dist_4_5 >= success_distance[4]

    details = {
        "condition_1": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} <= {success_distance[0]}",
            "value": dist_0_4,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[2], p[3]) = {dist_2_3:.2f} <= {success_distance[1]}",
            "value": dist_2_3,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} <= {success_distance[2]}",
            "value": dist_1_5,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[0], p[1]) = {dist_0_1:.2f} >= {success_distance[3]}",
            "value": dist_0_1,
            "threshold": success_distance[3],
            "passed": cond4,
        },
        "condition_5": {
            "description": f"dist(p[4], p[5]) = {dist_4_5:.2f} >= {success_distance[4]}",
            "value": dist_4_5,
            "threshold": success_distance[4],
            "passed": cond5,
        },
    }

    return cond1 and cond2 and cond3 and cond4 and cond5, details

def check_pant_long(p, success_distance):
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_0_2 = calculate_distance(p[0], p[2])   
    dist_1_3 = calculate_distance(p[1], p[3])
    dist_1_5 = calculate_distance(p[1], p[5])
    cond1 = dist_0_4 <= success_distance[0]
    cond2 = dist_0_2 >= success_distance[1]
    cond3 = dist_1_3 >= success_distance[2]
    cond4 = dist_1_5 <= success_distance[3]
    details = {
        "condition_1": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} <= {success_distance[0]}",
            "value": dist_0_4,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[0], p[2]) = {dist_0_2:.2f} >= {success_distance[1]}",
            "value": dist_0_2,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[1], p[3]) = {dist_1_3:.2f} >= {success_distance[2]}",
            "value": dist_1_3,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} <= {success_distance[3]}",
            "value": dist_1_5,
            "threshold": success_distance[3],
            "passed": cond4,
        },
    }
    return cond1 and cond2 and cond3 and cond4, details

def check_pant_short(p, success_distance):
    dist_0_1 = calculate_distance(p[0], p[1])
    dist_4_5 = calculate_distance(p[4], p[5])
    dist_0_4 = calculate_distance(p[0], p[4])
    dist_1_5 = calculate_distance(p[1], p[5])
    cond1 = dist_0_1 <= success_distance[0]
    cond2 = dist_4_5 <= success_distance[1]
    cond3 = dist_0_4 >= success_distance[2]
    cond4 = dist_1_5 >= success_distance[3]

    details = {
        "condition_1": {
            "description": f"dist(p[0], p[1]) = {dist_0_1:.2f} <= {success_distance[0]}",
            "value": dist_0_1,
            "threshold": success_distance[0],
            "passed": cond1,
        },
        "condition_2": {
            "description": f"dist(p[4], p[5]) = {dist_4_5:.2f} <= {success_distance[1]}",
            "value": dist_4_5,
            "threshold": success_distance[1],
            "passed": cond2,
        },
        "condition_3": {
            "description": f"dist(p[0], p[4]) = {dist_0_4:.2f} >= {success_distance[2]}",
            "value": dist_0_4,
            "threshold": success_distance[2],
            "passed": cond3,
        },
        "condition_4": {
            "description": f"dist(p[1], p[5]) = {dist_1_5:.2f} >= {success_distance[3]}",
            "value": dist_1_5,
            "threshold": success_distance[3],
            "passed": cond4,
        },
    }
    return cond1 and cond2 and cond3 and cond4, details


@step_interval(interval=10)
def success_checker_garment_fold(particle_object, garment_type: str):
    check_point_indices = particle_object.check_points  # list[int]
    success_distance = particle_object.success_distance  # list[int]
    p = get_object_particle_position(particle_object, check_point_indices)

    if garment_type == "top-long-sleeve" or garment_type == "top-short-sleeve":
        success, details = check_top_sleeve(p, success_distance)
    elif garment_type == "short-pant":
        success, details = check_pant_short(p, success_distance)
    elif garment_type == "long-pant":
        success, details = check_pant_long(p, success_distance)
    else:
        raise ValueError(f"Unknown garment_type: {garment_type}")

    result = {
        "success": bool(success),
        "garment_type": garment_type,
        "thresholds": success_distance,
        "details": details,
    }

    return result

@step_interval(interval=50)
def success_checker_fling(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    p = get_object_particle_position(particle_object, index_list)

    def xy_distance(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def z_distance(a, b):
        return abs(a[2] - b[2])

    success = (
        xy_distance(p[0], p[4]) > 18
        and z_distance(p[0], p[4]) < 2
        and xy_distance(p[1], p[5]) > 18
        and z_distance(p[1], p[5]) < 2
    )

    return bool(success)


@step_interval(interval=30)
def success_checker_burger(beef_pos, plate_pos):
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    # z distance
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # Success condition: xy < 0.045 and z < 0.03
    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2