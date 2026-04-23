import math
import numpy as np
import torch
from lehome.utils.logger import get_logger
logger = get_logger(__name__)


def step_interval(interval=50):
    """Factory function: Create a decorator with a customizable step size"""

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


@step_interval(interval=10)
def success_checker_pick(
    rigid_object_a, ori_z,env_id: int = 0
):

    pos_a = rigid_object_a.data.root_pos_w[env_id].clone()     # (3,)

    a_z = pos_a[2].item()
    
    success = (
        abs(a_z-ori_z)>0.05
    )
    return bool(success)


def success_checker_pick_once(
    rigid_object_a, ori_z,env_id: int = 0
):

    pos_a = rigid_object_a.data.root_pos_w[env_id].clone()     # (3,)

    a_z = pos_a[2].item()
    
    success = (
        abs(a_z-ori_z)>0.05
    )
    return bool(success)
    

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
    pos_a = rigid_object_a.data.root_pos_w[env_id]
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


def success_checker_lifted(
    rigid_object_a, ori_z, env_id: int = 0
) -> bool:
    pos_a = rigid_object_a.data.root_pos_w[env_id]

    a_z = pos_a[2].item()
    success = (a_z - ori_z) > 0.08
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
    """Calculate the straight-line distance between 2D coordinate points"""
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
    pos_a = rigid_object_a.data.root_pos_w[env_id]
    pos_b = rigid_object_b.data.root_pos_w[env_id]
    a_x = pos_a[0].item()
    a_y = pos_a[1].item()
    b_x = pos_b[0].item()
    b_y = pos_b[1].item()

    print(calculate_distance_2D((a_x, a_y), (b_x, b_y)))

    success = (
        calculate_distance_2D((a_x, a_y), (b_x, b_y))<= 0.05
    )
    return bool(success)


@step_interval(interval=5)
def success_checker_pull(
    rigid_object_a, ori_x, env_id: int = 0
):
    env_id = int(env_id)
    pos_a = rigid_object_a.data.root_pos_w[env_id] 
    a_x = pos_a[0].item()
    success = (
        abs(a_x-ori_x) > 0.03
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

@step_interval(interval=20)
def success_checker_pour_water(
    fluid_object,
    container_object,
    env_id: int = 0,
    xy_tolerance: float = 0.02,
    z_tolerance: float = 0.5,
    success_ratio: float = 0.01,
) -> bool:
    """
    General fluid pour checker. Determines if at least the success_ratio proportion of fluid falls into the target container (container_object).
    """
    particles, _, _ = fluid_object.get_particle_positions(visualize=False, global_coord=True)
    if len(particles) == 0:
        return False

    sampled = np.array(particles, dtype=np.float32)
    total_particles = len(sampled)

    # Get the current world coordinates of the target container
    pos_target = container_object.data.root_pos_w[env_id].detach().cpu().numpy()  # (3,)
    container_pos = np.array(pos_target, dtype=np.float32)
    # print(f"Pour Check: Container position: {container_pos}")

    # Calculate the difference between particles and the container center
    diff = np.abs(sampled - container_pos)

    # x and y directions are limited by the cup opening radius, z direction is only calculated downwards to a certain depth
    in_xy = (diff[:, 0] <= xy_tolerance) & (diff[:, 1] <= xy_tolerance)
    # print(f"Pour Check: {in_xy.sum()} particles within XY tolerance out of {total_particles} total particles.")
    in_z = diff[:, 2] <= z_tolerance

    inside = in_xy & in_z
    # print(f"Pour Check: {inside.sum()} particles inside the container out of {total_particles} total particles.")

    ratio = inside.mean() if len(inside) > 0 else 0.0
    # print(f"Pour Check: {inside.sum()}/{total_particles} particles inside, ratio={ratio:.4f}")
    return bool(ratio >= success_ratio)


@step_interval(interval=10)
def success_checker_push_button(contact_sensor, env_id: int = 0) -> bool:
    """
    Check if the button is touched by the robotic arm (or any specified contact object in the scene)
    Use contact_sensor to get the net forces or contact forces of the target object.
    """
    forces = contact_sensor.data.net_forces_w[env_id]
    
    # Calculate force magnitude
    force_magnitude = torch.norm(forces, dim=-1)
    
    threshold = 0.5
    
    # Determine if the force on any part exceeds this threshold
    success = (force_magnitude > threshold).any().item()
    
    return bool(success)


@step_interval(interval=10)
def success_checker_stack_cup(rigid_object_a, rigid_object_b, env_id: int = 0) -> bool:
    """
    Check if two cups are successfully stacked.
    Judgment criteria: XY horizontal distance less than 2.5 cm, Z-axis height difference less than 4 cm.
    """
    pos_a = rigid_object_a.data.root_pos_w[env_id]
    pos_b = rigid_object_b.data.root_pos_w[env_id]

    a_x, a_y, a_z = pos_a[0].item(), pos_a[1].item(), pos_a[2].item()
    b_x, b_y, b_z = pos_b[0].item(), pos_b[1].item(), pos_b[2].item()

    # Calculate Euclidean distance on the horizontal plane
    xy_distance = ((a_x - b_x) ** 2 + (a_y - b_y) ** 2) ** 0.5
    # Calculate vertical height difference
    z_distance = abs(a_z - b_z)

    # Set judgment thresholds: horizontal deviation < 0.025 m (2.5cm), height deviation < 0.04 m (4cm)
    success = (xy_distance <= 0.025) and (z_distance <= 0.04)

    return bool(success)
