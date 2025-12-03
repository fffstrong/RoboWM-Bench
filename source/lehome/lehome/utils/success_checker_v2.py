import math
import numpy as np
import torch


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


def calculate_distance(point_a, point_b):
    # 计算距离
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    return np.linalg.norm(point_a - point_b)


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


# Success checker for top-long-sleeve (长袖上衣)
@step_interval(interval=50)
def success_checker_top_long_sleeve_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    """Success checker for long-sleeve tops (长袖上衣).
    Uses the same particle indices as the original fold checker.
    TODO: Adjust particle indices based on actual garment geometry if needed.
    """
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


# Success checker for top-short-sleeve (短袖上衣)
@step_interval(interval=50)
def success_checker_top_short_sleeve_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    """Success checker for short-sleeve tops (短袖上衣).
    Uses the same particle indices as long-sleeve for now.
    TODO: Adjust particle indices based on actual garment geometry if needed.
    """
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


# Success checker for short-pant (短裤) - placeholder
@step_interval(interval=50)
def success_checker_short_pant_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    """Success checker for short pants (短裤) - placeholder implementation.
    TODO: Implement proper success criteria for pants when assets are available.
    """
    p = get_object_particle_position(particle_object, index_list)
    success = (
        calculate_distance(p[0], p[4]) <= 10
        and calculate_distance(p[2], p[3]) <= 16
        and calculate_distance(p[1], p[5]) <= 10
    )
    return bool(success)


# Success checker for long-pant (长裤) - placeholder
@step_interval(interval=50)
def success_checker_long_pant_fold(
    particle_object, index_list=[8077, 1711, 2578, 3942, 8738, 588]
):
    """Success checker for long pants (长裤) - placeholder implementation.
    TODO: Implement proper success criteria for pants when assets are available.
    """
    p = get_object_particle_position(particle_object, index_list)
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

    # z 距离
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # 成功条件：同时满足 xy < 0.05 且 z < 0.1
    success_mask = (dist_xy < 0.045) & (diff_z < 0.03)
    success = success_mask.any().item()

    return bool(success)


@step_interval(interval=6)
def success_checker_cut(sausage_count: int) -> bool:
    return sausage_count >= 2

