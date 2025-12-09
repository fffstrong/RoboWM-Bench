# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a lehome teleoperation with lehome manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="lehome teleoperation for lehome environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "bi-keyboard", "so101leader", "bi-so101leader"],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the teleop device:so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
# auto reset interval

# recorder_parameter
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help="whether to enable record function",
)
parser.add_argument(
    "--step_hz", type=int, default=120, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--recalibrate",
    action="store_true",
    default=False,
    help="recalibrate SO101-Leader or Bi-SO101Leader",
)
parser.add_argument("--num_episode", type=int, default=20, help="max num of episode ")
parser.add_argument(
    "--disable_depth",
    action="store_true",
    default=False,
    help="Disable using top depth observation in env and dataset.",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="fold the garment on the table",
    help=" Description of the task to be performed.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import torch
from pathlib import Path
import gymnasium as gym
import json
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.devices import Se3Keyboard, SO101Leader, BiSO101Leader, BiKeyboard
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    get_next_experiment_path_with_gap,
    RateLimiter,
    append_episode_initial_pose,
)
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def validate_task_and_device(args: argparse.Namespace) -> None:
    """在环境创建前检查 task 与 teleop_device 是否匹配。"""
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "Bi" in args.task:
        assert (
            args.teleop_device == "bi-so101leader"
            or args.teleop_device == "bi-keyboard"
        ), "only support bi-so101leader or bi-keyboardfor bi-arm task"
    else:
        assert (
            args.teleop_device == "so101leader" or args.teleop_device == "keyboard"
        ), "only support so101leader or keyboard for single-arm task"


def create_teleop_interface(env: DirectRLEnv, args: argparse.Namespace):
    """根据参数创建相应的遥操作接口。"""
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(
            env,
            left_port=args.left_arm_port,
            right_port=args.right_arm_port,
            recalibrate=args.recalibrate,
        )
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
    )


def register_teleop_callbacks(teleop_interface):
    """注册 S/N/D/ESC 四个按键的回调，并返回状态字典。"""
    flags = {
        "start": False,  # S：开始录制
        "success": False,  # N：成功/提前结束当前 episode
        "remove": False,  # D：丢弃当前 episode
        "abort": False,  # ESC：中止整个录制过程，清空当前 buffer
    }

    def on_start():
        flags["start"] = True

    def on_success():
        flags["success"] = True

    def on_remove():
        flags["remove"] = True

    def on_abort():
        flags["abort"] = True
        print("\n[ESC] 中止录制，正在清空当前 episode buffer...")

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("ESCAPE", on_abort)

    return flags


def create_dataset_if_needed(args: argparse.Namespace):
    """若开启记录，则创建 LeRobotDataset，并返回 (dataset, jsonl_path)。"""
    if not args.record:
        return None, None

    # 单臂每条手臂的 DoF 名称
    action_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    # 判定单双臂（和 validate_task_and_device 保持一致）
    is_bi_arm = ("Bi" in (args.task or "")) or (
        getattr(args, "teleop_device", "") or ""
    ).startswith("bi-")

    # ---------- 关节 / 动作 feature ----------
    if is_bi_arm:
        left_names = [f"left_{n}" for n in action_names]
        right_names = [f"right_{n}" for n in action_names]
        joint_names = left_names + right_names
    else:
        joint_names = action_names

    dim = len(joint_names)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
    }

    # 根据参数决定是否记录深度
    use_depth = not getattr(args, "disable_depth", False)
    if use_depth:
        features["observation.top_depth"] = {
            "dtype": "float32",
            "shape": (480, 640),
            "names": ["height", "width"],
        }

    # ---------- 图像 feature：根据单双臂选择相机 ----------
    if is_bi_arm:
        # 例子：双臂用 top + left + right
        image_keys = ["top_rgb", "left_rgb", "right_rgb"]
    else:
        # 例子：单臂只用 top + wrist（如果你单臂也有 right，就把它也加进来）
        image_keys = ["top_rgb", "wrist_rgb"]

    for key in image_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    root_path = Path("Datasets/record")
    dataset = LeRobotDataset.create(
        repo_id="abc",
        fps=30,
        root=get_next_experiment_path_with_gap(root_path),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )
    jsonl_path = dataset.root / "meta" / "object_initial_pose.jsonl"
    return dataset, jsonl_path


def run_idle_phase(env, teleop_interface, rate_limiter, args, count_render: int):
    """未按 S 之前的阶段：只用来做环境准备和初始化观测。"""
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

    actions = teleop_interface.advance()
    object_initial_pose = None

    if actions is None:
        env.render()
    else:
        env.step(actions)
        object_initial_pose = env.get_all_pose()

    if rate_limiter:
        rate_limiter.sleep(env)

    if count_render == 0:
        env.initialize_obs()
        count_render += 1

    return object_initial_pose, count_render


def run_recording_phase(
    env,
    teleop_interface,
    rate_limiter,
    args,
    flags,
    dataset,
    jsonl_path,
    initial_object_pose,
):
    """按下 S 之后并且开启 record 的录制阶段。"""
    episode_index = 0
    object_initial_pose = initial_object_pose

    while episode_index < args.num_episode:
        # 检查是否需要中止录制
        if flags["abort"]:
            dataset.clear_episode_buffer()
            dataset.finalize()
            print(f"已中止录制，共完成 {episode_index} 条 episode")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False

        # 单个 episode 内循环
        while not flags["success"]:
            # 检查是否需要中止录制
            if flags["abort"]:
                dataset.clear_episode_buffer()
                dataset.finalize()
                print(f"已中止录制，共完成 {episode_index} 条 episode")
                return object_initial_pose

            dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
            actions = teleop_interface.advance()

            if actions is None:
                env.render()
            else:
                env.step(actions)

            observations = env._get_observations()
            # 若禁用深度，则在保存前移除深度观测
            if (
                getattr(args, "disable_depth", False)
                and "observation.top_depth" in observations
            ):
                observations.pop("observation.top_depth")
            _, truncated = env._get_dones()
            # NOTE: 原代码里 task 写死为 "burger"，这里保持不变
            frame = {**observations, "task": args.task_description}
            dataset.add_frame(frame)

            if rate_limiter:
                rate_limiter.sleep(env)

            if truncated or flags["remove"]:
                dataset.clear_episode_buffer()
                print(f"Re-recording episode {episode_index}")
                env.reset()
                object_initial_pose = env.get_all_pose()
                flags["remove"] = False
                # 重新开始这一条 episode
                continue

        # 成功结束该 episode
        dataset.save_episode()
        append_episode_initial_pose(jsonl_path, episode_index, object_initial_pose)

        episode_index += 1
        print(
            f"Episode {episode_index - 1} 录制完成，进度: {episode_index}/{args.num_episode}"
        )
        env.reset()
        # for _ in range(1000):
        #     env.render()
        object_initial_pose = env.get_all_pose()
    dataset.clear_episode_buffer()
    dataset.finalize()
    print(f"所有 {args.num_episode} 条 episode 录制完成！")
    return object_initial_pose


def run_live_control_without_record(env, teleop_interface, rate_limiter, args):
    """
    已按 S 但未开启 record 时的逻辑：
    简单地进行遥操作控制，不写数据集。
    """
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
    actions = teleop_interface.advance()

    if actions is None:
        env.render()
    else:
        env.step(actions)

    if rate_limiter:
        rate_limiter.sleep(env)


def main():
    """Running lerobot teleoperation with lehome manipulation environment."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
    )
    task_name = args_cli.task

    # create environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    teleop_interface = create_teleop_interface(env, args_cli)
    flags = register_teleop_callbacks(teleop_interface)
    rate_limiter = RateLimiter(args_cli.step_hz)
    teleop_interface.reset()
    dataset, jsonl_path = create_dataset_if_needed(args_cli)
    count_render = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    # 预启动阶段：等待 S 被按下
                    pose, count_render = run_idle_phase(
                        env,
                        teleop_interface,
                        rate_limiter,
                        args_cli,
                        count_render,
                    )
                    if pose is not None:
                        object_initial_pose = pose

                elif args_cli.record and dataset is not None:
                    # 录制阶段：阻塞直到录完 num_episode
                    object_initial_pose = run_recording_phase(
                        env,
                        teleop_interface,
                        rate_limiter,
                        args_cli,
                        flags,
                        dataset,
                        jsonl_path,
                        object_initial_pose,
                    )
                    # 所有 episode 录制完后，可以选择 break 或继续等待下一次 S
                    # 这里简单退出循环
                    break

                else:
                    # 已按 S 但未开启录制：只进行正常遥操作
                    run_live_control_without_record(
                        env, teleop_interface, rate_limiter, args_cli
                    )

    except KeyboardInterrupt:
        print("\n[Ctrl+C] 检测到中断信号")
        # 如果在录制过程中按 Ctrl+C，清空当前 buffer
        if args_cli.record and dataset is not None and flags["start"]:
            print("清空当前 episode buffer...")
            dataset.clear_episode_buffer()
            print("Buffer 已清空，数据集保持完整性")
            dataset.finalize()
            print("数据集已保存")
    finally:
        # if cfg.push_to_hub:
        #     dataset.push_to_hub()

        # close the simulator
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
