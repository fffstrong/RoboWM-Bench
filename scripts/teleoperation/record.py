import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.devices import Se3Keyboard, SO101Leader, BiSO101Leader
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import get_next_experiment_path_with_gap, RateLimiter


# ------------------------------------------------------------------------
# 参数解析与应用启动
# ------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="lehome teleoperation for lehome environments."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to simulate.",
    )
    parser.add_argument(
        "--teleop_device",
        type=str,
        default="keyboard",
        choices=["keyboard", "so101leader", "bi-so101leader"],
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
        "--sensitivity",
        type=float,
        default=1.0,
        help="Sensitivity factor.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="whether to enable record function",
    )
    parser.add_argument(
        "--step_hz",
        type=int,
        default=60,
        help="Environment stepping rate in Hz.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="recalibrate SO101-Leader or Bi-SO101Leader",
    )
    parser.add_argument(
        "--num_episode",
        type=int,
        default=100,
        help="max num of episode ",
    )
    # 追加 AppLauncher 的通用参数
    AppLauncher.add_app_launcher_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args()


def create_sim_app(args: argparse.Namespace):
    """创建并返回 (app_launcher, simulation_app)。"""
    app_launcher = AppLauncher(vars(args))
    simulation_app = app_launcher.app
    return app_launcher, simulation_app


# ------------------------------------------------------------------------
# 环境 & Teleop 创建
# ------------------------------------------------------------------------


def validate_task_and_device(args: argparse.Namespace) -> None:
    """在环境创建前检查 task 与 teleop_device 是否匹配。"""
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "BiArm" in args.task:
        assert (
            args.teleop_device == "bi-so101leader"
        ), "only support bi-so101leader for bi-arm task"


def create_env(args: argparse.Namespace) -> DirectRLEnv:
    """根据 task 配置创建环境。"""
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env: DirectRLEnv = gym.make(args.task, cfg=env_cfg).unwrapped
    return env


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

    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader'."
    )


# ------------------------------------------------------------------------
# 回调 & 状态
# ------------------------------------------------------------------------


def register_teleop_callbacks(teleop_interface):
    """注册 S/N/D 三个按键的回调，并返回状态字典。"""
    flags = {
        "start": False,   # S：开始录制
        "success": False, # N：成功/提前结束当前 episode
        "remove": False,  # D：丢弃当前 episode
    }

    def on_start():
        flags["start"] = True

    def on_success():
        flags["success"] = True

    def on_remove():
        flags["remove"] = True

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)

    return flags


# ------------------------------------------------------------------------
# 数据集 / 录制相关
# ------------------------------------------------------------------------


def create_dataset_if_needed(args: argparse.Namespace):
    """若开启记录，则创建 LeRobotDataset，并返回 (dataset, jsonl_path)。"""
    if not args.record:
        return None, None

    action_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),
            "names": action_names * 2,
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": action_names * 2,
        },
        "observation.images.top_rgb": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.top_depth": {
            "dtype": "float32",
            "shape": (480, 640),
            "names": ["height", "width"],
        },
        "observation.images.left_rgb": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_rgb": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
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


def _ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    return obj


def append_episode_initial_pose(jsonl_path: Path, episode_idx: int, object_initial_pose):
    object_initial_pose = _ndarray_to_list(object_initial_pose)
    rec = {"episode_idx": episode_idx, "object_initial_pose": object_initial_pose}
    with open(jsonl_path, "a") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------
# 主循环阶段划分
# ------------------------------------------------------------------------


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
        flags["success"] = False
        flags["remove"] = False

        # 单个 episode 内循环
        while not flags["success"]:
            dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
            actions = teleop_interface.advance()

            if actions is None:
                env.render()
            else:
                env.step(actions)

            observations = env._get_observations()
            _, truncated = env._get_dones()
            # NOTE: 原代码里 task 写死为 "burger"，这里保持不变
            frame = {**observations, "task": "burger"}
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
        env.reset()
        for _ in range(1000):
            env.render()
        object_initial_pose = env.get_all_pose()

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


# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------


def main(args: argparse.Namespace | None = None):
    """Running lerobot teleoperation with lehome manipulation environment."""
    if args is None:
        args = parse_args()

    validate_task_and_device(args)

    app_launcher, simulation_app = create_sim_app(args)
    env = create_env(args)
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(teleop_interface)

    rate_limiter = RateLimiter(args.step_hz)
    teleop_interface.reset()

    dataset, jsonl_path = create_dataset_if_needed(args)

    count_render = 0
    object_initial_pose = None

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    # 预启动阶段：等待 S 被按下
                    pose, count_render = run_idle_phase(
                        env,
                        teleop_interface,
                        rate_limiter,
                        args,
                        count_render,
                    )
                    if pose is not None:
                        object_initial_pose = pose

                elif args.record and dataset is not None:
                    # 录制阶段：阻塞直到录完 num_episode
                    object_initial_pose = run_recording_phase(
                        env,
                        teleop_interface,
                        rate_limiter,
                        args,
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
                        env, teleop_interface, rate_limiter, args
                    )

        # 如果需要推送到 Hub，可以在这里处理
        # if cfg.push_to_hub:
        #     dataset.push_to_hub()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()