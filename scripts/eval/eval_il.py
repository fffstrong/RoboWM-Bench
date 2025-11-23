import multiprocessing

# Ensure the multiprocessing start method is "spawn", which is necessary for CUDA.
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

# Import necessary modules from Isaac Lab
from isaaclab.app import AppLauncher

# Create the parser
parser = argparse.ArgumentParser(
    description="A script for evaluating policy in lehome manipulation environments."
)
# Add core arguments, consistent with teleop_record.py
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=600,
    help="Maximum number of steps per evaluation episode.",
)
parser.add_argument(
    "--task",
    type=str,
    default="LeHome-BiSO101-Direct-Garment-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=5,
    help="Total number of evaluation episodes to run.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
parser.add_argument(
    "--save_video",
    action="store_true",
    help="If set, save evaluation episodes as video.",
)

parser.add_argument(
    "--video_dir",
    type=str,
    default="outputs/eval_videos",
    help="Directory to save evaluation videos.",
)

parser.add_argument(
    "--save_datasets",
    action="store_true",
    help="If set, save evaluation episodes dataset(only success).",
)

parser.add_argument(
    "--eval_dataset_path",
    type=str,
    default="datasets/eval",
    help="Path to the pretrained policy checkpoint.",
)

parser.add_argument(
    "--eval_task",
    type=str,
    default="eval",
    help="dataset task name when eval.",
)

parser.add_argument(
    "--policy_path",
    type=str,
    default="outputs/train/diffusion_fold_1/checkpoints/100000/pretrained_model",
    help="Path to the pretrained policy checkpoint.",
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="Datasets/record/030",
    help="Path of the train dataset.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

from pathlib import Path
import os
import time
import torch
import gymnasium as gym
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
import numpy as np
from typing import Dict, Union
from torch import Tensor
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import cv2
import json


def preprocess_observation(
    obs_dict: Dict[str, Union[np.ndarray, Dict]], device: torch.device
) -> Dict[str, Tensor]:
    """
    Args:
        obs_dict: The observation dictionary from the environment, containing numpy arrays.
        device: The target PyTorch device (e.g., torch.device('cuda') or torch.device('cpu')).
    Returns:
        A dictionary with the same structure as the input, but with values processed
        into batched PyTorch Tensors on the specified device.
    """
    processed_dict = {}
    for key, value in obs_dict.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            processed_dict[key] = preprocess_observation(value, device)
            continue

        # Assume the value is a numpy array from this point
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected numpy array for key '{key}', but got {type(value)}"
            )
        processed_value = value

        # Check for image data
        if processed_value.ndim == 3 and processed_value.shape[-1] == 3:
            # 3. Process image data
            assert (
                processed_value.dtype == np.uint8
            ), f"Image for key '{key}' expected np.uint8, but got {processed_value.dtype}"
            processed_value = processed_value.astype(np.float32) / 255.0
            processed_value = np.transpose(processed_value, (2, 0, 1))

        batched_value = np.expand_dims(processed_value, axis=0)
        processed_dict[key] = torch.as_tensor(
            batched_value, dtype=torch.float32, device=device
        )
        processed_dict["task"] = args_cli.eval_task
    return processed_dict


def save_videos_from_observations(
    all_episode_frames, save_dir, episode_idx, success, fps=30
):
    # 判断 success，决定存放目录
    if success.item():  # 成功
        target_dir = os.path.join(save_dir, "success")
    else:  # 失败
        target_dir = os.path.join(save_dir, "failure")

    os.makedirs(target_dir, exist_ok=True)

    for key, frames in all_episode_frames.items():
        if len(frames) == 0:
            continue
        h, w, c = frames[0].shape
        out_path = os.path.join(
            target_dir, f"episode{episode_idx}_{key.replace('.', '_')}.mp4"
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()
        print(f"Saved video: {out_path}")


def run_evaluation_loop(env, policy, meta):
    """The core loop that drives the entire evaluation process."""
    if args_cli.save_datasets:
        root_path = Path(args_cli.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="abc",
            fps=30,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=meta.features,
        )
        jsonl_path = eval_dataset.root / "meta" / "object_initial_pose.jsonl"
        episode_index = 0
    all_episode_metrics = []
    print(
        f"\n[Evaluation Loop] Starting, running for {args_cli.num_episodes} episodes..."
    )
    rate_limiter = RateLimiter(args_cli.step_hz)
    env.reset()
    observation_dict = env._get_observations()
    if args_cli.save_video:
        episode_frames = {k: [] for k in observation_dict.keys() if "images" in k}
    for i in range(args_cli.num_episodes):
        env.reset()
        if args_cli.save_datasets:
            object_initial_pose = env.get_all_pose()
        observation_dict = env._get_observations()
        observation = preprocess_observation(observation_dict, policy.config.device)
        policy.reset()
        episode_return = 0.0
        episode_length = 0

        extra_steps = 0
        success_flag = False
        for st in range(args_cli.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            with torch.inference_mode():
                action = policy.select_action(observation)
            env.step(action)
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50

            observation_dict = env._get_observations()
            if args_cli.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args_cli.eval_task
                eval_dataset.add_frame(frame)
            observation = preprocess_observation(observation_dict, policy.config.device)

            if args_cli.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            reward = env.reward.item() if hasattr(env, "reward") else 0.0
            if not success_flag:
                episode_return += reward
                episode_length += 1

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break
        if args_cli.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(
                    jsonl_path, episode_index, object_initial_pose
                )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()
        if args_cli.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success,
                save_dir=args_cli.video_dir,
                episode_idx=i,
            )
            episode_frames = {k: [] for k in observation_dict.keys() if "images" in k}
        is_success = success.item()

        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        print(
            f"  Episode {i + 1}/{args_cli.num_episodes} complete: Total Return = {episode_return:.2f}, Length = {episode_length}, Success = {is_success}"
        )

    return all_episode_metrics


# Metrics Calculation and Display
def calculate_and_print_metrics(metrics: list):
    """Calculates and prints the final aggregated performance metrics."""
    if not metrics:
        print("[Results] No evaluation metrics were collected.")
        return

    total_returns = [m["return"] for m in metrics]
    total_successes = [1 if m["success"] else 0 for m in metrics]

    avg_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    success_rate = np.mean(total_successes)

    print("\n" + "=" * 35)
    print("      Evaluation Results Summary")
    print("=" * 35)
    print(f"Total Episodes:   {len(metrics)}")
    print(f"Average Return:   {avg_return:.2f} +/- {std_return:.2f}")
    print(f"Success Rate:     {success_rate:.2%}")
    print("=" * 35)


# Main Execution Function
def main():
    """Main function to organize and execute all evaluation steps."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    task_name = args_cli.task
    if "BiArm" in task_name:
        assert (
            args_cli.teleop_device == "bi-so101leader"
        ), "Only bi-so101leader is supported for bi-arm tasks"

    # Create the Gym environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    print("Environment created successfully.")

    env.initialize_obs()

    # Create the policy instance
    policy_cfg = PreTrainedConfig.from_pretrained(
        args_cli.policy_path, cli_overrides={}
    )
    policy_cfg.pretrained_path = args_cli.policy_path

    meta = LeRobotDatasetMetadata(repo_id="abc", root=args_cli.dataset_root)
    policy = make_policy(policy_cfg, ds_meta=meta)
    policy.eval()
    policy.reset()

    # Run the main evaluation loop
    episode_metrics = run_evaluation_loop(env=env, policy=policy, meta=meta)

    # Calculate and print the final metrics
    calculate_and_print_metrics(episode_metrics)

    # Cleanup resources
    env.close()
    simulation_app.close()
    print("[Cleanup] Environment and simulation app closed successfully.")


if __name__ == "__main__":
    main()
