# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay recorded episodes from LeRobotDataset in lehome environments."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay recorded episodes in lehome manipulation environments."
)
parser.add_argument(
    "--task",
    type=str,
    default="LeIsaac-BiSO101-Direct-loftburger-v0",
    help="Name of the task environment.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default="Datasets/record/023",
    help="Root directory of the dataset to replay.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default=None,
    help="Root directory to save replayed episodes (if None, replay only without saving).",
)
parser.add_argument(
    "--num_replays",
    type=int,
    default=1,
    help="Number of times to replay each episode.",
)
parser.add_argument(
    "--save_successful_only",
    action="store_true",
    default=False,
    help="Only save episodes that achieve success during replay.",
)
parser.add_argument(
    "--disable_depth",
    action="store_true",
    default=False,
    help="Disable depth observation during replay.",
)
parser.add_argument(
    "--start_episode",
    type=int,
    default=0,
    help="Starting episode index (inclusive).",
)
parser.add_argument(
    "--end_episode",
    type=int,
    default=None,
    help="Ending episode index (exclusive). If None, replay all episodes.",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="fold the garment on the table",
    help=" Description of the task to be performed.",
)
parser.add_argument(
    "--garment_type",
    type=str,
    default=None,
    choices=["top-long-sleeve", "top-short-sleeve", "short-pant", "long-pant"],
    help="Type of garment to use for replay (optional). If not specified, will use the garment type from the original recording.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import json
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lehome.utils.record import get_next_experiment_path_with_gap, RateLimiter


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    dataset_path = Path(args.dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"Dataset root does not exist: {args.dataset_root}")

    info_json = dataset_path / "meta" / "info.json"
    if not info_json.exists():
        raise ValueError(f"Dataset info.json not found: {info_json}")

    if args.num_replays < 1:
        raise ValueError(f"num_replays must be >= 1, got {args.num_replays}")


def load_dataset(dataset_root: str) -> LeRobotDataset:
    """Load the LeRobotDataset from the specified root directory."""
    print(f"[Info] Loading dataset from: {dataset_root}")
    dataset = LeRobotDataset(repo_id="replay_source", root=dataset_root)
    print(
        f"[Info] Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
    )
    return dataset


def load_initial_pose(dataset_root: str, episode_index: int):
    """Load the initial object pose for a given episode."""
    pose_file = Path(dataset_root) / "meta" / "object_initial_pose.jsonl"
    if not pose_file.exists():
        print(f"[Warning] Initial pose file not found: {pose_file}")
        return None

    with open(pose_file, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["episode_idx"] == episode_index:
                return record["object_initial_pose"]

    print(f"[Warning] Initial pose not found for episode {episode_index}")
    return None


def create_replay_dataset(args: argparse.Namespace, source_dataset: LeRobotDataset):
    """Create a new dataset for saving replayed episodes."""
    if args.output_root is None:
        return None, None

    output_path = Path(args.output_root)
    root = get_next_experiment_path_with_gap(output_path)

    # Use the same features as the source dataset
    features = source_dataset.meta.features

    # Optionally remove depth if disabled
    if args.disable_depth and "observation.top_depth" in features:
        features = {k: v for k, v in features.items() if k != "observation.top_depth"}

    print(f"[Info] Creating replay dataset at: {root}")
    replay_dataset = LeRobotDataset.create(
        repo_id="replay_output",
        fps=source_dataset.fps,
        root=root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )

    jsonl_path = replay_dataset.root / "meta" / "object_initial_pose.jsonl"
    return replay_dataset, jsonl_path


def replay_episode(
    env: DirectRLEnv,
    episode_data,
    rate_limiter,
    initial_pose,
    args: argparse.Namespace,
    replay_dataset=None,
    disable_depth=False,
) -> bool:
    """
    Replay a single episode.

    Returns:
        bool: True if episode was successful, False otherwise.
    """
    # Reset environment and set initial pose
    env.reset()
    if initial_pose is not None:
        env.set_all_pose(initial_pose)

    success_achieved = False

    # Replay each frame
    for idx in range(len(episode_data)):
        if rate_limiter:
            rate_limiter.sleep(env)

        # Get action from recorded data
        action = episode_data[idx]["action"].to(args.device).unsqueeze(0)

        # Step environment
        env.step(action)

        # If saving, record observations
        if replay_dataset is not None:
            observations = env._get_observations()

            # Remove depth if disabled
            if disable_depth and "observation.top_depth" in observations:
                observations = {
                    k: v
                    for k, v in observations.items()
                    if k != "observation.top_depth"
                }
            frame = {**observations, "task": args.task_description}
            replay_dataset.add_frame(frame)

        # Check for success
        success = env._get_success().item()
        if success:
            success_achieved = True

    return success_achieved


def _ndarray_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    else:
        return obj


def append_episode_initial_pose(jsonl_path, episode_idx, object_initial_pose):
    """Append initial pose information to the JSONL file."""
    object_initial_pose = _ndarray_to_list(object_initial_pose)
    rec = {"episode_idx": episode_idx, "object_initial_pose": object_initial_pose}
    with open(jsonl_path, "a") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    """Main replay loop."""
    # Validate arguments
    validate_args(args_cli)

    # Load dataset
    dataset = load_dataset(args_cli.dataset_root)

    # Create environment
    print(f"[Info] Creating environment: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    # Set garment configuration if specified (optional for replay)
    if args_cli.garment_type is not None:
        if hasattr(env_cfg, "garment_type"):
            env_cfg.garment_type = args_cli.garment_type
        if hasattr(env_cfg, "garment_index"):
            env_cfg.garment_index = None  # Use recorded initial pose, so keep None

    env: DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Initialize observations
    env.initialize_obs()

    # Create rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz > 0 else None

    # Create replay dataset if output is specified
    replay_dataset, jsonl_path = create_replay_dataset(args_cli, dataset)

    # Determine episode range
    start_idx = args_cli.start_episode
    end_idx = (
        args_cli.end_episode
        if args_cli.end_episode is not None
        else dataset.num_episodes
    )
    end_idx = min(end_idx, dataset.num_episodes)

    print(f"[Info] Replaying episodes {start_idx} to {end_idx-1}")

    # Statistics
    total_attempts = 0
    total_successes = 0
    saved_episodes = 0

    # Iterate through episodes
    for episode_idx in range(start_idx, end_idx):
        print(f"\n{'='*60}")
        print(f"[Episode {episode_idx}/{end_idx-1}]")
        print(f"{'='*60}")

        # Load initial pose
        initial_pose = load_initial_pose(args_cli.dataset_root, episode_idx)

        # Filter episode data
        episode_data = dataset.hf_dataset.filter(
            lambda x: x["episode_index"].item() == episode_idx
        )

        if len(episode_data) == 0:
            print(f"[Warning] Episode {episode_idx} has no data, skipping...")
            continue

        print(f"[Info] Episode length: {len(episode_data)} frames")

        # Replay multiple times if requested
        for replay_idx in range(args_cli.num_replays):
            total_attempts += 1

            print(f"[Replay {replay_idx + 1}/{args_cli.num_replays}]", end=" ")

            # Clear buffer if saving
            if replay_dataset is not None:
                replay_dataset.clear_episode_buffer()

            # Replay the episode
            success = replay_episode(
                env=env,
                episode_data=episode_data,
                rate_limiter=rate_limiter,
                initial_pose=initial_pose,
                args=args_cli,
                replay_dataset=replay_dataset,
                disable_depth=args_cli.disable_depth,
            )

            if success:
                total_successes += 1
                print("✓ Success")
            else:
                print("✗ Failed")

            # Save episode if conditions are met
            should_save = replay_dataset is not None and (
                not args_cli.save_successful_only or success
            )

            if should_save:
                replay_dataset.save_episode()
                append_episode_initial_pose(jsonl_path, saved_episodes, initial_pose)
                saved_episodes += 1
                print(f"  → Saved as episode {saved_episodes - 1}")
            elif replay_dataset is not None:
                replay_dataset.clear_episode_buffer()
    replay_dataset.clear_episode_buffer()
    replay_dataset.finalize()
    # Print statistics
    print(f"\n{'='*60}")
    print("[Statistics]")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Total successes: {total_successes}")
    if total_attempts > 0:
        print(f"  Success rate: {100.0 * total_successes / total_attempts:.1f}%")
    if replay_dataset is not None:
        print(f"  Saved episodes: {saved_episodes}")
    print(f"{'='*60}")

    # Cleanup
    env.close()
    simulation_app.close()
    print("[Info] Cleanup completed successfully.")


if __name__ == "__main__":
    main()
