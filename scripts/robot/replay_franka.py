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
    "--output_root",
    type=str,
    default=None,
    help="Root directory to save replayed episodes (if None, replay only without saving).",
)

parser.add_argument(
    "--task_description",
    type=str,
    default="fold the garment on the table",
    help="Task description string stored with saved frames.",
)

parser.add_argument(
    "--json_root",
    type=str,
    default=".",
    help="Directory containing per-episode JSON files and file_indices.txt.",
)

parser.add_argument(
    "--action_tolerance",
    type=float,
    default=0.01,
    help="Tolerance for determining if the robot has reached the target action.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Set kit args to error level
args_cli.kit_args = '--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error'
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import json
import logging
from pathlib import Path
import gymnasium as gym
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lehome.utils.record import get_next_experiment_path_with_gap
import torch
# Create logger at module level
logger = logging.getLogger(__name__)


FRANKA_JOINT_LIMITS = {
    "panda_joint1": {"radians": (-2.897, 2.897)},
    "panda_joint2": {"radians": (-1.763, 1.763)},
    "panda_joint3": {"radians": (-2.897, 2.897)},
    "panda_joint4": {"radians": (-3.072, -0.070)},
    "panda_joint5": {"radians": (-2.897, 2.897)},
    "panda_joint6": {"radians": (-0.018, 3.753)},
    "panda_joint7": {"radians": (-2.897, 2.897)},
    "panda_finger_joint1": {"radians": (0.000, 0.040)},
    "panda_finger_joint2": {"radians": (0.000, 0.040)},
}

JOINT_LIMITS = [FRANKA_JOINT_LIMITS[f"panda_joint{i+1}"]["radians"] for i in range(7)] + [
    FRANKA_JOINT_LIMITS["panda_finger_joint1"]["radians"],
    FRANKA_JOINT_LIMITS["panda_finger_joint2"]["radians"],
]

def clip_action(action: torch.Tensor) -> torch.Tensor:
    """Clamp joint actions to Franka limits (7 arm joints + 2 gripper joints).

    Args:
        action: Tensor shaped (1, 9) or (9,)

    Returns:
        Tensor with the same shape as the input.
    """
    min_limits = torch.tensor([limit[0] for limit in JOINT_LIMITS], dtype=action.dtype, device=action.device)
    max_limits = torch.tensor([limit[1] for limit in JOINT_LIMITS], dtype=action.dtype, device=action.device)
    return torch.clamp(action, min=min_limits, max=max_limits)

def create_replay_dataset(args: argparse.Namespace):
    """Create a new dataset for saving replayed episodes."""
    if args.output_root is None:
        return None

    output_path = Path(args.output_root)
    root = get_next_experiment_path_with_gap(output_path)

    action_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": action_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),
            "names": action_names,
        },
        "observation.images.top_rgb": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channels"],
        },
    }

    logger.info(f"Creating replay dataset at: {root}")
    replay_dataset = LeRobotDataset.create(
        repo_id="replay_output",
        fps=30,
        root=root,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )

    return replay_dataset


def replay_episode(
    env: DirectRLEnv,
    json_file_name,
    rate_limiter,
    args: argparse.Namespace,
    replay_dataset=None,
) -> bool:
    """Replay a single episode from a JSON file."""
    try:
        file_path = Path(args.json_root) / f"{json_file_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r") as f:
            actions = [json.loads(line.strip()) for line in f]

        env.reset()
        success_achieved = False

        for action_values in actions:
            if rate_limiter:
                rate_limiter.sleep(env)

            action = torch.tensor(action_values, dtype=torch.float32).unsqueeze(0)

            if rate_limiter:
                rate_limiter.sleep(env)

            action = clip_action(action)
            env.step(action)

            if replay_dataset is not None:
                observations = env._get_observations()
                frame = {**observations, "task": args.task_description}
                replay_dataset.add_frame(frame)

            success = env._get_success().item()
            if success:
                success_achieved = True

        return success_achieved
    except Exception as e:
        logger.error(f"Error during episode replay for episode {json_file_name}: {e}", exc_info=True)
        return False
    
def main():
    """Main replay loop."""
    # Configure logging
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(handler)
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    # Create environment
    logger.info(f"Creating environment: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    env: DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Initialize observations
    env.initialize_obs()

    # Replay without pacing; pass `RateLimiter(step_hz)` from lehome.utils.record to cap speed.
    rate_limiter = None
    # Create replay dataset only when output_root is provided.
    replay_dataset = create_replay_dataset(args_cli) if args_cli.output_root is not None else None

    total_attempts = 0
    saved_episodes = 0
    json_root = Path(args_cli.json_root)
    if not json_root.exists():
        raise ValueError(f"JSON root directory does not exist: {json_root}")

    # 读取 file_indices.txt 文件
    file_indices_path = json_root / "file_indices.txt"
    if not file_indices_path.exists():
        raise ValueError(f"file_indices.txt does not exist in {json_root}.")

    try:
        with open(file_indices_path, "r") as f:
            json_file_names = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        raise ValueError(f"Failed to read file_indices.txt: {e}")

    total_episodes = len(json_file_names)
    if total_episodes == 0:
        raise ValueError(f"No JSON file names found in {file_indices_path}.")

    # Iterate through file_indices.txt
    try:
        for episode_idx, json_file_name in enumerate(json_file_names):
            display_episode_num = episode_idx + 1

            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"Episode {display_episode_num}/{total_episodes}")
            logger.info(f"{'='*60}")

            for replay_idx in range(1):
                total_attempts += 1

                if replay_dataset is not None:
                    replay_dataset.clear_episode_buffer()

                # Replay the episode
                success = replay_episode(
                    env=env,
                    json_file_name=json_file_name,
                    rate_limiter=rate_limiter,
                    args=args_cli,
                    replay_dataset=replay_dataset,
                )
                logger.info("Replay result: episode=%s success=%s", json_file_name, success)

                should_save = replay_dataset is not None

                if should_save:
                    try:
                        if replay_dataset.episode_buffer:
                            replay_dataset.save_episode()
                            saved_episodes += 1
                            logger.info(f"  → Saved as episode {saved_episodes - 1}")
                        else:
                            logger.warning(f"No frames to save for episode {json_file_name}. Skipping save.")
                    except Exception as e:
                        logger.error(f"Failed to save episode: {e}", exc_info=True)

            # Remove the processed episode name from file_indices.txt to support resume.
            try:
                with open(file_indices_path, "r") as f:
                    lines = f.readlines()

                updated_lines = [line for line in lines if line.strip() != json_file_name]

                with open(file_indices_path, "w") as f:
                    f.writelines(updated_lines)

            except Exception as e:
                logger.error(f"Failed to update file_indices.txt: {e}", exc_info=True)

            # Add separator between episodes
            if episode_idx < total_episodes - 1:
                logger.info("")
    finally:
        # Ensure dataset is finalized even if an error occurs
        if replay_dataset is not None:
            try:
                replay_dataset.clear_episode_buffer()
                replay_dataset.finalize()
            except Exception as e:
                logger.error(f"Error finalizing dataset: {e}", exc_info=True)
    # Print statistics
    logger.info("")
    logger.info(f"{'='*60}")
    logger.info("Statistics")
    logger.info(f"  Total attempts: {total_attempts}")
    if replay_dataset is not None:
        logger.info(f"  Saved episodes: {saved_episodes}")
    
    # Cleanup
    env.close()
    simulation_app.close()

    logger.info("Cleanup completed successfully.")

if __name__ == "__main__":
    main()
