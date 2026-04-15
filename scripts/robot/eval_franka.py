# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded episodes from JSON trajectories in lehome Franka environments."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

# NOTE: Do not import isaaclab.envs (or other Omniverse-backed modules) before
# AppLauncher starts SimulationApp — see Isaac Lab / Carbonite import order.

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
    "--step_hz",
    type=int,
    default=60,
    help="Environment stepping rate in Hz (used only if rate limiting is enabled below).",
)

parser.add_argument(
    "--output_root",
    type=str,
    default=None,
    help="Root directory to save replayed episodes (if None, replay only without saving).",
)
parser.add_argument(
    "--save_dataset",
    action="store_true",
    help="If set, create and save a replay dataset; otherwise skip saving.",
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
    help="Directory containing per-episode JSON files and optional pose.jsonl.",
)

parser.add_argument(
    "--action_tolerance",
    type=float,
    default=0.01,
    help="Tolerance for determining if the robot has reached the target action.",
)

parser.add_argument(
    "--episode_index",
    type=int,
    default=None,
    help="If set, only replay the JSON for this episode index (e.g. 0 for 0.json).",
)
parser.add_argument(
    "--part_scores",
    action="store_true",
    help="If set, print per-episode part/scores and mean part & scores over all replays.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.kit_args = (
    "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
)
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import json
import logging
import re
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.utils.record import get_next_experiment_path_with_gap

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

JOINT_LIMITS = [FRANKA_JOINT_LIMITS[f"panda_joint{i + 1}"]["radians"] for i in range(7)] + [
    FRANKA_JOINT_LIMITS["panda_finger_joint1"]["radians"],
    FRANKA_JOINT_LIMITS["panda_finger_joint2"]["radians"],
]


def clip_action(action: torch.Tensor) -> torch.Tensor:
    """Clamp joint actions to Franka limits (7 arm joints + 2 gripper joints)."""
    min_limits = torch.tensor(
        [limit[0] for limit in JOINT_LIMITS], dtype=action.dtype, device=action.device
    )
    max_limits = torch.tensor(
        [limit[1] for limit in JOINT_LIMITS], dtype=action.dtype, device=action.device
    )
    return torch.clamp(action, min=min_limits, max=max_limits)


def convert_quat_xyzw_to_wxyz(quat_xyzw):
    """Convert quaternion from [x, y, z, w] to [w, x, y, z]. Returns None if invalid."""
    if isinstance(quat_xyzw, list) and len(quat_xyzw) == 4:
        return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return None


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

    logger.info("Creating replay dataset at: %s", root)
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
):
    """
    Replay a single episode from a JSON file.

    Returns:
        success (bool), or (success, part, scores) if args.part_scores is True.
        part/scores are the last values from env.get_part_scores() when available.
    """
    part, scores = None, None
    try:
        file_path = Path(args.json_root) / f"{json_file_name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r") as f:
            actions = [json.loads(line.strip()) for line in f]

        pose_path = Path(args.json_root) / "pose.jsonl"
        pose_name = None
        pose_xyz = None
        pose_quat_wxyz = None
        pose_objects = None
        pose_joint = None

        if pose_path.exists():
            with open(pose_path, "r") as f:
                pose_lines = [line.strip() for line in f if line.strip()]
            pose_index = (
                int(json_file_name)
                if json_file_name.isdigit()
                else int(json_file_name.split("_")[-1])
            )
            if pose_index < 0 or pose_index >= len(pose_lines):
                raise IndexError(f"pose.jsonl index out of range: {pose_index}")
            pose_record = json.loads(pose_lines[pose_index])

            if "objects" in pose_record:
                objects = pose_record.get("objects", [])
                if len(objects) >= 2:
                    obj1, obj2 = objects[0], objects[1]
                    pose_objects = [
                        {
                            "name": obj1.get("name"),
                            "xyz": obj1.get("xyz"),
                            "quat_wxyz": convert_quat_xyzw_to_wxyz(obj1.get("quat")),
                        },
                        {
                            "name": obj2.get("name"),
                            "xyz": obj2.get("xyz"),
                            "quat_wxyz": convert_quat_xyzw_to_wxyz(obj2.get("quat")),
                        },
                    ]
                elif len(objects) == 1:
                    obj = objects[0]
                    pose_name = obj.get("name")
                    pose_xyz = obj.get("xyz")
                    pose_quat_wxyz = convert_quat_xyzw_to_wxyz(obj.get("quat"))
            else:
                pose_name = pose_record.get("name")
                if pose_name == "drawer" and "joint" in pose_record:
                    pose_name = "drawer"
                    pose_xyz = None
                    pose_quat_wxyz = None
                    pose_joint = pose_record.get("joint")
                else:
                    pose_xyz = pose_record.get("xyz")
                    pose_quat_xyzw = pose_record.get("quat")
                    if isinstance(pose_quat_xyzw, list) and len(pose_quat_xyzw) == 4:
                        pose_quat_wxyz = [
                            pose_quat_xyzw[3],
                            pose_quat_xyzw[0],
                            pose_quat_xyzw[1],
                            pose_quat_xyzw[2],
                        ]

        if pose_objects is not None:
            obj1, obj2 = pose_objects[0], pose_objects[1]
            env.reset(
                pose_name=obj1["name"],
                pose_xyz=obj1["xyz"],
                pose_quat_wxyz=obj1["quat_wxyz"],
                pose_name2=obj2["name"],
                pose_xyz2=obj2["xyz"],
                pose_quat_wxyz2=obj2["quat_wxyz"],
            )
        else:
            if pose_name != "drawer":
                env.reset(pose_name=pose_name, pose_xyz=pose_xyz, pose_quat_wxyz=pose_quat_wxyz)
            else:
                env.reset(joint=pose_joint)

        settle_action = torch.tensor(
            [
                -0.02829,
                -0.2848,
                -0.00921,
                -2.1854,
                0.0016,
                1.8824,
                0.7581,
                0.035,
                0.035,
            ],
            dtype=torch.float32,
            device=args_cli.device,
        ).unsqueeze(0)
        for _ in range(30):
            env.step(settle_action)

        success_achieved = False
        env.get_pose()

        for action_values in actions:
            if rate_limiter:
                rate_limiter.sleep(env)

            action = torch.tensor(action_values, dtype=torch.float32).unsqueeze(0)
            if action[0, 7] < 0.02:
                action[0, 7] = 0
                action[0, 8] = 0
            if pose_name == "banana":
                if action[0, 7] < 0.025:
                    action[0, 7] = 0
                    action[0, 8] = 0

            if rate_limiter:
                rate_limiter.sleep(env)

            action = clip_action(action)
            env.step(action)

            if replay_dataset is not None:
                observations = env._get_observations()
                frame = {**observations, "task": args.task_description}
                replay_dataset.add_frame(frame)

            success = env._get_success().item()
            if args_cli.part_scores:
                res = env.get_part_scores()
                if res is not None:
                    part, scores = res

            if success:
                success_achieved = True

        if args_cli.part_scores:
            return success_achieved, part, scores
        return success_achieved

    except Exception as e:
        logger.error(
            "Error during episode replay for episode %s: %s", json_file_name, e, exc_info=True
        )
        if args_cli.part_scores:
            return False, part, scores
        return False


def main():
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.propagate = False

    logger.info("Creating environment: %s", args_cli.task)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)
    env: DirectRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    # Replay without pacing; pass `RateLimiter(step_hz)` from lehome.utils.record to cap speed.
    rate_limiter = None

    replay_dataset = create_replay_dataset(args_cli) if args_cli.save_dataset else None

    total_attempts = 0
    success_count = 0
    saved_episodes = 0
    part_total = 0.0
    scores_total = 0.0
    part_scores_episodes = 0

    json_root = Path(args_cli.json_root)
    if not json_root.exists():
        raise ValueError(f"JSON root directory does not exist: {json_root}")

    json_file_names = sorted(
        [p.stem for p in json_root.glob("*.json") if p.is_file() and p.stem.isdigit()],
        key=lambda s: int(s),
    )

    if not json_file_names:
        json_file_names = sorted(
            [
                p.stem
                for p in json_root.glob("episode_*.json")
                if p.is_file() and re.fullmatch(r"episode_\d+", p.stem)
            ],
            key=lambda s: int(s.split("_")[-1]),
        )

    if args_cli.episode_index is not None:
        json_file_names = [
            s
            for s in json_file_names
            if (int(s) if s.isdigit() else int(s.split("_")[-1])) == args_cli.episode_index
        ]
        if not json_file_names:
            raise ValueError(
                f"No JSON file found for episode_index={args_cli.episode_index} in {json_root}."
            )

    total_episodes = len(json_file_names)
    if total_episodes == 0:
        raise ValueError(f"No JSON files found in {json_root}.")

    try:
        for episode_idx, json_file_name in enumerate(json_file_names):
            display_episode_num = episode_idx + 1

            logger.info("")
            logger.info("%s", "=" * 60)
            logger.info("Episode %s/%s", display_episode_num, total_episodes)
            logger.info("%s", "=" * 60)

            for _replay_idx in range(1):
                total_attempts += 1

                if replay_dataset is not None:
                    replay_dataset.clear_episode_buffer()

                if args_cli.part_scores:
                    success, part, score = replay_episode(
                        env=env,
                        json_file_name=json_file_name,
                        rate_limiter=rate_limiter,
                        args=args_cli,
                        replay_dataset=replay_dataset,
                    )
                    if part is not None and score is not None:
                        part_total += float(part)
                        scores_total += float(score)
                        part_scores_episodes += 1
                    logger.info(
                        "part_scores: episode=%s part=%s scores=%s",
                        json_file_name,
                        part,
                        score,
                    )
                else:
                    success = replay_episode(
                        env=env,
                        json_file_name=json_file_name,
                        rate_limiter=rate_limiter,
                        args=args_cli,
                        replay_dataset=replay_dataset,
                    )

                logger.info("Replay result: episode=%s success=%s", json_file_name, success)
                if success:
                    success_count += 1

                if replay_dataset is not None:
                    try:
                        if replay_dataset.episode_buffer:
                            replay_dataset.save_episode()
                            saved_episodes += 1
                            logger.info("  → Saved as episode %s", saved_episodes - 1)
                        else:
                            logger.warning(
                                "No frames to save for episode %s. Skipping save.", json_file_name
                            )
                    except Exception as e:
                        logger.error("Failed to save episode: %s", e, exc_info=True)

            if episode_idx < total_episodes - 1:
                logger.info("")
    finally:
        if replay_dataset is not None:
            try:
                replay_dataset.clear_episode_buffer()
                replay_dataset.finalize()
            except Exception as e:
                logger.error("Error finalizing dataset: %s", e, exc_info=True)

    logger.info("")
    logger.info("%s", "=" * 60)
    logger.info("Statistics")
    logger.info("  Total attempts: %s", total_attempts)
    success_rate = success_count / total_attempts if total_attempts > 0 else 0.0
    logger.info("  Successes: %s", success_count)
    logger.info("  Success rate: %.2f%%", 100.0 * success_rate)
    if args_cli.part_scores:
        if part_scores_episodes > 0:
            mean_part = part_total / part_scores_episodes
            mean_scores = scores_total / part_scores_episodes
            logger.info(
                "  Mean part (over %s replays with part_scores): %.4f",
                part_scores_episodes,
                mean_part,
            )
            logger.info(
                "  Mean scores (over %s replays with part_scores): %.4f",
                part_scores_episodes,
                mean_scores,
            )
        else:
            logger.info("  Mean part / mean scores: no valid part_scores samples")
    if replay_dataset is not None:
        logger.info("  Saved episodes: %s", saved_episodes)

    env.close()
    simulation_app.close()
    logger.info("Cleanup completed successfully.")


if __name__ == "__main__":
    main()
