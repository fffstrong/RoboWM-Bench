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
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--recalibrate",
    action="store_true",
    default=False,
    help="recalibrate SO101-Leader or Bi-SO101Leader",
)
parser.add_argument("--num_episode", type=int, default=100, help="max num of episode ")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app
from pathlib import Path
import os
import time
import torch
import gymnasium as gym
import json
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from lehome.devices import Se3Keyboard, SO101Leader, BiSO101Leader
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import get_next_experiment_path_with_gap
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    """Running lerobot teleoperation with lehome manipulation environment."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)

    task_name = args_cli.task

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert (
            args_cli.teleop_device == "bi-so101leader"
        ), "only support bi-so101leader for bi-arm task"

    # create environment
    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # create controller
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        teleop_interface = SO101Leader(
            env, port=args_cli.port, recalibrate=args_cli.recalibrate
        )
    elif args_cli.teleop_device == "bi-so101leader":
        teleop_interface = BiSO101Leader(
            env,
            left_port=args_cli.left_arm_port,
            right_port=args_cli.right_arm_port,
            recalibrate=args_cli.recalibrate,
        )
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'so101leader', 'bi-so101leader'."
        )

    start_detected = False

    def reset_start():
        nonlocal start_detected
        start_detected = True

    teleop_interface.add_callback("S", reset_start)

    # add teleoperation key for task success
    success_detected = False

    def success_reset():
        nonlocal success_detected
        success_detected = True

    teleop_interface.add_callback("N", success_reset)  # 提前结束，并重置

    # add teleoperation key for env reset
    remove_detected = False

    def reset_remove():
        nonlocal remove_detected
        remove_detected = True

    teleop_interface.add_callback("D", reset_remove)  # 重置环境，丢弃当前

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    teleop_interface.reset()
    if args_cli.record:

        action_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        # Configure dataset features based on environment spaces
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (12,),
                "names": action_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (12,),
                "names": action_names,
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
        # root_path = Path("Datasets/record")
        # dataset = LeRobotDataset.create(
        #     repo_id="abc",
        #     fps=30,
        #     root=get_next_experiment_path_with_gap(root_path),
        #     use_videos=True,
        #     image_writer_threads=8,
        #     image_writer_processes=0,
        #     features=features,
        # )
        # jsonl_path = dataset.root / "meta" / "object_initial_pose.jsonl"

    count_render = 0
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            if start_detected == False:
                # env.initialize_obs()
                dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
                actions = teleop_interface.advance()
                if actions == None:
                    env.render()
                else:
                    env.step(actions)
                    object_initial_pose = env.get_all_pose()
                if rate_limiter:
                    rate_limiter.sleep(env)
                if count_render == 0:
                    env.initialize_obs()
                    count_render += 1
            elif start_detected == True and args_cli.record == True:
                episode_index = 0
                while episode_index < args_cli.num_episode:
                    while success_detected == False:
                        dynamic_reset_gripper_effort_limit_sim(
                            env, args_cli.teleop_device
                        )
                        actions = teleop_interface.advance()

                        if actions == None:
                            env.render()
                        else:
                            env.step(actions)
                        observations = env._get_observations()
                        _, truncated = env._get_dones()
                        frame = {**observations}
                        # dataset.add_frame(frame, task="burger")
                        if rate_limiter:
                            rate_limiter.sleep(env)
                        if truncated == True or remove_detected == True:
                            # dataset.clear_episode_buffer()
                            print(f"Re-recording episode {episode_index}")
                            env.reset()
                            object_initial_pose = env.get_all_pose()
                            remove_detected = False
                            continue
                    # dataset.save_episode()
                    append_episode_initial_pose(
                        jsonl_path, episode_index, object_initial_pose
                    )
                    episode_index += 1
                    env.reset()
                    for i in range(1000):
                        env.render()
                    object_initial_pose = env.get_all_pose()
                    success_detected = False
                    continue

    # if cfg.push_to_hub:
    #     dataset.push_to_hub()

    # close the simulator
    env.close()
    simulation_app.close()


def _ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ndarray_to_list(x) for x in obj]
    else:
        return obj


def append_episode_initial_pose(jsonl_path, episode_idx, object_initial_pose):
    object_initial_pose = _ndarray_to_list(object_initial_pose)
    rec = {"episode_idx": episode_idx, "object_initial_pose": object_initial_pose}
    with open(jsonl_path, "a") as fout:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # run the main function
    main()
