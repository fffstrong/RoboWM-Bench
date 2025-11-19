# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with BiSO101 Leader for Garment environment."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Teleoperation with BiSO101 Leader for Garment environment."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="LeHome-BiSO101-Direct-Garment-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--left_port",
    type=str,
    default="/dev/ttyACM0",
    help="Serial port for left SO101 Leader.",
)
parser.add_argument(
    "--right_port",
    type=str,
    default="/dev/ttyACM0",
    help="Serial port for right SO101 Leader.",
)
# parser.add_argument(
#     "--device",
#     type=str,
#     default="cuda",
#     help="Device to run the simulation on.",
# )
parser.add_argument(
    "--recalibrate",
    action="store_true",
    default=False,
    help="Recalibrate the SO101 Leader devices.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import lehome.tasks  # noqa: F401
from lehome.devices import BiSO101Leader


def main():
    """Teleoperation with BiSO101 Leader for Garment environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # initialize the BiSO101 Leader device
    print("\n" + "=" * 60)
    print("Initializing BiSO101 Leader device...")
    print("=" * 60)
    teleop_device = BiSO101Leader(
        env=env.unwrapped,
        left_port=args_cli.left_port,
        right_port=args_cli.right_port,
        recalibrate=args_cli.recalibrate,
    )
    print(teleop_device)
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'b' to start teleoperation")
    print("  - Move the BiSO101 Leader arms to control the follower robots")
    print("  - Press Ctrl+C to quit")
    print("=" * 60 + "\n")

    # reset environment
    # env.initialize_obs()
    # obs, info = env.reset()

    # initialize observation (for garment object)
    if hasattr(env.unwrapped, "initialize_obs"):
        env.unwrapped.initialize_obs()
    # simulate environment
    started = False
    env.reset()
    for i in range(50):
        env.render()
    env.reset()
    if hasattr(env.unwrapped, "initialize_obs"):
        env.unwrapped.initialize_obs()
    while simulation_app.is_running():
        # get device action
        device_action = teleop_device.input2action()

        # check if started
        if device_action.get("started", False) and not started:
            started = True
            print("[INFO]: Teleoperation started! You can now move the leader arms.")

        # check if reset is requested
        if device_action.get("reset", False):
            print("[INFO]: Resetting environment...")
            obs, info = env.reset()
            if hasattr(env.unwrapped, "initialize_obs"):
                env.unwrapped.initialize_obs()
            continue

        # if not started, apply zero actions
        if not started:
            with torch.inference_mode():
                actions = torch.zeros(
                    env.action_space.shape, device=env.unwrapped.device
                )
                obs, rewards, terminated, truncated, info = env.step(actions)
            continue

        # run everything in inference mode
        with torch.inference_mode():
            # preprocess device action to get tensor actions
            actions = env.unwrapped.preprocess_device_action(
                device_action, teleop_device
            )

            # apply actions
            obs, rewards, terminated, truncated, info = env.step(actions)

            # check if episode is done
            if terminated.any() or truncated.any():
                print("[INFO]: Episode finished!")
                # check for success
                # if hasattr(env.unwrapped, '_get_success'):
                #     success = env.unwrapped._get_success()
                #     if success.any():
                #         print("[SUCCESS]: Task completed successfully!")
                #     else:
                #         print("[INFO]: Task not completed.")

                # reset environment
                obs, info = env.reset()
                if hasattr(env.unwrapped, "initialize_obs"):
                    env.unwrapped.initialize_obs()
                started = False
                print("[INFO]: Press 'b' to start new episode.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function

    main()
    print("\n[INFO]: Teleoperation interrupted by user.")

    simulation_app.close()
