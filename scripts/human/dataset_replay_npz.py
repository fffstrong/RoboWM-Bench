import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.app import AppLauncher
import json
import os

TASKS_WITH_TWO_OBJECTS = ["Task04_Franka_Tableware_Banana_Plate",
                          "Task05_Franka_Tableware_Stack_Cup",
                          "Task07_Franka_Tableware_Pour_Water",
                          "Task06_Franka_Tableware_Stapler_Box",
                          "Task09_Franka_Tableware_Banana_Drawer"]

TASK_PART_SCORES_MAP = {
    "Task04_Franka_Tableware_Banana_Plate": {
        1: "Contact",
        2: "Lift",
        3: "In Plate"
    },
    "Task06_Franka_Tableware_Stapler_Box":{
        1: "Contact",
        2: "Lift",
        3: "In Box"
    },
    "Task09_Franka_Tableware_Banana_Drawer": {
        1: "Contact",
        2: "Lift",
        3: "Above Drawer",
        4: "In Drawer",
        5: "Close Drawer"
    },
}

# -- Args Setup --
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="Task05_Franka_Tableware_Stack_Cup", help="Task folder name, e.g., Task01_Franka-Tableware_Cube")
parser.add_argument("--model_name", type=str, default="human", help="Model name, e.g., human or veo")
parser.add_argument("--debug", action="store_true", help="Whether to print detailed error comparison info")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--pos_gain", type=float, default=6.0, help="Position P-Gain")
parser.add_argument("--rot_gain", type=float, default=5.0, help="Rotation P-Gain")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.task_dir = os.path.join("third_party/phantom/data/processed/hand_dataset", args.task_name)

args.init_json = os.path.join(
    "source/lehome/lehome/tasks/human_task",
    args.task_name,
    "initial_states.json"
)

print("=" * 50)
print(f"[Auto-Config] Task Name : {args.task_name}")
print(f"[Auto-Config] Task Dir  : {args.task_dir}")
print(f"[Auto-Config] Init JSON : {args.init_json}")
print("=" * 50)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import lehome.tasks.human_task


def calculate_action(
    target_pos, target_rot_mat, current_pos, current_quat, pos_gain, rot_gain, device
):
    """
    Calculate P-Control action: [dx, dy, dz, drot_x, drot_y, drot_z]
    """
    # 1. Position Error
    pos_err = target_pos - current_pos

    # 2. Rotation Error (Scipy handles conversion)
    # Isaac (w, x, y, z) -> Scipy (x, y, z, w)
    curr_quat_scipy = np.array(
        [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
    )
    r_target = Rotation.from_matrix(target_rot_mat)
    r_current = Rotation.from_quat(curr_quat_scipy)

    # R_diff = R_target * R_current.inv()
    r_diff = r_target * r_current.inv()
    rot_vec_err = r_diff.as_rotvec()  # (3,)

    # 3. Apply Gains
    action_pos = pos_err * pos_gain
    action_rot = rot_vec_err * rot_gain

    action_combined = np.concatenate([action_pos, action_rot])
    return torch.tensor(action_combined, dtype=torch.float32, device=device).unsqueeze(
        0
    )


def load_trajectory(file_path):
    data = np.load(file_path)
    pts = data["ee_pts"]
    oris = data["ee_oris"]
    widths = data["ee_widths"] if "ee_widths" in data else np.zeros(len(pts))

    # Calculate the maximum and minimum widths
    if len(widths) > 0:
        max_width = np.max(widths)
        min_width = np.min(widths)
    else:
        max_width = 0.08
        min_width = 0.0

    if args.debug:
        print(f"[INFO] Loaded trajectory. Gripper Width Range: [{min_width:.4f}, {max_width:.4f}]")
    return pts, oris, widths, max_width, min_width


def main():
    with open(args.init_json, "r") as f:
        init_states = json.load(f)

    # 2. Initialize Env in advance (moved outside the loop for reuse, avoiding crashes from reloading the environment every time)
    env_cfg = parse_env_cfg(args.task_name, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task_name, cfg=env_cfg).unwrapped

    results = {}
    total_trials = 0
    total_successes = 0
    
    task_steps_config = TASK_PART_SCORES_MAP.get(args.task_name, {})
    part_success_counts = {step: 0 for step in task_steps_config.keys()}

    # Outer loop: Iterate through 10 trajectories (0 to 9)
    for traj_id in range(10):
        # Dynamic path construction, e.g., pick/0/0_human/...
        traj_path = os.path.join(
            args.task_dir,
            str(traj_id),
            f"{traj_id}_{args.model_name}",
            "smoothing_processor",
            "smoothed_actions_left_single_arm.npz"
        )

        if not os.path.exists(traj_path):
            print(f"[WARNING] Trajectory file not found, skipping: {traj_path}")
            continue

        ee_pts, ee_oris, ee_widths, max_width, min_width = load_trajectory(traj_path)
        total_frames = len(ee_pts)
        success_in_this_traj = 0

        # Inner loop: Test each trajectory twice
        for trial_idx in range(2):
            print(f"\n{'='*20} Testing Trajectory {traj_id} | Trial {trial_idx + 1}/2 {'='*20}")

            # Dynamically modify the initial positions of objects based on JSON
            new_pos = init_states[str(traj_id)]
            print(f"[INFO] Setting initial object positions for Trajectory {traj_id}: {new_pos}")
            if args.task_name in TASKS_WITH_TWO_OBJECTS:
                # Two-object task: JSON format should be [[x1, y1, z1], [x2, y2, z2]]
                pos_A = new_pos[0]
                pos_B = new_pos[1]
                
                # Set object A
                if hasattr(env.object_A, "data"):
                    env.object_A.data.default_root_state[:, 0:3] = torch.tensor(pos_A, dtype=torch.float32, device=env.device)
                elif hasattr(env.object_A, "garment_config"):
                    # For cloth types, force override its reset range locked to pos_A
                    env.object_A.garment_config.soft_reset_pos_range = [pos_A[0], pos_A[1], pos_A[2], pos_A[0], pos_A[1], pos_A[2]]
                    
                # Set object B
                if hasattr(env.object_B, "data"):
                    env.object_B.data.default_root_state[:, 0:3] = torch.tensor(pos_B, dtype=torch.float32, device=env.device)
                elif hasattr(env.object_B, "garment_config"):
                    env.object_B.garment_config.soft_reset_pos_range = [pos_B[0], pos_B[1], pos_B[2], pos_B[0], pos_B[1], pos_B[2]]
            else:
                # Single-object task: JSON format should be [x, y, z]
                if hasattr(env.object_A, "data"):
                    env.object_A.data.default_root_state[:, 0:3] = torch.tensor(new_pos, dtype=torch.float32, device=env.device)
                elif hasattr(env.object_A, "garment_config"):
                    # For cloth types, force override its soft_reset_pos_range
                    env.object_A.garment_config.soft_reset_pos_range = [new_pos[0], new_pos[1], new_pos[2], new_pos[0], new_pos[1], new_pos[2]]
                else:
                    print(f"[INFO] Skipping default_root_state modification for unrecognized object: {type(env.object_A)}")

            # 3. Initial Reset & Setup
            env.reset()
            robot = env.robot
            try:
                ee_idx = robot.find_bodies("panda_hand")[0][0]
            except (IndexError, TypeError):
                ee_idx = robot.num_bodies - 1

            print("[INFO] Warming up to start pose (Pos + Rot)...")

            # 1. Prepare Target for Frame 0 (World Coordinates)
            base_state = robot.data.root_state_w[0, :7].cpu().numpy()
            base_pos, base_quat = base_state[0:3], base_state[3:7]
            r_base = Rotation.from_quat(
                [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
            )

            # Calculate world coordinate pose for Frame 0
            start_pos_w_np = r_base.apply(ee_pts[0]) + base_pos
            start_rot_w_mat = (r_base * Rotation.from_matrix(ee_oris[0])).as_matrix()

            start_pos_w = start_pos_w_np

            # 2. Warmup Loop
            current_joints = robot.data.joint_pos.clone()

            for i in range(160):
                robot.update(dt=0.0)
                ee_state = robot.data.body_state_w[:, ee_idx]
                curr_pos_np = ee_state[0, 0:3].cpu().numpy()
                curr_quat_np = ee_state[0, 3:7].cpu().numpy()  # w, x, y, z
                ik_cmd_tensor = calculate_action(
                    start_pos_w,  # Target Pos (Tensor)
                    start_rot_w_mat,  # Target Rot (Matrix)
                    curr_pos_np,  # Curr Pos (Numpy)
                    curr_quat_np,  # Curr Quat (Numpy)
                    pos_gain=1.0,
                    rot_gain=1.0,
                    device=env.device,
                )

                ik_cmd_tensor = torch.clamp(ik_cmd_tensor, min=-0.1, max=0.1)

                env.ik_controller.set_command(
                    ik_cmd_tensor, ee_pos=ee_state[:, 0:3], ee_quat=ee_state[:, 3:7]
                )

                jac = robot.root_physx_view.get_jacobians()[:, ee_idx, :, :]
                joint_target = env.ik_controller.compute(
                    ee_state[:, 0:3], ee_state[:, 3:7], jac, current_joints
                )

                # Keep gripper open
                joint_target[:, 7:9] = 0.04

                robot.write_joint_state_to_sim(joint_target, torch.zeros_like(joint_target))
                current_joints = joint_target.clone()

                if i % 20 == 0:
                    start_pos_w = torch.tensor(start_pos_w, device=env.device)
                    err = torch.norm(start_pos_w - ee_state[:, 0:3])
                    start_pos_w = start_pos_w.cpu().numpy()
                    if args.debug:
                        print(f"Warmup Step {i}, Pos Error: {err:.4f}")

            print("[INFO] Warmup Done.")

            # 3. Set the calculated perfect state as Default, and reset environment
            robot.data.default_joint_pos[:] = current_joints
            env.reset()
            robot.set_joint_position_target(current_joints)

            robot.update(dt=env.step_dt)

            print("[INFO] Starting P-Control Replay Loop...")

            prev_gripper_val = None
            max_gripper_delta = 0.004

            frame_idx = 0
            error_count = 0
            max_diff = 0.0
            min_diff = float("inf")
            episode_success = False
            
            max_part_achieved = 0  # Record the highest step achieved in the current trial

            while simulation_app.is_running():
                # 1. Get Target (and convert to world coordinates)
                target_pos_local = ee_pts[frame_idx]
                target_rot_local = ee_oris[frame_idx]

                target_pos_w = r_base.apply(target_pos_local) + base_pos
                target_rot_w = (r_base * Rotation.from_matrix(target_rot_local)).as_matrix()

                # 2. Get Current State
                ee_state = robot.data.body_state_w[0, ee_idx].cpu().numpy()
                curr_pos = ee_state[0:3]
                curr_quat = ee_state[3:7]  # w, x, y, z

                diff_vec = target_pos_w - curr_pos
                pos_err_mm = np.linalg.norm(diff_vec) * 1000.0
                
                # 2. Calculate rotation error (Geodesic distance / Angle difference)
                curr_quat_scipy = [curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]]
                r_curr = Rotation.from_quat(curr_quat_scipy)
                r_target = Rotation.from_matrix(target_rot_w)

                # Calculate the relative rotation between the two rotations
                r_diff = r_target * r_curr.inv()
                rot_err_deg = np.degrees(r_diff.magnitude())

                if args.debug:
                    print("-" * 60)
                    print(f"[Frame {frame_idx}] Error Monitor:")
                    print(f"  >>> Pos Error : {pos_err_mm:.2f} mm")
                    print(f"  >>> Rot Error : {rot_err_deg:.2f} deg")
                    print(
                        f"  Target Pos    : [{target_pos_w[0]:.4f}, {target_pos_w[1]:.4f}, {target_pos_w[2]:.4f}]"
                    )
                    print(
                        f"  Actual Pos    : [{curr_pos[0]:.4f}, {curr_pos[1]:.4f}, {curr_pos[2]:.4f}]"
                    )
                    print(
                        f"  Diff Vector   : [{diff_vec[0]:.4f}, {diff_vec[1]:.4f}, {diff_vec[2]:.4f}]"
                    )
                if pos_err_mm > 10.0:
                    error_count += 1
                    if args.debug:
                        print("  [WARNING] Huge error! Inconsistent coordinate system definitions (e.g., TCP offset) or unreachable target might exist.")
                elif pos_err_mm > 20.0:
                    error_count += 1
                    if args.debug:
                        print("  [WARNING] Obvious error. Check if P-Gain is too small or movement speed is too fast.")
                        print("-" * 60)

                max_diff = max(max_diff, pos_err_mm)
                min_diff = min(min_diff, pos_err_mm)

                # 3. 计算 P-Control Action (Pose Delta)
                ik_cmd_tensor = calculate_action(
                    target_pos_w,
                    target_rot_w,
                    curr_pos,
                    curr_quat,
                    args.pos_gain,
                    args.rot_gain,
                    env.device,
                )

                # 4. Calculate joint target (Differential IK)
                jac = robot.root_physx_view.get_jacobians()[:, ee_idx, :, :]

                env.ik_controller.set_command(
                    ik_cmd_tensor,
                    ee_pos=torch.tensor(curr_pos, device=env.device).unsqueeze(0),
                    ee_quat=torch.tensor(curr_quat, device=env.device).unsqueeze(0),
                )

                new_joint_target = env.ik_controller.compute(
                    torch.tensor(curr_pos, device=env.device).unsqueeze(0),
                    torch.tensor(curr_quat, device=env.device).unsqueeze(0),
                    jac,
                    robot.data.joint_pos,
                )

                # 5. Process gripper
                target_width = ee_widths[frame_idx]

                if max_width <= 0.001:
                    gripper_val = 0.04
                    print(
                        "[WARNING] Max gripper width is zero or negative. Defaulting to fully open (0.04m)."
                    )
                else:
                    ratio = target_width / max_width
                    gripper_val = ratio * 0.04
                    grip_offset = 0.02
                    if target_width < max_width * 0.85:
                        gripper_val -= grip_offset

                gripper_val = np.clip(gripper_val, 0.0, 0.04)

                if prev_gripper_val is not None:
                    if (
                        abs(gripper_val - prev_gripper_val) > max_gripper_delta
                        and frame_idx > 10
                    ):
                        if args.debug:
                            print(
                                f"  [WARNING] Gripper width change too large! Prev: {prev_gripper_val:.4f}, Target: {gripper_val:.4f}"
                            )
                        if gripper_val > prev_gripper_val:
                            gripper_val = prev_gripper_val + 0.001
                        else:
                            gripper_val = prev_gripper_val - 0.001

                prev_gripper_val = gripper_val

                if new_joint_target.shape[1] >= 9:
                    new_joint_target[:, 7] = gripper_val
                    new_joint_target[:, 8] = gripper_val

                robot.set_joint_position_target(new_joint_target)

                for _ in range(env_cfg.decimation):
                    robot.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)
                    
                if args.task_name in TASK_PART_SCORES_MAP:
                    part, score = env.get_part_scores()
                    if part > max_part_achieved:
                        max_part_achieved = part

                    score_strs = []
                    for step, name in task_steps_config.items():
                        mark = "✓" if max_part_achieved >= step else "✗"
                        score_strs.append(f"{name}: {mark}")
                    if frame_idx % 50 == 0:
                        print(f"[Part Scores] {' | '.join(score_strs)}")

                # 8. Render
                if frame_idx % 2 == 0:
                    env.sim.render()

                success_flags = env._get_success()
                if success_flags[0].item():
                    episode_success = True

                # Debug Print
                if frame_idx % 20 == 0 and args.debug:
                    diff = np.linalg.norm(target_pos_w - curr_pos) * 1000
                    print(f"Frame {frame_idx:03d} | Err: {diff:.1f} mm")

                # Next Frame
                frame_idx += 1
                if frame_idx >= total_frames:
                    print(f"--- Completed trajectory {traj_id} trial {trial_idx + 1} ---")
                    if episode_success:
                        print(f"[Result] Task SUCCESS!")
                        success_in_this_traj += 1
                        total_successes += 1
                    else:
                        print(f"[Result] Task FAILED.")
                        
                    total_trials += 1
                        
                    if args.task_name in TASK_PART_SCORES_MAP:
                        print("-" * 40)
                        print("PART SCORES REPORT (Process Accuracy for this trial):")
                        for step in task_steps_config.keys():
                            if max_part_achieved >= step:
                                part_success_counts[step] += 1
                                
                        for step, name in task_steps_config.items():
                            mark = "✓" if max_part_achieved >= step else "✗"
                            print(f"  Step {step} ({name}): {mark}")

                    error_count = 0
                    max_diff = 0.0
                    min_diff = float('inf')
                    episode_success = False
                    break

        results[traj_id] = success_in_this_traj / 2.0
        print(f">>> Trajectory {traj_id} finished. Success rate: {results[traj_id] * 100:.1f}% ({success_in_this_traj}/2)\n")

    print("\n" + "=" * 40)
    print("ALL TRAJECTORIES EVALUATION REPORT")
    print("=" * 40)
    for t_id, rate in results.items():
        print(f"Trajectory {t_id}: {rate * 100:.1f}%")
    if total_trials > 0:
        print(f"OVERALL SUCCESS RATE: {(total_successes / total_trials) * 100:.1f}% ({total_successes}/{total_trials})")
        if args.task_name in TASK_PART_SCORES_MAP:
            print("-" * 40)
            print("PART SCORES REPORT (Process Accuracy):")
            for step, name in task_steps_config.items():
                count = part_success_counts[step]
                print(f"  Step {step} ({name}): {(count / total_trials) * 100:.1f}% ({count}/{total_trials})")
    print("=" * 40)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
