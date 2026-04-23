# dataset_replay_npz_bi.py
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.app import AppLauncher
import json
import os

TASKS_WITH_TWO_OBJECTS = [
    "Task11_Bi_Franka_Tableware_Cook"
]
TASK_PART_SCORES_MAP = {
    "Task11_Bi_Franka_Tableware_Cook": {
        1: "Grasp spatula",
        2: "Grasp pan",
        3: "Lift spatula",
        4: "Lift pan",
        5: "Spatula--pan contact",
        6: "Place spatula",
        7: "Place pan"
    },
    "Task12_Bi_Franka_Tableware_Big_Box": {
        1: "Touch left",
        2: "Touch right",
        3: "Lift",
        4: "Place"
    }
}

# -- Args Setup --
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="Task11_Bi_Franka_Tableware_Cook", help="Task folder name")
parser.add_argument("--model_name", type=str, default="human", help="Model name, e.g., human or veo")
parser.add_argument("--debug", action="store_true", help="Print detailed error comparisons")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--pos_gain", type=float, default=6.0, help="Position P-Gain")
parser.add_argument("--rot_gain", type=float, default=5.0, help="Rotation P-Gain")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Construct Paths
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
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import lehome.tasks.human_task

def calculate_action(
    target_pos, target_rot_mat, current_pos, current_quat, pos_gain, rot_gain, device
):
    """Calculate P-Control actions for [dx, dy, dz, drot_x, drot_y, drot_z]"""
    # 1. Position Error
    pos_err = target_pos - current_pos

    # 2. Rotation Error (Scipy handles conversion)
    curr_quat_scipy = np.array(
        [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
    )
    r_target = Rotation.from_matrix(target_rot_mat)
    r_current = Rotation.from_quat(curr_quat_scipy)

    r_diff = r_target * r_current.inv()
    rot_vec_err = r_diff.as_rotvec() 

    # 3. Apply Gains
    action_pos = pos_err * pos_gain
    action_rot = rot_vec_err * rot_gain

    action_combined = np.concatenate([action_pos, action_rot])
    return torch.tensor(action_combined, dtype=torch.float32, device=device).unsqueeze(0)


def load_trajectory(file_path):
    data = np.load(file_path)
    pts = data["ee_pts"]
    oris = data["ee_oris"]
    widths = data["ee_widths"] if "ee_widths" in data else np.zeros(len(pts))

    if len(widths) > 0:
        max_width = np.max(widths)
        min_width = np.min(widths)
    else:
        max_width = 0.08
        min_width = 0.0

    if args.debug:
        print(f"[INFO] Loaded trajectory. Gripper Width Range: [{min_width:.4f}, {max_width:.4f}]")
    return pts, oris, widths, max_width, min_width


def get_gripper_action(target_width, max_width, prev_gripper_val):
    if max_width <= 0.001:
        return 0.04

    ratio = target_width / max_width
    gripper_val = ratio * 0.04
    grip_offset = 0.019
    
    if target_width < max_width * 0.85:
        gripper_val -= grip_offset

    gripper_val = np.clip(gripper_val, 0.0, 0.04)
    max_gripper_delta = 0.01
    
    if prev_gripper_val is not None:
        if abs(gripper_val - prev_gripper_val) > max_gripper_delta:
            gripper_val = prev_gripper_val + np.sign(gripper_val - prev_gripper_val) * 0.002

    return gripper_val


def main():
    with open(args.init_json, "r") as f:
        init_states = json.load(f)

    # Init Env
    env_cfg = parse_env_cfg(args.task_name, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task_name, cfg=env_cfg).unwrapped

    # Markers for visualization
    marker_cfg_l = FRAME_MARKER_CFG.copy()
    marker_cfg_l.prim_path = "/Visuals/TargetLeft"
    marker_cfg_l.markers["frame"].scale = (0.05, 0.05, 0.05)
    target_marker_l = VisualizationMarkers(marker_cfg_l)

    marker_cfg_r = FRAME_MARKER_CFG.copy()
    marker_cfg_r.prim_path = "/Visuals/TargetRight"
    marker_cfg_r.markers["frame"].scale = (0.05, 0.05, 0.05)
    target_marker_r = VisualizationMarkers(marker_cfg_r)

    # Statistics Tracking
    results = {}
    total_trials = 0
    total_successes = 0
    task_steps_config = TASK_PART_SCORES_MAP.get(args.task_name, {})
    part_success_counts = {step: 0 for step in task_steps_config.keys()}

    for traj_id in range(10):
        # Bi-manual Trajectory Paths
        traj_path_left = os.path.join(
            args.task_dir, str(traj_id), f"{traj_id}_{args.model_name}_left_black",
            "smoothing_processor", "smoothed_actions_left_single_arm.npz"
        )
        traj_path_right = os.path.join(
            args.task_dir, str(traj_id), f"{traj_id}_{args.model_name}_right_black",
            "smoothing_processor", "smoothed_actions_right_single_arm.npz"
        )

        if not os.path.exists(traj_path_left) or not os.path.exists(traj_path_right):
            print(f"[WARNING] Missing Bimanual Trajectory files for ID {traj_id}. Skipping.")
            continue

        pts_l, oris_l, widths_l, max_w_l, min_w_l = load_trajectory(traj_path_left)
        pts_r, oris_r, widths_r, max_w_r, min_w_r = load_trajectory(traj_path_right)
        total_frames = min(len(pts_l), len(pts_r))
        
        success_in_this_traj = 0

        for trial_idx in range(2):
            print(f"\n{'='*20} Testing Trajectory {traj_id} | Trial {trial_idx + 1}/2 {'='*20}")

            # 1. Set Initial Object Positions
            new_pos = init_states[str(traj_id)]
            print(f"[INFO] Setting initial positions for Traj {traj_id}: {new_pos}")
            if args.task_name in TASKS_WITH_TWO_OBJECTS:
                # Two-object task: JSON format should be [[x1, y1, z1], [x2, y2, z2]]
                pos_A = new_pos[0]
                pos_B = new_pos[1]
                
                # Set Object A
                if hasattr(env.object_A, "data"):
                    env.object_A.data.default_root_state[:, 0:3] = torch.tensor(pos_A, dtype=torch.float32, device=env.device)
                elif hasattr(env.object_A, "garment_config"):
                    # For cloth types, force override its reset range locked to pos_A
                    env.object_A.garment_config.soft_reset_pos_range = [pos_A[0], pos_A[1], pos_A[2], pos_A[0], pos_A[1], pos_A[2]]
                    
                # Set Object B
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

            # 2. Reset and Setup Arms
            env.reset()
            left_arm = env.left_arm
            right_arm = env.right_arm

            try:
                ee_idx_l = left_arm.find_bodies("panda_hand")[0][0]
                ee_idx_r = right_arm.find_bodies("panda_hand")[0][0]
            except (IndexError, TypeError):
                ee_idx_l = left_arm.num_bodies - 1
                ee_idx_r = right_arm.num_bodies - 1

            print("[INFO] Warming up to start pose (Pos + Rot)...")

            # Calculate World Start Poses L/R
            base_state_l = left_arm.data.root_state_w[0, :7].cpu().numpy()
            base_pos_l, base_quat_l = base_state_l[0:3], base_state_l[3:7]
            r_base_l = Rotation.from_quat([base_quat_l[1], base_quat_l[2], base_quat_l[3], base_quat_l[0]])
            
            start_pos_w_l = r_base_l.apply(pts_l[0]) + base_pos_l
            start_rot_w_l = (r_base_l * Rotation.from_matrix(oris_l[0])).as_matrix()

            base_state_r = right_arm.data.root_state_w[0, :7].cpu().numpy()
            base_pos_r, base_quat_r = base_state_r[0:3], base_state_r[3:7]
            r_base_r = Rotation.from_quat([base_quat_r[1], base_quat_r[2], base_quat_r[3], base_quat_r[0]])

            start_pos_w_r = r_base_l.apply(pts_r[0]) + base_pos_l
            start_rot_w_r = (r_base_l * Rotation.from_matrix(oris_r[0])).as_matrix()

            curr_joints_l = left_arm.data.joint_pos.clone()
            curr_joints_r = right_arm.data.joint_pos.clone()

            # Warmup Loop
            for i in range(160):
                left_arm.update(dt=0.0)
                right_arm.update(dt=0.0)

                # Left Arm Warmup
                ee_state_l = left_arm.data.body_state_w[:, ee_idx_l]
                cmd_l = calculate_action(
                    start_pos_w_l, start_rot_w_l, ee_state_l[0, 0:3].cpu().numpy(), ee_state_l[0, 3:7].cpu().numpy(), 1.0, 1.0, env.device
                )
                cmd_l = torch.clamp(cmd_l, -0.1, 0.1)
                env.ik_controller_left.set_command(cmd_l, ee_pos=ee_state_l[:, 0:3], ee_quat=ee_state_l[:, 3:7])
                jac_l = left_arm.root_physx_view.get_jacobians()[:, ee_idx_l, :, :]
                joint_target_l = env.ik_controller_left.compute(ee_state_l[:, 0:3], ee_state_l[:, 3:7], jac_l, curr_joints_l)
                joint_target_l[:, 7:9] = 0.04
                left_arm.write_joint_state_to_sim(joint_target_l, torch.zeros_like(joint_target_l))
                curr_joints_l = joint_target_l.clone()

                # Right Arm Warmup
                ee_state_r = right_arm.data.body_state_w[:, ee_idx_r]
                cmd_r = calculate_action(
                    start_pos_w_r, start_rot_w_r, ee_state_r[0, 0:3].cpu().numpy(), ee_state_r[0, 3:7].cpu().numpy(), 1.0, 1.0, env.device
                )
                cmd_r = torch.clamp(cmd_r, -0.1, 0.1)
                env.ik_controller_right.set_command(cmd_r, ee_pos=ee_state_r[:, 0:3], ee_quat=ee_state_r[:, 3:7])
                jac_r = right_arm.root_physx_view.get_jacobians()[:, ee_idx_r, :, :]
                joint_target_r = env.ik_controller_right.compute(ee_state_r[:, 0:3], ee_state_r[:, 3:7], jac_r, curr_joints_r)
                joint_target_r[:, 7:9] = 0.04
                right_arm.write_joint_state_to_sim(joint_target_r, torch.zeros_like(joint_target_r))
                curr_joints_r = joint_target_r.clone()

            print("[INFO] Warmup Done.")

            # Apply defaults and reset
            left_arm.data.default_joint_pos[:] = curr_joints_l
            right_arm.data.default_joint_pos[:] = curr_joints_r
            env.reset()
            left_arm.set_joint_position_target(curr_joints_l)
            right_arm.set_joint_position_target(curr_joints_r)
            left_arm.update(dt=env.step_dt)
            right_arm.update(dt=env.step_dt)

            # 3. Start P-Control Loop
            print("[INFO] Starting Bimanual P-Control Replay Loop...")
            prev_g_val_l = None
            prev_g_val_r = None
            frame_idx = 0
            episode_success = False
            max_part_achieved = 0

            while simulation_app.is_running():
                # Get L/R Targets
                target_pos_w_l = r_base_l.apply(pts_l[frame_idx]) + base_pos_l
                target_rot_w_l = (r_base_l * Rotation.from_matrix(oris_l[frame_idx])).as_matrix()

                target_pos_w_r = r_base_l.apply(pts_r[frame_idx]) + base_pos_l
                target_rot_w_r = (r_base_l * Rotation.from_matrix(oris_r[frame_idx])).as_matrix()

                # Visualizers
                vis_q_l = Rotation.from_matrix(target_rot_w_l).as_quat()
                target_marker_l.visualize(
                    translations=torch.tensor(target_pos_w_l, dtype=torch.float32, device=env.device).unsqueeze(0),
                    orientations=torch.tensor([vis_q_l[3], vis_q_l[0], vis_q_l[1], vis_q_l[2]], dtype=torch.float32, device=env.device).unsqueeze(0),
                )
                vis_q_r = Rotation.from_matrix(target_rot_w_r).as_quat()
                target_marker_r.visualize(
                    translations=torch.tensor(target_pos_w_r, dtype=torch.float32, device=env.device).unsqueeze(0),
                    orientations=torch.tensor([vis_q_r[3], vis_q_r[0], vis_q_r[1], vis_q_r[2]], dtype=torch.float32, device=env.device).unsqueeze(0),
                )

                # Get L/R States
                state_l = left_arm.data.body_state_w[0, ee_idx_l].cpu().numpy()
                state_r = right_arm.data.body_state_w[0, ee_idx_r].cpu().numpy()

                # P-Control Actions
                cmd_l = calculate_action(target_pos_w_l, target_rot_w_l, state_l[0:3], state_l[3:7], args.pos_gain, args.rot_gain, env.device)
                cmd_r = calculate_action(target_pos_w_r, target_rot_w_r, state_r[0:3], state_r[3:7], args.pos_gain, args.rot_gain, env.device)

                # Compute L/R IK
                jac_l = left_arm.root_physx_view.get_jacobians()[:, ee_idx_l, :, :]
                jac_r = right_arm.root_physx_view.get_jacobians()[:, ee_idx_r, :, :]

                env.ik_controller_left.set_command(cmd_l, ee_pos=torch.tensor(state_l[0:3], device=env.device).unsqueeze(0), ee_quat=torch.tensor(state_l[3:7], device=env.device).unsqueeze(0))
                env.ik_controller_right.set_command(cmd_r, ee_pos=torch.tensor(state_r[0:3], device=env.device).unsqueeze(0), ee_quat=torch.tensor(state_r[3:7], device=env.device).unsqueeze(0))

                new_joint_target_l = env.ik_controller_left.compute(torch.tensor(state_l[0:3], device=env.device).unsqueeze(0), torch.tensor(state_l[3:7], device=env.device).unsqueeze(0), jac_l, left_arm.data.joint_pos)
                new_joint_target_r = env.ik_controller_right.compute(torch.tensor(state_r[0:3], device=env.device).unsqueeze(0), torch.tensor(state_r[3:7], device=env.device).unsqueeze(0), jac_r, right_arm.data.joint_pos)

                # Gripper Logic
                g_val_l = get_gripper_action(widths_l[frame_idx], max_w_l, prev_g_val_l)
                g_val_r = get_gripper_action(widths_r[frame_idx], max_w_r, prev_g_val_r)
                prev_g_val_l, prev_g_val_r = g_val_l, g_val_r

                if new_joint_target_l.shape[1] >= 9:
                    new_joint_target_l[:, 7:9] = g_val_l
                if new_joint_target_r.shape[1] >= 9:
                    new_joint_target_r[:, 7:9] = g_val_r

                # Command & Step
                left_arm.set_joint_position_target(new_joint_target_l)
                right_arm.set_joint_position_target(new_joint_target_r)

                for _ in range(env_cfg.decimation):
                    left_arm.write_data_to_sim()
                    right_arm.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt=env.physics_dt)

                # Scores Tracker
                if args.task_name in TASK_PART_SCORES_MAP:
                    env.get_part_scores() # Call environment function to update internal env.part and env.scores
                    part = env.part
                    if part > max_part_achieved:
                        max_part_achieved = part
                        
                    if frame_idx % 50 == 0:
                        score_strs = []
                        for step, name in task_steps_config.items():
                            mark = "✓" if max_part_achieved >= step else "✗"
                            score_strs.append(f"{name}: {mark}")
                        print(f"[Part Scores] {' | '.join(score_strs)}")

                if frame_idx % 2 == 0:
                    env.sim.render()

                # Success Flag
                success_flags = env._get_success()
                if success_flags[0].item():
                    episode_success = True

                if frame_idx % 20 == 0 and args.debug:
                    diff_l = np.linalg.norm(target_pos_w_l - state_l[0:3]) * 1000
                    diff_r = np.linalg.norm(target_pos_w_r - state_r[0:3]) * 1000
                    print(f"Frame {frame_idx:03d} | L Err: {diff_l:.1f} mm | R Err: {diff_r:.1f} mm")

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
                            
                    break

        results[traj_id] = success_in_this_traj / 2.0
        print(f">>> Trajectory {traj_id} Finished. Success Rate: {results[traj_id] * 100:.1f}% ({success_in_this_traj}/2)\n")

    # Final Report
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
