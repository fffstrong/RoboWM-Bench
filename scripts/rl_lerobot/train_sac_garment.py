import argparse
import sys
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Launch Isaac Sim App (MUST BE DONE BEFORE IMPORTING ENV)
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

# Create the parser
parser = argparse.ArgumentParser(description="Train SAC policy for Garment Folding")

# Add Training specific arguments
parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v0", help="Task name")
# parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
parser.add_argument("--output_dir", type=str, default="outputs/garment_sac", help="Output directory")
parser.add_argument("--max_steps", type=int, default=1000000, help="Total training steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")

# Add AppLauncher arguments (e.g., --headless, --num_envs)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

#  Since GarmentEnv uses TiledCamera, we MUST enable cameras.
args.enable_cameras = True

# Launch the simulator
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from pathlib import Path
import torch
import torch.optim as optim
import numpy as np

# Import lehome environment and config
from lehome.tasks.bedroom.garment_bi import GarmentEnv
from lehome.tasks.bedroom.garment_bi_cfg import GarmentEnvCfg
from garment_sac_config import get_garment_sac_config

# Import LeRobot modules
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer

def adapter_process_obs(obs_dict, device_for_policy, target_size=(120, 160)):
    """
    Handles the conversion from Isaac Lab Env (GPU Tensors, Channel-Last)
    to:
    1. Policy Input (Tensor, Channel-First, on policy device)
    2. Buffer Input (Numpy, Channel-First, on CPU)
    
    Args:
        obs_dict: Dict of Tensors from DirectRLEnv. Shape (num_envs, ...).
        device_for_policy: torch.device
    
    Returns:
        policy_obs: Dict of Tensors for SAC forward pass.
        buffer_obs: Dict of Numpy arrays for Replay Buffer storage.
    """
    policy_obs = {}
    buffer_obs = {}
    
    for k, v in obs_dict.items():
        if k == "policy" or k == "action" or k == "depth": # Skip action & policy & depth in obs
            continue
        
        # Convert to Tensor    
        v_tensor = torch.from_numpy(v)
        
        # Add Batch dimension back: (1, ...)
        v_tensor = v_tensor.unsqueeze(0)
        
        # 1. Handle Images: (N, H, W, C) -> (N, C, H, W)
        if "rgb" in k:
            # Check dim to ensure it's image
            if v_tensor.dim() == 4: # (N, H, W, C)
                # Permute to (N, C, H, W)
                tensor_chw = v_tensor.permute(0, 3, 1, 2).contiguous()
            else:
                tensor_chw = v_tensor
                
            # 2. Resize to smaller resolution (e.g., 120x160) to save memory
            # Interpolate expects float usually, but we cast back later
            if tensor_chw.shape[-2:] != target_size:
                tensor_chw = F.interpolate(
                    tensor_chw.float(), 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                tensor_chw = tensor_chw.float()
        else:
            # States: (N, dim) - no change needed
            tensor_chw = v_tensor
            
        # 2. Create Policy Input (Keep on GPU or move to specified device)
        # Ensure float for policy net
        if tensor_chw.dtype == torch.uint8:
            policy_obs[k] = tensor_chw.float().to(device_for_policy)
        else:
            policy_obs[k] = tensor_chw.to(device_for_policy)
            
        # 3. Create Buffer Input (Move to CPU Numpy)
        buffer_obs[k] = tensor_chw.cpu()
        
    return policy_obs, buffer_obs


def train():
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"[TRAIN] Device: {device}")
    
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Environment
    print("[TRAIN] Initializing Isaac Sim Environment...")
    env_cfg = GarmentEnvCfg()
    # You might need to override task/device settings here depending on how arguments are passed
    env_cfg.sim.device = args.device 
    
    # Adjust config based on AppLauncher args if needed (e.g. headless)
    env_cfg.sim.headless = args.headless # AppLauncher handles this usually via simulation_app
    
    env = GarmentEnv(cfg=env_cfg, render_mode="rgb_array")
    
    # 2. Initialize Policy
    print("[TRAIN] Initializing SAC Policy...")
    policy_cfg = get_garment_sac_config(device=args.device)
    policy = SACPolicy(policy_cfg)
    policy.to(device)
    policy.train()

    # 3. Initialize Replay Buffer
    # We use only online buffer since we don't have offline data/HIL
    replay_buffer = ReplayBuffer(
        device=device, 
        state_keys=list(policy_cfg.input_features.keys()),
        capacity=policy_cfg.online_buffer_capacity
    )
    
    optimizer = optim.Adam(policy.parameters(), lr=policy_cfg.actor_lr)

    # 4. Training Loop
    global_step = 0
    episode_num = 0
    
    print("[TRAIN] Starting Training Loop...")
    
    try:
        # Main Simulation Loop
        while simulation_app.is_running():
            if global_step >= args.max_steps:
                print("[TRAIN] Max steps reached.")
                break
            env.initialize_obs()
            # Reset Environment
            obs_dict, extras = env.reset()
            # Process initial observation
            policy_obs, buffer_obs_np = adapter_process_obs(obs_dict, device)
            
            episode_reward = 0
            episode_step = 0
            # Since DirectRLEnv auto-resets internally, 'done' here implies
            # we want to track logical episodes for logging, but the loop continues
            # based on num_steps usually. However, let's write a standard loop.
            done = False
            
            # --- Episode Loop ---
            while not done:
                # Select Action
                with torch.no_grad():
                    if global_step < policy_cfg.online_step_before_learning:
                        # Random exploration
                        action_tensor = torch.rand((1, 12), device=device) * 2 - 1 # [-1, 1]
                    else:
                        action_tensor = policy.select_action(policy_obs)
                
                # Step Environment
                next_obs_dict, reward_val, terminated_tensor, truncated_tensor, extras = env.step(action_tensor)
                
                # --- Process Return Values ---
                next_policy_obs, next_buffer_obs_np = adapter_process_obs(next_obs_dict, device)
                
                reward_tensor_for_buffer = torch.tensor([reward_val], dtype=torch.float32)
                
                terminated = terminated_tensor.item()
                truncated = truncated_tensor.item()
                done = terminated or truncated
                
                action_np = action_tensor.cpu()
                
                # --- Store in Buffer ---
                transition = {
                    "state": buffer_obs_np,         # Dict of numpy arrays
                    "action": action_np,            # Numpy array
                    "reward": reward_tensor_for_buffer,  # Float
                    "next_state": next_buffer_obs_np, # Dict of numpy arrays
                    "done": terminated,             # Bool
                    "truncated": truncated          # Bool
                }
                replay_buffer.add(**transition)
                
                # Update Loop Vars
                policy_obs = next_policy_obs
                buffer_obs_np = next_buffer_obs_np
                episode_reward += reward_val
                global_step += 1
                episode_step += 1
                
                # Training Step
                if len(replay_buffer) > policy_cfg.online_step_before_learning:
                    batch = replay_buffer.sample(args.batch_size)
                    loss_dict = policy.forward(batch)
                    
                    optimizer.zero_grad()
                    loss = loss_dict["loss_critic"] + loss_dict.get("loss_actor", 0) + loss_dict.get("loss_temperature", 0)
                    loss.backward()
                    optimizer.step()
                    
                    if global_step % 100 == 0:
                        print(f"Step {global_step} | Loss: {loss.item():.4f} | Ep Reward: {episode_reward:.2f}")

                if global_step >= args.max_steps:
                    break
            
            episode_num += 1
            print(f"Episode {episode_num} finished. Length: {episode_step}, Reward: {episode_reward}")
            
            # Save checkpoint
            if episode_num % 50 == 0:
                print(f"[TRAIN] Saving checkpoint at episode {episode_num}")
                policy.save_pretrained(output_directory)

    except KeyboardInterrupt:
        print("[TRAIN] Interrupted by user.")
    except Exception as e:
        print(f"[TRAIN] Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[TRAIN] Saving final model and closing...")
        policy.save_pretrained(output_directory)
        if 'env' in locals():
            env.close()
        simulation_app.close()

if __name__ == "__main__":
    train()
    