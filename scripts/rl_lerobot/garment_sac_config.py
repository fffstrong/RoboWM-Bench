from dataclasses import dataclass
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import ACTION, OBS_STATE

# 用来解决 'dict' object has no attribute 'shape' 的报错
# modeling_sac.py 期望的是对象访问 (.shape)，而不是字典访问 (['shape'])
@dataclass
class FeatureSpec:
    shape: tuple
    dtype: str = "float32"

def get_garment_sac_config(device="cuda"):
    """
    Generates SAC configuration for the Garment folding task.
    """
    
    # Define features based on GarmentEnv outputs
    # Images: (C, H, W)
    IMG_H = 120
    IMG_W = 160
    input_features = {
        "observation.state": FeatureSpec(
            shape=(12,),  # 6 + 6
            dtype="float32"
        ),
        "observation.images.top_rgb": FeatureSpec(
            shape=(3, IMG_H, IMG_W), # (C, H, W)
            dtype="uint8"  # Assuming raw from camera
        ),
        "observation.images.left_rgb": FeatureSpec(
            shape=(3, IMG_H, IMG_W),
            dtype="uint8"
        ),
        "observation.images.right_rgb": FeatureSpec(
            shape=(3, IMG_H, IMG_W),
            dtype="uint8"
        ),
        # Assuming we don't use depth for the policy input yet to keep it simple, 
        # but if you want to use it, uncomment below:
        # "observation.top_depth": FeatureSpec(
        #     shape=(3, 480, 640),
        #     dtype="float32"
        # ),
    }

    # Action: 12 dims
    output_features = {
        "action": FeatureSpec(
            shape=(12,),
            dtype="float32"
        )
    }

    # Dataset stats for normalization
    # IMPORTANT: Since we don't have a dataset yet, we define reasonable bounds.
    # Joint positions (State) are typically in radians. Let's assume -pi to pi range roughly.
    # Actions are normalized outputs from policy, scaled by env.
    dataset_stats = {
        "observation.images.top_rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "observation.images.left_rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "observation.images.right_rgb": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "observation.state": {
            "min": [-3.14] * 12, 
            "max": [3.14] * 12
        },
        ACTION: {
            "min": [-1.0] * 12, 
            "max": [1.0] * 12
        }
    }

    cfg = SACConfig(
        device=device,
        input_features=input_features,
        output_features=output_features,
        dataset_stats=dataset_stats,
        
        # SAC Hyperparameters
        actor_lr=3e-4,
        critic_lr=3e-4,
        temperature_lr=3e-4,
        online_steps=1_000_000,
        online_buffer_capacity=100_00,
        online_step_before_learning=100, # Collect some random steps first
        
        # Encoder settings
        vision_encoder_name="microsoft/resnet-18", # Or keep None for simple CNN
        image_encoder_hidden_dim=256,
        latent_dim=256,
        
        # Misc
        use_torch_compile=False # Set True if your setup supports it
    )
    
    return cfg
