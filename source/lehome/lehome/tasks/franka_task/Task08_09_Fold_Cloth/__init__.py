import gymnasium as gym

gym.register(
    id="Franka-garment",
    entry_point=f"{__name__}.garment:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_cfg:GarmentEnvCfg",
    },
)

