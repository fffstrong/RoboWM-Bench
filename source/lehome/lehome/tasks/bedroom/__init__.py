import gymnasium as gym

gym.register(
    id="LeIsaac-BiSO101-Direct-Garment-v0",
    entry_point=f"{__name__}.garment_bi:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg:GarmentEnvCfg",
    },
)