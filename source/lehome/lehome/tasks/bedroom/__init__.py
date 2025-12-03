import gymnasium as gym

gym.register(
    id="LeHome-BiSO101-Direct-Garment-v0",
    entry_point=f"{__name__}.garment_bi:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg:GarmentEnvCfg",
    },
)

gym.register(
    id="LeHome-SO101-Direct-Garment-v0",
    entry_point=f"{__name__}.garment:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_cfg:GarmentEnvCfg",
    },
)

gym.register(
    id="LeHome-BiSO101-Direct-Garment-v2",
    entry_point=f"{__name__}.garment_bi_v2:GarmentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg_v2:GarmentEnvCfg",
    },
)

gym.register(
    id="LeHome-BiSO101-Direct-Marbel-v0",
    entry_point=f"{__name__}.test_bi:MarbelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_bi_cfg:MarbelEnvCfg",
    },
)