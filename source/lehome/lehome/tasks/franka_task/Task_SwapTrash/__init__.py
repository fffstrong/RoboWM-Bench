import gymnasium as gym

gym.register(
    id="Franka-Swaptrash",
    entry_point=f"{__name__}.swaptrash:SwapRubbishEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.swaptrash_cfg:SwapRubbishEnvCfg",
    },
)