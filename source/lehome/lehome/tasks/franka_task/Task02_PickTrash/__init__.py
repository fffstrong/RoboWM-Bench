import gymnasium as gym

gym.register(
    id="Franka-picktrash",
    entry_point=f"{__name__}.picktrash:TrashEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.picktrash_cfg:TrashEnvCfg",
    },
)
