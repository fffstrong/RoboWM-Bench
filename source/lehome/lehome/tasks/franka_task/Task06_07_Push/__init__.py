import gymnasium as gym

gym.register(
    id="Franka-pullandpush",
    entry_point=f"{__name__}.pullandpush:PullandPushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pullandpush_cfg:PullandPushEnvCfg",
    },
)