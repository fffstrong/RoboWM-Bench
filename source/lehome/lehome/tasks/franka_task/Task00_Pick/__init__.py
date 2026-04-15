import gymnasium as gym

gym.register(
    id="Franka-pick",
    entry_point=f"{__name__}.pick:PickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cfg:PickEnvCfg",
    },
)
