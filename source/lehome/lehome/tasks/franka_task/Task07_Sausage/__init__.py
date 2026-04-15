import gymnasium as gym

gym.register(
    id="Franka-sausage",
    entry_point=f"{__name__}.sausage:SausageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sausage_cfg:SausageEnvCfg",
    },
)
