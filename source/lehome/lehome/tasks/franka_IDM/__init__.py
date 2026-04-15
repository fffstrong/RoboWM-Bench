import gymnasium as gym

gym.register(
    id="Franka-IDM",
    entry_point=f"{__name__}.idm:IDMEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.idm_cfg:IDMEnvCfg",
    },
)
