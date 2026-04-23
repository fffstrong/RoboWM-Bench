import gymnasium as gym

gym.register(
    id="Task06_Franka_Tableware_Stapler_Box",
    entry_point=f"{__name__}.tableware:TablewareEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tableware_cfg:TablewareEnvCfg",
    },
)
