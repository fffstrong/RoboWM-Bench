import gymnasium as gym

gym.register(
    id="Franka-put_on_plate",
    entry_point=f"{__name__}.tableware:TablewareEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tableware_cfg:TablewareEnvCfg",
    },
)
