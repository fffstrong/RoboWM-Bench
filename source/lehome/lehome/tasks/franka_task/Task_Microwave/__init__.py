import gymnasium as gym

gym.register(
    id="Franka-microwave",
    entry_point=f"{__name__}.microwave:MicrowaveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.microwave_cfg:MicrowaveEnvCfg",
    },
)

