import gymnasium as gym

gym.register(
    id="Franka-buttom",
    entry_point=f"{__name__}.pressbuttom:PressButtomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pressbuttom_cfg:PressButtomEnvCfg",
    },
)
