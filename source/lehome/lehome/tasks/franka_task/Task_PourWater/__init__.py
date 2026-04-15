import gymnasium as gym

gym.register(
    id="Franka-Pourwater",
    entry_point=f"{__name__}.pour_water:PourWaterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pour_water_cfg:PourWaterEnvCfg",
    },
)
