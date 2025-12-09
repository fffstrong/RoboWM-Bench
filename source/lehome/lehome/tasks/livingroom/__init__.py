import gymnasium as gym

gym.register(
    id="LeHome-SO101-Direct-loftwater-v0",
    entry_point=f"{__name__}.loft_water:LoftWaterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loft_water_cfg:LoftWaterEnvCfg",
    },
)
