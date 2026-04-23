import gymnasium as gym

gym.register(
    id="Task10_Franka_Tableware_Towel",
    entry_point=f"{__name__}.towel:TowelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.towel_cfg:TowelEnvCfg",
    },
)

