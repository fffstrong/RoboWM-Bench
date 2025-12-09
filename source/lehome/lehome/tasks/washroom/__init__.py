import gymnasium as gym

gym.register(
    id="LeHome-SO101-Direct-loftwipe-v0",
    entry_point=f"{__name__}.loft_wipe:LoftWipeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loft_wipe_cfg:LoftWipeEnvCfg",
    },
)
