import gymnasium as gym

gym.register(
    id="Franka-close_drawer",
    entry_point=f"{__name__}.drawer:DrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drawer_cfg:DrawerEnvCfg",
    },
)
