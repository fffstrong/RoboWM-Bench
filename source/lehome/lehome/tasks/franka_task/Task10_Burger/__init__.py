import gymnasium as gym

gym.register(
    id="Franka-burger",
    entry_point=f"{__name__}.burger:BurgerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.burger_cfg:BurgerEnvCfg",
    },
)
