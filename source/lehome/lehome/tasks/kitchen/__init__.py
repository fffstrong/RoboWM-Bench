import gymnasium as gym

gym.register(
    id="LeHome-BiSO101-Direct-loftcut-v0",
    entry_point=f"{__name__}.loft_cut_bi:LoftCutEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loft_cut_bi_cfg:LoftCutEnvCfg",
    },
)
gym.register(
    id="LeHome-BiSO101-Direct-loftburger-v0",
    entry_point=f"{__name__}.loft_burger_bi:LoftBurgerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loft_burger_bi_cfg:LoftBurgerEnvCfg",
    },
)
