import gymnasium as gym

gym.register(
    id="Franka-faucet",
    entry_point=f"{__name__}.faucet:FaucetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.faucet_cfg:FaucetEnvCfg",
    },
)