import gymnasium as gym

gym.register(
    id="Franka-press_button",
    entry_point=f"{__name__}.pressbutton:PressButtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pressbutton_cfg:PressButtonEnvCfg",
    },
)
