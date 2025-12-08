# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-

import gymnasium as gym

gym.register(
    id="LeIsaac-BiSO101-Direct-loftcut-v0",
    entry_point=f"{__name__}.loft_cut_bi:LoftCutEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loft_cut_bi_cfg:LoftCutEnvCfg",
    },
)
gym.register(
    id="LeIsaac-BiSO101-Direct-loftburger-v0",
    entry_point=f"{__name__}.loft_burger_bi:LoftBurgerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loft_burger_bi_cfg:LoftBurgerEnvCfg",
    },
)
