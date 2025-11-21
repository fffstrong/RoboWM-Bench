from pathlib import Path

from lehome.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT)

KITCHEN_WITH_ORANGE_USD_PATH = str(SCENES_ROOT / "scenes" / "kitchen_with_orange" / "scene_v1.usd")

KITCHEN_WITH_ORANGE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_WITH_ORANGE_USD_PATH,
    )
)
