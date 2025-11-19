from pathlib import Path

from lehome.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

BEDROOM_USD_PATH = str(SCENES_ROOT / "robocasaliving_room-1-1" / "scene.usd")

KITCHEN_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BEDROOM_USD_PATH,
    )
)
