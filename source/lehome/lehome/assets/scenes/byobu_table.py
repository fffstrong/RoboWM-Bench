from pathlib import Path

from lehome.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT)

BYOBU_TABLE_USD_PATH = str(SCENES_ROOT / "human_assets" / "scenes" / "byobu_table" / "scene_v1.usd")

BYOBU_TABLE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BYOBU_TABLE_USD_PATH,
    )
)
