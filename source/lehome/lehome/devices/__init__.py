from .device_base import DeviceBase
from .keyboard import (
    BiKeyboard,
    Se3DroidKeyboard,
    Se3FrankaKeyboard,
    Se3Keyboard,
    XlerobotKeyboard,
)

# from .gamepad import XboxController
from .hybrid.xlerobot_hybrid_controller import XlerobotHybridController

# Optional LeRobot devices.
# These depend on additional assets / hardware-specific modules and should not
# break importing LeHome when they are unavailable (e.g., during Franka eval).
try:
    from .lerobot import SO101Leader, BiSO101Leader  # type: ignore
except Exception:  # pragma: no cover
    SO101Leader = None  # type: ignore
    BiSO101Leader = None  # type: ignore

__all__ = [
    "DeviceBase",
    "Se3Keyboard",
    "Se3FrankaKeyboard",
    "Se3DroidKeyboard",
    "BiKeyboard",
    "XlerobotKeyboard",
    "XlerobotHybridController",  # 添加这一行
    # "XboxController",  # 注释掉，因为可能不存在
]

# Only export LeRobot devices if they imported successfully.
if SO101Leader is not None:
    __all__.append("SO101Leader")
if BiSO101Leader is not None:
    __all__.append("BiSO101Leader")
