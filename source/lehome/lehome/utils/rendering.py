from __future__ import annotations

"""
Utilities to configure Isaac Sim / Omniverse Kit rendering and viewport defaults.
"""

from typing import Any

_TONEMAP_APPLIED = False
_LIGHTING_APPLIED = False


def _get_settings():
    try:
        import carb  # type: ignore
        return carb.settings.get_settings()
    except Exception:
        return None


def _safe_set_setting(settings, path: str, value: Any) -> bool:
    """Try to set a carb setting."""
    if settings is None:
        return False
    try:
        settings.set(path, value)
        return True
    except Exception:
        return False


def set_tone_mapping_fstop(fstop: float = 5.8, enabled: bool = True) -> bool:
    """Set RTX tone mapping fNumber (F-stop) to the desired value."""
    settings = _get_settings()
    if settings is None:
        return False

    ok1 = _safe_set_setting(settings, "/rtx/post/tonemap/enabled", bool(enabled))
    ok2 = _safe_set_setting(settings, "/rtx/post/tonemap/fNumber", float(fstop))
    return ok1 and ok2


def setup_default_lighting(task_name: str | None = None):
    """
    Directly create a 3-point light setup under '/World/DefaultLight' to emulate 
    the viewport 'Default' light rig.
    """
    try:
        import omni.usd # type: ignore
        from pxr import UsdLux, Gf, UsdGeom # type: ignore
        
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        base_path = "/World/DefaultLight"
        
        # Key / Fill / Rim parameters to match Isaac Sim 'Default' rig
        # Default rotation: (Rotate X, Rotate Y, Rotate Z)
        configs = [
            ("Key", 3500.0, (-35.0, 45.0, 0.0)),
            ("Fill", 900.0, (-20.0, -35.0, 0.0)),
            ("Rim", 600.0, (60.0, 180.0, 0.0)),
        ]

        # Special handling for burger task: rotate lights by 180 degrees around Y axis
        if task_name and "burger" in task_name.lower():
            print(f"[Rendering] Applying custom lighting for task: {task_name} (Rotated 180°)")
            new_configs = []
            for name, intensity, (rx, ry, rz) in configs:
                # Rotate 180 degrees on Y axis (Yaw)
                new_configs.append((name, intensity, (rx+100, ry-20, rz)))
            configs = new_configs
        if task_name and "tableware" in task_name.lower():
            print(f"[Rendering] Applying custom lighting for task: {task_name} (Rotated 180°)")
            new_configs = []
            for name, intensity, (rx, ry, rz) in configs:
                # Rotate 180 degrees on Y axis (Yaw)
                new_configs.append((name, intensity, (rx+100, ry-20, rz)))
            configs = new_configs

        created_lights = []
        for name, intensity, rot in configs:
            path = f"{base_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            
            # 创建或获取光源
            if not prim.IsValid():
                light = UsdLux.DistantLight.Define(stage, path)
                status = "created"
            else:
                light = UsdLux.DistantLight(prim)
                status = "updated"
            
            # 设置属性（无论是新创建还是已存在的光源）
            light.GetIntensityAttr().Set(float(intensity))
            light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            
            # 设置方向
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
            
            created_lights.append(f"{name}({status}, I={intensity})")
        
        # 打印详细信息
        print(f"[Rendering] ✓ Lights setup: {', '.join(created_lights)}")
        
        # 所有光源设置完成后才返回
        return True
    except Exception as e:
        print(f"[Rendering][Error] Failed to setup default lighting: {e}")
        return False

def setup_default_lighting_drawer(task_name: str | None = None):
    """
    Directly create a 3-point light setup under '/World/DefaultLight' to emulate 
    the viewport 'Default' light rig.
    """
    try:
        import omni.usd # type: ignore
        from pxr import UsdLux, Gf, UsdGeom # type: ignore
        
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False

        base_path = "/World/DefaultLight"
        
        # Key / Fill / Rim parameters to match Isaac Sim 'Default' rig
        # Default rotation: (Rotate X, Rotate Y, Rotate Z)
        configs = [
            ("Key", 3500.0, (-35.0, 45.0, 0.0)),
            ("Fill", 900.0, (-20.0, -35.0, 0.0)),
            ("Rim", 600.0, (60.0, 180.0, 0.0)),
        ]


        print(f"[Rendering] Applying custom lighting for task: {task_name} (Rotated 180°)")
        new_configs = []
        for name, intensity, (rx, ry, rz) in configs:
            # Rotate 180 degrees on Y axis (Yaw)
            new_configs.append((name, intensity, (rx-40, ry-20, rz+20)))
        configs = new_configs

        created_lights = []
        for name, intensity, rot in configs:
            path = f"{base_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            
            # 创建或获取光源
            if not prim.IsValid():
                light = UsdLux.DistantLight.Define(stage, path)
                status = "created"
            else:
                light = UsdLux.DistantLight(prim)
                status = "updated"
            
            # 设置属性（无论是新创建还是已存在的光源）
            light.GetIntensityAttr().Set(float(intensity))
            light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
            
            # 设置方向
            xform = UsdGeom.Xformable(light.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
            
            created_lights.append(f"{name}({status}, I={intensity})")
        
        # 打印详细信息
        print(f"[Rendering] ✓ Lights setup: {', '.join(created_lights)}")
        
        # 所有光源设置完成后才返回
        return True
    except Exception as e:
        print(f"[Rendering][Error] Failed to setup default lighting: {e}")
        return False


def apply_default_render_settings(
    *,
    tonemap_fstop: float = 5.8,
    enable_tonemap: bool = True,
    once_per_process: bool = True,
    task_name: str | None = None,
    **kwargs,  # Accept extra args for backward compatibility
) -> None:
    """
    Apply requested default rendering settings.
    - Sets Tone Mapping F-stop to 5.8.
    - Sets up simulated Default lighting.
    """
    global _TONEMAP_APPLIED, _LIGHTING_APPLIED

    # 1. Tone Mapping
    if not (once_per_process and _TONEMAP_APPLIED):
        tm_ok = set_tone_mapping_fstop(fstop=tonemap_fstop, enabled=enable_tonemap)
        if tm_ok:
            _TONEMAP_APPLIED = True
            print(f"[Rendering] ✓ Tone mapping F-stop set to {tonemap_fstop}")

    # 2. Default Lighting Emulation
    if not (once_per_process and _LIGHTING_APPLIED):
        if setup_default_lighting(task_name=task_name):
            _LIGHTING_APPLIED = True
        # Note: setup_default_lighting() now prints its own detailed message

def apply_default_render_settings_drawer(
    *,
    tonemap_fstop: float = 5.8,
    enable_tonemap: bool = True,
    once_per_process: bool = True,
    task_name: str | None = None,
    **kwargs,  # Accept extra args for backward compatibility
) -> None:
    """
    Apply requested default rendering settings.
    - Sets Tone Mapping F-stop to 5.8.
    - Sets up simulated Default lighting.
    """
    global _TONEMAP_APPLIED, _LIGHTING_APPLIED

    # 1. Tone Mapping
    if not (once_per_process and _TONEMAP_APPLIED):
        tm_ok = set_tone_mapping_fstop(fstop=tonemap_fstop, enabled=enable_tonemap)
        if tm_ok:
            _TONEMAP_APPLIED = True
            print(f"[Rendering] ✓ Tone mapping F-stop set to {tonemap_fstop}")

    # 2. Default Lighting Emulation
    if not (once_per_process and _LIGHTING_APPLIED):
        if setup_default_lighting_drawer(task_name=task_name):
            _LIGHTING_APPLIED = True
        # Note: setup_default_lighting() now prints its own detailed message
