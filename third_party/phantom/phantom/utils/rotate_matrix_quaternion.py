import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Original rotation matrix (your input)
camera_base_ori = np.array([
    [ 0.14020245, -0.74159521,  0.6560334 ],
    [-0.98983669, -0.08905137,  0.11087464],
    [-0.02380343, -0.66491082, -0.74654336]
])

# 2. Create original rotation object
r_orig = R.from_matrix(camera_base_ori)

# 3. Create 90-degree Z-axis rotation object
# 90 degrees = np.pi / 2
r_rot_90 = R.from_euler('z', 90, degrees=True)

# ---------------------------------------------------------
# 4. Perform rotation (choose one scenario)
# ---------------------------------------------------------

# [Scenario A]: Rotate along the "World Coordinate System" Z-axis (Global / Extrinsic)
# Applicable for: Correcting coordinate system orientation (e.g., changing 'forward' from X-axis to Y-axis)
# Algorithm: New rotation = Rotation90 * Old rotation (Left multiplication)
r_new_global = r_rot_90 * r_orig 

# [Scenario B]: Rotate along the "Camera's own" Z-axis (Local / Intrinsic)
# Applicable for: Camera position remains unchanged, only rotating the lens by 90 degrees (Roll)
# Algorithm: New rotation = Old rotation * Rotation90 (Right multiplication)
r_new_local = r_orig * r_rot_90

# 5. Output results (Convert to Isaac Sim default XYZ Euler angles)
euler_global = r_new_global.as_euler('xyz', degrees=True)
euler_local  = r_new_local.as_euler('xyz', degrees=True)

print("--- Angles converted from the original matrix ---")
print(f"{r_orig.as_euler('xyz', degrees=True)}")

print("\n--- Scenario A: Rotate +90 degrees along World Z-axis (Left multiplication) ---")
print(f"X: {euler_global[0]:.4f}")
print(f"Y: {euler_global[1]:.4f}")
print(f"Z: {euler_global[2]:.4f}")

print("\n--- Scenario B: Rotate +90 degrees along Local Z-axis (Right multiplication) ---")
print(f"X: {euler_local[0]:.4f}")
print(f"Y: {euler_local[1]:.4f}")
print(f"Z: {euler_local[2]:.4f}")
