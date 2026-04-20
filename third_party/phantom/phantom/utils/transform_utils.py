import numpy as np
import math

EPS = np.finfo(float).eps * 4.0

def transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts, np.ones((len(pts), 1))])
    pts = np.dot(T, pts.T).T
    return pts[:, :3]

def project_point_to_plane(point: np.ndarray, plane_coeffs: np.ndarray) -> np.ndarray:
    """
    Projects a 3D point onto a plane defined by its coefficients.

    Args:
        point (array-like): Coordinates of the point to be projected (x0, y0, z0).
        plane_coeffs (array-like): Coefficients of the plane (a, b, c, d) for ax + by + cz + d = 0.

    Returns:
        numpy.ndarray: The projected point's coordinates on the plane.
    """
    # Convert inputs to numpy arrays
    point = np.array(point)
    plane_coeffs = np.array(plane_coeffs)
    
    # Extract the plane normal vector and constant term
    normal = plane_coeffs[:3]  # [a, b, c]
    d = plane_coeffs[3]
    
    # Normalize the plane normal vector
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude == 0:
        raise ValueError("Invalid plane coefficients: normal vector cannot have zero magnitude.")
    normal /= normal_magnitude
    
    # Calculate the signed distance from the point to the plane
    distance = np.dot(normal, point) + d / normal_magnitude
    
    # Project the point onto the plane
    projected_point = point - distance * normal
    
    return projected_point

