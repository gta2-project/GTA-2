"""Coordinate transforms, keypoint projection, and geometric helpers."""

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R

from config import Config


def get_transformation_matrix_from_pose(pose):
    """
    Get the transformation matrix from a 6D pose vector x,y,z,rx,ry,rz.
    """
    T = np.eye(4)
    rotation = R.from_rotvec(pose[3:]).as_matrix()
    T[:3, :3] = rotation
    T[:3, 3] = pose[:3]
    return T


def get_pose_from_transformation_matrix(T):
    """
    Convert a 4x4 transformation matrix into a 6D pose vector of x,y,z,rx,ry,rz.
    """
    position = T[:3, 3]  # Extract translation (x, y, z)
    rotation = R.from_matrix(T[:3, :3])  # Extract rotation matrix
    orientation = rotation.as_rotvec()  # Convert rotation to rotation vector

    return np.concatenate([position, orientation])


def get_3d_position(pcd, keypoint_pixel_coords):
    """
    Generate a random 3D position.
    
    Returns:
        numpy array with shape (3,) containing [x, y, z] coordinates
    """
    print(f"[Robot Utils] Converting pixel coords {keypoint_pixel_coords} to 3D position")

    original_image = Image.open('environment.jpg')
    resized_image = Image.open('resized_img.png')
    resize_w, resize_h = resized_image.size # resized image is padded to be square, so resize_w == resize_h
    original_w, original_h = original_image.size

    keypoint = np.asarray(keypoint_pixel_coords) * original_w // resize_w
    keypoint[1] -= (original_w - original_h) // 2

    pcd_reshaped = pcd.reshape(original_h, original_w, 3)

    pcd_point = pcd_reshaped[keypoint[1], keypoint[0]]

    return pcd_point


def a_surface_normal(g1, pcd):
    """
    Compute surface normal axis at keypoint g1 using point cloud data.
    
    Args:
        g1: 3D position of the keypoint (numpy array with shape (3,))
        pcd: Point cloud data (numpy array with shape (N, 3))
    
    Returns:
        numpy array with shape (3,) representing the surface normal unit vector
    
    Note:
        This is a placeholder implementation that returns a fixed normal vector.
        In a real implementation, this would involve:
        1- Converting to open3d point cloud
        2- Estimating normals using open3d's built-in functions
        3- Finding the normal corresponding to the keypoint g1
    """
    print(f"[Robot Utils] Computing surface normal at keypoint {g1} using point cloud data")

    # 1 - Convert to open3d point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    # 2 - Estimate normals
    pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))
    # 3 - Find the normal corresponding to the keypoint g1
    # We can find the nearest neighbor in the point cloud to g1
    distances = np.linalg.norm(pcd - g1, axis=1)
    nearest_index = np.argmin(distances)

    normal = unit(np.asarray(pcd_o3d.normals)[nearest_index])

    camera_position = np.asarray(Config.T_BASE_CAM)[:3, 3]  # Extract camera position from T_BASE_CAM
    point_to_camera = camera_position - g1
    dot_product = np.sign(np.dot(normal, point_to_camera))
    normal = normal * dot_product  # Flip normal if it's pointing away from the camera

    return normal


def a_crossprod_axes(a1, a2, eps=1e-6):
    """
    Return a unit rotation axis for aligning a1 to a2.
    Always returns a unit vector, even if a1 and a2 are parallel or anti-parallel.
    """
    a1 = unit(a1)
    a2 = unit(a2)

    cross = np.cross(a1, a2)
    if np.linalg.norm(cross) > eps:
        return unit(cross)

    # parallel or anti-parallel: choose any perpendicular axis
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a1, candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])

    return unit(np.cross(a1, candidate))


def a_along_keypoints(g1, g2):
    """
    Compute axis pointing from keypoint g1 to keypoint g2.
    """
    g1, g2 = np.asarray(g1), np.asarray(g2)
    
    direction = g2 - g1
    
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("Keypoints g1 and g2 are at the same location")

    return direction / norm


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def choose_perp_axis(a):
    # pick something not parallel to a
    a = unit(a)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, candidate)
    return unit(u)
