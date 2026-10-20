"""Camera acquisition helpers for generated robot scripts."""

from config import Config
from scene.camera import Camera


def get_point_cloud_data():
    """
    Simulate retrieval of point cloud data from the robot's sensors.
    
    Returns:
        A numpy array of shape (N, 3) representing N points in 3D space.
    """
    print("[Robot Utils] Retrieving point cloud data from sensors")
    cam = Camera(
                camera_node=Config.CAMERA_NODE,
                camera_model=Config.CAMERA_MODEL,
                rosmaster_ip=Config.ROSMASTER_IP,
                rosmaster_port=Config.ROSMASTER_PORT,
                camera_backend=Config.CAMERA_BACKEND,
                camera_image_path=Config.CAMERA_IMAGE_PATH
            )
    pcd = cam.get_point_cloud()
    return pcd


def get_rgb_image():
    """
    Simulate retrieval of an RGB image from the robot's camera.
    
    Returns:
        A numpy array representing the RGB image.
    """
    print("[Robot Utils] Retrieving RGB image from camera")
    cam = Camera(
                camera_node=Config.CAMERA_NODE,
                camera_model=Config.CAMERA_MODEL,
                rosmaster_ip=Config.ROSMASTER_IP,
                rosmaster_port=Config.ROSMASTER_PORT,
                camera_backend=Config.CAMERA_BACKEND,
                camera_image_path=Config.CAMERA_IMAGE_PATH
            )
    rgb_image = cam.get_rgb_image()
    return rgb_image
