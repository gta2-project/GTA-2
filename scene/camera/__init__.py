"""Camera entry point: select a backend without importing unused ROS clients."""

from config import Config
from .base import BaseCamera


class Camera:
    """Create the configured camera backend through the shared interface.

    Preserve the existing instance policy: file/ROS 1 share the cached camera
    here, while ROS 2 owns its own singleton. Repeated construction does not
    reconfigure an existing camera.
    """

    _instance = None

    def __new__(
        cls,
        camera_node=Config.CAMERA_NODE,
        camera_model=Config.CAMERA_MODEL,
        rosmaster_ip=Config.ROSMASTER_IP,
        rosmaster_port=Config.ROSMASTER_PORT,
        camera_backend=Config.CAMERA_BACKEND,
        camera_image_path=Config.CAMERA_IMAGE_PATH,
        debug=False,
    ) -> BaseCamera:
        requested_backend = (camera_backend or "ros1").lower()
        if requested_backend not in {"file", "ros1", "ros2"}:
            raise ValueError(f"Unsupported camera backend: {requested_backend}")
        settings = dict(
            camera_node=camera_node,
            camera_model=camera_model,
            rosmaster_ip=rosmaster_ip,
            rosmaster_port=rosmaster_port,
            camera_backend=requested_backend,
            camera_image_path=camera_image_path,
            debug=debug,
        )

        if requested_backend == "ros2":
            from .ros2 import CameraROS2

            return CameraROS2(**settings)

        if cls._instance is not None:
            return cls._instance

        if requested_backend == "file":
            from .file import CameraFile

            backend = CameraFile
        elif requested_backend == "ros1":
            from .ros1 import CameraROS1

            backend = CameraROS1
        # Publish only an initialized instance; failed construction stays retryable.
        cls._instance = backend(**settings)
        return cls._instance
