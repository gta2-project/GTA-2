"""Image-file camera backend; no ROS or point-cloud connection."""

import os

import cv2

from config import Config
from .base import BaseCamera


class CameraFile(BaseCamera):
    """Read the configured image file, using the existing RGB frame cache."""

    def __init__(
        self,
        camera_node=Config.CAMERA_NODE,
        camera_model=Config.CAMERA_MODEL,
        rosmaster_ip=Config.ROSMASTER_IP,
        rosmaster_port=Config.ROSMASTER_PORT,
        camera_backend="file",
        camera_image_path=Config.CAMERA_IMAGE_PATH,
        debug=False,
    ):
        self.debug = debug
        self.camera_backend = (camera_backend or "file").lower()
        if self.camera_backend != "file":
            raise ValueError(f"Unsupported camera backend: {self.camera_backend}")
        self.camera_image_path = camera_image_path
        self._last_rgb_image = None
        self.__camera_model = camera_model
        self.__camera_node = camera_node
        self.__topic_prefix = camera_node
        self.args = self.__get_default_args()
        print(f"[Camera] Image file mode: {camera_image_path}")

    def get_rgb_image(self, use_new_frame: bool = True):
        """Read a fresh RGB image, or return a copy of the cached frame."""
        if not use_new_frame and self._last_rgb_image is not None:
            return self._last_rgb_image.copy()

        color_image = self.__get_image_from_file()
        self._last_rgb_image = color_image.copy()
        return color_image

    def get_point_cloud(self, save_points: bool = False):
        raise RuntimeError(
            "File camera mode only provides RGB images. Point clouds require CAMERA_BACKEND=ros1 or ros2."
        )

    def __get_image_from_file(self):
        """Read the configured image file as an RGB numpy array."""
        if not os.path.exists(self.camera_image_path):
            raise FileNotFoundError(f"Camera image file not found: {self.camera_image_path}")

        image = cv2.imread(self.camera_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read camera image file: {self.camera_image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __get_default_args(self):
        class args:
            # realsense defaults
            topic_color_info = '/color/camera_info'
            topic_depth_info = '/depth/camera_info'
            topic_infra_info = '/infra1/camera_info'
            topic_color_image = '/color/image_raw'
            topic_depth_image = '/depth/image_rect_raw'
            topic_stereo_left = '/infra1/image_rect_raw'
            topic_stereo_right = '/infra2/image_rect_raw'
            topic_stereo_info = topic_infra_info
            topic_points = '/transform_igev_points'

        if self.__camera_model == 'd405':
            args.camera_seperation = 0.018
        if self.__camera_model == 'zed2i':
            args.camera_seperation = 0.120
            args.topic_color_info = '/rgb_rect/left/camera_info'
            args.topic_color_image = '/rgb_rect/left/image'
            args.topic_stereo_left = '/rgb_rect/left/image'
            args.topic_stereo_right = '/rgb_rect/right/image'
            args.topic_stereo_info = '/rgb_rect/left/camera_info'
            args.topic_points = '/transform_igev_points'

        return args
