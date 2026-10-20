"""ROS 2 camera interface matching :mod:`scene.camera`."""

import base64
import math
import os
import struct
import sys
import threading
from ctypes import POINTER, c_float, c_uint32, cast, pointer
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import numpy as np
import open3d as o3d

from config import Config
from .base import BaseCamera


ROS2_TOPIC_TIMEOUT_S = float(os.getenv("CAMERA_TOPIC_TIMEOUT", "30.0"))


class CameraROS2(BaseCamera):
    """Synchronous RGB and point-cloud access backed by ROS 2 subscriptions.

    The public constructor and methods intentionally match ``scene.camera.Camera``
    so existing agents, utilities, and generated scripts can use either backend
    solely through ``CAMERA_BACKEND``.
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
    ):
        if cls._instance is not None:
            return cls._instance

        instance = super().__new__(cls)
        cls._instance = instance

        instance.debug = debug
        instance.camera_backend = (camera_backend or "ros2").lower()
        instance.camera_image_path = camera_image_path
        instance._last_rgb_image = None
        instance.__camera_model = camera_model
        instance.__camera_node = camera_node
        instance.__topic_prefix = camera_node
        instance.__absolute_topics = {
            topic
            for topic in (
                Config.CAMERA_COLOR_TOPIC,
                Config.CAMERA_POINT_CLOUD_TOPIC,
            )
            if topic
        }
        instance.__node = None
        instance.__spin_lock = threading.RLock()
        instance.__current_msg = {}
        instance.__subscriptions = {}

        # These ROS 1 connection settings remain in the signature for drop-in API
        # compatibility. ROS 2 discovery is configured through its normal
        # environment variables, including ROS_DOMAIN_ID.
        del rosmaster_ip, rosmaster_port

        if instance.camera_backend != "ros2":
            cls._instance = None
            raise ValueError(
                f"CameraROS2 requires camera_backend='ros2', got {instance.camera_backend!r}"
            )

        # Match the ROS 1 topic-prefix behavior for RealSense cameras.
        if instance.__camera_model == "d405" and "/" in instance.__topic_prefix:
            instance.__topic_prefix = instance.__topic_prefix[
                : instance.__topic_prefix.rfind("/")
            ]
        instance.args = instance.__get_default_args()

        try:
            instance.__init_ros2()
            instance.__assert_image_topic()
        except Exception:
            instance.close()
            cls._instance = None
            raise

        return instance

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Destroy this class's ROS 2 node without shutting down shared rclpy."""
        node = getattr(self, "_CameraROS2__node", None)
        if node is not None:
            try:
                subscriptions = getattr(
                    self,
                    "_CameraROS2__subscriptions",
                    {},
                )
                for subscription, _topic_name, _msg_type in subscriptions.values():
                    try:
                        node.destroy_subscription(subscription)
                    except Exception:
                        pass
                subscriptions.clear()
                node.destroy_node()
            finally:
                self.__node = None
                self.__current_msg.clear()
        if CameraROS2._instance is self:
            CameraROS2._instance = None

    def __init_ros2(self):
        try:
            import rclpy
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CompressedImage, Image, PointCloud2
        except ImportError as exc:
            raise RuntimeError(
                "CAMERA_BACKEND=ros2 requires rclpy and ROS 2 sensor_msgs. "
                "Source your ROS 2 environment before starting the pipeline."
            ) from exc

        if not rclpy.ok():
            init_kwargs = {"args": None}
            try:
                from rclpy.signals import SignalHandlerOptions

                init_kwargs["signal_handler_options"] = SignalHandlerOptions.NO
            except (ImportError, AttributeError):
                pass

            try:
                rclpy.init(**init_kwargs)
            except TypeError:
                # Older ROS 2 distributions do not expose signal_handler_options.
                rclpy.init(args=None)

        self.__rclpy = rclpy
        self.__qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.__ImageMsg = Image
        self.__CompressedImageMsg = CompressedImage
        self.__PointCloud2Msg = PointCloud2

        try:
            self.__node = rclpy.create_node("gta_camera_ros2")
        except Exception as exc:
            raise RuntimeError(
                "Could not create the GTA ROS 2 camera node. "
                "Check that the ROS 2 environment is sourced and the selected "
                "ROS_DOMAIN_ID is correct."
            ) from exc

        domain_id = os.getenv("ROS_DOMAIN_ID", "default")
        print(f"[CameraROS2] ROS 2 mode using rclpy (domain {domain_id})")

    def __full_topic(self, topic_name: str) -> str:
        if topic_name in self.__absolute_topics:
            return topic_name
        return f"{self.__topic_prefix}{topic_name}"

    def __wait_for_message(
        self,
        key: str,
        topic_name: str,
        msg_type,
        timeout: float = ROS2_TOPIC_TIMEOUT_S,
    ):
        """Wait for a fresh cached message while actively spinning the camera node."""
        with self.__spin_lock:
            existing = self.__subscriptions.get(key)
            if existing is None or existing[1:] != (topic_name, msg_type):
                if existing is not None:
                    self.__node.destroy_subscription(existing[0])
                subscription = self.__node.create_subscription(
                    msg_type,
                    topic_name,
                    lambda incoming, message_key=key: self.__current_msg.__setitem__(
                        message_key,
                        incoming,
                    ),
                    self.__qos_profile,
                )
                self.__subscriptions[key] = (subscription, topic_name, msg_type)

            # Clear the cached value so every call waits for a newly delivered frame.
            self.__current_msg[key] = None
            start = self.__node.get_clock().now()
            while self.__current_msg[key] is None:
                if not self.__rclpy.ok():
                    raise RuntimeError(
                        f"ROS 2 shut down while waiting for topic {topic_name}"
                    )
                self.__rclpy.spin_once(self.__node, timeout_sec=0.1)
                if self.__current_msg[key] is not None:
                    break
                elapsed = (
                    self.__node.get_clock().now() - start
                ).nanoseconds / 1e9
                if elapsed > timeout:
                    count_publishers = getattr(
                        self.__node,
                        "count_publishers",
                        None,
                    )
                    publisher_count = (
                        count_publishers(topic_name)
                        if callable(count_publishers)
                        else None
                    )
                    if publisher_count == 0:
                        detail = "no publishers were discovered"
                    elif publisher_count is None:
                        detail = "publisher availability could not be determined"
                    else:
                        detail = (
                            f"{publisher_count} publisher(s) were discovered, "
                            "but none delivered a message"
                        )
                    raise TimeoutError(
                        f"Timed out waiting for ROS 2 topic {topic_name} "
                        f"after {timeout:.1f}s; {detail}"
                    )

            return self.__current_msg[key]

    def __assert_image_topic(self):
        topic_name = self.__full_topic(self.args.topic_color_image)
        print(f"[CameraROS2] Checking ROS 2 image topic {topic_name}")
        image_msg = self.__wait_for_message("rgb", topic_name, self.__ImageMsg)
        self._last_rgb_image = self.__image_msg_to_rgb(image_msg)

    def get_rgb_image(
        self, use_new_frame: bool = True
    ) -> Union[Optional[np.ndarray], Any]:
        """Capture and return an RGB image, matching ``Camera.get_rgb_image``."""
        if not use_new_frame and self._last_rgb_image is not None:
            return self._last_rgb_image.copy()

        color_image = self.__get_image_from_rostopic(self.args.topic_color_image)
        color_image = np.array(color_image[:, :, 0:3])
        self._last_rgb_image = color_image.copy()
        return color_image

    def get_point_cloud(
        self, save_points: bool = False
    ) -> Union[o3d.geometry.PointCloud, np.ndarray]:
        """Capture and return the point cloud, matching ``Camera.get_point_cloud``."""
        point_cloud = self.__get_pointcloud_from_rostopic(self.args.topic_points)

        if save_points:
            now = datetime.now().strftime("%m%d_%H%M")
            camera_label = self.__camera_node.strip("/")[:4] or "ros2"
            points_filename = f"points_{camera_label}_{now}.ply"
            cloud_to_save = point_cloud
            if isinstance(point_cloud, np.ndarray):
                cloud_to_save = o3d.geometry.PointCloud()
                cloud_to_save.points = o3d.utility.Vector3dVector(point_cloud)
            o3d.io.write_point_cloud(points_filename, cloud_to_save)

        return point_cloud

    def __get_image_from_rostopic(self, topic_name: str):
        full_topic_name = self.__full_topic(topic_name)
        print(f"Getting image from ROS 2 topic {full_topic_name}")
        image_msg = self.__wait_for_message(
            "rgb",
            full_topic_name,
            self.__ImageMsg,
        )
        return self.__image_msg_to_rgb(image_msg)

    def __image_msg_to_rgb(self, img_msg):
        def encoding_to_dtype_with_channels(encoding):
            if encoding == "mono8":
                return np.uint8, 1
            if encoding == "bgr8":
                return np.uint8, 3
            if encoding == "rgb8":
                return np.uint8, 3
            if encoding == "mono16":
                return np.uint16, 1
            if encoding == "rgba8":
                return np.uint8, 4
            if encoding == "bgra8":
                return np.uint8, 4
            if encoding == "32FC1":
                return np.float32, 1
            if encoding == "32FC2":
                return np.float32, 2
            if encoding == "32FC3":
                return np.float32, 3
            if encoding == "32FC4":
                return np.float32, 4
            raise TypeError(f"Unsupported encoding: {encoding}")

        encoding = img_msg.encoding
        dtype, n_channels = encoding_to_dtype_with_channels(encoding)
        dtype = np.dtype(dtype).newbyteorder(">" if img_msg.is_bigendian else "<")
        img_buf = np.frombuffer(img_msg.data, dtype=dtype)

        if n_channels == 1:
            image = np.ndarray(
                shape=(img_msg.height, int(img_msg.step / dtype.itemsize)),
                dtype=dtype,
                buffer=img_buf,
            )
            image = np.ascontiguousarray(image[: img_msg.height, : img_msg.width])
        else:
            image = np.ndarray(
                shape=(
                    img_msg.height,
                    int(img_msg.step / dtype.itemsize / n_channels),
                    n_channels,
                ),
                dtype=dtype,
                buffer=img_buf,
            )
            image = np.ascontiguousarray(
                image[: img_msg.height, : img_msg.width, :]
            )

        if img_msg.is_bigendian == (sys.byteorder == "little"):
            image = image.byteswap().newbyteorder()

        if encoding == "bgr8":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if n_channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image[:, :, :3]

    def __get_compressedimage_from_rostopic(self, topic_name: str):
        full_topic_name = self.__full_topic(topic_name)
        print(f"Getting compressed image from ROS 2 topic {full_topic_name}")
        img_msg = self.__wait_for_message(
            "compressed_rgb",
            full_topic_name,
            self.__CompressedImageMsg,
        )
        img_buf = np.frombuffer(img_msg.data, dtype=np.uint8)
        image = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(
                f"Could not decode compressed image from {full_topic_name}"
            )
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __get_pointcloud_from_rostopic(
        self, topic_name: str, return_numpy: bool = True
    ) -> Union[o3d.geometry.PointCloud, np.ndarray]:
        convert_rgb_uint32_to_tuple = lambda rgb_uint32: (
            (rgb_uint32 & 0x00FF0000) >> 16,
            (rgb_uint32 & 0x0000FF00) >> 8,
            rgb_uint32 & 0x000000FF,
        )
        convert_rgb_float_to_tuple = lambda rgb_float: convert_rgb_uint32_to_tuple(
            int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
        )

        full_topic_name = self.__full_topic(topic_name)
        print(f"Getting point cloud from ROS 2 topic {full_topic_name}")
        point_cloud_message = self.__wait_for_message(
            "point_cloud",
            full_topic_name,
            self.__PointCloud2Msg,
        )

        field_names = [
            self.__field_value(field, "name")
            for field in point_cloud_message.fields
        ]
        cloud_data = list(
            self.__read_points(
                point_cloud_message,
                skip_nans=True,
                field_names=field_names,
            )
        )

        open3d_cloud = o3d.geometry.PointCloud()
        if not cloud_data:
            print("Converting an empty cloud")
            return None

        if "rgb" in field_names:
            rgb_index = 3
            xyz = [(x, y, z) for x, y, z, rgb in cloud_data]
            if type(cloud_data[0][rgb_index]) is float:
                rgb = [
                    convert_rgb_float_to_tuple(rgb_value)
                    for x, y, z, rgb_value in cloud_data
                ]
            else:
                rgb = [
                    convert_rgb_uint32_to_tuple(rgb_value)
                    for x, y, z, rgb_value in cloud_data
                ]
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz) * 1000)
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb) / 255.0)
        else:
            xyz = [(x, y, z) for x, y, z in cloud_data]
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

        if return_numpy:
            return np.asarray(open3d_cloud.points)
        return open3d_cloud

    def __read_points(self, cloud, field_names=None, skip_nans=False, uvs=None):
        if uvs is None:
            uvs = []

        datatypes = {
            1: ("b", 1),
            2: ("B", 1),
            3: ("h", 2),
            4: ("H", 2),
            5: ("i", 4),
            6: ("I", 4),
            7: ("f", 4),
            8: ("d", 8),
        }

        def get_struct_fmt(is_bigendian, fields, selected_field_names=None):
            fmt = ">" if is_bigendian else "<"
            offset = 0
            for field in (
                item
                for item in sorted(
                    fields, key=lambda item: self.__field_value(item, "offset")
                )
                if selected_field_names is None
                or self.__field_value(item, "name") in selected_field_names
            ):
                field_offset = self.__field_value(field, "offset")
                field_datatype = self.__field_value(field, "datatype")
                field_count = self.__field_value(field, "count")

                if offset < field_offset:
                    fmt += "x" * (field_offset - offset)
                    offset = field_offset
                if field_datatype not in datatypes:
                    print(
                        "Skipping unknown PointField datatype [%d]"
                        % field_datatype,
                        file=sys.stderr,
                    )
                else:
                    datatype_fmt, datatype_length = datatypes[field_datatype]
                    fmt += field_count * datatype_fmt
                    offset += field_count * datatype_length
            return fmt

        fmt = get_struct_fmt(
            self.__msg_value(cloud, "is_bigendian"),
            self.__msg_value(cloud, "fields"),
            field_names,
        )
        width = self.__msg_value(cloud, "width")
        height = self.__msg_value(cloud, "height")
        point_step = self.__msg_value(cloud, "point_step")
        row_step = self.__msg_value(cloud, "row_step")
        data = self.__msg_value(cloud, "data")
        if isinstance(data, str):
            data = base64.b64decode(data)
        data = np.frombuffer(data, dtype=np.uint8)
        unpack_from = struct.Struct(fmt).unpack_from

        if uvs:
            offsets = [
                (row_step * vertical) + (point_step * horizontal)
                for horizontal, vertical in uvs
            ]
        else:
            offsets = [
                (row_step * vertical) + (point_step * horizontal)
                for vertical in range(height)
                for horizontal in range(width)
            ]

        for offset in offsets:
            point = unpack_from(data, offset)
            if skip_nans and any(
                isinstance(value, float) and math.isnan(value) for value in point
            ):
                continue
            yield point

    @staticmethod
    def __msg_value(message, name):
        if isinstance(message, dict):
            return message[name]
        return getattr(message, name)

    @staticmethod
    def __field_value(field, name):
        if isinstance(field, dict):
            return field[name]
        return getattr(field, name)

    def __get_default_args(self):
        class args:
            topic_color_info = "/color/camera_info"
            topic_depth_info = "/depth/camera_info"
            topic_infra_info = "/infra1/camera_info"
            topic_color_image = "/color/image_raw"
            topic_depth_image = "/depth/image_rect_raw"
            topic_stereo_left = "/infra1/image_rect_raw"
            topic_stereo_right = "/infra2/image_rect_raw"
            topic_stereo_info = topic_infra_info
            topic_points = "/transform_igev_points"

        if self.__camera_model == "d405":
            args.camera_seperation = 0.018
        if self.__camera_model == "zed2i":
            args.camera_seperation = 0.120
            args.topic_color_info = "/rgb_rect/left/camera_info"
            args.topic_color_image = "/rgb_rect/left/image"
            args.topic_stereo_left = "/rgb_rect/left/image"
            args.topic_stereo_right = "/rgb_rect/right/image"
            args.topic_stereo_info = "/rgb_rect/left/camera_info"
            args.topic_points = "/transform_igev_points"

        if Config.CAMERA_COLOR_TOPIC:
            args.topic_color_image = Config.CAMERA_COLOR_TOPIC
        if Config.CAMERA_POINT_CLOUD_TOPIC:
            args.topic_points = Config.CAMERA_POINT_CLOUD_TOPIC

        return args


if __name__ == "__main__":
    camera = CameraROS2(camera_backend="ros2")
    print(camera.get_rgb_image().shape)
    print(camera.get_point_cloud().shape)
