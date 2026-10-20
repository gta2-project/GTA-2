"""ROS 1 camera backend."""

import base64
import math
import os
import struct
import sys
from ctypes import POINTER, c_float, c_uint32, cast, pointer
from datetime import datetime
from typing import Any, Optional, Union
from config import Config
from .base import BaseCamera
import cv2
import numpy as np
import open3d as o3d


ROS_TOPIC_TIMEOUT_S = float(os.getenv("CAMERA_TOPIC_TIMEOUT", "15.0"))


class CameraROS1(BaseCamera):
    """
    This class represents a camera object for capturing RGB and pointcloud data.
    """

    # data members
    __rospy         = None
    __ImageMsg      = None
    __CompressedImageMsg = None
    __PointCloud2Msg = None
    __camera_node   :str = None
    __camera_model   :str = None # 'd405'

    def __init__(self, camera_node=Config.CAMERA_NODE,
                 camera_model=Config.CAMERA_MODEL,
                 rosmaster_ip=Config.ROSMASTER_IP,
                 rosmaster_port=Config.ROSMASTER_PORT,
                 camera_backend='ros1',
                 camera_image_path=Config.CAMERA_IMAGE_PATH,
                 debug=False):
        self.debug = debug
        self.camera_backend = (camera_backend or 'ros1').lower()
        self.camera_image_path = camera_image_path
        self._last_rgb_image = None

        if self.camera_backend != 'ros1':
            raise ValueError(f"Unsupported camera backend: {self.camera_backend}")

        self.__camera_model = camera_model
        self.__camera_node = camera_node
        self.__topic_prefix = camera_node
        # remove 'realsense2_camera' from the node name
        if self.__camera_model == 'd405' and '/' in self.__topic_prefix:
            self.__topic_prefix = self.__topic_prefix[:self.__topic_prefix.rfind('/')]
        self.args = self.__get_default_args()

        self.__init_rospy(rosmaster_ip, rosmaster_port)
        self.__assert_image_topic()

    def __del__(self):
        pass

    def __init_rospy(self, rosmaster_ip: str, rosmaster_port: int):
        """Initialize a normal rospy node and assert the ROS master is reachable."""
        if not os.environ.get("ROS_MASTER_URI"):
            os.environ["ROS_MASTER_URI"] = f"http://{rosmaster_ip}:{rosmaster_port}"

        try:
            import rospy
            from sensor_msgs.msg import CompressedImage, Image, PointCloud2
        except ImportError as exc:
            raise RuntimeError(
                "CAMERA_BACKEND=ros1 requires rospy and sensor_msgs. "
                "Source your ROS environment before starting the pipeline."
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node("gta_camera", anonymous=True, disable_signals=True)

        self.__rospy = rospy
        self.__ImageMsg = Image
        self.__CompressedImageMsg = CompressedImage
        self.__PointCloud2Msg = PointCloud2

        master_uri = os.environ.get("ROS_MASTER_URI")
        try:
            rospy.get_master().getSystemState()
        except Exception as exc:
            raise RuntimeError(
                f"Could not contact ROS master at {master_uri}. "
                "Start roscore or set ROS_MASTER_URI/ROSMASTER_IP/ROSMASTER_PORT correctly."
            ) from exc

        print(f"[Camera] ROS mode using rospy ({master_uri})")

    def __full_topic(self, topic_name: str) -> str:
        return f"{self.__topic_prefix}{topic_name}"

    def __wait_for_message(self, topic_name: str, msg_type):
        try:
            return self.__rospy.wait_for_message(
                topic_name,
                msg_type,
                timeout=ROS_TOPIC_TIMEOUT_S,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Timed out waiting for ROS topic {topic_name} "
                f"after {ROS_TOPIC_TIMEOUT_S:.1f}s"
            ) from exc

    def __assert_image_topic(self):
        topic_name = self.__full_topic(self.args.topic_color_image)
        print(f"[Camera] Checking ROS image topic {topic_name}")
        image_msg = self.__wait_for_message(topic_name, self.__ImageMsg)
        self._last_rgb_image = self.__image_msg_to_rgb(image_msg)

    def get_rgb_image(self, use_new_frame: bool=True) -> Union[Optional[np.array], Any]:
        """
        Captures and returns an RGB image from the camera.

        Args:
            use_new_frame (bool): If True, captures a new frame. If False, uses the last saved image.

        Returns:
            color_image (numpy.ndarray): RGB image captured from the camera.
        """
        if not use_new_frame and self._last_rgb_image is not None:
            return self._last_rgb_image.copy()

        color_image = self.__get_image_from_rostopic(self.args.topic_color_image)
        color_image = np.array(color_image[:, :, 0:3])
        self._last_rgb_image = color_image.copy()

        return color_image

    def get_point_cloud(self,  save_points: bool = False) -> o3d.geometry.PointCloud:
        """
        Captures and returns a point cloud from the camera.

        Returns:
            point_cloud (o3d.geometry.PointCloud): Point cloud captured from the camera.
        """
        point_cloud = self.__get_pointcloud_from_rostopic(self.args.topic_points)

        if (save_points):
            now = datetime.now().strftime("%m%d_%H%M")
            points_filename = f"points_{self.__camera_node[1:5]}_{now}.ply"
            o3d.io.write_point_cloud(points_filename, point_cloud)

        return point_cloud

    def __get_image_from_rostopic(self, topic_name: str):
        """
        Get image from ROS topic.
        Based on cv_bridge implementation
        https://github.com/ros-perception/vision_opencv/blob/a34a0c261984e4ab71f267d093f6a48820801d80/cv_bridge/python/cv_bridge/core.py#L147
        """
        full_topic_name = self.__full_topic(topic_name)
        print(f"Getting image from ROS topic {full_topic_name}")
        image_msg = self.__wait_for_message(full_topic_name, self.__ImageMsg)
        return self.__image_msg_to_rgb(image_msg)

    def __image_msg_to_rgb(self, img_msg):
        def encoding_to_dtype_with_channels(encoding):
            if encoding == 'mono8':
                return np.uint8, 1
            elif encoding == 'bgr8':
                return np.uint8, 3
            elif encoding == 'rgb8':
                return np.uint8, 3
            elif encoding == 'mono16':
                return np.uint16, 1
            elif encoding == 'rgba8':
                return np.uint8, 4
            elif encoding == 'bgra8':
                return np.uint8, 4
            elif encoding == '32FC1':
                return np.float32, 1
            elif encoding == '32FC2':
                return np.float32, 2
            elif encoding == '32FC3':
                return np.float32, 3
            elif encoding == '32FC4':
                return np.float32, 4
            else:
                raise TypeError(f'Unsupported encoding: {encoding}')

        encoding = img_msg.encoding
        dtype, n_channels = encoding_to_dtype_with_channels(encoding)
        dtype = np.dtype(dtype).newbyteorder('>' if img_msg.is_bigendian else '<')
        img_buf = np.frombuffer(img_msg.data, dtype=dtype)

        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, int(img_msg.step / dtype.itemsize)),
                            dtype=dtype, buffer=img_buf)
            im = np.ascontiguousarray(im[:img_msg.height, :img_msg.width])
        else:
            im = np.ndarray(shape=(img_msg.height, int(img_msg.step / dtype.itemsize / n_channels), n_channels),
                            dtype=dtype, buffer=img_buf)
            im = np.ascontiguousarray(im[:img_msg.height, :img_msg.width, :])

        # If the byte order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()

        if encoding == 'bgr8':
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if encoding == 'bgra8':
            return cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
        if encoding == 'rgba8':
            return cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        if n_channels == 1:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        return im[:, :, :3]

    def __get_compressedimage_from_rostopic(self, topic_name: str):
        """
        Get image from ROS topic.
        Based on cv_bridge implementation
        https://github.com/ros-perception/vision_opencv/blob/a34a0c261984e4ab71f267d093f6a48820801d80/cv_bridge/python/cv_bridge/core.py#L147
        """

        full_topic_name = self.__full_topic(topic_name)
        print(f"Getting compressed image from ROS topic {full_topic_name}")
        img_msg = self.__wait_for_message(full_topic_name, self.__CompressedImageMsg)
        img_buf = np.frombuffer(img_msg.data, dtype=np.uint8)
        image = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not decode compressed image from {full_topic_name}")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __get_pointcloud_from_rostopic(self, topic_name: str, return_numpy: bool=True) -> Union[o3d.geometry.PointCloud, np.ndarray]:
        """
        Get pointcloud from ROS topic.
        https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py
        https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
        """
        convert_rgbUint32_to_tuple = lambda rgb_uint32: (
            (rgb_uint32 & 0x00FF0000) >> 16,
            (rgb_uint32 & 0x0000FF00) >> 8,
            (rgb_uint32 & 0x000000FF),
        )
        convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
            int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
        )

        full_topic_name = self.__full_topic(topic_name)
        print(f"Getting point cloud from ROS topic {full_topic_name}")
        pcd = self.__wait_for_message(full_topic_name, self.__PointCloud2Msg)

        # Get cloud data from ros_cloud
        field_names = [self.__field_value(field, "name") for field in pcd.fields]
        cloud_data = list(
            self.__read_points(pcd, skip_nans=True, field_names=field_names)
        )

        # Check empty
        open3d_cloud = o3d.geometry.PointCloud()

        if len(cloud_data) == 0:
            print("Converting an empty cloud")
            return None

        # Set open3d_cloud
        if "rgb" in field_names:
            IDX_RGB_IN_FIELD = 3  # x, y, z, rgb

            # Get xyz
            xyz = [
                (x, y, z) for x, y, z, rgb in cloud_data
            ]  # (why cannot put this line below rgb?)

            # Get rgb
            # Check whether int or float
            if (
                type(cloud_data[0][IDX_RGB_IN_FIELD]) == float
            ):  # if float (from pcl::toROSMsg)
                rgb = [convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data]
            else:
                rgb = [convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data]

            # combine
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz) * 1000)
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb) / 255.0)
        else:
            xyz = [(x, y, z) for x, y, z in cloud_data]  # get xyz
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

        # return
        if return_numpy:
            return np.asarray(open3d_cloud.points)
        else:
            return open3d_cloud

    def __read_points(self, cloud, field_names=None, skip_nans=False, uvs=[]):
        """
        Read points from a L{sensor_msgs.PointCloud2} message.
        Implementation based on code from:
        https://github.com/ros/common_msgs/blob/20a833b56f9d7fd39655b8491a2ec1226d2639b3/sensor_msgs/src/sensor_msgs/point_cloud2.py#L61

        @param cloud: The point cloud to read from.
        @param field_names: The names of fields to read. If None, read all fields. [default: None]
        @type  field_names: iterable
        @param skip_nans: If True, then don't return any point with a NaN value.
        @type  skip_nans: bool [default: False]
        @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
        @type  uvs: iterable
        @return: Generator which yields a list of values for each point.
        @rtype:  generator
        """
        _DATATYPES = {}

        _DATATYPES[1] = ('b', 1)  # _DATATYPES[PointField.INT8]    = ('b', 1)
        _DATATYPES[2] = ('B', 1)  # _DATATYPES[PointField.UINT8]   = ('B', 1)
        _DATATYPES[3] = ('h', 2)  # _DATATYPES[PointField.INT16]   = ('h', 2)
        _DATATYPES[4] = ('H', 2)  # _DATATYPES[PointField.UINT16]  = ('H', 2)
        _DATATYPES[5] = ('i', 4)  # _DATATYPES[PointField.INT32]   = ('i', 4)
        _DATATYPES[6] = ('I', 4)  # _DATATYPES[PointField.UINT32]  = ('I', 4)
        _DATATYPES[7] = ('f', 4)  # _DATATYPES[PointField.FLOAT32] = ('f', 4)
        _DATATYPES[8] = ('d', 8)  # _DATATYPES[PointField.FLOAT64] = ('d', 8)

        def get_struct_fmt(is_bigendian, fields, field_names=None):
            fmt = ">" if is_bigendian else "<"

            offset = 0
            for field in (
                f
                for f in sorted(fields, key=lambda f: self.__field_value(f, "offset"))
                if field_names is None or self.__field_value(f, "name") in field_names
            ):
                field_offset = self.__field_value(field, "offset")
                field_datatype = self.__field_value(field, "datatype")
                field_count = self.__field_value(field, "count")

                if offset < field_offset:
                    fmt += "x" * (field_offset - offset)
                    offset = field_offset
                if field_datatype not in _DATATYPES:
                    print(
                        "Skipping unknown PointField datatype [%d]" % field_datatype,
                        file=sys.stderr,
                    )
                else:
                    datatype_fmt, datatype_length = _DATATYPES[field_datatype]
                    fmt += field_count * datatype_fmt
                    offset += field_count * datatype_length

            return fmt

        fmt = get_struct_fmt(self.__msg_value(cloud, "is_bigendian"), self.__msg_value(cloud, "fields"), field_names)
        width, height, point_step, row_step, data, isnan = (
            self.__msg_value(cloud, "width"),
            self.__msg_value(cloud, "height"),
            self.__msg_value(cloud, "point_step"),
            self.__msg_value(cloud, "row_step"),
            self.__msg_value(cloud, "data"),
            math.isnan,
        )
        if isinstance(data, str):
            data = base64.b64decode(data)
        data = np.frombuffer(data, dtype=np.uint8)
        unpack_from = struct.Struct(fmt).unpack_from

        if skip_nans:
            if uvs:
                for u, v in uvs:
                    p = unpack_from(data, (row_step * v) + (point_step * u))
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
            else:
                for v in range(height):
                    offset = row_step * v
                    for u in range(width):
                        p = unpack_from(data, offset)
                        has_nan = False
                        for pv in p:
                            if isnan(pv):
                                has_nan = True
                                break
                        if not has_nan:
                            yield p
                        offset += point_step
        else:
            if uvs:
                for u, v in uvs:
                    yield unpack_from(data, (row_step * v) + (point_step * u))
            else:
                for v in range(height):
                    offset = row_step * v
                    for u in range(width):
                        yield unpack_from(data, offset)
                        offset += point_step

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
