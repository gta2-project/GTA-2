"""Offline tests for the ROS 2 camera backend."""

from __future__ import annotations

import struct
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from scene.camera import Camera
from scene.camera.ros2 import CameraROS2


class FakeImage:
    pass


class FakeCompressedImage:
    pass


class FakePointCloud2:
    pass


class FakeQoSProfile:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTime:
    def __init__(self):
        self.nanoseconds = time.monotonic_ns()

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)


class FakeNode:
    def __init__(self, messages):
        self.messages = {message_type: list(values) for message_type, values in messages.items()}
        self.subscribed_topics = []
        self.subscriptions = []
        self.destroyed_subscriptions = []
        self.destroyed = False

    def create_subscription(self, message_type, topic, callback, qos_profile):
        subscription = SimpleNamespace(
            message_type=message_type,
            topic=topic,
            callback=callback,
            qos_profile=qos_profile,
        )
        self.subscribed_topics.append(topic)
        self.subscriptions.append(subscription)
        return subscription

    def dispatch_one(self):
        for subscription in self.subscriptions:
            queued = self.messages.get(subscription.message_type, [])
            if queued:
                subscription.callback(queued.pop(0))
                return

    def get_clock(self):
        return SimpleNamespace(now=FakeTime)

    def count_publishers(self, _topic):
        return 1

    def destroy_subscription(self, subscription):
        self.destroyed_subscriptions.append(subscription)

    def destroy_node(self):
        self.destroyed = True


def image_message(bgr_pixels):
    pixels = np.asarray(bgr_pixels, dtype=np.uint8)
    message = FakeImage()
    message.encoding = "bgr8"
    message.is_bigendian = False
    message.height = pixels.shape[0]
    message.width = pixels.shape[1]
    message.step = pixels.shape[1] * pixels.shape[2]
    message.data = pixels.tobytes()
    return message


def point_cloud_message():
    message = FakePointCloud2()
    message.fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
    ]
    message.is_bigendian = False
    message.width = 2
    message.height = 1
    message.point_step = 12
    message.row_step = 24
    message.data = struct.pack("<ffffff", 0.1, 0.2, 0.3, float("nan"), 1.0, 2.0)
    return message


def fake_ros2_modules(node):
    state = SimpleNamespace(ok=False, init_kwargs=None, spin_calls=0)

    rclpy = types.ModuleType("rclpy")

    def ok():
        return state.ok

    def init(**kwargs):
        state.ok = True
        state.init_kwargs = kwargs

    def create_node(name):
        state.node_name = name
        return node

    def spin_once(spun_node, timeout_sec):
        state.spin_calls += 1
        state.last_spin = (spun_node, timeout_sec)
        spun_node.dispatch_one()

    rclpy.ok = ok
    rclpy.init = init
    rclpy.create_node = create_node
    rclpy.spin_once = spin_once

    qos = types.ModuleType("rclpy.qos")
    qos.QoSProfile = FakeQoSProfile
    qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    qos.qos_profile_sensor_data = object()

    signals = types.ModuleType("rclpy.signals")
    signals.SignalHandlerOptions = SimpleNamespace(NO=object())

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = FakeImage
    sensor_msgs_msg.CompressedImage = FakeCompressedImage
    sensor_msgs_msg.PointCloud2 = FakePointCloud2
    sensor_msgs.msg = sensor_msgs_msg

    modules = {
        "rclpy": rclpy,
        "rclpy.qos": qos,
        "rclpy.signals": signals,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
    }
    return modules, state


class CameraROS2Tests(unittest.TestCase):
    def tearDown(self):
        instance = CameraROS2._instance
        if instance is not None:
            instance.close()
        CameraROS2._instance = None
        Camera._instance = None

    def test_camera_entry_point_routes_ros2_without_loading_ros1(self):
        sentinel = object()
        with patch("scene.camera.ros2.CameraROS2", return_value=sentinel) as ros2_camera:
            result = Camera(
                camera_node="/zed_node",
                camera_model="zed2i",
                camera_backend="ros2",
            )

        self.assertIs(result, sentinel)
        ros2_camera.assert_called_once()
        self.assertEqual(ros2_camera.call_args.kwargs["camera_backend"], "ros2")
        self.assertIsNone(Camera._instance)

    def test_ros2_backend_matches_rgb_and_point_cloud_api(self):
        initial = image_message([[[1, 2, 3], [4, 5, 6]]])
        refreshed = image_message([[[10, 20, 30], [40, 50, 60]]])
        cloud = point_cloud_message()
        node = FakeNode(
            {
                FakeImage: [initial, refreshed],
                FakePointCloud2: [cloud],
            }
        )
        modules, state = fake_ros2_modules(node)

        with patch.dict(sys.modules, modules):
            camera = CameraROS2(
                camera_node="/zed_node",
                camera_model="zed2i",
                camera_backend="ros2",
            )
            cached_rgb = camera.get_rgb_image(use_new_frame=False)
            refreshed_rgb = camera.get_rgb_image(use_new_frame=True)
            points = camera.get_point_cloud()
            camera.close()

        np.testing.assert_array_equal(
            cached_rgb,
            np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            refreshed_rgb,
            np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8),
        )
        np.testing.assert_allclose(points, [[0.1, 0.2, 0.3]], rtol=1e-6)
        self.assertEqual(
            node.subscribed_topics,
            [
                "/zed_node/rgb_rect/left/image",
                "/zed_node/transform_igev_points",
            ],
        )
        self.assertEqual(state.node_name, "gta_camera_ros2")
        self.assertIsNotNone(state.init_kwargs)
        self.assertGreaterEqual(state.spin_calls, 3)
        self.assertEqual(len(node.destroyed_subscriptions), 2)
        self.assertTrue(node.destroyed)
        self.assertTrue(
            all(
                subscription.qos_profile.depth == 1
                and subscription.qos_profile.reliability == "reliable"
                for subscription in node.subscriptions
            )
        )

    def test_d405_uses_same_prefix_rule_as_ros1_camera(self):
        node = FakeNode(
            {
                FakeImage: [image_message([[[0, 0, 0]]])],
            }
        )
        modules, _ = fake_ros2_modules(node)

        with patch.dict(sys.modules, modules):
            CameraROS2(
                camera_node="/camera/realsense2_camera",
                camera_model="d405",
                camera_backend="ros2",
            )

        self.assertEqual(node.subscribed_topics, ["/camera/color/image_raw"])

    def test_full_ros2_topic_overrides_bypass_camera_prefix(self):
        node = FakeNode(
            {
                FakeImage: [image_message([[[0, 0, 0]]])],
                FakePointCloud2: [point_cloud_message()],
            }
        )
        modules, _ = fake_ros2_modules(node)

        with (
            patch.dict(sys.modules, modules),
            patch.object(Config, "CAMERA_COLOR_TOPIC", "/custom/rgb"),
            patch.object(Config, "CAMERA_POINT_CLOUD_TOPIC", "/custom/points"),
        ):
            camera = CameraROS2(
                camera_node="/ignored_prefix",
                camera_model="zed2i",
                camera_backend="ros2",
            )
            camera.get_point_cloud()

        self.assertEqual(node.subscribed_topics, ["/custom/rgb", "/custom/points"])


if __name__ == "__main__":
    unittest.main()
