"""Mocked tests for ROS 2 GTA command publishing and backend selection."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
import gta.skill_execution as dispatcher
from gta.skill_execution import ros1 as ros1_backend
from gta.skill_execution import ros2 as backend


class ExecutionLayoutTests(unittest.TestCase):
    def test_selector_and_backends_live_in_skill_execution_package(self):
        components = Path(__file__).resolve().parents[1] / "gta"
        package = components / "skill_execution"
        self.assertEqual(Path(dispatcher.__file__).resolve(), package / "__init__.py")
        self.assertFalse((components / "skill_execution.py").exists())
        self.assertFalse((components / "execution" / "__init__.py").exists())
        for name, module in (("ros1", ros1_backend), ("ros2", backend)):
            with self.subTest(backend=name):
                self.assertEqual(Path(module.__file__).resolve(), package / f"{name}.py")
                self.assertFalse((components / "execution" / f"{name}.py").exists())
                self.assertFalse((components / f"skill_execution{name.upper()}.py").exists())

    def test_public_import_loads_only_selected_backend_without_ros(self):
        probe = """
import importlib
import sys
from config import Config

Config.GTA_BACKEND = sys.argv[1]
# Keep the probe offline even when ROS is installed on the test machine.
sys.modules['rospy'] = None
sys.modules['rclpy'] = None
import gta as components
import gta.skill_execution as execution

selected = importlib.import_module(f'{execution.__name__}.{Config.GTA_BACKEND}')
other = 'ros2' if Config.GTA_BACKEND == 'ros1' else 'ros1'
assert f'{execution.__name__}.{other}' not in sys.modules
assert execution.BACKEND == Config.GTA_BACKEND
assert execution.ROS_AVAILABLE is False
for name in execution.__all__:
    if name not in ('BACKEND', 'ROS_AVAILABLE'):
        assert getattr(execution, name) is getattr(selected, name), name
        assert getattr(components, name) is getattr(selected, name), name
for name in ('rtde_control', 'rtde_receive', 'open3d'):
    assert name not in sys.modules, name
"""
        for gta_backend in ('ros1', 'ros2'):
            with self.subTest(backend=gta_backend):
                result = subprocess.run(
                    [sys.executable, "-c", probe, gta_backend],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_helpers_and_duration_defaults_are_preserved(self):
        projection = importlib.import_module("gta.error_projection")
        managed_run = importlib.import_module("gta.managed_run")
        for module in (ros1_backend, backend):
            with self.subTest(backend=module.__name__):
                self.assertIs(module.project_errors, projection.project_errors)
                self.assertEqual(inspect.signature(module.execute_subtask).parameters["duration"].default, 15.0)
        self.assertIs(backend.get_runtime, managed_run.get_runtime)
        self.assertIs(backend.should_skip, managed_run.should_skip)


class Message:
    def __init__(self, data=None):
        self.data = [] if data is None else data


class FakeQoSProfile:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePublisher:
    def __init__(self, topic, qos, subscription_count=1):
        self.topic = topic
        self.qos = qos
        self.subscription_count = subscription_count
        self.messages = []

    def get_subscription_count(self):
        return self.subscription_count

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self, subscription_count=1):
        self.subscription_count = subscription_count
        self.publishers = []
        self.destroyed = False

    def create_publisher(self, _message_type, topic, qos):
        publisher = FakePublisher(topic, qos, self.subscription_count)
        self.publishers.append(publisher)
        return publisher

    def destroy_node(self):
        self.destroyed = True


class FakeRclpy:
    def __init__(self, node):
        self.node = node
        self.running = False
        self.spin_calls = []

    def ok(self):
        return self.running

    def init(self, args=None):
        self.running = True

    def create_node(self, name):
        self.node.name = name
        return self.node

    def spin_once(self, node, timeout_sec):
        self.spin_calls.append((node, timeout_sec))


def ros2_patches(node):
    fake_rclpy = FakeRclpy(node)
    policy = SimpleNamespace(KEEP_LAST="keep_last", RELIABLE="reliable", VOLATILE="volatile")
    return fake_rclpy, patch.multiple(
        backend,
        ROS_AVAILABLE=True,
        rclpy=fake_rclpy,
        QoSProfile=FakeQoSProfile,
        HistoryPolicy=policy,
        ReliabilityPolicy=policy,
        DurabilityPolicy=policy,
        Float32=Message,
        Float32MultiArray=Message,
        String=Message,
    )


class SkillExecutionROS2Tests(unittest.TestCase):
    def setUp(self):
        backend._destroy_node()

    def tearDown(self):
        backend._destroy_node()

    def test_dispatcher_selects_gta_backend_independently_of_camera(self):
        try:
            for camera_backend in ('file', 'ros1', 'ros2'):
                for gta_backend, expected in (('ros1', ros1_backend), ('ros2', backend)):
                    with self.subTest(camera_backend=camera_backend, gta_backend=gta_backend), \
                         patch.multiple(Config, CAMERA_BACKEND=camera_backend, GTA_BACKEND=gta_backend):
                        selected = importlib.reload(dispatcher)
                        self.assertEqual(selected.BACKEND, gta_backend)
                        self.assertIs(selected.execute_subtask, expected.execute_subtask)
                        for name in (
                            "initialize_ros_publishers", "start_skill", "update_control_commands",
                            "end_skill", "execute_subtask",
                        ):
                            self.assertIs(getattr(selected, name), getattr(expected, name))
                        self.assertIs(selected.ROS_AVAILABLE, expected.ROS_AVAILABLE)
        finally:
            importlib.reload(dispatcher)

    def test_invalid_gta_backend_does_not_silently_select_ros1(self):
        try:
            with patch.object(Config, "GTA_BACKEND", "local"):
                with self.assertRaisesRegex(ValueError, 'GTA_BACKEND must be ros1 or ros2'):
                    importlib.reload(dispatcher)
        finally:
            importlib.reload(dispatcher)

    def test_topics_qos_payloads_status_and_force_clear(self):
        node = FakeNode(subscription_count=1)
        fake_rclpy, ros_patch = ros2_patches(node)
        with ros_patch, patch("time.sleep"):
            self.assertTrue(backend.initialize_ros_publishers())
            backend.start_skill()
            self.assertTrue(
                backend.update_control_commands(
                    np.array([0.1, -0.2, 0.3]),
                    np.arange(9, dtype=float).reshape(3, 3),
                    np.array([0.0, 0.0, 1.0, 7.5]),
                    1.0,
                )
            )
            self.assertTrue(
                backend.update_control_commands(
                    np.zeros(3),
                    np.eye(3),
                    None,
                    None,
                )
            )
            backend.end_skill()

        self.assertEqual(
            [publisher.topic for publisher in node.publishers],
            list(backend._TOPICS),
        )
        self.assertEqual([publisher.qos.depth for publisher in node.publishers], [1, 1, 1, 10, 10])
        self.assertTrue(all(pub.qos.reliability == "reliable" for pub in node.publishers))
        self.assertTrue(all(pub.qos.history == "keep_last" for pub in node.publishers))
        self.assertEqual(node.publishers[0].messages[0].data, [0.1, -0.2, 0.3])
        self.assertEqual(node.publishers[1].messages[0].data, list(map(float, range(9))))
        self.assertEqual(node.publishers[2].messages[0].data, [0.0, 0.0, 1.0, 7.5])
        self.assertEqual(node.publishers[2].messages[1].data, [])
        self.assertEqual(node.publishers[3].messages[0].data, 1.0)
        self.assertEqual(
            [message.data for message in node.publishers[4].messages],
            ["skill_started", "skill_ended"],
        )
        self.assertEqual(node.name, "gta_skill_executor")
        self.assertTrue(fake_rclpy.running)

    def test_initialization_times_out_when_subscribers_are_missing(self):
        node = FakeNode(subscription_count=0)
        _, ros_patch = ros2_patches(node)
        with ros_patch, patch.object(Config, "ROS2_PUBLISHER_MATCH_TIMEOUT_S", 0.0):
            self.assertFalse(backend.initialize_ros_publishers())
        self.assertTrue(node.destroyed)
        self.assertFalse(backend._ros_initialized)

    def test_missing_rclpy_is_reported_without_creating_a_node(self):
        with patch.multiple(backend, ROS_AVAILABLE=False, rclpy=None):
            self.assertFalse(backend.initialize_ros_publishers())
        self.assertIsNone(backend._ros_node)

    def test_non_finite_commands_are_rejected_before_publish(self):
        with self.assertRaises(ValueError):
            backend.update_control_commands(
                np.array([np.nan, 0.0, 0.0]),
                np.eye(3),
                None,
                None,
            )


if __name__ == "__main__":
    unittest.main()
