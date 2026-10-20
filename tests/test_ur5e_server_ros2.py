"""Offline ROS 2 UR5e server tests with fake RTDE and action clients."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
import scene.robot.ur5e.ros2 as server_module
from scene.gripper.robotiq import ros2 as gripper_module


class FakeRTDEControl:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return True

        return record


class FakeRobot:
    def __init__(self, pose=None):
        self.pose = np.zeros(6) if pose is None else np.asarray(pose, dtype=float)
        self.disconnected = False

    def get_tcp_pose(self):
        return self.pose.copy()

    def close(self):
        self.disconnected = True


class FakeFuture:
    def __init__(self, result=None, exception=None, done=True):
        self._result = result
        self._exception = exception
        self._done = done

    def done(self):
        return self._done

    def exception(self):
        return self._exception

    def result(self):
        return self._result


class FakeGripperCommand:
    class Goal:
        def __init__(self):
            self.command = SimpleNamespace(position=None, max_effort=None)


class FakeActionClient:
    def __init__(self, accepted=True, status=4, reached_goal=True, available=True):
        self.available = available
        self.goals = []
        self.destroyed = False
        result = SimpleNamespace(reached_goal=reached_goal)
        wrapped = SimpleNamespace(status=status, result=result)
        self.goal_handle = SimpleNamespace(
            accepted=accepted,
            get_result_async=lambda: FakeFuture(wrapped),
        )

    def wait_for_server(self, timeout_sec):
        self.wait_timeout = timeout_sec
        return self.available

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return FakeFuture(self.goal_handle)

    def destroy(self):
        self.destroyed = True


def make_gripper():
    gripper = object.__new__(gripper_module.Gripper)
    gripper._goal_type = FakeGripperCommand.Goal
    gripper._succeeded = 4
    gripper._spin = lambda future, seconds: None
    gripper._check_cancel = None
    gripper._poll_interval = 0.05
    gripper._cancel_on_timeout = True
    gripper._pending = gripper._active = None
    gripper._destroyed = False
    gripper.client = FakeActionClient()
    return gripper


def make_server():
    instance = object.__new__(server_module.UR5eROS2Server)
    instance.latest_position_error = np.zeros(3)
    instance.latest_rotation_error = np.eye(3)
    instance.latest_force_command = None
    instance.latest_gripper_command = None
    instance.pending_gripper_command = None
    instance.skill_active = False
    instance.skill_start_time = None
    instance.force_mode_was_active = False
    instance.reconnect_requested = False
    instance.zero_ft_requested = False
    instance.previous_gripper_command = None
    instance.reconnect_settle_ticks = 0
    instance.rtde_c = FakeRTDEControl()
    instance.robot = FakeRobot()
    instance.gripper = make_gripper()
    return instance


class UR5eROS2ServerTests(unittest.TestCase):
    def test_stable_server_path_dispatches_ros1_and_ros2(self):
        module_name = "scene.robot.server"

        def fake_backend(name):
            module = types.ModuleType(name)
            module.main = lambda: None
            return module

        ros1_name = "scene.robot.ur5e.ros1"
        ros2_name = "scene.robot.ur5e.ros2"
        replacements = {
            ros1_name: fake_backend(ros1_name),
            ros2_name: fake_backend(ros2_name),
        }
        sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, replacements), patch.object(Config, "ROBOT_TYPE", "ur5e"):
                with patch.object(Config, "GTA_BACKEND", "ros2"):
                    selected = importlib.import_module(module_name)
                    self.assertIs(selected._load_backend("ur5e", "ros2"), replacements[ros2_name])
                with patch.object(Config, "GTA_BACKEND", "ros1"):
                    selected = importlib.reload(selected)
                    self.assertIs(selected._load_backend("ur5e", "ros1"), replacements[ros1_name])
                self.assertTrue(callable(selected.main))
        finally:
            sys.modules.pop(module_name, None)

    def test_callbacks_validate_payloads_and_empty_force_deactivates(self):
        instance = make_server()
        instance.position_error_callback(SimpleNamespace(data=[1.0, 2.0, 3.0]))
        instance.rotation_error_callback(SimpleNamespace(data=list(range(9))))
        instance.force_command_callback(SimpleNamespace(data=[0.0, 0.0, 1.0, 5.0]))
        np.testing.assert_allclose(instance.latest_position_error, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(instance.latest_rotation_error, np.arange(9).reshape(3, 3))
        np.testing.assert_allclose(instance.latest_force_command, [0.0, 0.0, 1.0, 5.0])

        instance.force_command_callback(SimpleNamespace(data=[]))
        self.assertIsNone(instance.latest_force_command)
        old_position = instance.latest_position_error.copy()
        instance.position_error_callback(SimpleNamespace(data=[1.0, np.nan, 3.0]))
        np.testing.assert_array_equal(instance.latest_position_error, old_position)

    def test_motion_step_matches_ros1_step_limiting_and_servo_parameters(self):
        instance = make_server()
        instance.skill_active = True
        instance.latest_position_error = np.array([1.0, 0.0, 0.0])
        instance.latest_rotation_error = np.eye(3)
        self.assertTrue(instance.control_step())

        servo_call = next(call for call in instance.rtde_c.calls if call[0] == "servoL")
        pose, acceleration, velocity, dt, lookahead, gain = servo_call[1]
        self.assertAlmostEqual(pose[0], server_module.V_MPS * server_module.DT)
        np.testing.assert_allclose(pose[1:], np.zeros(5), atol=1e-12)
        self.assertEqual((acceleration, velocity), (0, 0))
        self.assertEqual((dt, lookahead, gain), (server_module.DT, server_module.LOOKAHEAD, server_module.GAIN))

        # The ROS 1 server's direct-RTDE rule is v_mps * dt along the error axis.
        np.testing.assert_allclose(
            pose[:3],
            [server_module.V_MPS * server_module.DT, 0.0, 0.0],
        )

    def test_force_mode_and_empty_force_trigger_control_reconnect(self):
        instance = make_server()
        instance.skill_active = True
        instance.latest_force_command = np.array([0.0, 0.0, 1.0, 6.0])
        instance.control_step()
        self.assertTrue(instance.force_mode_was_active)
        force_call = next(call for call in instance.rtde_c.calls if call[0] == "forceMode")
        self.assertEqual(force_call[1][1], [0, 0, 1, 0, 0, 0])
        self.assertEqual(force_call[1][2], [0.0, 0.0, 6.0, 0.0, 0.0, 0.0])

        old_control = instance.rtde_c
        replacement = FakeRTDEControl()
        instance.force_command_callback(SimpleNamespace(data=[]))
        with patch.object(server_module, "restart_rtde_control", return_value=replacement) as restart:
            instance.control_step()
        restart.assert_called_once_with(
            old_control,
            sleep_s=server_module.CONTROL_RECONNECT_SLEEP_S,
        )
        self.assertIs(instance.rtde_c, replacement)
        self.assertFalse(instance.force_mode_was_active)

    def test_gripper_open_and_close_use_standard_action_contract(self):
        instance = make_server()
        action_client = FakeActionClient(status=4)
        instance.gripper.client = action_client
        fake_rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None
        )
        fake_goal_status = SimpleNamespace(STATUS_SUCCEEDED=4)
        with patch.multiple(
            server_module,
            rclpy=fake_rclpy,

        ):
            instance.send_gripper_command(1.0, timeout_s=2.0)
            instance.send_gripper_command(0.0, timeout_s=2.0)

        self.assertEqual(
            [goal.command.position for goal in action_client.goals],
            [gripper_module.GRIPPER_OPEN_POSITION, gripper_module.GRIPPER_CLOSED_POSITION],
        )
        self.assertTrue(
            all(
                goal.command.max_effort == gripper_module.GRIPPER_MAX_EFFORT
                for goal in action_client.goals
            )
        )

    def test_unavailable_action_fails_startup_check(self):
        instance = make_server()
        instance.gripper.client = FakeActionClient(available=False)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            instance.wait_for_gripper_server(timeout_s=0.25)
        self.assertEqual(instance.gripper.client.wait_timeout, 0.25)

    def test_rejected_gripper_goal_stops_active_skill_safely(self):
        instance = make_server()
        instance.skill_active = True
        instance.pending_gripper_command = 1.0
        instance.gripper.client = FakeActionClient(accepted=False)
        fake_rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None
        )
        with patch.multiple(
            server_module,
            rclpy=fake_rclpy,

        ):
            self.assertFalse(instance._run_pending_gripper_command())
        self.assertFalse(instance.skill_active)
        self.assertIn("servoStop", [name for name, _args, _kwargs in instance.rtde_c.calls])

    def test_failed_gripper_result_stops_active_skill_safely(self):
        instance = make_server()
        instance.skill_active = True
        instance.pending_gripper_command = 1.0
        instance.gripper.client = FakeActionClient(status=6)
        fake_rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None
        )
        fake_goal_status = SimpleNamespace(STATUS_SUCCEEDED=4)
        with patch.multiple(
            server_module,
            rclpy=fake_rclpy,

        ):
            self.assertFalse(instance._run_pending_gripper_command())
        self.assertFalse(instance.skill_active)
        self.assertIn("servoStop", [name for name, _args, _kwargs in instance.rtde_c.calls])

    def test_succeeded_gripper_result_without_reached_goal_continues_skill(self):
        instance = make_server()
        instance.skill_active = True
        instance.pending_gripper_command = 1.0
        instance.gripper.client = FakeActionClient(status=4, reached_goal=False)
        fake_rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None
        )
        fake_goal_status = SimpleNamespace(STATUS_SUCCEEDED=4)
        with patch.multiple(
            server_module,
            rclpy=fake_rclpy,

        ):
            self.assertTrue(instance._run_pending_gripper_command())

        self.assertTrue(instance.skill_active)
        self.assertEqual(instance.previous_gripper_command, 1.0)
        self.assertNotIn(
            "servoStop", [name for name, _args, _kwargs in instance.rtde_c.calls]
        )

    def test_gripper_timeout_stops_active_skill_and_cleanup_disconnects(self):
        instance = make_server()
        instance.skill_active = True
        instance.pending_gripper_command = 0.0
        instance.gripper.client.send_goal_async = lambda _goal: FakeFuture(done=False)
        fake_rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *_args, **_kwargs: None
        )
        with patch.multiple(
            server_module,
            rclpy=fake_rclpy,

        ):
            self.assertFalse(instance._run_pending_gripper_command())
        self.assertFalse(instance.skill_active)

        action_client = instance.gripper.client
        with patch.object(server_module, 'rclpy', fake_rclpy):
            instance.close()
        call_names = [name for name, _args, _kwargs in instance.rtde_c.calls]
        self.assertIn("servoStop", call_names)
        self.assertIn("stopScript", call_names)
        self.assertIn("disconnect", call_names)
        self.assertTrue(instance.robot.disconnected)
        self.assertTrue(action_client.destroyed)


if __name__ == "__main__":
    unittest.main()
