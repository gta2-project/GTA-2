"""Managed server safety paths: all ROS, RTDE and action calls are fakes."""
import threading
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gta.managed_run import RunCancelled
from scene.robot.ur5e import ros2 as server
from test_ur5e_server_ros2 import make_gripper


class Runtime:
    def __init__(self):
        self.cancelled = threading.Event()
        self.events = []

    def check_cancel(self):
        if self.cancelled.is_set():
            raise RunCancelled('Cancelled')

    def emit(self, kind, **payload):
        self.events.append((kind, payload))


@pytest.mark.parametrize('control_failure', [False, True])
def test_main_readiness_follows_all_hardware_and_cleanup(control_failure):
    runtime = Runtime()
    order = []
    node = Mock()
    node.close.side_effect = lambda: order.append('close') or []
    node.wait_for_gripper_server.side_effect = lambda: order.append('gripper')
    control = Mock()
    robot = Mock()
    robot.get_tcp_pose.side_effect = lambda: order.append('receive') or [0] * 6
    def step():
        order.append('step')
        if control_failure:
            return False
        runtime.cancelled.set()
        return True
    node.control_step.side_effect = step
    original_emit = runtime.emit
    def emit(kind, **payload):
        order.append(kind)
        original_emit(kind, **payload)
    runtime.emit = emit
    fake_ros = Mock(); fake_ros.ok.return_value = True
    with patch.object(server, 'get_runtime', return_value=runtime), \
         patch.object(server, 'ROS2_AVAILABLE', True), patch.object(server, 'rclpy', fake_ros), \
         patch.object(server, 'UR5eROS2Server', side_effect=lambda: order.append('subscriptions') or node), \
         patch.object(server, 'RTDEControlInterface', side_effect=lambda _ip: order.append('control') or control), \
         patch.dict(sys.modules, {'scene.robot': SimpleNamespace(RobotState=lambda **_: robot)}):
        with pytest.raises(RuntimeError if control_failure else RunCancelled):
            server.main()
    assert order[:5] == ['subscriptions', 'gripper', 'control', 'receive', 'server_ready']
    assert order.index('close') < order.index('server_cleaned')
    node.destroy_node.assert_called_once()
    fake_ros.shutdown.assert_called_once()
    assert any(kind == 'server_failed' for kind, _ in runtime.events) == control_failure


def test_unavailable_gripper_does_not_initialize_rtde():
    runtime = Runtime()
    node = Mock(); node.wait_for_gripper_server.side_effect = RuntimeError('unavailable')
    node.close.return_value = []
    with patch.object(server, 'get_runtime', return_value=runtime), \
         patch.object(server, 'ROS2_AVAILABLE', True), patch.object(server, 'rclpy', Mock()), \
         patch.object(server, 'UR5eROS2Server', return_value=node), \
         patch.object(server, 'RTDEControlInterface') as control, \
         patch.dict(sys.modules, {'scene.robot': SimpleNamespace(RobotState=Mock())}):
        with pytest.raises(RuntimeError, match='unavailable'):
            server.main()
    control.assert_not_called()
    assert not any(kind == 'server_ready' for kind, _ in runtime.events)
    node.close.assert_called_once()


def test_gripper_availability_wait_is_cancellable():
    runtime = Runtime()
    node = object.__new__(server.UR5eROS2Server)
    node.gripper = make_gripper()
    node.gripper._check_cancel = runtime.check_cancel
    node.gripper.client = Mock()
    def wait(**kwargs):
        assert kwargs['timeout_sec'] <= 0.1
        runtime.cancelled.set()
        return False
    node.gripper.client.wait_for_server.side_effect = wait
    with patch.object(server, 'get_runtime', return_value=runtime):
        with pytest.raises(RunCancelled):
            node.wait_for_gripper_server(10)


def test_cancel_during_reconnect_closes_new_interface():
    runtime = Runtime()
    old, replacement = Mock(), Mock()
    waits = []
    def interrupt(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            raise RunCancelled()
    with patch.object(server, 'RTDEControlInterface', return_value=replacement), \
         patch.object(server, 'interruptible_sleep', side_effect=interrupt):
        with pytest.raises(RunCancelled):
            server.restart_rtde_control(old)
    old.disconnect.assert_called_once()
    replacement.servoStop.assert_called_once()
    replacement.stopScript.assert_called_once()
    replacement.disconnect.assert_called_once()


def test_cleanup_cancels_acknowledged_goal_and_reports_failures():
    node = object.__new__(server.UR5eROS2Server)
    node.gripper = make_gripper()
    node.rtde_c, node.robot, node.gripper.client = Mock(), Mock(), Mock()
    goal = Mock(); goal.accepted = True
    acknowledgement = Mock(); acknowledgement.done.return_value = True; acknowledgement.result.return_value = goal
    node.gripper._pending = acknowledgement
    acknowledgement.exception.return_value = None
    cancellation = Mock(); cancellation.done.return_value = True; cancellation.exception.return_value = None
    cancellation.result.return_value = SimpleNamespace(return_code=0)
    goal.cancel_goal_async.return_value = cancellation
    node.rtde_c.servoStop.return_value = False
    with patch.object(server, 'rclpy', Mock()):
        errors = node.close()
    goal.cancel_goal_async.assert_called_once()
    node.rtde_c.forceModeStop.assert_called_once()
    node.rtde_c.stopScript.assert_called_once()
    node.rtde_c.disconnect.assert_called_once()
    node.robot.close.assert_called_once()
    node.gripper.client.destroy.assert_called_once()
    assert any('servoStop' in error for error in errors)
