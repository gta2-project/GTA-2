"""Manual controls use only fake RTDE/ROS; never connect to robot hardware."""
import ast
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from config import Config
from gta.managed_run import RunCancelled
from scene.robot import server
from scene.robot.ur5e import helpers
from test_ur5e_gripper_scripts import ros2, settings


def reset(runtime, errors, control_factory=None, receive_factory=None):
    return helpers.Helpers().reset_joints(
        check_cancel=runtime.check_cancel, on_started=lambda: runtime.emit('manual_started'),
        errors=errors, wait=runtime.cancelled.wait,
        control_factory=control_factory, receive_factory=receive_factory,
    )


@pytest.fixture
def runtime():
    value = SimpleNamespace(role='manual', cancelled=threading.Event(), emit=Mock())
    def check():
        if value.cancelled.is_set():
            raise RunCancelled()
    value.check_cancel = check
    return value


@pytest.fixture
def hardware():
    control = Mock()
    control.getAsyncOperationProgress.side_effect = [-1, 0, -2]
    receive = Mock()
    receive.getActualQ.side_effect = [[0] * 6, helpers.RESET_JOINTS[:]]
    return control, receive


def test_reset_pose_and_speed_match_existing_standalone_helper():
    helper = Path(__file__).resolve().parents[1] / 'scripts' / 'ur5e' / 'reset_joints.py'
    tree = ast.parse(helper.read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)}
    for name in ('RESET_JOINTS', 'JOINT_SPEED', 'JOINT_ACCELERATION'):
        assert getattr(helpers, name) == constants[name]


def test_async_reset_verifies_pose_and_cleans_up(runtime, hardware):
    control, receive = hardware
    factories = Mock(return_value=control), Mock(return_value=receive)
    errors = []
    reset(runtime, errors, *factories)
    for factory in factories:
        factory.assert_called_once_with(Config.ROBOT_IP)
    control.moveJ.assert_called_once_with(helpers.RESET_JOINTS, 0.2, 0.2, True)
    control.stopJ.assert_called_once_with(0.5)
    control.stopScript.assert_called_once()
    control.disconnect.assert_called_once()
    receive.disconnect.assert_called_once()
    assert errors == []


def test_async_reset_does_not_mistake_old_progress_for_completion(runtime, hardware):
    control, receive = hardware
    control.getAsyncOperationProgress.side_effect = [-1, -1, -1, -2]
    reset(runtime, [], lambda _: control, lambda _: receive)
    assert control.getAsyncOperationProgress.call_count == 4


@pytest.mark.parametrize('stage', ['before_connect', 'after_connect', 'during_move'])
def test_reset_cancellation_and_partial_connection_cleanup(runtime, hardware, stage):
    control, receive = hardware
    def connect(_):
        if stage == 'after_connect':
            runtime.cancelled.set()
        return control
    if stage == 'before_connect':
        runtime.cancelled.set()
    if stage == 'during_move':
        control.moveJ.side_effect = lambda *_: runtime.cancelled.set() or True
    with pytest.raises(RunCancelled):
        reset(runtime, [], connect, lambda _: receive)
    if stage != 'during_move':
        control.moveJ.assert_not_called()
    else:
        control.stopJ.assert_called_once()
        receive.disconnect.assert_called_once()
    if stage != 'before_connect':
        control.disconnect.assert_called_once()


@pytest.mark.parametrize('failure', ['rejected', 'wrong_pose', 'nonfinite', 'timeout', 'active'])
def test_reset_failures_do_not_report_success_and_release_hardware(runtime, hardware, failure, monkeypatch):
    control, receive = hardware
    if failure == 'rejected':
        control.moveJ.return_value = False
    elif failure == 'wrong_pose':
        receive.getActualQ.side_effect = [[0] * 6, [0] * 6]
    elif failure == 'nonfinite':
        receive.getActualQ.side_effect = [[float('nan')] * 6]
    elif failure == 'timeout':
        control.getAsyncOperationProgress.side_effect = None
        control.getAsyncOperationProgress.return_value = -1
        monkeypatch.setattr(helpers, 'RESET_TIMEOUT_S', 0)
    else:
        control.getAsyncOperationProgress.side_effect = [0]
    with pytest.raises((RuntimeError, TimeoutError)):
        reset(runtime, [], lambda _: control, lambda _: receive)
    control.disconnect.assert_called_once()
    receive.disconnect.assert_called_once()
    if failure in {'nonfinite', 'active'}:
        control.moveJ.assert_not_called()


def test_reset_reports_stop_failure_but_still_disconnects(runtime, hardware):
    control, receive = hardware
    control.stopJ.side_effect = RuntimeError('stop failed')
    errors = []
    reset(runtime, errors, lambda _: control, lambda _: receive)
    assert errors == ['stopJ: stop failed']
    control.disconnect.assert_called_once()
    receive.disconnect.assert_called_once()


@pytest.mark.parametrize('operation,position', [('open_gripper', 0.0), ('close_gripper', 0.85)])
@pytest.mark.parametrize('failure', [None, 'unavailable', 'rejected', 'cancelled'])
def test_gripper_uses_shared_client_without_robot_server(runtime, ros2, operation, position, failure):
    if failure == 'unavailable':
        ros2.client.wait_for_server.side_effect = lambda **kw: ros2.clock.advance(kw['timeout_sec']) or False
    elif failure == 'rejected':
        ros2.handle.accepted = False
    elif failure == 'cancelled':
        ros2.client.wait_for_server.side_effect = lambda **kw: runtime.cancelled.set() or True
    errors = []
    if failure:
        with pytest.raises((RuntimeError, RunCancelled)):
            server._run_gripper(operation, runtime, errors)
    else:
        server._run_gripper(operation, runtime, errors)
    if failure in {'unavailable', 'cancelled'}:
        ros2.client.send_goal_async.assert_not_called()
    else:
        assert ros2.client.send_goal_async.call_args.args[0].command.position == position
    ros2.node.create_subscription.assert_not_called()
    ros2.client.destroy.assert_called_once()
    ros2.executor.shutdown.assert_called_once()
    ros2.node.destroy_node.assert_called_once()
    ros2.ros.shutdown.assert_called_once_with(context=ros2.context)
    assert errors == []


def test_helper_cannot_start_without_supervisor(monkeypatch):
    import scene.robot
    with patch.object(server, 'get_runtime', return_value=None), patch.object(scene.robot, 'RobotHelpers') as reset:
        with pytest.raises(RuntimeError, match='GUI supervisor'):
            server.main(['--operation', 'reset_joints'])
    reset.assert_not_called()


def test_helper_cleanup_failure_does_not_emit_completion(runtime, monkeypatch):
    import scene.robot
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    helper = Mock()
    helper.reset_joints.side_effect = lambda **kw: kw['errors'].append('stop failed')
    with patch.object(server, 'get_runtime', return_value=runtime), \
         patch.object(scene.robot, 'RobotHelpers', return_value=helper):
        assert server.main(['--operation', 'reset_joints']) == 1
    assert not any(call.args[0] == 'manual_completed' for call in runtime.emit.call_args_list)
    runtime.emit.assert_called_with('manual_cleaned', errors=['stop failed'])
