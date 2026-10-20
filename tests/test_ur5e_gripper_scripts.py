"""Offline standalone-gripper tests. All ROS/action clients are fakes."""

import ast
import builtins
from pathlib import Path
import runpy
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from config import Config
from scripts.ur5e import gripper_control as gripper


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = [('open_gripper.py', 'open'), ('close_gripper.py', 'close'),
              ('half_open_gripper.py', 'half_open')]


class Future:
    def __init__(self, result=None, done=True, error=None):
        self.value, self.ready, self.error = result, done, error

    def done(self):
        return self.ready

    def exception(self):
        return self.error

    def result(self):
        return self.value


class Clock:
    now = 0.0

    def advance(self, seconds):
        self.now += seconds


def module(monkeypatch, name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)
    return value


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'ur5e')
    monkeypatch.setattr(Config, 'GRIPPER_TYPE', 'robotiq')
    monkeypatch.setattr(Config, 'ROS2_GRIPPER_ACTION_NAME', '/test/gripper_cmd')
    monkeypatch.setattr(Config, 'ROS2_GRIPPER_WAIT_TIMEOUT_S', 0.3)
    clock = Clock()
    monkeypatch.setattr(gripper.time, 'monotonic', lambda: clock.now)
    return clock


@pytest.fixture
def ros2(monkeypatch, settings):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    context = SimpleNamespace(ok=Mock(return_value=True))
    node = Mock()
    executor = Mock()
    executor.shutdown.return_value = True
    executor.spin_once.side_effect = lambda timeout_sec: settings.advance(timeout_sec)
    result = Future(SimpleNamespace(status=4, result=SimpleNamespace(reached_goal=True)))
    cancel = Future(SimpleNamespace(return_code=0))
    handle = SimpleNamespace(accepted=True, get_result_async=Mock(return_value=result),
                             cancel_goal_async=Mock(return_value=cancel))
    sent = Future(handle)
    client = Mock()
    client.destroy.return_value = None
    client.wait_for_server.return_value = True
    client.send_goal_async.return_value = sent

    class Goal:
        def __init__(self):
            self.command = SimpleNamespace(position=None, max_effort=None)

    ros = module(monkeypatch, 'rclpy', init=Mock(), shutdown=Mock(),
                 create_node=Mock(return_value=node))
    context_factory = Mock(return_value=context)
    executor_factory = Mock(return_value=executor)
    client_factory = Mock(return_value=client)
    module(monkeypatch, 'rclpy.context', Context=context_factory)
    module(monkeypatch, 'rclpy.executors', SingleThreadedExecutor=executor_factory)
    module(monkeypatch, 'rclpy.action', ActionClient=client_factory)
    module(monkeypatch, 'rclpy.signals', SignalHandlerOptions=SimpleNamespace(NO='no-signals'))
    module(monkeypatch, 'action_msgs')
    module(monkeypatch, 'action_msgs.msg', GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4))
    module(monkeypatch, 'control_msgs')
    module(monkeypatch, 'control_msgs.action', GripperCommand=SimpleNamespace(Goal=Goal))
    from scene.gripper.robotiq import ros2 as server
    return SimpleNamespace(ros=ros, context=context, node=node, executor=executor,
                           executor_factory=executor_factory, context_factory=context_factory,
                           client=client, client_factory=client_factory, result=result, cancel=cancel,
                           handle=handle, sent=sent, clock=settings, server=server)


@pytest.fixture
def ros1(monkeypatch, settings):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros1')
    client = Mock()
    client.wait_for_server.return_value = True
    client.wait_for_result.return_value = True
    client.get_state.return_value = 3
    ros = module(monkeypatch, 'rospy', init_node=Mock(), signal_shutdown=Mock(), Duration=lambda value: value)
    action = module(monkeypatch, 'actionlib', SimpleActionClient=Mock(return_value=client))
    module(monkeypatch, 'actionlib_msgs')
    module(monkeypatch, 'actionlib_msgs.msg', GoalStatus=SimpleNamespace(SUCCEEDED=3))
    module(monkeypatch, 'robotiq_mm_ros')
    module(monkeypatch, 'robotiq_mm_ros.msg', GripperCommandAction=object, GripperCommandGoal=SimpleNamespace)
    return SimpleNamespace(ros=ros, action=action, client=client, clock=settings)


@pytest.mark.parametrize('filename,operation', OPERATIONS)
def test_entrypoint_resolves_repo_and_calls_only_its_operation(monkeypatch, tmp_path, filename, operation):
    main = Mock(return_value=0)
    monkeypatch.setattr(gripper, 'main', main)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', [p for p in sys.path if p and p != str(ROOT)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / 'scripts/ur5e' / filename), run_name='__main__')
    assert result.value.code == 0
    main.assert_called_once_with(operation)
    assert sys.path[0] == str(ROOT)


def test_importing_utilities_never_loads_ros_rtde_or_config():
    probe = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'config', 'rclpy', 'rospy', 'actionlib', 'robotiq_mm_ros',
                              'rtde_control', 'rtde_receive', 'robot_server'}:
        raise AssertionError('Unexpected import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from scripts.ur5e import open_gripper, close_gripper, half_open_gripper, gripper_control
'''
    result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('operation,position', [('open', 0.0), ('close', 0.85), ('half_open', 0.425)])
def test_ros2_mapping_explicit_executor_and_cleanup(ros2, operation, position):
    assert gripper.main(operation) == 0
    goal = ros2.client.send_goal_async.call_args.args[0]
    assert goal.command.position == position
    assert goal.command.max_effort == 165
    ros2.ros.init.assert_called_once_with(args=None, context=ros2.context, signal_handler_options='no-signals')
    ros2.executor_factory.assert_called_once_with(context=ros2.context)
    ros2.executor.add_node.assert_called_once_with(ros2.node)
    assert ros2.client_factory.call_args.args[2] == '/test/gripper_cmd'
    ros2.node.create_subscription.assert_not_called()
    ros2.client.destroy.assert_called_once()
    ros2.executor.shutdown.assert_called_once()
    ros2.node.destroy_node.assert_called_once()
    ros2.ros.shutdown.assert_called_once_with(context=ros2.context)
    ros2.handle.cancel_goal_async.assert_not_called()


@pytest.mark.parametrize('operation,position', [('open', 0.1), ('close', 0.7), ('half_open', 0.4)])
def test_ros2_mapping_really_comes_from_shared_backend(ros2, operation, position, monkeypatch):
    monkeypatch.setattr(ros2.server, "GRIPPER_OPEN_POSITION", 0.1)
    monkeypatch.setattr(ros2.server, "GRIPPER_CLOSED_POSITION", 0.7)
    monkeypatch.setattr(ros2.server, "GRIPPER_MAX_EFFORT", 42.0)
    assert gripper.main(operation) == 0
    goal = ros2.client.send_goal_async.call_args.args[0]
    assert goal.command.position == pytest.approx(position)
    assert goal.command.max_effort == 42


def test_ros2_success_spins_the_owned_executor_for_acknowledgement_and_result(ros2):
    ros2.sent.ready = ros2.result.ready = False
    def spin(timeout_sec):
        ros2.clock.advance(timeout_sec)
        if not ros2.sent.ready:
            ros2.sent.ready = True
        else:
            ros2.result.ready = True
    ros2.executor.spin_once.side_effect = spin
    assert gripper.main('open') == 0
    assert ros2.executor.spin_once.call_count == 2


def test_ros2_does_not_need_any_ros1_dependencies(ros2, monkeypatch):
    for name in ('rospy', 'actionlib', 'robotiq_mm_ros'):
        monkeypatch.setitem(sys.modules, name, None)
    assert gripper.main('open') == 0


def test_ros1_does_not_need_ros2(ros1, monkeypatch):
    monkeypatch.setitem(sys.modules, 'rclpy', None)
    assert gripper.main('open') == 0


@pytest.mark.parametrize('operation,position', [('open', 0.085), ('close', 0.0), ('half_open', 0.06)])
def test_ros1_original_wire_contract_and_cleanup(ros1, operation, position):
    assert gripper.main(operation) == 0
    goal = ros1.client.send_goal.call_args.args[0]
    assert vars(goal) == {'position': position, 'velocity': 0.05, 'force': 165}
    assert ros1.action.SimpleActionClient.call_args.args[0] == '/ur5/move_gripper'
    ros1.ros.init_node.assert_called_once_with('gta_gripper_utility', anonymous=True, disable_signals=True)
    ros1.ros.signal_shutdown.assert_called_once()
    ros1.client.cancel_goal.assert_not_called()


@pytest.mark.parametrize('failure', ['unavailable', 'rejected', 'failed', 'timeout', 'send_timeout', 'future_error'])
def test_ros2_failures_cleanup_and_never_report_success(ros2, capsys, failure):
    if failure == 'unavailable':
        ros2.client.wait_for_server.side_effect = lambda timeout_sec: ros2.clock.advance(timeout_sec) or False
    elif failure == 'rejected':
        ros2.handle.accepted = False
    elif failure == 'failed':
        ros2.result.value.status = 6
    elif failure == 'timeout':
        ros2.result.ready = False
    elif failure == 'send_timeout':
        ros2.sent.ready = False
    else:
        ros2.result.error = RuntimeError('result failed')
    assert gripper.main('open') == 1
    assert 'action completed' not in capsys.readouterr().out
    if failure == 'unavailable':
        ros2.client.send_goal_async.assert_not_called()
    if failure in {'failed', 'timeout', 'future_error'}:
        ros2.handle.cancel_goal_async.assert_called_once()
    ros2.client.destroy.assert_called_once()
    ros2.node.destroy_node.assert_called_once()
    ros2.executor.shutdown.assert_called_once()
    ros2.ros.shutdown.assert_called_once()


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('stage', ['discovery', 'acknowledgement', 'result'])
def test_ros2_signals_cancel_own_goal_before_shutdown(ros2, sig, stage):
    previous = signal.getsignal(sig)
    def interrupt(value):
        signal.getsignal(sig)(sig, None)
        return value
    if stage == 'discovery':
        ros2.client.wait_for_server.side_effect = lambda **_: interrupt(True)
    elif stage == 'acknowledgement':
        ros2.client.send_goal_async.side_effect = lambda _: interrupt(ros2.sent)
    else:
        ros2.handle.get_result_async.side_effect = lambda: interrupt(ros2.result)
    assert gripper.main('open') == 130
    assert signal.getsignal(sig) == previous
    if stage == 'discovery':
        ros2.client.send_goal_async.assert_not_called()
        ros2.handle.cancel_goal_async.assert_not_called()
    else:
        ros2.handle.cancel_goal_async.assert_called_once()
    ros2.ros.shutdown.assert_called_once()


def test_cancellation_precedes_ros_resource_shutdown(ros2):
    events = []
    ros2.client.send_goal_async.side_effect = lambda _: (
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None) or ros2.sent)
    ros2.handle.cancel_goal_async.side_effect = lambda: events.append('cancel') or ros2.cancel
    ros2.client.destroy.side_effect = lambda: events.append('client')
    ros2.executor.shutdown.side_effect = lambda **_: events.append('executor') or True
    ros2.node.destroy_node.side_effect = lambda: events.append('node')
    ros2.ros.shutdown.side_effect = lambda **_: events.append('shutdown')
    assert gripper.main('close') == 130
    assert events == ['cancel', 'client', 'executor', 'node', 'shutdown']


def test_ros2_waits_for_late_acknowledgement_and_cancels_it(ros2):
    ros2.sent.ready = False
    def spin(timeout_sec):
        ros2.clock.advance(timeout_sec)
        if ros2.clock.now >= 0.4:
            ros2.sent.ready = True
    ros2.executor.spin_once.side_effect = spin
    assert gripper.main('open') == 1
    ros2.handle.cancel_goal_async.assert_called_once()


@pytest.mark.parametrize('failure', ['cancel_rejected', 'cancel_timeout', 'destroy', 'node', 'executor'])
def test_ros2_reports_unverified_cleanup(ros2, capsys, failure):
    if failure.startswith('cancel_'):
        ros2.result.ready = False
        if failure == 'cancel_rejected':
            ros2.cancel.value.return_code = 1
        else:
            ros2.cancel.ready = False
    elif failure == 'destroy':
        ros2.client.destroy.side_effect = RuntimeError('cleanup failed')
    elif failure == 'node':
        ros2.node.destroy_node.side_effect = RuntimeError('cleanup failed')
    else:
        ros2.executor.shutdown.return_value = False
    assert gripper.main('open') == 1
    output = capsys.readouterr()
    assert 'Cleanup failed' in output.err
    assert 'action completed' not in output.out
    ros2.ros.shutdown.assert_called_once()


@pytest.mark.parametrize('stage', ['node', 'executor', 'client'])
def test_ros2_partial_startup_cleans_initialized_resources(ros2, stage):
    factory = {'node': ros2.ros.create_node, 'executor': ros2.executor_factory,
               'client': ros2.client_factory}[stage]
    factory.side_effect = RuntimeError('startup failed')
    assert gripper.main('open') == 1
    ros2.client.send_goal_async.assert_not_called()
    ros2.ros.shutdown.assert_called_once()
    if stage != 'node':
        ros2.node.destroy_node.assert_called_once()
    if stage == 'client':
        ros2.executor.shutdown.assert_called_once()


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_ros2_invalid_timeout_fails_before_initialization(ros2, monkeypatch, timeout):
    monkeypatch.setattr(Config, 'ROS2_GRIPPER_WAIT_TIMEOUT_S', timeout)
    assert gripper.main('open') == 1
    ros2.ros.init.assert_not_called()
    ros2.client.send_goal_async.assert_not_called()


@pytest.mark.parametrize('failure', ['unavailable', 'failed', 'timeout', 'interrupted'])
def test_ros1_failures_cancel_own_goal_and_shutdown(ros1, failure):
    if failure == 'unavailable':
        ros1.client.wait_for_server.side_effect = lambda seconds: ros1.clock.advance(seconds) or False
    elif failure == 'failed':
        ros1.client.get_state.return_value = 4
    elif failure == 'timeout':
        ros1.client.wait_for_result.side_effect = lambda seconds: (
            True if ros1.client.cancel_goal.called else ros1.clock.advance(seconds) or False)
    else:
        ros1.client.send_goal.side_effect = lambda _: signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert gripper.main('open') == (130 if failure == 'interrupted' else 1)
    if failure == 'unavailable':
        ros1.client.send_goal.assert_not_called()
        ros1.client.cancel_goal.assert_not_called()
    else:
        ros1.client.cancel_goal.assert_called_once()
    ros1.ros.signal_shutdown.assert_called_once()


def test_ros1_unacknowledged_cancellation_reports_cleanup_failure(ros1, capsys):
    ros1.client.send_goal.side_effect = lambda _: signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    ros1.client.wait_for_result.return_value = False
    assert gripper.main('open') == 1
    assert 'Gripper stop could not be verified' in capsys.readouterr().err
    ros1.ros.signal_shutdown.assert_called_once()


@pytest.mark.parametrize('backend,blocked', [('ros1', 'robotiq_mm_ros'), ('ros2', 'rclpy')])
def test_missing_dependency_reports_selected_backend(settings, monkeypatch, capsys, backend, blocked):
    monkeypatch.setattr(Config, 'GTA_BACKEND', backend)
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.split('.')[0] == blocked:
            raise ModuleNotFoundError(blocked)
        # Never fall through to any real ROS imports during this test.
        if backend == 'ros1' and name in ('actionlib', 'actionlib_msgs.msg'):
            return SimpleNamespace(GoalStatus=object)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    assert gripper.main('open') == 1
    assert f'GTA_BACKEND={backend} requires' in capsys.readouterr().err


@pytest.mark.parametrize('name,value', [('ROBOT_TYPE', 'robot_x'), ('GTA_BACKEND', 'invalid')])
def test_invalid_configuration_does_not_initialize_any_backend(settings, monkeypatch, name, value):
    ros1, ros2 = Mock(), Mock()
    monkeypatch.setattr(gripper, '_ros1', ros1)
    monkeypatch.setattr(gripper, '_ros2', ros2)
    monkeypatch.setattr(Config, name, value)
    assert gripper.main('open') == 1
    ros1.assert_not_called()
    ros2.assert_not_called()
