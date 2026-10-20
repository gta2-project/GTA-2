"""Device extension contracts and launch guards. No hardware/provider IO."""

import builtins
import importlib
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from config import Config
from scene import robot, gripper
from scene.robot import server
from test_robot_helpers import runtime
from test_ur5e_gripper_scripts import ros2, settings, module


ROOT = Path(__file__).resolve().parents[1]


class OtherHelpers(robot.BaseRobotHelpers):
    calls = []

    def reset_joints(self, *, check_cancel, on_started, errors, wait):
        check_cancel()
        on_started()
        self.calls.append('reset')


class OtherGripper(gripper.BaseGripper):
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.calls = []

    def wait_for_server(self, timeout_s=None):
        self.calls.append('ready')

    def open(self, timeout_s=None, **kwargs):
        self.calls.append('open')

    def close(self, timeout_s=None, **kwargs):
        self.calls.append('close')

    def cancel(self, **kwargs):
        self.calls.append('cancel')
        return []

    def shutdown(self):
        self.calls.append('shutdown')
        return []


@pytest.mark.parametrize('backend', ['ros1', 'ros2'])
def test_robot_and_gripper_are_independently_selected(monkeypatch, backend):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    monkeypatch.setattr(Config, 'GRIPPER_TYPE', 'gripper_y')
    monkeypatch.setattr(Config, 'GTA_BACKEND', backend)
    helpers = ModuleType('scene.robot.robot_x.helpers')
    helpers.Helpers = OtherHelpers
    device = ModuleType(f'scene.gripper.gripper_y.{backend}')
    device.Gripper = OtherGripper
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setitem(sys.modules, device.__name__, device)
    imported = []
    def only_selected(name):
        imported.append(name)
        assert name in {helpers.__name__, device.__name__}
        return sys.modules[name]
    monkeypatch.setattr(robot, 'import_module', only_selected)
    monkeypatch.setattr(gripper, 'import_module', only_selected)
    assert isinstance(robot.RobotHelpers(), OtherHelpers)
    client = gripper.Gripper(node='borrowed')
    assert isinstance(client, OtherGripper)
    assert client.settings == {'node': 'borrowed'}
    client.open(); client.close()
    assert client.calls == ['open', 'close']
    with pytest.raises(NotImplementedError, match='half_open'):
        client.half_open()
    assert imported == [helpers.__name__, device.__name__]


@pytest.mark.parametrize('contract', [robot.BaseRobotHelpers, gripper.BaseGripper])
def test_contract_is_abstract(contract):
    with pytest.raises(TypeError):
        contract()


@pytest.mark.parametrize('target,setting', [(robot, 'ROBOT_TYPE'), (gripper, 'GRIPPER_TYPE')])
@pytest.mark.parametrize('value', [None, '', '../ur5e', 'bad.name', '1robot', '/tmp/robot'])
def test_invalid_device_identifier_never_imports_backend(monkeypatch, target, setting, value):
    monkeypatch.setattr(Config, setting, value)
    loader = Mock(side_effect=AssertionError('Unexpected backend import'))
    monkeypatch.setattr(target, 'import_module', loader)
    factory = target.RobotHelpers if target is robot else target.Gripper
    with pytest.raises(ValueError, match=setting):
        factory()
    loader.assert_not_called()


@pytest.mark.parametrize('target,setting', [(robot, 'ROBOT_TYPE'), (gripper, 'GRIPPER_TYPE')])
def test_missing_device_is_not_a_fallback(monkeypatch, target, setting):
    monkeypatch.setattr(Config, setting, 'unavailable_device')
    factory = target.RobotHelpers if target is robot else target.Gripper
    with pytest.raises(RuntimeError, match='no fallback'):
        factory()


@pytest.mark.parametrize('target', [robot, gripper])
@pytest.mark.parametrize('kind', ['absent', 'function', 'unrelated', 'abstract'])
def test_device_backend_must_implement_its_contract(monkeypatch, target, kind):
    module = ModuleType('example')
    name = 'Helpers' if target is robot else 'Gripper'
    contract = robot.BaseRobotHelpers if target is robot else gripper.BaseGripper
    if kind != 'absent':
        setattr(module, name, {'function': lambda: None, 'unrelated': object, 'abstract': contract}[kind])
    monkeypatch.setattr(target, 'import_module', lambda _: module)
    factory = target.RobotHelpers if target is robot else target.Gripper
    with pytest.raises(RuntimeError, match='concrete'):
        factory()


@pytest.mark.parametrize('target', [robot, gripper])
def test_dependency_failure_identifies_dependency(monkeypatch, target):
    error = ModuleNotFoundError('missing sdk', name='custom_sdk')
    monkeypatch.setattr(target, 'import_module', Mock(side_effect=error))
    factory = target.RobotHelpers if target is robot else target.Gripper
    with pytest.raises(ImportError, match='custom_sdk'):
        factory()


@pytest.mark.parametrize('value,expected', [(None, 'robotiq'), (' ROBOTIQ ', 'robotiq'), ('gripper_y', 'gripper_y')])
def test_gripper_type_config_default_and_normalization(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('GRIPPER_TYPE', raising=False)
    else:
        monkeypatch.setenv('GRIPPER_TYPE', value)
    with patch('dotenv.load_dotenv'):
        settings = runpy.run_path(str(ROOT / 'config.py'))['Config']
    assert settings.GRIPPER_TYPE == expected


@pytest.mark.parametrize('operation', ['reset_joints', 'open_gripper', 'close_gripper'])
@pytest.mark.parametrize('role', [None, 'script', 'server'])
def test_operation_rejects_missing_or_wrong_supervisor_before_device_import(monkeypatch, operation, role):
    monkeypatch.setattr(server, 'get_runtime', lambda: None if role is None else SimpleNamespace(role=role))
    with patch.object(robot, 'RobotHelpers') as helper, patch.object(gripper, 'Gripper') as client:
        with pytest.raises(RuntimeError, match='GUI supervisor'):
            server.main(['--operation', operation])
    helper.assert_not_called(); client.assert_not_called()


def test_ros1_supervised_operations_remain_unsupported(monkeypatch, runtime):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros1')
    monkeypatch.setattr(server, 'get_runtime', lambda: runtime)
    with patch.object(robot, 'RobotHelpers') as helper:
        with pytest.raises(RuntimeError, match='GTA_BACKEND=ros2'):
            server.main(['--operation', 'reset_joints'])
    helper.assert_not_called()


def test_other_robot_reset_uses_callbacks_not_gui_events(monkeypatch, runtime):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    helpers = ModuleType('scene.robot.robot_x.helpers')
    helpers.Helpers = OtherHelpers
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setattr(server, 'get_runtime', lambda: runtime)
    assert server.main(['--operation', 'reset_joints']) == 0
    assert [call.args[0] for call in runtime.emit.call_args_list] == [
        'manual_started', 'manual_completed', 'manual_cleaned']


@pytest.mark.parametrize('mode', ['import', 'file_help', 'module_help'])
def test_public_imports_and_help_are_hardware_free(mode):
    # Pure Python child, not a server process: all hardware imports are blocked.
    probe = '''
import builtins, runpy, sys
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'rclpy', 'rospy', 'rtde_control', 'rtde_receive',
                             'actionlib', 'robotiq_mm_ros', 'control_msgs'}:
        raise AssertionError('Unexpected hardware import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from scene.robot import RobotState, RobotHelpers
from scene.gripper import Gripper
import scene.robot.server
assert 'scene.robot.ur5e.state' not in sys.modules
assert 'scene.gripper.robotiq.ros2' not in sys.modules
'''
    if mode != 'import':
        probe += "sys.argv = ['server.py', '--help']\n"
        probe += ("runpy.run_path('scene/robot/server.py', run_name='__main__')\n" if mode == 'file_help'
                  else "runpy.run_module('scene.robot.server', run_name='__main__')\n")
    result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_old_source_entrypoints_are_removed():
    for path in ('robot_server/server.py', 'robot_script_components/robot_state/__init__.py',
                 'scripts/robot_manual_control.py'):
        assert not (ROOT / path).exists()


def test_gripper_contract_makes_no_arm_or_command_subscription_calls():
    for path in (ROOT / 'scene/gripper').rglob('*.py'):
        source = path.read_text()
        assert 'import rtde_' not in source
        assert 'from rtde_' not in source
        assert 'create_subscription(' not in source


@pytest.mark.parametrize('failure', ['missing_gripper', 'action_client', 'subscription'])
def test_ros2_server_constructor_failure_destroys_node_before_rtde(monkeypatch, ros2, runtime, failure):
    nodes = []

    class Node:
        def __init__(self, name):
            self.destroy_node = Mock()
            self.create_subscription = Mock()
            if failure == 'subscription':
                self.create_subscription.side_effect = RuntimeError('subscription creation failed')
            nodes.append(self)

    module(monkeypatch, 'rclpy.node', Node=Node)
    module(monkeypatch, 'rclpy.qos',
           DurabilityPolicy=SimpleNamespace(VOLATILE=1), HistoryPolicy=SimpleNamespace(KEEP_LAST=1),
           ReliabilityPolicy=SimpleNamespace(RELIABLE=1), QoSProfile=lambda **kw: kw)
    module(monkeypatch, 'std_msgs')
    module(monkeypatch, 'std_msgs.msg', Float32=object, Float32MultiArray=object, String=object)
    control = Mock(side_effect=AssertionError('RTDE must not initialize'))
    module(monkeypatch, 'rtde_control', RTDEControlInterface=control)
    monkeypatch.setattr(ros2.ros, 'ok', lambda: True, raising=False)
    if failure == 'missing_gripper':
        monkeypatch.setattr(Config, 'GRIPPER_TYPE', 'unknown_gripper')
    else:
        ros2.client_factory.side_effect = RuntimeError('action client could not initialize')
    namespace = runpy.run_path(str(ROOT / 'scene/robot/ur5e/ros2.py'))
    namespace['main'].__globals__['get_runtime'] = lambda: runtime
    with pytest.raises(RuntimeError):
        namespace['main']()
    assert len(nodes) == 1
    nodes[0].destroy_node.assert_called_once()
    control.assert_not_called()
    ros2.ros.shutdown.assert_called_once()
    events = [call.args[0] for call in runtime.emit.call_args_list]
    assert 'server_failed' in events and 'server_cleaned' in events
    assert 'server_ready' not in events


def test_server_stops_arm_before_gripper_cancellation_and_releases_every_owner():
    from test_ur5e_server_ros2 import make_server
    node = make_server()
    order = []
    node.rtde_c = Mock()
    node.robot = Mock()
    node.gripper = Mock()
    for name in ('forceModeStop', 'servoStop', 'stopScript', 'disconnect'):
        getattr(node.rtde_c, name).side_effect = lambda name=name: order.append(name)
    node.robot.close.side_effect = lambda: order.append('receive_close')
    node.gripper.cancel.side_effect = lambda: order.append('gripper_cancel') or []
    node.gripper.shutdown.side_effect = lambda: order.append('gripper_shutdown') or []
    assert node.close() == []
    assert order == ['forceModeStop', 'servoStop', 'gripper_cancel', 'stopScript',
                     'disconnect', 'receive_close', 'gripper_shutdown']


@pytest.mark.parametrize('operation', ['open', 'close'])
def test_ros1_server_policy_stays_unbounded_and_does_not_validate_status(monkeypatch, operation):
    from scene.gripper.robotiq.ros1 import Gripper
    client = object.__new__(Gripper)
    client._goal_type = SimpleNamespace
    client._check_cancel = lambda: None
    client.client = Mock()
    getattr(client, operation)(require_success=False)
    client.client.wait_for_result.assert_called_once_with()
    client.client.get_state.assert_not_called()
    client.client.get_result.assert_not_called()
    assert client._outstanding is False


@pytest.mark.parametrize('backend', ['', 'file', 'ROS2'])
def test_invalid_gripper_backend_does_not_fall_back(monkeypatch, backend):
    with patch.object(gripper, 'import_module') as load:
        with pytest.raises(ValueError, match='GTA_BACKEND'):
            gripper.Gripper(gta_backend=backend)
    load.assert_not_called()
