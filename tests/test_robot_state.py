"""Offline state selection and tool-frame regressions; every robot is a fake."""

from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from config import Config
import gta as components
from gta.object_handler import ObjectHandler
import scene.robot as state_api
from scene.robot import BaseRobotState, RobotStateError
from test_robot_address import robot_state


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_exports_contract_without_old_robot_alias():
    assert not hasattr(components, 'RobotState')
    assert state_api.RobotStateError is RobotStateError
    assert set(state_api.__all__) == {'RobotState', 'BaseRobotState', 'RobotStateError', 'RobotHelpers', 'BaseRobotHelpers'}
    assert not hasattr(components, 'Robot')
    assert not (ROOT / 'gta' / 'robot.py').exists()
    assert Path(state_api.__file__).resolve() == ROOT / 'scene/robot/__init__.py'
    with pytest.raises(TypeError):
        BaseRobotState()


@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
@pytest.mark.parametrize('camera_backend', ['file', 'ros1', 'ros2'])
def test_state_selection_is_independent_of_command_and_camera_backends(
    robot_state, monkeypatch, gta_backend, camera_backend,
):
    monkeypatch.setattr(Config, 'GTA_BACKEND', gta_backend)
    monkeypatch.setattr(Config, 'CAMERA_BACKEND', camera_backend)
    reader = state_api.RobotState()
    assert isinstance(reader, BaseRobotState)
    assert isinstance(reader, robot_state.module.RobotState)
    assert state_api.RobotState() is reader
    robot_state.factory.assert_called_once_with(Config.ROBOT_IP)


def test_another_robot_supplies_state_without_connecting_ur5e(robot_state, monkeypatch):
    pose = np.array([0.1, -0.2, 0.3, 0, 0, np.pi / 2])

    class OtherState(BaseRobotState):
        def get_tcp_pose(self):
            return pose.copy()

        def close(self):
            self.closed = True

    module = ModuleType('scene.robot.robot_x.state')
    module.RobotState = OtherState
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    reader = state_api.RobotState()
    assert isinstance(reader, OtherState)
    np.testing.assert_array_equal(reader.get_tcp_pose(), pose)
    obj = ObjectHandler(name='tool', frame='robot', keypoints={'tip': np.array([1., 0, 0])})
    np.testing.assert_allclose(obj.get_k('tip'), [0.1, 0.8, 0.3])
    reader.close()
    assert reader.closed
    robot_state.factory.assert_not_called()


@pytest.mark.parametrize('value', ['', '../ur5e', 'ur5e.ros2', '/tmp/robot', 'robot-x', '1robot', None])
def test_invalid_robot_type_fails_before_import_or_connection(monkeypatch, value):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', value)
    loader = Mock(side_effect=AssertionError('Must not load a hardware backend'))
    monkeypatch.setattr(state_api, 'import_module', loader)
    with pytest.raises(ValueError, match='ROBOT_TYPE'):
        state_api.RobotState()
    loader.assert_not_called()


def test_missing_robot_state_does_not_fall_back_to_ur5e(robot_state, monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'missing_robot')
    with pytest.raises(RuntimeError, match='scene/robot/missing_robot/state.py'):
        state_api.RobotState()
    robot_state.factory.assert_not_called()


def test_missing_backend_dependency_is_not_reported_as_an_unknown_robot(monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    cause = ModuleNotFoundError('missing SDK', name='robot_x_sdk')
    loader = Mock(side_effect=cause)
    monkeypatch.setattr(state_api, 'import_module', loader)
    with pytest.raises(ImportError, match='robot_x_sdk') as error:
        state_api.RobotState()
    assert error.value.__cause__ is cause
    loader.assert_called_once_with('scene.robot.robot_x.state')


@pytest.mark.parametrize('kind', ['missing', 'function', 'unrelated_class', 'abstract'])
def test_backend_must_implement_the_state_contract(monkeypatch, kind):
    module = ModuleType('scene.robot.robot_x.state')
    if kind == 'function':
        module.RobotState = Mock(side_effect=AssertionError('Must not be called'))
    elif kind == 'unrelated_class':
        module.RobotState = type('UnrelatedState', (), {})
    elif kind == 'abstract':
        module.RobotState = BaseRobotState
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    with pytest.raises(RuntimeError, match='concrete RobotState subclass'):
        state_api.RobotState()


def test_offline_imports_and_other_robots_do_not_require_the_ur_sdk():
    probe = '''
import builtins
import sys
from types import ModuleType
original_import = builtins.__import__
def without_hardware(name, *args, **kwargs):
    if name.split('.')[0] == 'rtde_receive':
        raise ModuleNotFoundError('UR SDK unavailable', name='rtde_receive')
    if name.split('.')[0] in {'rtde_control', 'rospy', 'rclpy'}:
        raise AssertionError('Unexpected hardware import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_hardware
from scene.robot import RobotState, BaseRobotState
import gta
from gta.object_handler import ObjectHandler
from config import Config
assert not hasattr(gta, 'RobotState')
assert 'scene.robot.ur5e.state' not in sys.modules
world = ObjectHandler(name='world', keypoints={'origin': [0, 0, 0]})
assert world.get_k('origin') == [0, 0, 0]
Config.ROBOT_TYPE = 'ur5e'
try:
    RobotState()
except ImportError as exc:
    assert 'rtde_receive' in str(exc)
else:
    raise AssertionError('UR state must require its missing SDK')
class OtherState(BaseRobotState):
    def get_tcp_pose(self):
        import numpy as np
        return np.zeros(6)
    def close(self):
        pass
other = ModuleType('scene.robot.robot_x.state')
other.RobotState = OtherState
sys.modules[other.__name__] = other
Config.ROBOT_TYPE = 'robot_x'
assert isinstance(RobotState(), OtherState)
'''
    result = subprocess.run(
        [sys.executable, '-c', probe], cwd=ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('pose', [
    [0., 0, 0, 0, 0, 0],
    [0.1, -0.5, 0.3, np.pi / 2, 0, 0],
    [-0.2, 0.1, 0.4, 0, np.pi / 2, 0],
    [0.3, -0.4, 0.2, 0.2, -0.3, 0.4],
])
def test_ur5e_tool_offset_and_axes_keep_the_existing_transform(robot_state, pose):
    robot_state.receive.getActualTCPPose.return_value = pose
    reader = state_api.RobotState()
    # The state reader must return the RTDE pose itself, never a tool-offset pose.
    np.testing.assert_array_equal(reader.get_tcp_pose(), pose)
    offset = np.array([0., 0, 0.1628])
    axes = {'x': np.array([0., -1, 0]), 'y': np.array([-1., 0, 0]), 'z': np.array([0., 0, 1])}
    eef = ObjectHandler(name='eef', frame='robot', keypoints={'TCP': offset.copy()},
                        axes={name: value.copy() for name, value in axes.items()})
    rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    np.testing.assert_allclose(eef.get_k('TCP'), np.asarray(pose[:3]) + rotation @ offset, atol=1e-14)
    for name, axis in axes.items():
        np.testing.assert_allclose(eef.get_a(name), rotation @ axis, atol=1e-14)
        np.testing.assert_array_equal(eef.axes[name], axis)
    np.testing.assert_array_equal(eef.keypoints['TCP'], offset)

    # Further queries still read live state rather than caching world positions.
    next_pose = np.asarray(pose, dtype=float) + np.array([0.1, 0.2, 0.3, 0, 0, 0])
    robot_state.receive.getActualTCPPose.return_value = next_pose
    np.testing.assert_allclose(eef.get_k('TCP'), next_pose[:3] + rotation @ offset, atol=1e-14)
    robot_state.factory.assert_called_once_with(Config.ROBOT_IP)


def test_object_attachment_and_detachment_keep_the_existing_frame_math(robot_state):
    pose = np.array([0.1, -0.2, 0.3, 0.2, 0.1, -0.4])
    robot_state.receive.getActualTCPPose.return_value = pose
    rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
    point, axis = np.array([0.4, -0.6, 0.9]), np.array([0., 0, 1])
    obj = ObjectHandler(name='object', frame='world', keypoints={'tip': point.copy()},
                        axes={'z': axis.copy()})
    local_point = rotation.T @ (point - pose[:3])
    local_axis = rotation.T @ axis
    obj.set_frame('robot')
    np.testing.assert_allclose(obj.keypoints['tip'], local_point, atol=1e-14)
    np.testing.assert_allclose(obj.axes['z'], local_axis, atol=1e-14)
    np.testing.assert_allclose(obj.get_k('tip'), point, atol=1e-14)
    np.testing.assert_allclose(obj.get_a('z'), axis, atol=1e-14)

    next_pose = np.array([0.5, -0.1, 0.2, -0.3, 0.2, 0.1])
    robot_state.receive.getActualTCPPose.return_value = next_pose
    next_rotation = Rotation.from_rotvec(next_pose[3:]).as_matrix()
    expected_point = next_pose[:3] + next_rotation @ local_point
    expected_axis = next_rotation @ local_axis
    obj.set_frame('world')
    robot_state.receive.getActualTCPPose.side_effect = AssertionError('World objects must not read RTDE')
    np.testing.assert_allclose(obj.get_k('tip'), expected_point, atol=1e-14)
    np.testing.assert_allclose(obj.get_a('z'), expected_axis, atol=1e-14)
    robot_state.factory.assert_called_once_with(Config.ROBOT_IP)


@pytest.mark.parametrize('disconnect_failure', [False, True])
def test_public_close_preserves_receive_cleanup_without_reconnecting(robot_state, disconnect_failure):
    reader = state_api.RobotState()
    if disconnect_failure:
        robot_state.receive.disconnect.side_effect = RuntimeError('disconnected')
    assert reader.close() is None
    robot_state.receive.disconnect.assert_called_once_with()
    robot_state.clock.sleep.assert_not_called()
    robot_state.factory.assert_called_once_with(Config.ROBOT_IP)
    assert not hasattr(reader, '_disconnect')
    assert state_api.RobotState() is reader
