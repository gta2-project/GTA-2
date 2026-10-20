"""Offline layout and launch checks; never initialize ROS or hardware."""

import importlib
from pathlib import Path
import runpy
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from config import Config


ROOT = Path(__file__).resolve().parents[1]


def launch():
    namespace = runpy.run_path(str(ROOT / 'scene/robot/server.py'))
    namespace['main']([])
    return namespace


def fake_backend(name):
    module = ModuleType(name)
    for attribute in ('main', 'get_safe_target_position', 'get_safe_target_rotvec',
                      'restart_rtde_control', '_safe_call'):
        setattr(module, attribute, Mock(name=attribute))
    return module


def test_robot_server_contains_only_launcher_and_robot_backends():
    server_dir = ROOT / 'scene' / 'robot'
    assert {path.name for path in server_dir.iterdir() if path.name != '__pycache__'} == {
        '__init__.py', 'base.py', 'server.py', 'ur5e',
    }
    assert {path.name for path in (server_dir / 'ur5e').iterdir() if path.name != '__pycache__'} == {
        '__init__.py', 'state.py', 'helpers.py', 'ros1.py', 'ros2.py',
    }


@pytest.mark.parametrize('camera_backend', ['file', 'ros1', 'ros2'])
@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
def test_current_launcher_preserves_selection_and_calls_backend_main(gta_backend, camera_backend, tmp_path, monkeypatch):
    ros1 = fake_backend('scene.robot.ur5e.ros1')
    ros2 = fake_backend('scene.robot.ur5e.ros2')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', [path for path in sys.path if path and path != str(ROOT)])
    monkeypatch.setattr(Config, 'GTA_BACKEND', gta_backend)
    monkeypatch.setattr(Config, 'CAMERA_BACKEND', camera_backend)
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'ur5e')
    with patch.dict(sys.modules, {ros1.__name__: ros1, ros2.__name__: ros2}):
        namespace = launch()
    selected, other = (ros2, ros1) if gta_backend == 'ros2' else (ros1, ros2)
    selected.main.assert_called_once_with()
    other.main.assert_not_called()
    for name in ('get_safe_target_position', 'get_safe_target_rotvec', 'restart_rtde_control'):
        assert name not in namespace
        assert callable(getattr(selected, name))
    assert sys.path[0] == str(ROOT)


@pytest.mark.parametrize('value,expected', [(None, 'ur5e'), ('ur5e', 'ur5e'), (' UR5E ', 'ur5e'), ('robot_x', 'robot_x')])
def test_robot_type_configuration_default_and_normalization(value, expected, monkeypatch):
    if value is None:
        monkeypatch.delenv('ROBOT_TYPE', raising=False)
    else:
        monkeypatch.setenv('ROBOT_TYPE', value)
    # Do not let the developer's .env override this isolated config check.
    with patch('dotenv.load_dotenv'):
        namespace = runpy.run_path(str(ROOT / 'config.py'))
    assert namespace['Config'].ROBOT_TYPE == expected


@pytest.mark.parametrize('source,gta_backend', [('ros2', 'ros2'), ('ros1', 'ros1')])
def test_robot_selector_launches_other_backend_requiring_only_main(source, gta_backend, monkeypatch):
    name = f'scene.robot.robot_x.{gta_backend}'
    backend = ModuleType(name)
    backend.main = Mock()
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    monkeypatch.setattr(Config, 'GTA_BACKEND', source)
    with patch.dict(sys.modules, {name: backend}):
        namespace = launch()
    backend.main.assert_called_once_with()
    assert callable(namespace['main'])
    assert 'restart_rtde_control' not in namespace


@pytest.mark.parametrize('robot_type', ['', '../ur5e', 'ur5e.ros2', '/tmp/robot', 'robot-x', '1robot'])
def test_invalid_robot_selector_is_rejected_before_import(robot_type, monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', robot_type)
    with patch('importlib.import_module') as load:
        with pytest.raises(ValueError, match='ROBOT_TYPE must be a robot directory name'):
            launch()
    load.assert_not_called()


def test_invalid_gta_backend_is_rejected_before_import(monkeypatch):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'local')
    with patch('importlib.import_module') as load:
        with pytest.raises(ValueError, match='GTA_BACKEND must be ros1 or ros2'):
            launch()
    load.assert_not_called()


def test_missing_robot_is_reported_without_ur5e_fallback(monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'missing_robot')
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    ur5e = fake_backend('scene.robot.ur5e.ros2')
    with patch.dict(sys.modules, {ur5e.__name__: ur5e}):
        with pytest.raises(RuntimeError, match='Expected scene/robot/missing_robot/ros2.py'):
            launch()
    ur5e.main.assert_not_called()


def test_missing_transport_is_reported_without_switching_to_other_transport(monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    package = ModuleType('scene.robot.robot_x')
    package.__path__ = []
    ros1 = fake_backend('scene.robot.robot_x.ros1')
    with patch.dict(sys.modules, {package.__name__: package, ros1.__name__: ros1}):
        with pytest.raises(RuntimeError, match='Expected scene/robot/robot_x/ros2.py'):
            launch()
    ros1.main.assert_not_called()


def test_missing_backend_dependency_is_not_misreported_as_missing_robot(monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    error = ModuleNotFoundError("No module named 'robot_x_sdk'", name='robot_x_sdk')
    with patch('importlib.import_module', side_effect=error) as load:
        with pytest.raises(ModuleNotFoundError) as raised:
            launch()
    assert raised.value is error
    load.assert_called_once_with('scene.robot.robot_x.ros2')


@pytest.mark.parametrize('main', [None, 'not callable'])
def test_backend_must_define_callable_main(main, monkeypatch):
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    backend = ModuleType('scene.robot.robot_x.ros2')
    if main is not None:
        backend.main = main
    with patch.dict(sys.modules, {backend.__name__: backend}):
        with pytest.raises(RuntimeError, match='must define a callable main'):
            launch()


def test_reloading_selector_switches_backend_without_starting_hardware(monkeypatch):
    module_name = 'scene.robot.server'
    ur5e = fake_backend('scene.robot.ur5e.ros2')
    robot_x = ModuleType('scene.robot.robot_x.ros2')
    robot_x.main = Mock()
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    monkeypatch.setattr(Config, 'ROBOT_TYPE', 'ur5e')
    with patch.dict(sys.modules, {ur5e.__name__: ur5e, robot_x.__name__: robot_x}):
        sys.modules.pop(module_name, None)
        selected = importlib.import_module(module_name)
        assert not hasattr(selected, '_backend')
        assert not hasattr(selected, 'restart_rtde_control')
        monkeypatch.setattr(Config, 'ROBOT_TYPE', 'robot_x')
        selected = importlib.reload(selected)
        assert not hasattr(selected, '_backend')
        assert selected._load_backend('robot_x', 'ros2') is robot_x
        assert not hasattr(selected, 'restart_rtde_control')
        ur5e.main.assert_not_called()
        robot_x.main.assert_not_called()


@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
def test_backend_file_import_from_another_directory_is_hardware_free(gta_backend, tmp_path, monkeypatch):
    # Stub hardware packages even when the developer's machine has ROS installed.
    names = ['rospy', 'actionlib', 'std_msgs', 'std_msgs.msg', 'robotiq_mm_ros',
             'robotiq_mm_ros.msg', 'rtde_control', 'gta',
             'scene.robot',
             'gta.managed_run']
    modules = {name: ModuleType(name) for name in names}
    messages = modules['std_msgs.msg']
    messages.String = messages.Float32 = messages.Float32MultiArray = object
    modules['robotiq_mm_ros'].msg = modules['robotiq_mm_ros.msg']
    blocked = Mock(side_effect=AssertionError('No hardware initialization is allowed'))
    modules['rospy'].init_node = blocked
    modules['rtde_control'].RTDEControlInterface = blocked
    modules['scene.robot'].RobotState = blocked
    modules['gta.managed_run'].get_runtime = blocked
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', [path for path in sys.path if path and path != str(ROOT)])
    with patch.dict(sys.modules, {**modules, 'rclpy': None}):
        namespace = runpy.run_path(str(ROOT / 'scene/robot/ur5e' / f'{gta_backend}.py'))
    assert callable(namespace['main'])
    assert sys.path[0] == str(ROOT)
    blocked.assert_not_called()
