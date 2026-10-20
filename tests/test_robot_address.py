"""Robot-address regression tests; every ROS and RTDE interface is fake."""

import importlib.util
import inspect
from pathlib import Path
import runpy
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from config import Config
import scene.robot as state_api


ROOT = Path(__file__).resolve().parents[1]
TEST_IP = "192.0.2.42"


def load_isolated_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def robot_state(monkeypatch):
    monkeypatch.setattr(Config, "ROBOT_IP", TEST_IP)
    monkeypatch.setattr(Config, "ROBOT_TYPE", "ur5e")
    receive = Mock()
    receive.isConnected.return_value = True
    receive.getTimestamp.return_value = 1.0
    receive.getActualTCPPose.return_value = [0.1, -0.2, 0.3, 0, 0, 0]
    factory = Mock(return_value=receive)
    fake_rtde = SimpleNamespace(RTDEReceiveInterface=factory)
    with patch.dict(sys.modules, {"rtde_receive": fake_rtde}):
        module = load_isolated_module(
            "scene.robot.ur5e._ur5e_address_test",
            "scene/robot/ur5e/state.py",
        )
        monkeypatch.setitem(sys.modules, "scene.robot.ur5e.state", module)
        clock = SimpleNamespace(time=Mock(return_value=100.0), sleep=Mock())
        monkeypatch.setattr(module, "time", clock)
        yield SimpleNamespace(module=module, factory=factory, receive=receive, clock=clock)


@pytest.mark.parametrize("address", [None, TEST_IP, "192.0.2.99"])
def test_robot_ip_configuration_is_read_from_environment(monkeypatch, address):
    if address is None:
        monkeypatch.delenv("ROBOT_IP", raising=False)
    else:
        monkeypatch.setenv("ROBOT_IP", address)
    with patch("dotenv.load_dotenv"):
        config = runpy.run_path(str(ROOT / "config.py"))["Config"]
    assert config.ROBOT_IP == (address if address is not None else "192.168.1.10")


def test_robot_uses_config_at_first_construction_not_import_time(robot_state, monkeypatch):
    robot_state.factory.assert_not_called()
    monkeypatch.setattr(Config, "ROBOT_IP", "192.0.2.99")
    robot = state_api.RobotState()
    robot_state.factory.assert_called_once_with("192.0.2.99")
    assert robot.robot_ip == Config.ROBOT_IP
    assert not inspect.signature(state_api.RobotState).parameters
    np.testing.assert_array_equal(robot.get_tcp_pose(), [0.1, -0.2, 0.3, 0, 0, 0])


@pytest.mark.parametrize("style", ["positional", "keyword", "none"])
def test_robot_rejects_address_overrides_without_connecting(robot_state, style):
    with pytest.raises(TypeError):
        if style == "positional":
            state_api.RobotState("192.0.2.99")
        elif style == "keyword":
            state_api.RobotState(robot_ip="192.0.2.99")
        else:
            state_api.RobotState(robot_ip=None)
    robot_state.factory.assert_not_called()
    assert robot_state.module.RobotState._instance is None


def test_existing_singleton_is_not_silently_retargeted(robot_state, monkeypatch):
    robot = state_api.RobotState()
    monkeypatch.setattr(Config, "ROBOT_IP", "192.0.2.99")
    assert state_api.RobotState() is robot
    assert robot.robot_ip == TEST_IP
    robot_state.factory.assert_called_once_with(TEST_IP)


def test_receive_reconnect_keeps_configured_address_and_existing_delay(robot_state):
    robot = state_api.RobotState()
    robot.get_tcp_pose()
    robot.reconnect()
    assert robot_state.factory.call_args_list == [call(TEST_IP), call(TEST_IP)]
    robot_state.receive.disconnect.assert_called_once_with()
    robot_state.clock.sleep.assert_called_once_with(0.25)
    assert robot._last_timestamp is None
    assert robot._last_timestamp_wall_time is None


def test_disconnected_receive_retries_using_the_same_address(robot_state):
    replacement = Mock()
    replacement.isConnected.return_value = True
    replacement.getTimestamp.return_value = 2.0
    replacement.getActualTCPPose.return_value = [0] * 6
    robot_state.factory.side_effect = [robot_state.receive, replacement]
    robot_state.receive.isConnected.return_value = False
    robot = state_api.RobotState()
    np.testing.assert_array_equal(robot.get_tcp_pose(), np.zeros(6))
    assert robot_state.factory.call_args_list == [call(TEST_IP), call(TEST_IP)]
    robot_state.receive.disconnect.assert_called_once_with()
    robot_state.clock.sleep.assert_called_once_with(0.25)


@pytest.mark.parametrize("pose", [[0] * 5, [0, 0, 0, 0, 0, float("nan")]])
def test_pose_validation_and_retry_behavior_are_preserved(robot_state, pose):
    robot_state.receive.getActualTCPPose.return_value = pose
    with pytest.raises(robot_state.module.RobotStateError, match="Unable to read fresh TCP pose"):
        state_api.RobotState().get_tcp_pose()
    assert robot_state.factory.call_args_list == [call(TEST_IP), call(TEST_IP)]


def test_stale_pose_protection_is_preserved(robot_state):
    robot = state_api.RobotState()
    robot_state.clock.time.side_effect = [100.0, 100.3]
    robot._check_pose_freshness()
    with pytest.raises(robot_state.module.RobotStateError, match="timestamp has not advanced"):
        robot._check_pose_freshness()
    assert robot.max_pose_stale_s == 0.25


def test_object_handler_default_robot_reads_from_configured_address(robot_state):
    objects = load_isolated_module("robot_address_objects", "gta/object_handler.py")
    obj = objects.ObjectHandler(
        name="eef", frame="robot", keypoints={"TCP": np.zeros(3)},
        axes={"z": np.array([0., 0, 1])},
    )
    np.testing.assert_allclose(obj.get_k("TCP"), [0.1, -0.2, 0.3])
    np.testing.assert_allclose(obj.get_a("z"), [0, 0, 1])
    robot_state.factory.assert_called_once_with(TEST_IP)


@pytest.fixture(params=["ros1", "ros2"])
def server_backend(request, monkeypatch, robot_state):
    # Isolate imports even on a laptop that has real ROS and RTDE installed.
    names = ("actionlib_msgs.msg", "rospy", "std_msgs", "std_msgs.msg", "robotiq_mm_ros",
             "robotiq_mm_ros.msg", "actionlib", "rtde_control")
    modules = {name: ModuleType(name) for name in names}
    modules['actionlib_msgs.msg'].GoalStatus = SimpleNamespace(SUCCEEDED=3)
    modules["rospy"].init_node = Mock()
    modules["rospy"].Subscriber = Mock()
    messages = modules["std_msgs.msg"]
    messages.String = messages.Float32 = messages.Float32MultiArray = object
    modules["std_msgs"].msg = messages
    modules["robotiq_mm_ros"].msg = modules["robotiq_mm_ros.msg"]
    modules["robotiq_mm_ros.msg"].GripperCommandAction = object
    modules["robotiq_mm_ros.msg"].GripperCommandGoal = object
    modules["actionlib"].SimpleActionClient = Mock()
    control = Mock()
    control_factory = Mock(return_value=control)
    modules["rtde_control"].RTDEControlInterface = control_factory
    modules["rclpy"] = None
    robot_factory = Mock(side_effect=state_api.RobotState)
    modules["scene.robot"] = SimpleNamespace(RobotState=robot_factory)
    monkeypatch.setattr(sys, "path", sys.path[:])
    with patch.dict(sys.modules, modules):
        backend = load_isolated_module(
            f"robot_address_{request.param}", f"scene/robot/ur5e/{request.param}.py",
        )
        backend.signal = SimpleNamespace(
            SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM, SIG_IGN=signal.SIG_IGN,
            getsignal=Mock(return_value=signal.SIG_DFL), signal=Mock(),
        )
        sleeper = Mock()
        backend.time = SimpleNamespace(time=Mock(side_effect=[0.0, 1001.0]), sleep=sleeper)
        if request.param == "ros2":
            backend.ROS2_AVAILABLE = True
            backend.rclpy = Mock()
            backend.rclpy.ok.return_value = False
            backend.get_runtime = lambda: None
            backend.interruptible_sleep = sleeper
            node = Mock()
            node.close.return_value = []
            backend.UR5eROS2Server = Mock(return_value=node)
        yield SimpleNamespace(
            module=backend, control=control, factory=control_factory,
            robot_factory=robot_factory, state=robot_state, sleeper=sleeper,
        )


def test_both_servers_start_control_and_receive_at_the_configured_address(server_backend):
    server_backend.module.main()
    server_backend.factory.assert_called_once_with(TEST_IP)
    server_backend.state.factory.assert_called_once_with(TEST_IP)
    server_backend.robot_factory.assert_called_once_with()
    server_backend.control.servoL.assert_not_called()
    server_backend.control.forceMode.assert_not_called()


def test_both_servers_reconnect_control_using_config_without_address_argument(server_backend):
    old = Mock()
    replacement = server_backend.module.restart_rtde_control(old)
    assert replacement is server_backend.control
    server_backend.factory.assert_called_once_with(TEST_IP)
    assert old.method_calls == [call.servoStop(), call.stopScript(), call.disconnect()]
    assert server_backend.sleeper.call_args_list == [call(0.75), call(0.35)]
    server_backend.state.factory.assert_not_called()
    signature = inspect.signature(server_backend.module.restart_rtde_control)
    assert list(signature.parameters) == ["rtde_c", "sleep_s"]
