"""Exercise the exact startup probe with fake ROS modules; never start hardware."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from api.execution import check_server_conflicts


TOPICS = [
    '/robot/position_error', '/robot/rotation_error', '/robot/force_command',
    '/robot/gripper_command', '/robot/skill_status',
]


@pytest.fixture
def probe():
    # Capture the real subprocess source while bypassing all real processes/ROS.
    with patch.object(Path, 'iterdir', return_value=iter([])), \
         patch('api.execution.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
        check_server_conflicts()
    args = run.call_args.args[0]
    assert args[:2] == [sys.executable, '-c']
    assert run.call_args.kwargs == {'capture_output': True, 'text': True, 'timeout': 8}
    return compile(args[2], '<managed-startup-probe>', 'exec')


@pytest.fixture
def ros(monkeypatch):
    cleanup = []
    context = object()
    executor = Mock()
    executor.shutdown.side_effect = lambda: cleanup.append('executor')
    node = Mock()
    node.destroy_node.side_effect = lambda: cleanup.append('node')
    node.get_subscriptions_info_by_topic.return_value = []

    rclpy = ModuleType('rclpy')
    contexts = ModuleType('rclpy.context')
    executors = ModuleType('rclpy.executors')
    contexts.Context = Mock(return_value=context)
    executors.SingleThreadedExecutor = Mock(return_value=executor)
    rclpy.init = Mock()
    rclpy.create_node = Mock(return_value=node)
    rclpy.shutdown = Mock(side_effect=lambda **kwargs: cleanup.append('context'))

    def spin(actual_node, *, executor=None, timeout_sec=None):
        assert actual_node is node
        assert executor is ros_executor, 'Probe must not use the default ROS executor'
        assert timeout_sec == 0.05
        executors.SingleThreadedExecutor.assert_called_once_with(context=context)

    ros_executor = executor
    rclpy.spin_once = Mock(side_effect=spin)
    clock = ModuleType('time')
    clock.monotonic = Mock(side_effect=[0.0, 0.0, 2.0])
    for name, module in {'rclpy': rclpy, 'rclpy.context': contexts,
                         'rclpy.executors': executors, 'time': clock}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(rclpy=rclpy, contexts=contexts, executors=executors,
                           context=context, executor=executor, node=node, cleanup=cleanup)


def test_probe_spins_using_its_initialized_context_and_checks_all_topics(probe, ros):
    exec(probe, {})
    ros.rclpy.init.assert_called_once_with(context=ros.context)
    ros.rclpy.create_node.assert_called_once_with('gta_execution_probe', context=ros.context)
    ros.rclpy.spin_once.assert_called_once_with(ros.node, executor=ros.executor, timeout_sec=0.05)
    assert [call.args[0] for call in ros.node.get_subscriptions_info_by_topic.call_args_list] == TOPICS
    assert ros.cleanup == ['executor', 'node', 'context']
    ros.rclpy.shutdown.assert_called_once_with(context=ros.context)


def test_probe_still_rejects_existing_robot_subscriptions(probe, ros):
    ros.node.get_subscriptions_info_by_topic.return_value = [
        SimpleNamespace(node_namespace='/external', node_name='ur5e')]
    with pytest.raises(RuntimeError, match='Robot command subscriptions already exist: /external/ur5e'):
        exec(probe, {})
    assert ros.cleanup == ['executor', 'node', 'context']


@pytest.mark.parametrize('stage,cleanup', [
    ('node', ['context']),
    ('executor', ['node', 'context']),
    ('spin', ['executor', 'node', 'context']),
    ('graph', ['executor', 'node', 'context']),
])
def test_probe_cleans_up_after_partial_startup_or_discovery_failure(probe, ros, stage, cleanup):
    operation = {'node': ros.rclpy.create_node,
                 'executor': ros.executors.SingleThreadedExecutor,
                 'spin': ros.rclpy.spin_once,
                 'graph': ros.node.get_subscriptions_info_by_topic}[stage]
    operation.side_effect = RuntimeError('injected failure')
    with pytest.raises(RuntimeError, match='injected failure'):
        exec(probe, {})
    assert ros.cleanup == cleanup


@pytest.mark.parametrize('stage', ['executor', 'node'])
def test_probe_context_cleanup_survives_other_cleanup_failures(probe, ros, stage):
    def fail():
        ros.cleanup.append(stage)
        raise RuntimeError('cleanup failed')
    operation = ros.executor.shutdown if stage == 'executor' else ros.node.destroy_node
    operation.side_effect = fail
    with pytest.raises(RuntimeError, match='cleanup failed'):
        exec(probe, {})
    assert ros.cleanup == ['executor', 'node', 'context']


def test_probe_failure_is_reported_to_supervisor():
    failure = subprocess.CompletedProcess([], 1, '', 'ROS graph check failed')
    with patch.object(Path, 'iterdir', return_value=iter([])), \
         patch('api.execution.subprocess.run', return_value=failure):
        with pytest.raises(RuntimeError, match='Cannot start a managed server: ROS graph check failed'):
            check_server_conflicts()


@pytest.mark.parametrize('command,cwd', [
    (['python', '/repo/scene/robot/server.py'], '/tmp'),
    (['python', '/repo/scene/robot/server.py', '--operation', 'open_gripper'], '/tmp'),
    (['python', 'scene/robot/server.py'], '/repo'),
    (['python', 'server.py', '--operation', 'reset_joints'], '/repo/scene/robot'),
    (['python', 'ros1.py'], '/repo/scene/robot/ur5e'),
    (['python', 'ros2.py'], '/repo/scene/robot/robot_x'),
    (['python', '../server.py'], '/repo/scene/robot/ur5e'),
    *[(['python', '-m', module], '/tmp') for module in (
        'scene.robot.server', 'scene.robot.ur5e.ros1', 'scene.robot.ur5e.ros2',
        'scene.robot.robot_x.ros2',
    )],
    (['python', '/repo/robot_server/ur5e_real_time_control_reconnect.py'], '/repo'),
    (['python', '/repo/robot_server/ur5e_real_time_control_reconnectROS1.py'], '/repo'),
    (['python', '/repo/robot_server/ur5e_real_time_control_reconnectROS2.py'], '/repo'),
    (['python', '/repo/scripts/robot_manual_control.py'], '/repo'),
    (['python', '/repo/robot_server/server.py'], '/tmp'),
    (['python', '/repo/robot_server/ur5e/ros1.py'], '/tmp'),
    (['python', '/repo/robot_server/ur5e/ros2.py'], '/tmp'),
    (['python', 'robot_server/server.py'], '/repo'),
    (['python', 'robot_server/ur5e/ros2.py'], '/repo'),
    (['python', 'server.py'], '/repo/robot_server'),
    (['python', 'ros1.py'], '/repo/robot_server/ur5e'),
    (['python', 'ros2.py'], '/repo/robot_server/ur5e'),
    (['python', 'ur5e/ros2.py'], '/repo/robot_server'),
    (['python', '../server.py'], '/repo/robot_server/ur5e'),
    *[(['python', '-u', '-m', name], '/repo') for name in (
        'robot_server.server', 'robot_server.ur5e.ros1', 'robot_server.ur5e.ros2',
        'robot_server.ur5e_real_time_control_reconnect',
        'robot_server.ur5e_real_time_control_reconnectROS1',
        'robot_server.ur5e_real_time_control_reconnectROS2', 'scripts.robot_manual_control',
    )],
])
def test_local_server_conflict_prevents_ros_probe(tmp_path, command, cwd):
    directory = tmp_path / str(sys.maxsize)
    directory.mkdir()
    (directory / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in command) + b'\0')
    (directory / 'cwd').symlink_to(cwd)
    with patch.object(Path, 'iterdir', return_value=iter([directory])), \
         patch('api.execution.subprocess.run') as run:
        with pytest.raises(RuntimeError, match='An external GTA robot server is running'):
            check_server_conflicts()
    run.assert_not_called()


@pytest.mark.parametrize('command,cwd', [
    (['python', 'server.py'], '/some_other_project'),
    (['python', '/other/ros2.py'], '/repo/robot_server/ur5e'),
    (['python', 'ros2.py'], '/other/ur5e'),
    (['python', '-m', 'another_project.server'], '/repo'),
    (['python', 'client.py', 'robot_server.server'], '/repo'),
    (['python', 'server.py'], None),  # Process disappears before its cwd is read.
])
def test_unrelated_process_does_not_block_ros_probe(tmp_path, command, cwd):
    directory = tmp_path / str(sys.maxsize)
    directory.mkdir()
    (directory / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in command) + b'\0')
    if cwd is not None:
        (directory / 'cwd').symlink_to(cwd)
    with patch.object(Path, 'iterdir', return_value=iter([directory])), \
         patch('api.execution.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
        check_server_conflicts()
    run.assert_called_once()
