"""Process supervision tests use only harmless IPC fixtures, never real RTDE."""

import os
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from api.execution import ExecutionSupervisor
from api.service import GtaWebService
from config import Config
from gta import managed_run


ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / 'tests/fixtures/library_child.py'


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Timed out waiting for harmless child')


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    monkeypatch.setattr(Config, 'CAMERA_BACKEND', 'file')
    service = GtaWebService(tmp_path, scoped=True)
    service._state.update(revision=1, grounding_revision=1, script_revision=1)
    shutil.copy2(CHILD, tmp_path / 'robot_script.py')
    service._state['script_sha256'] = hashlib.sha256((tmp_path / 'robot_script.py').read_bytes()).hexdigest()
    scenario = {'mode': 'complete'}
    spawned = []

    def fake_hardware_popen(args, **kwargs):
        original_args = list(args)
        role = kwargs['env']['GTA_CHILD_ROLE']
        kwargs['env']['GTA_TEST_SCENARIO'] = scenario['mode']
        # Rewrite the server command BEFORE spawning. No robot code is imported.
        if role in {'server', 'manual'}:
            args = [sys.executable, '-u', str(CHILD)]
        spawned.append((role, time.monotonic(), args, {'original_args': original_args, 'cwd': kwargs['cwd'],
                        'env': {key: kwargs['env'][key] for key in ('PYTHONUNBUFFERED', 'CAMERA_IMAGE_PATH')}}))
        return subprocess.Popen(args, **kwargs)

    execution = ExecutionSupervisor(ROOT, service._lock, popen=fake_hardware_popen,
                                    conflict_check=lambda: None, startup_timeout=0.4, shutdown_timeout=0.3,
                                    lock_path=tmp_path / '.test-robot-execution.lock')
    service.execution_guard = execution.assert_idle
    yield execution, service, scenario, spawned
    execution.shutdown()
    assert all(process.poll() is not None for process in execution.children.values())


def test_readiness_normal_completion_and_owned_server_cleanup(supervisor):
    execution, service, _, spawned = supervisor
    run = execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'completed')
    assert [role for role, *_ in spawned] == ['server', 'script']
    assert spawned[0][3]['original_args'] == [sys.executable, '-u', str(ROOT / 'scene/robot/server.py')]
    assert spawned[1][1] - spawned[0][1] >= 0.03
    assert spawned[1][3]['cwd'] == str(service.root_dir)
    assert spawned[1][2][0] == sys.executable
    assert spawned[1][2][2] == str(ROOT / 'gta/runner.py')
    assert spawned[1][3]['env']['PYTHONUNBUFFERED'] == '1'
    assert all(child[3]['env']['CAMERA_IMAGE_PATH'] == str(service.root_dir / 'environment.jpg')
               for child in spawned)
    assert [task['status'] for task in execution.snapshot()['subtasks']] == ['completed', 'completed']
    assert execution.server_cleaned
    assert all(child.poll() is not None for child in execution.children.values())
    run_dir = service.root_dir / 'execution_logs' / run['run_id']
    assert execution.run_dir == run_dir
    assert {path.name for path in run_dir.iterdir()} == {
        'robot_script.py', 'events.jsonl', 'server.log', 'script.log'}
    assert (run_dir / 'robot_script.py').read_bytes() == (service.root_dir / 'robot_script.py').read_bytes()
    assert spawned[1][2][-1] == str(run_dir / 'robot_script.py')
    assert 'fake server started' in (run_dir / 'server.log').read_text()
    assert 'fake script started' in (run_dir / 'script.log').read_text()
    events = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text().splitlines()]
    assert events and all(event['content']['run_id'] == run['run_id'] for event in events)
    assert not list(service.root_dir.glob('*.log'))
    assert not list(service.root_dir.glob('*-events.jsonl'))
    assert not list(service.root_dir.glob('*-robot_script.py'))


def test_repeated_runs_keep_separate_records(supervisor):
    execution, service, _, _ = supervisor
    first = execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'completed' and execution._hardware_lock is None)
    first_dir = service.root_dir / 'execution_logs' / first['run_id']
    originals = {path.name: path.read_bytes() for path in first_dir.iterdir()}
    second = execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'completed' and execution._hardware_lock is None)
    assert first['run_id'] != second['run_id']
    assert {path.name: path.read_bytes() for path in first_dir.iterdir()} == originals
    assert execution.run_dir == service.root_dir / 'execution_logs' / second['run_id']
    assert (execution.run_dir / 'robot_script.py').read_bytes() == originals['robot_script.py']


def test_skip_identity_repeated_click_and_last_subtask(supervisor):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = 'run'
    run = execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'running')
    with pytest.raises(RuntimeError):
        execution.skip('old-run', '1')
    # Keep event handling blocked while verifying the duplicate click rejection.
    with execution.lock:
        execution.skip(run['run_id'], '1')
        with pytest.raises(RuntimeError):
            execution.skip(run['run_id'], '1')
    wait_for(lambda: execution.snapshot()['subtask'] == '2')
    with pytest.raises(RuntimeError):
        execution.skip(run['run_id'], '1')
    execution.skip(run['run_id'], '2')
    wait_for(lambda: execution.snapshot()['status'] == 'completed')
    assert [task['status'] for task in execution.snapshot()['subtasks']] == ['skipped', 'skipped']


@pytest.mark.parametrize('mode,stage', [('preview', 'preview'), ('run', 'running'), ('timeout', 'starting_server')])
def test_cancel_latches_and_never_enters_next_subtask(supervisor, mode, stage):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = mode
    run = execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == stage)
    # Hold the state lock while checking Stopping: startup cancellation can
    # otherwise finish cleanup before the second-Run assertion gets scheduled.
    with execution.lock:
        execution.cancel(run['run_id'])
        assert execution.snapshot()['status'] == 'stopping'
        with pytest.raises(RuntimeError, match='execution is active'):
            execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'cancelled')
    tasks = execution.snapshot()['subtasks']
    assert len(tasks) <= 1
    assert not any(task['status'] == 'skipped' for task in tasks)


@pytest.mark.parametrize('mode', ['timeout', 'scriptfail', 'serverfail', 'hardkill'])
def test_failures_cleanup_and_escalation(supervisor, mode):
    execution, service, scenario, spawned = supervisor
    scenario['mode'] = mode
    execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert execution.snapshot()['error']
    assert all(child.poll() is not None for child in execution.children.values())
    if mode == 'timeout':
        assert [role for role, *_ in spawned] == ['server']
        assert 'timed out' in execution.snapshot()['error']
    elif mode == 'hardkill':
        assert 'Forced termination' in execution.snapshot()['error']
    elif mode == 'serverfail':
        assert len(execution.snapshot()['subtasks']) <= 1


def test_conflict_starts_no_children(supervisor):
    execution, service, _, spawned = supervisor
    execution.conflict_check = Mock(side_effect=RuntimeError('External server found'))
    execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert spawned == []
    assert execution.snapshot()['error'] == 'External server found'


def test_ros2_camera_does_not_enable_ros1_managed_execution(supervisor, monkeypatch):
    execution, service, _, spawned = supervisor
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros1')
    monkeypatch.setattr(Config, 'CAMERA_BACKEND', 'ros2')
    with pytest.raises(RuntimeError, match='requires GTA_BACKEND=ros2'):
        execution.start(service, 'session')
    assert spawned == []
    assert execution.snapshot()['status'] == 'idle'


def test_control_channel_closure_stops_owned_child(supervisor):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = 'run'
    execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'running')
    import socket
    execution.channels['script'].shutdown(socket.SHUT_RDWR)
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert len(execution.snapshot()['subtasks']) <= 1


def test_preview_skip_disabled_and_shutdown_cancel(supervisor):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = 'preview'
    run = execution.start(service, 'session')
    wait_for(lambda: execution.snapshot()['status'] == 'preview')
    with pytest.raises(RuntimeError):
        execution.skip(run['run_id'], '1')
    execution.shutdown()
    assert execution.snapshot()['status'] == 'cancelled'


def test_cooperative_flag_and_boundary_cancellation():
    runtime = SimpleNamespace(cancelled=threading.Event(), lock=threading.RLock(),
                              skip_requested=True, skipped=False, subtask=None, emit=Mock())
    def check():
        if runtime.cancelled.is_set():
            raise managed_run.RunCancelled()
    runtime.check_cancel = check
    with patch.object(managed_run, 'get_runtime', return_value=runtime):
        assert managed_run.should_skip() and runtime.skipped
        runtime.cancelled.set()
        with pytest.raises(managed_run.RunCancelled):
            with managed_run.managed_subtask(2, 'Must not start'):
                pytest.fail('Entered a new subtask after Cancel')
        runtime.emit.assert_not_called()


def test_managed_open3d_preview_polls_and_closes_on_cancel():
    from gta import utils
    from gta.utils import visualization
    runtime = Mock()
    runtime.check_cancel.side_effect = managed_run.RunCancelled()
    visualizer = Mock()
    with patch.object(managed_run, 'get_runtime', return_value=runtime), \
         patch.object(visualization.o3d.visualization, 'Visualizer', return_value=visualizer):
        with pytest.raises(managed_run.RunCancelled):
            utils.visualize_point_cloud([[0.0, -0.1, 0.1]], show_coordinate_frame=False)
    visualizer.run.assert_not_called()
    visualizer.poll_events.assert_called_once()
    visualizer.destroy_window.assert_called_once()


def test_run_rejects_modified_script(supervisor):
    execution, service, _, spawned = supervisor
    (service.root_dir / 'robot_script.py').write_text('print("changed")')
    with pytest.raises(RuntimeError, match='script has changed'):
        execution.start(service, 'session')
    assert not spawned


@pytest.mark.parametrize('operation', ['reset_joints', 'open_gripper', 'close_gripper'])
def test_manual_controls_need_no_script_and_share_execution_lock(supervisor, operation):
    execution, service, scenario, spawned = supervisor
    service.invalidate()
    scenario['mode'] = 'run'
    run = execution.start_manual(service, 'session', operation)
    wait_for(lambda: execution.snapshot()['status'] == 'running')
    assert execution.snapshot()['operation'] == operation
    assert [role for role, *_ in spawned] == ['manual']
    with pytest.raises(RuntimeError, match='execution is active'):
        execution.start(service, 'session')
    with pytest.raises(RuntimeError, match='execution is active'):
        execution.start_manual(service, 'session', 'open_gripper')
    with pytest.raises(RuntimeError):
        execution.skip(run['run_id'], None)
    with pytest.raises(RuntimeError, match='execution is active'):
        service.requery_vision()
    execution.cancel(run['run_id'])
    wait_for(lambda: execution.snapshot()['status'] == 'cancelled')
    assert execution.manual_cleaned
    assert execution.snapshot()['subtasks'] == []


def test_manual_normal_completion_and_cleanup(supervisor):
    execution, service, _, spawned = supervisor
    run = execution.start_manual(service, 'session', 'reset_joints')
    wait_for(lambda: execution.snapshot()['status'] == 'completed')
    assert execution.manual_completed and execution.manual_cleaned
    assert [role for role, *_ in spawned] == ['manual']
    assert all(child.poll() is not None for child in execution.children.values())
    run_dir = service.root_dir / 'execution_logs' / run['run_id']
    assert {path.name for path in run_dir.iterdir()} == {'manual.log', 'events.jsonl'}
    assert 'fake manual started' in (run_dir / 'manual.log').read_text()
    assert not list(service.root_dir.glob('*.log'))
    assert not list(service.root_dir.glob('*-events.jsonl'))


@pytest.mark.parametrize('mode', ['manualfail', 'unclean', 'cleanupfail', 'timeout'])
def test_manual_failure_timeout_and_unverified_cleanup(supervisor, mode):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = mode
    execution.start_manual(service, 'session', 'close_gripper')
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert execution.snapshot()['error']
    assert all(child.poll() is not None for child in execution.children.values())


def test_manual_cancel_escalation_stays_failed_if_hardware_cleanup_unknown(supervisor):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = 'hardkill'
    run = execution.start_manual(service, 'session', 'reset_joints')
    wait_for(lambda: execution.snapshot()['status'] == 'running')
    execution.cancel(run['run_id'])
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert 'Forced termination' in execution.snapshot()['error']


def test_manual_conflict_starts_no_children(supervisor):
    execution, service, _, spawned = supervisor
    execution.conflict_check = Mock(side_effect=RuntimeError('External server found'))
    execution.start_manual(service, 'session', 'open_gripper')
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert spawned == []


def test_manual_control_channel_loss_stops_child(supervisor):
    import socket
    execution, service, scenario, _ = supervisor
    scenario['mode'] = 'run'
    execution.start_manual(service, 'session', 'reset_joints')
    wait_for(lambda: execution.snapshot()['status'] == 'running')
    execution.channels['manual'].shutdown(socket.SHUT_RDWR)
    wait_for(lambda: execution.snapshot()['status'] == 'failed')
    assert all(child.poll() is not None for child in execution.children.values())


def test_late_manual_started_event_cannot_leave_stopping(supervisor):
    execution, _, _, _ = supervisor
    execution.state.update(status='stopping', run_id='manual-run')
    execution.handle_event('manual', {'type': 'manual_started', 'run_id': 'manual-run'})
    assert execution.snapshot()['status'] == 'stopping'
    execution.state['status'] = 'idle'


def test_execution_lease_is_process_wide_and_released_after_cleanup(supervisor):
    execution, service, scenario, _ = supervisor
    scenario['mode'] = 'run'
    run = execution.start_manual(service, 'session', 'open_gripper')
    wait_for(lambda: execution.snapshot()['status'] == 'running')
    probe = '''
import fcntl, sys
with open(sys.argv[1], 'a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(2)
'''
    def acquire_in_other_process():
        return subprocess.run([sys.executable, '-c', probe, str(execution.lock_path)],
                              capture_output=True, text=True, timeout=5).returncode
    assert acquire_in_other_process() == 2
    with execution.lock:
        execution.cancel(run['run_id'])
        assert acquire_in_other_process() == 2
    execution.shutdown()
    assert all(child.poll() is not None for child in execution.children.values())
    assert acquire_in_other_process() == 0


def test_external_server_conflict_releases_dashboard_lease(supervisor):
    execution, service, _, _ = supervisor
    execution.conflict_check = Mock(side_effect=RuntimeError('External server found'))
    execution.start_manual(service, 'session', 'open_gripper')
    execution.shutdown()
    assert execution._hardware_lock is None
