"""Main dashboard robot controls; workers and hardware are never executed."""

import hashlib
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import pytest

from api.app import create_app
from api.library_app import create_library_app
from api.service import GtaWebService
from config import Config


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    service = GtaWebService(tmp_path)
    app = create_app(service, initialize_on_startup=False, serve_frontend=False)
    execution = app.state.execution_supervisor
    execution.lock_path = tmp_path / '.gta-robot-execution.lock'
    execution.popen = Mock(side_effect=AssertionError('No hardware process may start'))
    worker = Mock()
    worker.return_value.is_alive.return_value = False
    with TestClient(app) as client, patch('api.execution.threading',
                                        SimpleNamespace(Thread=worker, Event=threading.Event, RLock=threading.RLock)):
        yield SimpleNamespace(client=client, service=service, execution=execution, worker=worker, app=app)
    execution.popen.assert_not_called()
    assert execution._hardware_lock is None


def ready_script(service):
    source = b'print("reviewed script; never executed in this test")\n'
    (service.root_dir / 'robot_script.py').write_bytes(source)
    service._state.update(revision=1, grounding_revision=1, script_revision=1,
                          script_sha256=hashlib.sha256(source).hexdigest())
    return source


def test_state_and_page_refresh_never_start_motion(api):
    for _ in range(2):
        assert api.client.get('/api/execution').json()['status'] == 'idle'
        assert api.client.get('/api/state').status_code == 200
    api.worker.assert_not_called()
    assert not hasattr(api.app.state, 'library_manager')


@pytest.mark.parametrize('operation', ['reset_joints', 'open_gripper', 'close_gripper'])
def test_main_manual_controls_work_without_script_or_provider_initialization(api, operation):
    assert not api.service.initialized
    assert not api.service.state()['script_ready']
    result = api.client.post('/api/execution/manual', json={'operation': operation, 'confirmed': True})
    assert result.status_code == 200
    assert result.json()['operation'] == operation
    assert result.json()['session_id'] == 'main'
    assert result.json()['status'] == 'preparing'
    api.worker.return_value.start.assert_called_once()
    assert api.execution.service is api.service


def test_reset_requires_confirmation_and_unknown_operations_are_rejected(api):
    assert api.client.post('/api/execution/manual', json={'operation': 'reset_joints'}).status_code == 400
    assert api.client.post('/api/execution/manual', json={'operation': 'shell_command'}).status_code == 422
    api.worker.assert_not_called()


def test_run_freezes_only_current_reviewed_script(api):
    source = ready_script(api.service)
    result = api.client.post('/api/execution/run', json={'path': '/not/an/executable/override'})
    assert result.status_code == 200
    assert result.json()['session_id'] == 'main'
    assert result.json()['status'] == 'starting_server'
    assert result.json()['operation'] == 'run'
    assert api.execution.run_script.read_bytes() == source
    run_dir = api.service.root_dir / 'execution_logs' / result.json()['run_id']
    assert api.execution.run_script == run_dir / 'robot_script.py'
    assert (api.service.root_dir / 'robot_script.py').read_bytes() == source
    assert not list(api.service.root_dir.glob('*-robot_script.py'))
    api.worker.return_value.start.assert_called_once()


@pytest.mark.parametrize('operation', ['run', 'manual'])
def test_record_directory_failure_starts_no_worker_and_releases_lease(api, operation):
    ready_script(api.service)
    (api.service.root_dir / 'execution_logs').write_text('not a directory')
    result = api.client.post('/api/execution/' + operation,
                             json={'operation': 'open_gripper'})
    assert result.status_code == 409
    assert 'Cannot create execution records' in result.json()['detail']
    assert api.execution.state['status'] == 'failed'
    assert api.execution._hardware_lock is None
    api.worker.assert_not_called()


def test_snapshot_write_failure_starts_no_worker_and_releases_lease(api):
    source = ready_script(api.service)
    with patch('pathlib.Path.write_bytes', side_effect=OSError('disk full')):
        result = api.client.post('/api/execution/run')
    assert result.status_code == 409
    assert 'Cannot create execution records' in result.json()['detail']
    assert api.execution._hardware_lock is None
    assert (api.service.root_dir / 'robot_script.py').read_bytes() == source
    api.worker.assert_not_called()


@pytest.mark.parametrize('reason', ['ungrounded', 'stale', 'missing', 'modified'])
def test_run_rejects_unready_or_modified_script(api, reason):
    ready_script(api.service)
    if reason == 'ungrounded':
        api.service._state['grounding_revision'] = None
    elif reason == 'stale':
        api.service.invalidate()
    elif reason == 'missing':
        (api.service.root_dir / 'robot_script.py').unlink()
    else:
        (api.service.root_dir / 'robot_script.py').write_text('print("changed")')
    assert api.client.post('/api/execution/run').status_code == 409
    api.worker.assert_not_called()


@pytest.mark.parametrize('operation', ['run', 'manual'])
@pytest.mark.parametrize('busy', ['stage', 'handlers', 'agent', 'execution', 'ros1'])
def test_robot_operations_reject_active_work_and_wrong_backend(api, monkeypatch, operation, busy):
    ready_script(api.service)
    if busy == 'stage':
        api.service._state['busy_stage'] = 'Requerying vision'
    elif busy == 'handlers':
        api.service.message_bus = SimpleNamespace(active_handlers=1)
    elif busy == 'agent':
        api.service.agents['ParameterSetter'] = SimpleNamespace(processing=True)
    elif busy == 'execution':
        api.execution.state['status'] = 'stopping'
    else:
        monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros1')
    result = api.client.post('/api/execution/' + operation, json={'operation': 'open_gripper'})
    assert result.status_code == 409
    api.worker.assert_not_called()


@pytest.mark.parametrize('status', ['preparing', 'preview', 'running', 'stopping'])
def test_planning_mutations_are_blocked_until_execution_finishes(api, status):
    api.service.initialized = True  # No provider setup: only exercise the guards.
    api.execution.state['status'] = status
    with patch.object(api.service, 'capture_environment_image') as capture:
        for path, body in [
            ('/api/workflow/start', {'task': 'must not start'}),
            ('/api/agents/ParameterSetter/message', {'message': 'must not refine'}),
            ('/api/vision/requery', {}),
            ('/api/script/generate', {}),
            ('/api/workflow/save', {}),
        ]:
            assert api.client.post(path, json=body).status_code == 409, path
        capture.assert_not_called()


def test_skip_and_cancel_require_current_run_and_subtask(api):
    execution = api.execution
    execution.state.update(status='running', session_id='main', run_id='current', subtask='2')
    with patch.object(execution, '_send') as send:
        assert api.client.post('/api/execution/skip', json={'run_id': 'old', 'subtask': '2'}).status_code == 409
        assert api.client.post('/api/execution/skip', json={'run_id': 'current', 'subtask': '1'}).status_code == 409
        assert api.client.post('/api/execution/cancel', json={'run_id': 'old'}).status_code == 409
        send.assert_not_called()
        result = api.client.post('/api/execution/skip', json={'run_id': 'current', 'subtask': '2'})
        assert result.json()['skip_pending']
        send.assert_called_once_with('script', 'skip', subtask='2')
        assert api.client.post('/api/execution/skip', json={'run_id': 'current', 'subtask': '2'}).status_code == 409
        send.reset_mock()
        result = api.client.post('/api/execution/cancel', json={'run_id': 'current'})
        assert result.json()['status'] == 'stopping'
        assert all(call.args[1] == 'cancel' for call in send.call_args_list)
        assert execution.cancel_event.is_set()


def test_main_and_library_cannot_acquire_robot_at_the_same_time(api, tmp_path):
    assert api.client.post('/api/execution/manual', json={'operation': 'open_gripper'}).status_code == 200
    library = create_library_app(root=tmp_path, initialize_on_startup=False)
    manager = library.state.library_manager
    manager.execution.popen = Mock(side_effect=AssertionError('Must not start hardware'))
    manager.session = GtaWebService(tmp_path, scoped=True)
    manager.session_info = {'id': 'library-test'}
    manager._attach(manager.session)
    with TestClient(library) as client:
        result = client.post('/api/library/sessions/library-test/execution/manual', json={'operation': 'open_gripper'})
        assert result.status_code == 409
        assert 'Another GTA dashboard' in result.json()['detail']
        api.execution.cancel(api.execution.state['run_id'])
        # The lease remains held during cleanup, not only while moving.
        assert client.post('/api/library/sessions/library-test/execution/manual', json={'operation': 'open_gripper'}).status_code == 409
        api.execution.shutdown()
        assert client.post('/api/library/sessions/library-test/execution/manual', json={'operation': 'open_gripper'}).status_code == 200
        assert api.client.post('/api/execution/manual', json={'operation': 'close_gripper'}).status_code == 409
    manager.execution.popen.assert_not_called()


def test_app_shutdown_cancels_owned_work_and_releases_lease(api):
    result = api.client.post('/api/execution/manual', json={'operation': 'open_gripper'})
    assert result.status_code == 200
    api.execution.shutdown()
    assert api.execution.cancel_event.is_set()
    api.worker.return_value.join.assert_called_once()
    assert api.execution._hardware_lock is None


def test_worker_start_failure_releases_lease(api):
    api.worker.return_value.start.side_effect = RuntimeError('could not start worker')
    result = api.client.post('/api/execution/manual', json={'operation': 'open_gripper'})
    assert result.status_code == 409
    assert api.execution.state['status'] == 'failed'
    assert api.execution._hardware_lock is None
