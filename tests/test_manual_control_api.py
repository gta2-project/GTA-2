"""Manual-control HTTP guards, with worker creation disabled throughout."""
from unittest.mock import Mock, patch
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api.library_app import create_library_app
from api.service import GtaWebService
from config import Config


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros2')
    monkeypatch.setattr(Config, 'CAMERA_BACKEND', 'file')
    app = create_library_app(root=tmp_path, initialize_on_startup=False)
    manager = app.state.library_manager
    manager.session = GtaWebService(tmp_path / 'session', scoped=True)
    manager.session_info = {'id': 'test-session'}
    manager._attach(manager.session)
    manager.execution.popen = Mock(side_effect=AssertionError('No process may start in API tests'))
    worker = Mock()
    worker.return_value.is_alive.return_value = False
    # Do not replace threading.Thread globally: TestClient needs its own threads.
    with TestClient(app) as client, patch('api.execution.threading', SimpleNamespace(Thread=worker)):
        yield client, manager, worker
    manager.execution.popen.assert_not_called()


@pytest.mark.parametrize('operation', ['reset_joints', 'open_gripper', 'close_gripper'])
def test_manual_api_available_without_grounding_or_script(api, operation):
    client, manager, worker = api
    assert not manager.session.state()['script_ready']
    response = client.post('/api/library/sessions/test-session/execution/manual',
                           json={'operation': operation, 'confirmed': True})
    assert response.status_code == 200
    assert response.json()['operation'] == operation
    worker.return_value.start.assert_called_once()


def test_manual_api_validation_confirmation_and_stale_session(api):
    client, _, worker = api
    url = '/api/library/sessions/test-session/execution/manual'
    assert client.post(url, json={'operation': 'shell_command'}).status_code == 422
    assert client.post(url, json={'operation': 'reset_joints'}).status_code == 400
    assert client.post('/api/library/sessions/old/execution/manual', json={'operation': 'open_gripper'}).status_code == 404
    worker.assert_not_called()


@pytest.mark.parametrize('busy', ['execution', 'admin', 'session', 'ros1'])
def test_manual_api_rejects_conflicting_operations(api, busy, monkeypatch):
    client, manager, worker = api
    if busy == 'execution':
        manager.execution.state['status'] = 'stopping'
    elif busy == 'admin':
        manager.admin._state['busy_stage'] = 'parameters'
    elif busy == 'session':
        manager.session._state['busy_stage'] = 'vision'
    else:
        monkeypatch.setattr(Config, 'GTA_BACKEND', 'ros1')
        monkeypatch.setattr(Config, 'CAMERA_BACKEND', 'ros2')
    response = client.post('/api/library/sessions/test-session/execution/manual', json={'operation': 'open_gripper'})
    assert response.status_code == 409
    worker.assert_not_called()
