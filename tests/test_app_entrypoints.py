"""The normal dashboard and library application are separate launch targets."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app
from api.library_app import create_library_app
from api.service import GtaWebService


def test_web_launchers_do_not_require_retired_tk_widgets():
    root = Path(__file__).resolve().parents[1]
    assert not (root / 'gui_components').exists()
    assert 'customtkinter' not in (root / 'requirements.txt').read_text().lower()
    probe = '''
import importlib.abc
import sys

class NoLegacyWidgets(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gui_components', 'customtkinter'}:
            raise AssertionError('Unexpected legacy GUI import: ' + fullname)

sys.meta_path.insert(0, NoLegacyWidgets())
import main
import library
from gta import runner
assert callable(main.main)
assert callable(library.main)
assert callable(runner.main)
'''
    result = subprocess.run([sys.executable, '-c', probe], cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_main_has_robot_controls_without_library_pages_or_checkpoint_apis(tmp_path):
    service = GtaWebService(tmp_path)
    app = create_app(service, initialize_on_startup=False)
    assert app.state.gta_service is service
    assert not hasattr(app.state, 'library_manager')
    with TestClient(app) as client:
        assert client.get('/api/state').status_code == 200
        assert client.get('/api/execution').json()['status'] == 'idle'
        for path in ('/admin', '/library', '/run', '/library.html', '/api/checkpoints', '/api/library/session'):
            assert client.get(path).status_code == 404
        response = client.get('/')
        assert response.status_code == 200
        if 'text/html' in response.headers['content-type']:
            assert '<title>GTA Sessions Dashboard</title>' in response.text


def test_library_has_three_pages_and_its_own_workspace(tmp_path):
    main_service = GtaWebService(tmp_path)
    app = create_library_app(root=tmp_path, initialize_on_startup=False)
    manager = app.state.library_manager
    assert manager.admin is not main_service
    assert manager.admin.root_dir == tmp_path / 'library_sessions' / 'admin'
    assert manager.store.root == tmp_path / 'library_checkpoints'
    with TestClient(app) as client:
        redirect = client.get('/', follow_redirects=False)
        assert redirect.status_code == 303 and redirect.headers['location'] == '/admin'
        for path in ('/admin', '/library', '/run'):
            response = client.get(path)
            assert response.status_code == 200
            if 'text/html' in response.headers['content-type']:
                assert '<title>GTA-2 Library</title>' in response.text
        assert client.get('/api/checkpoints').json() == []
        assert client.get('/api/library/session').json() is None
        assert client.get('/api/execution').json()['status'] == 'idle'
        for retired_path in ('/demo', '/demo.html', '/api/demo/session'):
            assert client.get(retired_path).status_code == 404
        assert client.post('/api/demo/sessions', json={}).status_code == 404
        # The normal dashboard's unscoped control endpoints must not be installed
        # in the library app; it has its own session-scoped supervisor routes.
        assert client.post('/api/execution/run').status_code == 404


def test_library_launcher_uses_separate_app_and_port(monkeypatch):
    import library
    monkeypatch.delenv('GTA_LIBRARY_HOST', raising=False)
    monkeypatch.delenv('GTA_LIBRARY_PORT', raising=False)
    with patch.object(library, 'load_dotenv'), patch.object(library.uvicorn, 'run') as run:
        library.main()
    run.assert_called_once_with('api.library_app:app', host='127.0.0.1', port=8001)


def test_library_launcher_honors_renamed_address_settings(monkeypatch):
    import library
    monkeypatch.setenv('GTA_LIBRARY_HOST', '127.0.0.2')
    monkeypatch.setenv('GTA_LIBRARY_PORT', '8002')
    with patch.object(library, 'load_dotenv'), patch.object(library.uvicorn, 'run') as run:
        library.main()
    run.assert_called_once_with('api.library_app:app', host='127.0.0.2', port=8002)


def test_normal_launcher_still_uses_original_app(monkeypatch):
    import main
    monkeypatch.setenv('GTA_WEB_HOST', '127.0.0.1')
    monkeypatch.setenv('GTA_WEB_PORT', '8000')
    monkeypatch.setenv('GTA_WEB_RELOAD', 'false')
    with patch.object(main, 'load_dotenv'), patch.object(main.os, 'chdir'), patch.object(main.uvicorn, 'run') as run:
        main.main()
    run.assert_called_once_with('api.app:app', host='127.0.0.1', port=8000, reload=False)
