"""Local operator checkpoint, workspace and execution API."""

import hashlib
from fastapi import HTTPException, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from api.service import ARTIFACT_NAMES
from api.checkpoints import read_json
from api.events import stream_events
from api.execution_routes import Control, ManualControl


class SaveCheckpoint(BaseModel):
    name: str
    notes: str = ''
    visible: bool = False
    source: str = 'admin'


class UpdateCheckpoint(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    visible: Optional[bool] = None


class OpenCheckpoint(BaseModel):
    checkpoint_id: str
    target: str = 'run'


class Feedback(BaseModel):
    message: str


def install_library_routes(app, manager):
    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    def session(session_id):
        return call(manager.get_session, session_id)

    def operate(session_id, method, *args):
        with manager.lock:
            return call(getattr(session(session_id), method), *args)

    @app.get('/api/checkpoints')
    def checkpoints(visible_only: bool = False):
        return call(manager.store.list, visible_only)

    @app.post('/api/checkpoints')
    def save(payload: SaveCheckpoint):
        return call(manager.save, **payload.model_dump())

    @app.patch('/api/checkpoints/{checkpoint_id}')
    def update(checkpoint_id: str, payload: UpdateCheckpoint):
        with manager.lock:
            return call(manager.store.update, checkpoint_id, **payload.model_dump(exclude_none=True))

    @app.get('/api/checkpoints/{checkpoint_id}/artifacts/{name}')
    def checkpoint_artifact(checkpoint_id: str, name: str):
        if name not in ARTIFACT_NAMES:
            raise HTTPException(400, 'Unknown artifact')
        directory = call(manager.store.directory, checkpoint_id)
        manifest = call(read_json, directory / 'manifest.json')
        if name not in manifest['artifacts']:
            raise HTTPException(404, 'Artifact not found')
        path = directory / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['artifacts'][name]:
            raise HTTPException(409, 'Checkpoint artifact failed its integrity check')
        return FileResponse(path)

    @app.post('/api/library/sessions')
    def open_session(payload: OpenCheckpoint):
        result = call(manager.open, payload.checkpoint_id, payload.target)
        app.state.gta_service = manager.admin
        return result

    @app.get('/api/library/session')
    def current_session():
        return manager.session_info

    @app.get('/api/library/sessions/{session_id}/state')
    def session_state(session_id: str):
        with manager.lock:
            return {**operate(session_id, 'state'), 'session': manager.session_info}

    @app.get('/api/library/sessions/{session_id}/health')
    def health(session_id: str):
        return operate(session_id, 'health')

    @app.post('/api/library/sessions/{session_id}/reset')
    def reset(session_id: str):
        with manager.lock:
            session(session_id)
            return call(manager.open, manager.session_info['checkpoint_id'])

    @app.post('/api/library/sessions/{session_id}/agents/{agent_name}/message')
    def feedback(session_id: str, agent_name: str, payload: Feedback):
        return operate(session_id, 'send_agent_message', agent_name, payload.message)

    @app.post('/api/library/sessions/{session_id}/vision/requery')
    def requery(session_id: str):
        return operate(session_id, 'requery_vision')

    @app.post('/api/library/sessions/{session_id}/script/generate')
    def generate(session_id: str):
        return operate(session_id, 'generate_script')

    @app.post('/api/library/sessions/{session_id}/workflow/save')
    def export(session_id: str):
        return operate(session_id, 'save_workflow')

    @app.get('/api/library/sessions/{session_id}/artifacts/{name}')
    def artifact(session_id: str, name: str):
        return FileResponse(operate(session_id, 'artifact_path', name))

    @app.get('/api/library/sessions/{session_id}/reference/{name}')
    def reference(session_id: str, name: str):
        if name not in {'environment.jpg', 'annotated_keypoints.png'}:
            raise HTTPException(400, 'Unknown reference artifact')
        path = session(session_id).root_dir / 'reference' / name
        if not path.is_file():
            raise HTTPException(404, 'Reference image not found')
        return FileResponse(path)

    @app.websocket('/api/library/sessions/{session_id}/events')
    async def events(websocket: WebSocket, session_id: str):
        try:
            service = manager.get_session(session_id)
        except FileNotFoundError:
            await websocket.close(code=1008)
            return
        await stream_events(websocket, service)

    @app.get('/api/execution')
    def execution():
        return manager.execution.snapshot()

    @app.post('/api/library/sessions/{session_id}/execution/run')
    def run(session_id: str):
        with manager.lock:
            call(manager.admin.assert_idle)
            return call(manager.execution.start, session(session_id), session_id)

    @app.post('/api/library/sessions/{session_id}/execution/skip')
    def skip(session_id: str, payload: Control):
        with manager.lock:
            session(session_id)
            return call(manager.execution.skip, payload.run_id, payload.subtask)

    @app.post('/api/library/sessions/{session_id}/execution/manual')
    def manual(session_id: str, payload: ManualControl):
        with manager.lock:
            service = session(session_id)
            if payload.operation == 'reset_joints' and not payload.confirmed:
                raise HTTPException(400, 'Confirm that the path to the recorded joint pose is clear')
            call(manager.admin.assert_idle)
            return call(manager.execution.start_manual, service, session_id, payload.operation)

    @app.post('/api/library/sessions/{session_id}/execution/cancel')
    def cancel(session_id: str, payload: Control):
        with manager.lock:
            session(session_id)
            return call(manager.execution.cancel, payload.run_id)
