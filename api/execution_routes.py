"""Robot-control endpoints for the normal dashboard, sharing the library supervisor."""

from typing import Literal, Optional

from fastapi import HTTPException
from pydantic import BaseModel


MAIN_WORKSPACE_ID = 'main'


class Control(BaseModel):
    run_id: str
    subtask: Optional[str] = None


class ManualControl(BaseModel):
    operation: Literal['reset_joints', 'open_gripper', 'close_gripper']
    confirmed: bool = False


def install_execution_routes(app, service, execution):
    def call(fn, *args):
        try:
            with service._lock:
                return fn(*args)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/execution')
    def state():
        return execution.snapshot()

    @app.post('/api/execution/run')
    def run():
        return call(execution.start, service, MAIN_WORKSPACE_ID)

    @app.post('/api/execution/skip')
    def skip(payload: Control):
        return call(execution.skip, payload.run_id, payload.subtask)

    @app.post('/api/execution/cancel')
    def cancel(payload: Control):
        return call(execution.cancel, payload.run_id)

    @app.post('/api/execution/manual')
    def manual(payload: ManualControl):
        if payload.operation == 'reset_joints' and not payload.confirmed:
            raise HTTPException(400, 'Confirm that the path to the recorded joint pose is clear')
        return call(execution.start_manual, service, MAIN_WORKSPACE_ID, payload.operation)
