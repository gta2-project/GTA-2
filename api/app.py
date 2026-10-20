"""FastAPI routes for the GTA Sessions web dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.service import GtaWebService
from api.events import stream_events
from api.execution import ExecutionSupervisor
from api.execution_routes import install_execution_routes


ROOT_DIR = Path(__file__).resolve().parents[1]


class TaskRequest(BaseModel):
    task: str


class AgentMessageRequest(BaseModel):
    message: str


def create_app(
    service: Optional[GtaWebService] = None,
    initialize_on_startup: bool = True,
    serve_frontend: bool = True,
    enable_execution: bool = True,
) -> FastAPI:
    service = service or GtaWebService(ROOT_DIR)
    app = FastAPI(title="GTA Sessions Web Dashboard")
    app.state.gta_service = service

    if enable_execution:
        execution = ExecutionSupervisor(ROOT_DIR, service._lock)
        app.state.execution_supervisor = execution
        service.execution_guard = execution.assert_idle
        install_execution_routes(app, service, execution)

        @app.on_event('shutdown')
        def _shutdown_execution():
            execution.shutdown()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if initialize_on_startup:
        @app.on_event("startup")
        def _startup() -> None:
            service.initialize()

    @app.get("/api/health")
    def health():
        return service.health()

    @app.get("/api/state")
    def state():
        return service.state()

    @app.post("/api/workflow/start")
    def start_workflow(payload: TaskRequest):
        try:
            result = service.start_workflow(payload.task)
            return JSONResponse(status_code=202, content=result)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/agents/{agent_name}/message")
    def send_agent_message(agent_name: str, payload: AgentMessageRequest):
        try:
            result = service.send_agent_message(agent_name, payload.message)
            return JSONResponse(status_code=202, content=result)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/vision/requery")
    def requery_vision():
        try:
            result = service.requery_vision()
            return JSONResponse(status_code=202, content=result)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/script/generate")
    def generate_script():
        try:
            result = service.generate_script()
            return {
                "path": result["output_path"],
                "script": result["script"],
                "keypoint_count": result["keypoint_count"],
                "skill_count": result["skill_count"],
            }
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/workflow/save")
    def save_workflow():
        try:
            return service.save_workflow()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/artifacts/{name:path}")
    def get_artifact(name: str):
        try:
            return FileResponse(service.artifact_path(name))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Artifact not found")

    @app.websocket("/api/events")
    async def events(websocket: WebSocket):
        # Pin the owner for the lifetime of this websocket, including when the
        # separate library app uses a replaceable admin working copy.
        owner = getattr(service, 'event_service', service)
        await stream_events(websocket, owner)

    if not serve_frontend:
        return app
    dist_dir = ROOT_DIR / "frontend" / "dist"
    if dist_dir.exists():
        @app.get('/')
        def interface():
            return FileResponse(dist_dir / 'index.html')

        app.mount('/assets', StaticFiles(directory=str(dist_dir / 'assets')), name='frontend-assets')
    else:
        @app.get("/")
        def frontend_not_built():
            return {
                "message": "GTA Sessions backend is running. Start the Vite dev server or build frontend/dist.",
                "api": "/api/health",
            }

    return app


app = create_app()
