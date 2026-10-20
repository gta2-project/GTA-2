#!/usr/bin/env python3
"""Minimal review-before-run web UI for the CaP benchmark.

Launch with::

    conda run -n ur5 python -m benchmarks.cap.web

The UI deliberately keeps generation and physical execution as separate user
actions. The generated source is editable, parsed again for the minimum CaP
interface, and only then executed after explicit operator approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPOSITORY_ROOT / ".env")

from .artifacts import json_safe
from .conditions import CONDITIONS, ApiKind, get_condition
from .controller_server import ManagedControllerServer
from .perception import PerceptionSnapshot, VisionPerceptionLocalizer
from .runner import _capture_live_point_cloud, run_trial
from .validation import validate_policy
from .visualization import (
    render_annotated_scene,
    render_point_cloud_preview,
    show_interactive_point_cloud,
)


MAX_TASK_LENGTH = 10_000
MAX_POLICY_LENGTH = 250_000


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return slug[:80] or "cap_task"


def _capture_fresh_scene(destination: Path) -> Path:
    """Capture one new RGB frame and freeze it for generation and execution."""
    from config import Config
    from scene.camera import Camera

    print("[CaP Camera] Requesting a fresh RGB image...", flush=True)
    camera = Camera(
        camera_node=Config.CAMERA_NODE,
        camera_model=Config.CAMERA_MODEL,
        rosmaster_ip=Config.ROSMASTER_IP,
        rosmaster_port=Config.ROSMASTER_PORT,
        camera_backend=Config.CAMERA_BACKEND,
        camera_image_path=Config.CAMERA_IMAGE_PATH,
    )
    rgb = np.asarray(camera.get_rgb_image(use_new_frame=True))
    print(f"[CaP Camera] Fresh RGB image received: shape={rgb.shape}", flush=True)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError("Camera did not return an RGB image")
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise RuntimeError("Camera returned an empty RGB image")
    rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        if not np.all(np.isfinite(rgb)):
            raise RuntimeError("Camera image contains non-finite values")
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(destination, format="JPEG", quality=95)
    print(f"[CaP Camera] Frozen scene image saved: {destination}", flush=True)
    return destination


class GenerateRequest(BaseModel):
    task: str
    condition: str = "cartesian_zeroshot"


class PrepareRequest(BaseModel):
    session_id: str
    code: str
    fresh_scene: bool = False


class ExecuteRequest(BaseModel):
    session_id: str
    preparation_id: str
    code: str
    operator_approved: bool = False


class CapWebService:
    """State holder behind the intentionally single-user CaP UI."""

    def __init__(
        self,
        output_root: Path = REPOSITORY_ROOT / "benchmark_runs" / "cap",
        trial_runner: Callable[..., Dict[str, Any]] = run_trial,
        image_capturer: Callable[[Path], Path] = _capture_fresh_scene,
        point_cloud_capturer: Callable[[], np.ndarray] = _capture_live_point_cloud,
        perception_localizer=None,
        interactive_visualizer: Callable[..., None] = show_interactive_point_cloud,
        controller_server_factory: Optional[Callable[[Path], Any]] = None,
    ):
        self.output_root = Path(output_root).expanduser().resolve()
        self.session_root = self.output_root / "_ui_sessions"
        self.session_root.mkdir(parents=True, exist_ok=True)
        self._trial_runner = trial_runner
        self._image_capturer = image_capturer
        self._point_cloud_capturer = point_cloud_capturer
        self._perception_localizer = perception_localizer
        self._interactive_visualizer = interactive_visualizer
        self._controller_server_factory = controller_server_factory or (
            lambda log_path: ManagedControllerServer(log_path=log_path)
        )
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._generation_jobs: Dict[str, Dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()

    def _exclusive_operation(self, action: str):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(
                f"Another CaP generation or execution is already active; cannot {action}"
            )
        return self._operation_lock

    def start_generation(self, task: str, condition_name: str) -> Dict[str, Any]:
        """Start policy generation without tying it to one browser connection."""
        task = str(task).strip()
        if not task:
            raise ValueError("Enter a task prompt before generating code")
        if len(task) > MAX_TASK_LENGTH:
            raise ValueError(f"Task prompt exceeds {MAX_TASK_LENGTH} characters")
        condition = get_condition(condition_name)
        job_id = uuid.uuid4().hex
        with self._state_lock:
            self._generation_jobs[job_id] = {
                "job_id": job_id,
                "status": "running",
                "task": task,
                "condition": condition.name.value,
                "result": None,
                "error": None,
            }

        worker = threading.Thread(
            target=self._run_generation_job,
            args=(job_id, task, condition.name.value),
            name=f"cap-generation-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return {"job_id": job_id, "status": "running"}

    def _run_generation_job(
        self, job_id: str, task: str, condition_name: str
    ) -> None:
        print(f"[CaP Generate] Background job {job_id} started.", flush=True)
        try:
            result = self.generate(task, condition_name)
        except Exception as exc:
            with self._state_lock:
                job = self._generation_jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
            print(
                f"[CaP Generate] Background job {job_id} failed: {exc}",
                flush=True,
            )
            return

        with self._state_lock:
            job = self._generation_jobs[job_id]
            job["status"] = "completed"
            job["result"] = result
        print(f"[CaP Generate] Background job {job_id} completed.", flush=True)

    def generation_status(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._generation_jobs.get(job_id)
            if job is None:
                raise ValueError("Unknown or expired generation job")
            response = {
                "job_id": job["job_id"],
                "status": job["status"],
            }
            if job["status"] == "completed":
                response["result"] = job["result"]
            elif job["status"] == "failed":
                response["error"] = job["error"]
        return json_safe(response)

    def generate(self, task: str, condition_name: str) -> Dict[str, Any]:
        task = str(task).strip()
        if not task:
            raise ValueError("Enter a task prompt before generating code")
        if len(task) > MAX_TASK_LENGTH:
            raise ValueError(f"Task prompt exceeds {MAX_TASK_LENGTH} characters")
        condition = get_condition(condition_name)

        operation = self._exclusive_operation("generate code")
        try:
            session_id = uuid.uuid4().hex
            session_dir = self.session_root / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            image_path = self._image_capturer(session_dir / "scene_image.jpg").resolve()
            if not image_path.is_file():
                raise RuntimeError("Scene image capture did not create an image file")
            print(f"[CaP Generate] Scene image ready: {image_path}", flush=True)

            manifest_path = session_dir / "task.yaml"
            manifest = {
                "id": _slug(task),
                "task": task,
                "image": str(image_path),
                "point_cloud_frame": "base",
            }
            manifest_path.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )

            result = self._trial_runner(
                condition_name=condition.name.value,
                task_manifest=manifest_path,
                image_path=image_path,
                mode="generate",
                output_root=self.output_root,
            )
            artifact_dir = Path(result["artifact_dir"]).resolve()
            policy_path = artifact_dir / "policy.py"
            if not policy_path.is_file():
                raise RuntimeError(f"Generated policy is missing: {policy_path}")
            code = policy_path.read_text(encoding="utf-8")

            session = {
                "id": session_id,
                "task": task,
                "condition": condition.name.value,
                "session_dir": session_dir,
                "manifest_path": manifest_path,
                "image_path": image_path,
                "generation_artifact_dir": artifact_dir,
                "last_execution_artifact_dir": None,
                "preparations": {},
            }
            with self._state_lock:
                self._sessions[session_id] = session

            return json_safe(
                {
                    "session_id": session_id,
                    "condition": condition.name.value,
                    "api_kind": condition.api_kind.value,
                    "code": code,
                    "image_url": f"/api/sessions/{session_id}/image",
                    "artifact_dir": artifact_dir,
                    "model": result.get("model", {}),
                    "message": (
                        "Code generated. Review the displayed source, then "
                        "prepare perception."
                    ),
                }
            )
        finally:
            operation.release()

    @staticmethod
    def _validate_displayed_policy(code: str, api_kind):
        if not isinstance(code, str) or not code.strip():
            raise ValueError("The displayed policy is empty")
        if len(code) > MAX_POLICY_LENGTH:
            raise ValueError(f"Policy exceeds {MAX_POLICY_LENGTH} characters")
        return validate_policy(code, api_kind)

    @staticmethod
    def _policy_digest(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_manifest(path: Path, task: str, image_path: Path) -> Path:
        path.write_text(
            yaml.safe_dump(
                {
                    "id": _slug(task),
                    "task": task,
                    "image": str(image_path),
                    "point_cloud_frame": "base",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def prepare(self, session_id: str, code: str, fresh_scene: bool) -> Dict[str, Any]:
        """Freeze perception and render previews without opening robot interfaces."""
        with self._state_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("Unknown or expired UI session; generate the policy again")
            session = dict(session)

        condition = get_condition(session["condition"])
        validated = self._validate_displayed_policy(code, condition.api_kind)
        operation = self._exclusive_operation("prepare perception")
        try:
            preparation_id = uuid.uuid4().hex
            preparation_dir = Path(session["session_dir"]) / "preparations" / preparation_id
            preparation_dir.mkdir(parents=True, exist_ok=False)
            image_path = preparation_dir / "scene_image.jpg"
            if fresh_scene:
                image_path = self._image_capturer(image_path).resolve()
            else:
                shutil.copy2(Path(session["image_path"]), image_path)
                image_path = image_path.resolve()
            if not image_path.is_file():
                raise RuntimeError("Perception preparation has no frozen scene image")
            print(f"[CaP Prepare] Scene image ready: {image_path}", flush=True)

            manifest_path = self._write_manifest(
                preparation_dir / "task.yaml", session["task"], image_path
            )
            policy_path = preparation_dir / "reviewed_policy.py"
            policy_path.write_text(validated.source + "\n", encoding="utf-8")

            print("[CaP Prepare] Requesting point cloud...", flush=True)
            point_cloud = np.asarray(self._point_cloud_capturer(), dtype=float)
            print(
                f"[CaP Prepare] Point cloud received: shape={point_cloud.shape}",
                flush=True,
            )
            if point_cloud.ndim not in {2, 3} or point_cloud.shape[-1] != 3:
                raise RuntimeError(
                    f"Camera returned invalid point cloud shape {point_cloud.shape}"
                )
            point_cloud_path = preparation_dir / "point_cloud.npy"
            np.save(point_cloud_path, point_cloud, allow_pickle=False)

            if validated.keypoint_queries:
                localizer = self._perception_localizer or VisionPerceptionLocalizer()
                snapshot = localizer.localize(
                    queries=validated.keypoint_queries,
                    image_path=image_path,
                    point_cloud=point_cloud,
                    point_cloud_frame="base",
                )
            else:
                snapshot = PerceptionSnapshot(points={}, normals={}, pixels={})
            snapshot_path = preparation_dir / "perception_snapshot.json"
            snapshot_path.write_text(
                json.dumps(json_safe(snapshot.as_dict()), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            annotated_path = render_annotated_scene(
                image_path,
                snapshot.pixels,
                preparation_dir / "annotated_keypoints.png",
            )
            cloud_preview_path = render_point_cloud_preview(
                point_cloud,
                snapshot.points,
                snapshot.normals,
                preparation_dir / "point_cloud_preview.png",
            )
            # This is the same blocking Open3D view used by generated GTA robot
            # scripts. Preparation cannot complete—and execution cannot become
            # available—until the operator closes the popup.
            self._interactive_visualizer(
                point_cloud=point_cloud,
                image_path=image_path,
                points=snapshot.points,
                normals=snapshot.normals,
            )
            digest = self._policy_digest(validated.source)
            metadata = {
                "preparation_id": preparation_id,
                "fresh_scene": bool(fresh_scene),
                "policy_sha256": digest,
                "keypoint_query_count": len(validated.keypoint_queries),
                "interactive_visualization_closed": True,
                "image_path": str(image_path),
                "point_cloud_path": str(point_cloud_path),
                "perception_snapshot_path": str(snapshot_path),
            }
            (preparation_dir / "preparation.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            preparation = {
                **metadata,
                "directory": preparation_dir,
                "manifest_path": manifest_path,
                "policy_path": policy_path,
                "annotated_path": annotated_path,
                "cloud_preview_path": cloud_preview_path,
                "execution_started": False,
                "invalidated": False,
            }
            with self._state_lock:
                current = self._sessions.get(session_id)
                if current is None:
                    raise RuntimeError("UI session expired during perception preparation")
                for previous in current["preparations"].values():
                    previous["invalidated"] = True
                current["preparations"][preparation_id] = preparation
                if fresh_scene:
                    current["manifest_path"] = manifest_path
                    current["image_path"] = image_path

            base_url = f"/api/sessions/{session_id}/preparations/{preparation_id}"
            return json_safe(
                {
                    "preparation_id": preparation_id,
                    "fresh_scene": bool(fresh_scene),
                    "annotated_image_url": base_url + "/annotated_keypoints.png",
                    "point_cloud_preview_url": base_url + "/point_cloud_preview.png",
                    "scene_image_url": base_url + "/scene_image.jpg",
                    "points": snapshot.as_dict()["points"],
                    "normals": snapshot.as_dict()["normals"],
                    "preparation_dir": preparation_dir,
                    "message": (
                        "Perception prepared without opening robot control. The Open3D "
                        "review window was closed; inspect the annotated image, then "
                        "approve execution."
                    ),
                }
            )
        finally:
            operation.release()

    def execute_prepared(
        self,
        session_id: str,
        preparation_id: str,
        code: str,
        operator_approved: bool,
    ) -> Dict[str, Any]:
        if not operator_approved:
            raise ValueError(
                "Physical execution requires operator confirmation that the previews, "
                "cell, and robot-server state were checked"
            )

        with self._state_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("Unknown or expired UI session; generate the policy again")
            preparation = session["preparations"].get(preparation_id)
            if preparation is None:
                raise ValueError("Unknown or expired preparation; prepare perception again")
            if preparation["invalidated"]:
                raise ValueError(
                    "A newer preparation replaced this one; review the latest previews"
                )
            if preparation["execution_started"]:
                raise ValueError(
                    "This preparation has already been used; prepare perception again"
                )
            condition_name = session["condition"]

        condition = get_condition(condition_name)
        validated = self._validate_displayed_policy(code, condition.api_kind)
        if self._policy_digest(validated.source) != preparation["policy_sha256"]:
            raise ValueError("Policy changed after preparation; prepare perception again")

        operation = self._exclusive_operation("execute a prepared policy")
        try:
            with self._state_lock:
                current = self._sessions.get(session_id)
                active = current and current["preparations"].get(preparation_id)
                if (
                    active is None
                    or active["invalidated"]
                    or active["execution_started"]
                ):
                    raise ValueError("Preparation is no longer available")
                # One preparation authorizes at most one physical attempt. This
                # remains true if execution raises after partial robot motion.
                active["execution_started"] = True

            def execute_policy() -> Dict[str, Any]:
                return self._trial_runner(
                    condition_name=condition.name.value,
                    task_manifest=Path(preparation["manifest_path"]),
                    image_path=Path(preparation["image_path"]),
                    mode="robot",
                    output_root=self.output_root,
                    policy_file=Path(preparation["policy_path"]),
                    perception_path=Path(preparation["perception_snapshot_path"]),
                    approve_robot=True,
                )

            server_log_path = Path(preparation["directory"]) / "controller_server.log"
            managed_controller_server = condition.api_kind == ApiKind.CONTROLLER
            if managed_controller_server:
                with self._controller_server_factory(server_log_path) as server:
                    result = execute_policy()
                    server.assert_running("after policy execution")
            else:
                result = execute_policy()
            execution_artifact_dir = Path(result["artifact_dir"]).resolve()
            artifact_warnings = []
            artifact_sources = [
                (Path(preparation["annotated_path"]), "annotated_keypoints.png"),
                (Path(preparation["cloud_preview_path"]), "point_cloud_preview.png"),
                (Path(preparation["point_cloud_path"]), "point_cloud.npy"),
                (Path(preparation["directory"]) / "preparation.json", "preparation.json"),
            ]
            if managed_controller_server:
                artifact_sources.append((server_log_path, "controller_server.log"))
            for source, name in artifact_sources:
                try:
                    shutil.copy2(source, execution_artifact_dir / name)
                except Exception as exc:
                    # Physical execution has already returned successfully. Do
                    # not report a motion failure merely because optional
                    # artifact enrichment failed after the robot stopped.
                    artifact_warnings.append(f"Could not copy {name}: {exc}")
            with self._state_lock:
                if session_id in self._sessions:
                    self._sessions[session_id]["last_execution_artifact_dir"] = Path(
                        result["artifact_dir"]
                    ).resolve()
            return json_safe(
                {
                    "status": result["status"],
                    "artifact_dir": result["artifact_dir"],
                    "artifact_warnings": artifact_warnings,
                    "message": (
                        "Prepared policy execution completed. "
                        + (
                            "The CAP-managed controller robot server was stopped. "
                            if managed_controller_server
                            else ""
                        )
                        + "Prepare perception again before another physical run."
                        + (
                            " Artifact warning: " + " | ".join(artifact_warnings)
                            if artifact_warnings
                            else ""
                        )
                    ),
                }
            )
        finally:
            operation.release()

    def image_path(self, session_id: str) -> Path:
        with self._state_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("Unknown or expired UI session")
            path = Path(session["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def preparation_artifact_path(
        self, session_id: str, preparation_id: str, name: str
    ) -> Path:
        allowed = {
            "scene_image.jpg": "image_path",
            "annotated_keypoints.png": "annotated_path",
            "point_cloud_preview.png": "cloud_preview_path",
        }
        key = allowed.get(name)
        if key is None:
            raise ValueError("Unknown preparation artifact")
        with self._state_lock:
            session = self._sessions.get(session_id)
            preparation = session and session["preparations"].get(preparation_id)
            if preparation is None:
                raise ValueError("Unknown or expired preparation")
            path = Path(preparation[key])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def create_cap_web_app(service: Optional[CapWebService] = None) -> FastAPI:
    service = service or CapWebService()
    app = FastAPI(title="GTA-2 CaP Review and Run")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    @app.get("/api/config")
    async def config():
        return {
            "conditions": sorted(CONDITIONS),
            "default_condition": "cartesian_zeroshot",
        }

    @app.post("/api/generate", status_code=202)
    def generate(request: GenerateRequest):
        try:
            return service.start_generation(request.task, request.condition)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/generate/{job_id}")
    async def generation_status(job_id: str):
        try:
            return service.generation_status(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/prepare")
    def prepare(request: PrepareRequest):
        try:
            return service.prepare(
                request.session_id,
                request.code,
                request.fresh_scene,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/execute")
    def execute(request: ExecuteRequest):
        try:
            return service.execute_prepared(
                request.session_id,
                request.preparation_id,
                request.code,
                request.operator_approved,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/image")
    async def scene_image(session_id: str):
        try:
            return FileResponse(service.image_path(session_id), media_type="image/jpeg")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/api/sessions/{session_id}/preparations/{preparation_id}/{artifact_name}"
    )
    async def preparation_artifact(
        session_id: str, preparation_id: str, artifact_name: str
    ):
        try:
            path = service.preparation_artifact_path(
                session_id, preparation_id, artifact_name
            )
            media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            return FileResponse(path, media_type=media_type)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CaP Review & Run</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0b1020; color: #e7eaf2; }
    main { width: min(1180px, calc(100% - 32px)); margin: 28px auto; }
    h1 { margin: 0 0 6px; font-size: 1.7rem; }
    .subtitle { color: #9da8bd; margin: 0 0 20px; }
    .panel { background: #121a2d; border: 1px solid #26324b; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    label { display: block; font-weight: 650; margin-bottom: 8px; }
    textarea, select { width: 100%; border: 1px solid #35425f; border-radius: 8px; background: #080d19; color: #eef2ff; padding: 11px; }
    #task { min-height: 88px; resize: vertical; }
    #code { min-height: 430px; resize: vertical; font: 13px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; tab-size: 4; }
    .row { display: grid; grid-template-columns: 1fr 220px; gap: 14px; align-items: end; }
    button { border: 0; border-radius: 8px; padding: 11px 16px; font-weight: 700; cursor: pointer; background: #5b7cfa; color: white; }
    button.prepare { background: #387cc9; }
    button.fresh { background: #b96b24; }
    button.execute { background: #d14b53; }
    button:disabled { cursor: not-allowed; opacity: .42; }
    .actions { display: flex; gap: 12px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
    .confirm { display: flex; gap: 9px; align-items: flex-start; color: #c8cfdd; font-size: .93rem; }
    .confirm input { margin-top: 3px; }
    #status { white-space: pre-wrap; padding: 11px 13px; border-radius: 8px; background: #080d19; color: #aeb9cc; min-height: 44px; }
    #status.error { color: #ff9da3; border: 1px solid #80373c; }
    #scene { display: none; max-width: 100%; max-height: 330px; margin-top: 12px; border-radius: 8px; border: 1px solid #35425f; }
    #previewPanel { display: none; }
    .previews { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .previews figure { margin: 0; min-width: 0; }
    .previews figcaption { color: #aeb9cc; font-weight: 650; margin-bottom: 8px; }
    .previews img { width: 100%; max-height: 480px; object-fit: contain; border-radius: 8px; border: 1px solid #35425f; background: #080d19; }
    #preparedVectors { margin: 12px 0 0; padding: 10px; max-height: 180px; overflow: auto; border-radius: 8px; background: #080d19; color: #aeb9cc; font: 12px/1.45 ui-monospace, monospace; }
    .path { color: #8592aa; font: 12px/1.4 ui-monospace, monospace; overflow-wrap: anywhere; }
    @media (max-width: 720px) { .row, .previews { grid-template-columns: 1fr; } main { width: min(100% - 20px, 1180px); } }
  </style>
</head>
<body>
<main>
  <h1>CaP Review & Run</h1>
  <p class="subtitle">Generate the policy, prepare perception in the interactive Open3D view, then explicitly approve execution.</p>

  <section class="panel">
    <div class="row">
      <div>
        <label for="task">Task prompt</label>
        <textarea id="task" placeholder="Describe the manipulation task..."></textarea>
      </div>
      <div>
        <label for="condition">Benchmark condition</label>
        <select id="condition">
          <option value="cartesian_zeroshot">Cartesian · zero-shot</option>
          <option value="cartesian_fewshot">Cartesian · few-shot</option>
          <option value="controller_zeroshot">Controller · zero-shot</option>
          <option value="controller_fewshot">Controller · few-shot</option>
        </select>
      </div>
    </div>
    <div class="actions">
      <button id="generate">Generate code</button>
      <span id="generationPath" class="path"></span>
    </div>
    <img id="scene" alt="Frozen scene used for generation">
  </section>

  <section class="panel">
    <label for="code">Generated policy — review or edit before running</label>
    <textarea id="code" spellcheck="false" disabled></textarea>
    <div class="actions">
      <button id="prepare" class="prepare" disabled>Prepare current scene</button>
      <button id="prepareFresh" class="fresh" disabled>Prepare fresh-scene rerun</button>
    </div>
  </section>

  <section id="previewPanel" class="panel">
    <label>Prepared perception — inspect before execution</label>
    <div class="previews">
      <figure>
        <figcaption>Annotated frozen RGB</figcaption>
        <img id="annotatedPreview" alt="Prepared keypoints over the frozen RGB image">
      </figure>
      <figure>
        <figcaption>Static audit copy of the interactive Open3D review</figcaption>
        <img id="cloudPreview" alt="Prepared point cloud visualization">
      </figure>
    </div>
    <pre id="preparedVectors"></pre>
    <div class="actions">
      <label class="confirm">
        <input id="approved" type="checkbox" disabled>
        <span id="approvalText">I checked these previews, the robot cell, and server state.</span>
      </label>
      <button id="execute" class="execute" disabled>Execute prepared policy</button>
    </div>
  </section>

  <section class="panel">
    <label>Status</label>
    <div id="status">Ready. Generation captures a fresh camera frame.</div>
    <div id="executionPath" class="path"></div>
  </section>
</main>
<script>
  const task = document.getElementById('task');
  const condition = document.getElementById('condition');
  const code = document.getElementById('code');
  const generateButton = document.getElementById('generate');
  const prepareButton = document.getElementById('prepare');
  const prepareFreshButton = document.getElementById('prepareFresh');
  const executeButton = document.getElementById('execute');
  const approved = document.getElementById('approved');
  const approvalText = document.getElementById('approvalText');
  const status = document.getElementById('status');
  const scene = document.getElementById('scene');
  const previewPanel = document.getElementById('previewPanel');
  const annotatedPreview = document.getElementById('annotatedPreview');
  const cloudPreview = document.getElementById('cloudPreview');
  const preparedVectors = document.getElementById('preparedVectors');
  const generationPath = document.getElementById('generationPath');
  const executionPath = document.getElementById('executionPath');
  let sessionId = null;
  let preparationId = null;

  function updateApprovalText() {
    const cartesian = condition.value.startsWith('cartesian_');
    approvalText.textContent = cartesian
      ? 'I reviewed the Open3D popup and annotated RGB, checked the cell and reset pose, and confirmed the regular RTDE robot server is STOPPED.'
      : 'I reviewed the Open3D popup and annotated RGB, checked the cell and reset pose, and confirmed no controller ROS/RTDE robot server is already running. CAP will start and stop it automatically.';
  }

  function showStatus(message, isError = false) {
    status.textContent = message;
    status.classList.toggle('error', isError);
  }

  async function api(url, payload) {
    const response = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    return body;
  }

  async function apiGet(url) {
    const response = await fetch(url, {method: 'GET'});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    return body;
  }

  function delay(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
  }

  async function waitForGeneration(jobId) {
    const started = Date.now();
    let consecutivePollFailures = 0;
    while (true) {
      let job;
      try {
        job = await apiGet(`/api/generate/${jobId}`);
        consecutivePollFailures = 0;
      } catch (error) {
        consecutivePollFailures += 1;
        if (consecutivePollFailures >= 10) throw error;
        showStatus(
          `Generation is still running; status connection failed ` +
          `(${consecutivePollFailures}/10), retrying...`
        );
        await delay(1000);
        continue;
      }

      if (job.status === 'completed') return job.result;
      if (job.status === 'failed') {
        throw new Error(job.error || 'Policy generation failed');
      }
      const elapsed = Math.round((Date.now() - started) / 1000);
      showStatus(`Generating policy in the background... ${elapsed}s elapsed`);
      await delay(1000);
    }
  }

  function invalidatePreparation(message = null) {
    preparationId = null;
    approved.checked = false;
    approved.disabled = true;
    executeButton.disabled = true;
    previewPanel.style.display = 'none';
    annotatedPreview.removeAttribute('src');
    cloudPreview.removeAttribute('src');
    preparedVectors.textContent = '';
    if (message) showStatus(message);
  }

  condition.addEventListener('change', () => {
    updateApprovalText();
    if (sessionId) {
      sessionId = null;
      code.value = '';
      code.disabled = true;
      prepareButton.disabled = true;
      prepareFreshButton.disabled = true;
      invalidatePreparation();
      scene.style.display = 'none';
      generationPath.textContent = '';
      executionPath.textContent = '';
      showStatus('Condition changed. Generate a new policy.');
    }
  });

  code.addEventListener('input', () => {
    if (preparationId) {
      invalidatePreparation('Policy edited. Prepare perception again before execution.');
    }
  });

  approved.addEventListener('change', () => {
    executeButton.disabled = !preparationId || !approved.checked;
  });

  generateButton.addEventListener('click', async () => {
    generateButton.disabled = true;
    prepareButton.disabled = true;
    prepareFreshButton.disabled = true;
    invalidatePreparation();
    sessionId = null;
    code.disabled = true;
    code.value = '';
    executionPath.textContent = '';
    scene.style.display = 'none';
    showStatus('Starting background policy generation...');
    try {
      const job = await api('/api/generate', {
        task: task.value,
        condition: condition.value,
      });
      const result = await waitForGeneration(job.job_id);
      sessionId = result.session_id;
      code.value = result.code;
      code.disabled = false;
      prepareButton.disabled = false;
      prepareFreshButton.disabled = false;
      scene.src = `${result.image_url}?v=${Date.now()}`;
      scene.style.display = 'block';
      generationPath.textContent = `Generation artifacts: ${result.artifact_dir}`;
      showStatus(
        `${result.message}\nPrepare current scene uses this image. Prepare fresh-scene ` +
        `rerun captures the latest RGB frame without another policy-model call.`
      );
    } catch (error) {
      showStatus(error.message, true);
    } finally {
      generateButton.disabled = false;
    }
  });

  async function preparePerception(freshScene) {
    invalidatePreparation();
    generateButton.disabled = true;
    prepareButton.disabled = true;
    prepareFreshButton.disabled = true;
    code.disabled = true;
    executionPath.textContent = '';
    showStatus(
      freshScene
        ? 'Capturing fresh RGB and point cloud. An interactive Open3D window will open; inspect it and close it to finish preparation...'
        : 'Capturing the point cloud. An interactive Open3D window will open; inspect it and close it to finish preparation...'
    );
    try {
      const result = await api('/api/prepare', {
        session_id: sessionId,
        code: code.value,
        fresh_scene: freshScene,
      });
      preparationId = result.preparation_id;
      scene.src = `${result.scene_image_url}?v=${Date.now()}`;
      scene.style.display = 'block';
      annotatedPreview.src = `${result.annotated_image_url}?v=${Date.now()}`;
      cloudPreview.src = `${result.point_cloud_preview_url}?v=${Date.now()}`;
      preparedVectors.textContent = JSON.stringify(
        {points: result.points, normals: result.normals}, null, 2
      );
      previewPanel.style.display = 'block';
      approved.disabled = false;
      approved.checked = false;
      executeButton.disabled = true;
      showStatus(result.message);
    } catch (error) {
      invalidatePreparation();
      showStatus(error.message, true);
    } finally {
      generateButton.disabled = false;
      code.disabled = false;
      prepareButton.disabled = !sessionId;
      prepareFreshButton.disabled = !sessionId;
    }
  }

  prepareButton.addEventListener('click', async () => {
    await preparePerception(false);
  });

  prepareFreshButton.addEventListener('click', async () => {
    await preparePerception(true);
  });

  executeButton.addEventListener('click', async () => {
    executeButton.disabled = true;
    generateButton.disabled = true;
    prepareButton.disabled = true;
    prepareFreshButton.disabled = true;
    code.disabled = true;
    approved.disabled = true;
    showStatus(
      condition.value.startsWith('controller_')
        ? 'Starting the CAP-managed controller robot server, waiting for it to settle, then executing the exact prepared policy...'
        : 'Executing the exact prepared policy and perception snapshot...'
    );
    try {
      const result = await api('/api/execute', {
        session_id: sessionId,
        preparation_id: preparationId,
        code: code.value,
        operator_approved: approved.checked,
      });
      executionPath.textContent = `Execution artifacts: ${result.artifact_dir}`;
      showStatus(result.message);
    } catch (error) {
      showStatus(`${error.message}\nPrepare perception again before retrying.`, true);
    } finally {
      preparationId = null;
      approved.checked = false;
      approved.disabled = true;
      executeButton.disabled = true;
      generateButton.disabled = false;
      code.disabled = false;
      prepareButton.disabled = !sessionId;
      prepareFreshButton.disabled = !sessionId;
    }
  });

  updateApprovalText();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the minimal CaP review-and-run UI")
    parser.add_argument(
        "--host", default=os.getenv("CAP_WEB_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("CAP_WEB_PORT", "8010"))
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "benchmark_runs" / "cap",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    service = CapWebService(output_root=args.output)
    app = create_cap_web_app(service)
    print(f"CaP review-and-run UI: http://{args.host}:{args.port}")
    print(f"Trial artifacts: {service.output_root}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
