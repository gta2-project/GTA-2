"""Application service for the GTA Sessions web dashboard."""

from __future__ import annotations

import asyncio
import importlib.util
import hashlib
import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agents import (
    ParameterSettingAgent,
    SkillGeneratorAgent,
    TaskDecomposerAgent,
    VisionModuleAgent,
)
from communication import Message, MessageBus
from config import Config
from gta.script_generator import generate_robot_script
from gta.plan_validation import validate_plan


AGENT_NAMES = {
    "TaskDecomposer",
    "SkillGenerator",
    "ParameterSetter",
    "VisionModule",
}

ARTIFACT_NAMES = {
    "annotated_keypoints.png",
    "environment.jpg",
    "keypoint_locations.json",
    "resized_img.png",
    "robot_script.py",
    "skills_output.json",
}

WORKFLOW_SAVE_ARTIFACTS = (
    "annotated_keypoints.png",
    "environment.jpg",
    "keypoint_locations.json",
    "resized_img.png",
    "robot_script.py",
    "skills_output.json",
)


class GtaWebService:
    """Owns the long-lived agents, workflow state, and event stream."""

    def __init__(self, root_dir: Path, scoped: bool = False):
        self.root_dir = Path(root_dir)
        self.scoped = scoped
        self.api_prefix = '/api'
        self.execution_guard = lambda: None
        self.message_bus: Optional[MessageBus] = None
        self.agents: Dict[str, Any] = {}
        self.initialized = False
        self.init_error: Optional[str] = None

        self._lock = threading.RLock()
        self._clients: List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._events: List[Dict[str, Any]] = []
        self._state: Dict[str, Any] = self._empty_state()

    def _empty_state(self) -> Dict[str, Any]:
        return {
            "task": None,
            "subtasks": [],
            "skills": [],
            "parameters": [],
            "keypoints": {},
            "messages": [],
            "events": [],
            "artifacts": {},
            "busy_stage": None,
            "last_error": None,
            "script": None,
            "script_path": None,
            "revision": 0,
            "grounding_revision": None,
            "script_revision": None,
            "script_sha256": None,
            "image_revision": None,
        }

    def assert_idle(self):
        self.execution_guard()
        if (self._state.get('busy_stage') or
                (self.message_bus and getattr(self.message_bus, 'active_handlers', 0)) or
                any(getattr(agent, 'processing', False) for agent in self.agents.values())):
            raise RuntimeError('Workflow is busy; wait for all agent work to finish')

    def invalidate(self):
        self._state['revision'] += 1
        self._state['grounding_revision'] = None
        self._state['script_revision'] = None
        self._state['script_sha256'] = None
        self._state['script'] = None
        self._state['script_path'] = None
        self._state['last_error'] = None

    def _create_agents(self, snapshots=None):
        self.message_bus = MessageBus()
        self.message_bus.lifecycle_lock = self._lock
        self.message_bus.workspace = self.root_dir
        self.message_bus.lazy_clients = True
        self.message_bus.agent_snapshots = snapshots or {}
        self.message_bus.subscribe('broadcast', self._on_bus_message)
        self.agents = {
            'TaskDecomposer': TaskDecomposerAgent(self.message_bus),
            'SkillGenerator': SkillGeneratorAgent(self.message_bus),
            'ParameterSetter': ParameterSettingAgent(self.message_bus),
            'VisionModule': VisionModuleAgent(self.message_bus, str(self._artifact_file_path('environment.jpg'))),
        }
        # Restore only active agents; older checkpoints can also contain a retired agent.
        for name, agent in self.agents.items():
            if name in (snapshots or {}):
                agent.restore_checkpoint_state(snapshots[name])

    def initialize(self) -> bool:
        """Initialize agents once and retain any configuration error for health."""
        with self._lock:
            if self.initialized:
                return True

            try:
                Config.validate()
                self._create_agents()
                self.initialized = True
                self.init_error = None
                self._emit_event(
                    {
                        "type": "system",
                        "sender": "System",
                        "receiver": "ALL",
                        "message_type": "initialized",
                        "content": {
                            "agents": list(self.agents.keys()),
                            "environment_image": str(self._artifact_file_path('environment.jpg')),
                        },
                        "level": "info",
                    }
                )
                return True
            except Exception as exc:
                self.initialized = False
                self.init_error = str(exc)
                self._set_error(str(exc))
                return False

    def health(self) -> Dict[str, Any]:
        robot_deps = {
            "rospy": importlib.util.find_spec("rospy") is not None,
            "rclpy": importlib.util.find_spec("rclpy") is not None,
            "rtde_receive": importlib.util.find_spec("rtde_receive") is not None,
        }
        agent_models = {
            name: {
                "provider": getattr(agent, "provider", None),
                "model": getattr(agent, "model", None),
            }
            for name, agent in self.agents.items()
        }
        return {
            "ready": self.initialized and self.init_error is None,
            "initialized": self.initialized,
            "init_error": self.init_error,
            "camera_backend": Config.CAMERA_BACKEND,
            "gta_backend": Config.GTA_BACKEND,
            "camera_image_path": Config.CAMERA_IMAGE_PATH,
            "robot_dependencies": robot_deps,
            "agent_models": agent_models,
        }

    def state(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = self._json_safe(self._state)
            snapshot["events"] = list(self._events[-100:])
            snapshot["messages"] = self._message_history()
            snapshot["artifacts"] = self.artifact_status()
            snapshot['active_handlers'] = getattr(self.message_bus, 'active_handlers', 0)
            snapshot['grounding_ready'] = self._state['grounding_revision'] == self._state['revision']
            snapshot['script_ready'] = snapshot['grounding_ready'] and self._state['script_revision'] == self._state['revision']
            return snapshot

    def artifact_status(self) -> Dict[str, Dict[str, Any]]:
        artifacts = {}
        for name in sorted(ARTIFACT_NAMES):
            path = self._artifact_file_path(name)
            exists = path.exists()
            artifacts[name] = {
                "name": name,
                "exists": exists,
                "url": f"{self.api_prefix}/artifacts/{name}" if exists else None,
                "size": path.stat().st_size if exists else None,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
            }
        return artifacts

    def artifact_path(self, name: str) -> Path:
        if "/" in name or "\\" in name or name not in ARTIFACT_NAMES:
            raise ValueError("Unknown artifact")

        path = self._artifact_file_path(name)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def _artifact_file_path(self, name: str) -> Path:
        if name == "environment.jpg" and not self.scoped:
            path = Path(Config.CAMERA_IMAGE_PATH)
            return path if path.is_absolute() else self.root_dir / path
        return self.root_dir / name

    @staticmethod
    def _folder_slug(value: Optional[str]) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", (value or "").strip().lower()).strip("_")
        return slug or "task"

    def capture_environment_image(self) -> Dict[str, Any]:
        """Fetch the configured camera image and refresh the environment artifact."""
        from scene.camera import Camera

        path = self._artifact_file_path("environment.jpg")
        camera = Camera(
            camera_node=Config.CAMERA_NODE,
            camera_model=Config.CAMERA_MODEL,
            rosmaster_ip=Config.ROSMASTER_IP,
            rosmaster_port=Config.ROSMASTER_PORT,
            camera_backend=Config.CAMERA_BACKEND,
            camera_image_path=str(path),
        )
        rgb_image = camera.get_rgb_image(use_new_frame=True)
        if rgb_image is None or len(rgb_image.shape) < 3 or rgb_image.shape[2] < 3:
            raise RuntimeError("Camera did not return an RGB image")

        saved = False
        if Config.CAMERA_BACKEND in {"ros1", "ros2"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_image[:, :, :3]).save(path)
            saved = True
        elif not path.exists():
            raise FileNotFoundError(f"Camera image file not found: {path}")

        self._emit_event(
            {
                "type": "artifact",
                "sender": "System",
                "receiver": "WebDashboard",
                "message_type": "environment_captured",
                "content": {
                    "path": str(path),
                    "source": Config.CAMERA_BACKEND,
                    "saved": saved,
                },
                "level": "info",
            }
        )
        return {"path": str(path), "source": Config.CAMERA_BACKEND, "saved": saved}

    def start_workflow(self, task: str) -> Dict[str, Any]:
        self._require_initialized()
        if not task.strip():
            raise ValueError("Task cannot be empty")

        with self._lock:
            self.assert_idle()
            # Refresh the GUI environment artifact before resetting the prior
            # workflow so a capture failure leaves the existing state intact.
            self.capture_environment_image()
            self._reset_workflow_state(task.strip())
            self.invalidate()
            self._set_busy("Decomposing task")

        self._emit_event(
            {
                "type": "workflow",
                "sender": "User",
                "receiver": "TaskDecomposer",
                "message_type": "workflow_started",
                "content": {"task": task.strip()},
                "level": "info",
            }
        )
        self.message_bus.run_background(self._run_start_workflow, task.strip())
        return {"accepted": True, "busy_stage": "Decomposing task"}

    def send_agent_message(self, agent_name: str, message: str) -> Dict[str, Any]:
        self._require_initialized()
        if agent_name not in AGENT_NAMES:
            raise ValueError(f"Unknown agent: {agent_name}")
        if not message.strip():
            raise ValueError("Message cannot be empty")

        with self._lock:
            self.assert_idle()
            if agent_name == 'VisionModule' and self._state['grounding_revision'] != self._state['revision']:
                raise RuntimeError('Requery the current scene before refining its keypoints')
            self.invalidate()
            self._set_busy(f"Processing {agent_name}")

        self._emit_event(
            {
                "type": "user_message",
                "sender": "User",
                "receiver": agent_name,
                "message_type": "manual_message",
                "content": {"message": message.strip()},
                "level": "info",
            }
        )
        self.message_bus.run_background(self._run_agent_message, agent_name, message.strip())
        return {"accepted": True, "busy_stage": f"Processing {agent_name}"}

    def requery_vision(self) -> Dict[str, Any]:
        self._require_initialized()
        with self._lock:
            self.assert_idle()
            self.invalidate()
            self._set_busy("Requerying vision")

        assert self.message_bus is not None
        self.message_bus.publish(Message("WebDashboard", "VisionModule", "requery_vision",
                                         {'skills': self._state.get('parameters') or []}))
        return {"accepted": True, "busy_stage": "Requerying vision"}

    def generate_script(self) -> Dict[str, Any]:
        with self._lock:
            self.assert_idle()
            if (self.initialized or self.scoped) and self._state['grounding_revision'] != self._state['revision']:
                raise RuntimeError('Requery and review the current scene before generating a script')
            return self._generate_script()

    def _generate_script(self) -> Dict[str, Any]:
        self._state['script_revision'] = None
        self._state['script'] = None
        self._state['script_path'] = None
        try:
            if self.initialized or self.scoped:
                validate_plan(self._state['parameters'])
            result = generate_robot_script(
                keypoint_locations_path=self.root_dir / "keypoint_locations.json",
                skills_output_path=self.root_dir / "skills_output.json",
                output_path=self.root_dir / "robot_script.py",
                strict=self.initialized or self.scoped,
            )
        except Exception as exc:
            self._set_error(str(exc))
            self._emit_event(
                {
                    "type": "error",
                    "sender": "System",
                    "receiver": "WebDashboard",
                    "message_type": "script_generation_failed",
                    "content": {"error": str(exc)},
                    "level": "error",
                }
            )
            raise

        with self._lock:
            self._state["script"] = result["script"]
            self._state["script_path"] = result["output_path"]
            self._state["last_error"] = None
            self._state['script_revision'] = self._state['revision']
            self._state['script_sha256'] = hashlib.sha256(result['script'].encode('utf-8')).hexdigest()

        self._emit_event(
            {
                "type": "artifact",
                "sender": "System",
                "receiver": "WebDashboard",
                "message_type": "script_generated",
                "content": {
                    "path": result["output_path"],
                    "keypoint_count": result["keypoint_count"],
                    "skill_count": result["skill_count"],
                },
                "level": "info",
            }
        )
        return result

    def save_workflow(self) -> Dict[str, Any]:
        with self._lock:
            self.assert_idle()
            return self._save_workflow()

    def _save_workflow(self) -> Dict[str, Any]:
        saved_at = datetime.now()
        sources = {
            name: self._artifact_file_path(name)
            for name in WORKFLOW_SAVE_ARTIFACTS
        }
        missing = [
            name
            for name, path in sources.items()
            if not path.exists() or not path.is_file()
        ]
        if missing:
            error = f"Missing workflow artifacts: {', '.join(missing)}"
            self._emit_event(
                {
                    "type": "error",
                    "sender": "System",
                    "receiver": "WebDashboard",
                    "message_type": "workflow_save_failed",
                    "content": {"error": error, "missing": missing},
                    "level": "error",
                }
            )
            raise FileNotFoundError(error)

        generated_plans_dir = self.root_dir / "generated_plans"
        timestamp = saved_at.strftime("%Y%m%d_%H%M%S")
        task_slug = self._folder_slug(self._state.get("task"))
        save_dir = generated_plans_dir / f"{task_slug}_{timestamp}"
        if save_dir.exists():
            save_dir = generated_plans_dir / f"{task_slug}_{timestamp}_{saved_at.strftime('%f')}"

        try:
            save_dir.mkdir(parents=True, exist_ok=False)
            copied_files = []
            for name, source_path in sources.items():
                destination = save_dir / name
                shutil.copy2(source_path, destination)
                copied_files.append(name)
        except Exception as exc:
            if save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)
            self._emit_event(
                {
                    "type": "error",
                    "sender": "System",
                    "receiver": "WebDashboard",
                    "message_type": "workflow_save_failed",
                    "content": {"error": str(exc)},
                    "level": "error",
                }
            )
            raise

        folder = save_dir.relative_to(self.root_dir).as_posix()
        result = {
            "folder": folder,
            "saved_at": saved_at.isoformat(),
            "files": copied_files,
        }
        self._emit_event(
            {
                "type": "artifact",
                "sender": "System",
                "receiver": "WebDashboard",
                "message_type": "workflow_saved",
                "content": result,
                "level": "info",
            }
        )
        return result

    def register_client(self) -> Tuple[asyncio.Queue, List[Dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._clients.append((loop, queue))
            backlog = list(self._events[-50:])
        return queue, backlog

    def unregister_client(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._clients = [
                (loop, client_queue)
                for loop, client_queue in self._clients
                if client_queue is not queue
            ]

    def _require_initialized(self) -> None:
        if not self.initialized:
            self.initialize()
        if not self.initialized:
            raise RuntimeError(self.init_error or "Backend failed to initialize")

    def _reset_workflow_state(self, task: str) -> None:
        self._state = self._empty_state()
        self._state["task"] = task
        self._events = []
        if self.message_bus is not None:
            self.message_bus.clear_history()

        task_decomposer = self.agents.get("TaskDecomposer")
        skill_generator = self.agents.get("SkillGenerator")
        parameter_setter = self.agents.get("ParameterSetter")
        vision_module = self.agents.get("VisionModule")

        if task_decomposer is not None:
            task_decomposer.current_task = None
            task_decomposer.current_subtasks = []
            task_decomposer.reset_conversation()
        if skill_generator is not None:
            skill_generator.current_skills = []
            skill_generator.processing = False
            skill_generator.reset_conversation()
        if parameter_setter is not None:
            parameter_setter.current_skills = []
            parameter_setter.processing = False
            parameter_setter.reset_conversation()
        if vision_module is not None:
            vision_module.current_skills = []
            vision_module.keypoint_locations = {}
            vision_module.current_keypoints = []
            vision_module.current_queries = []
            vision_module.last_image_np = None
            vision_module.last_image_size = None
            vision_module.last_resized_path = None
            vision_module.processing = False
            vision_module.reset_conversation()

    def _run_start_workflow(self, task: str) -> None:
        try:
            result = self.agents["TaskDecomposer"].generate_subtasks(task)
            if result.get("error"):
                self._set_error(result["error"])
                self._set_busy(None)
            elif not result.get("subtasks"):
                self._set_error("No subtasks were generated.")
                self._set_busy(None)
        except Exception as exc:
            self._set_error(str(exc))
            self._set_busy(None)
            self._emit_exception("TaskDecomposer", exc)

    def _run_agent_message(self, agent_name: str, message: str) -> None:
        try:
            response: Any = None
            if agent_name == "TaskDecomposer":
                agent = self.agents["TaskDecomposer"]
                if agent.current_task is None:
                    response = agent.generate_subtasks(message)
                else:
                    response = agent.refine_subtasks(message)
            elif agent_name == "SkillGenerator":
                agent = self.agents["SkillGenerator"]
                if agent.get_current_skills():
                    response = agent.refine_skills(message)
                else:
                    response = {"warning": "No skills have been generated yet."}
                    self._set_busy(None)
            elif agent_name == "ParameterSetter":
                agent = self.agents["ParameterSetter"]
                if agent.get_current_skills():
                    response = agent.refine_parameters(message)
                    agent.notify_agent(
                        agent_name="VisionModule",
                        notification_type="parameters_completed",
                        data={"skills": response, "action": "refined"},
                    )
                else:
                    response = {"warning": "No parameters have been set yet."}
                    self._set_busy(None)
            elif agent_name == "VisionModule":
                response = self.agents["VisionModule"].refine_keypoints(message)
                if response.get("error"):
                    self._set_error(response["error"])
                    self._set_busy(None)
            else:
                response = {"warning": f"Unknown agent: {agent_name}"}
                self._set_busy(None)

            if response is not None:
                if isinstance(response, dict) and response.get('error'):
                    self._set_error(response['error'])
                    self._set_busy(None)
                self._emit_event(
                    {
                        "type": "agent_response",
                        "sender": agent_name,
                        "receiver": "WebDashboard",
                        "message_type": "manual_response",
                        "content": response,
                        "level": "warning" if isinstance(response, dict) and response.get("warning") else "info",
                    }
                )
        except Exception as exc:
            self._set_error(str(exc))
            self._set_busy(None)
            self._emit_exception(agent_name, exc)

    def _on_bus_message(self, message: Message) -> None:
        message_dict = message.to_dict()
        with self._lock:
            if message.message_type == "subtasks_updated" and message.sender == "TaskDecomposer":
                self._state["task"] = message.content.get("task", self._state.get("task"))
                self._state["subtasks"] = message.content.get("subtasks", [])
                self._state["last_error"] = None
                self._state["busy_stage"] = "Generating skills"
            elif message.message_type == "skills_updated" and message.sender == "SkillGenerator":
                self._state["skills"] = message.content.get("skills", [])
                self._state["last_error"] = None
                self._state["busy_stage"] = "Setting parameters"
            elif message.message_type == "parameters_completed" and message.sender == "ParameterSetter":
                skills = message.content.get("skills", [])
                self._state["parameters"] = skills
                self._state["last_error"] = None
                self._state["busy_stage"] = "Detecting keypoints" if skills else None
                if not skills:
                    self._state["last_error"] = "No parameterized skills are available for vision."
            elif message.message_type == "vision_completed" and message.sender == "VisionModule":
                self._state["keypoints"] = message.content.get("keypoint_locations", {})
                self._state["last_error"] = message.content.get("warning")
                self._state["busy_stage"] = None
                if not message.content.get('warning'):
                    self._state['grounding_revision'] = self._state['revision']
                    self._state['image_revision'] = self._state['revision']
            elif message.message_type in {"agent_error", "vision_failed"}:
                self._state["last_error"] = message.content.get("error", "Agent failed")
                self._state["busy_stage"] = None
                self._state['grounding_revision'] = None
                self._state['script_revision'] = None

        event_level = "error" if message.message_type in {"agent_error", "vision_failed"} else "info"
        self._emit_event(
            {
                "type": "message",
                "sender": message.sender,
                "receiver": message.receiver,
                "message_type": message.message_type,
                "content": message_dict.get("content"),
                "level": event_level,
            }
        )

    def _message_history(self) -> List[Dict[str, Any]]:
        if self.message_bus is None:
            return []
        return [message.to_dict() for message in self.message_bus.get_all_messages()]

    def _set_busy(self, stage: Optional[str]) -> None:
        with self._lock:
            self._state["busy_stage"] = stage

    def _set_error(self, error: str) -> None:
        with self._lock:
            self._state["last_error"] = error

    def _emit_exception(self, sender: str, exc: Exception) -> None:
        self._emit_event(
            {
                "type": "error",
                "sender": sender,
                "receiver": "WebDashboard",
                "message_type": "exception",
                "content": {"error": str(exc)},
                "level": "error",
            }
        )

    def _emit_event(self, partial: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "id": partial.get("id", str(uuid.uuid4())),
            "timestamp": datetime.now().isoformat(),
            "type": partial.get("type", "message"),
            "sender": partial.get("sender", "System"),
            "receiver": partial.get("receiver", "WebDashboard"),
            "message_type": partial.get("message_type", "event"),
            "content": self._json_safe(partial.get("content", {})),
            "level": partial.get("level", "info"),
        }

        with self._lock:
            if event['type'] != 'execution':
                self._events.append(event)
                self._state["events"] = list(self._events[-100:])
            clients = list(self._clients)

        for loop, queue in clients:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, event)
        return event

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:
            return str(value)
