"""Lifecycle management for the controller benchmark's robot server process."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_SCRIPT = (
    REPOSITORY_ROOT / "scene" / "robot" / "server.py"
)
READY_MARKER = "[UR5e Server] Waiting for skill execution commands..."
CONTROLLER_TOPICS = {
    "/robot/position_error",
    "/robot/rotation_error",
    "/robot/force_command",
    "/robot/gripper_command",
    "/robot/skill_status",
}


class ControllerServerError(RuntimeError):
    pass


def active_controller_nodes() -> list[str]:
    """Return ROS nodes already subscribed to every controller command topic."""
    try:
        import rosgraph

        _, subscribers, _ = rosgraph.Master(
            "/cap_controller_server_manager"
        ).getSystemState()
    except Exception:
        return []

    nodes_by_topic = {
        str(topic): {str(node) for node in nodes}
        for topic, nodes in subscribers
    }
    common_nodes: Optional[set[str]] = None
    for topic in CONTROLLER_TOPICS:
        topic_nodes = nodes_by_topic.get(topic, set())
        common_nodes = (
            set(topic_nodes)
            if common_nodes is None
            else common_nodes.intersection(topic_nodes)
        )
    return sorted(common_nodes or set())


class ManagedControllerServer:
    """Start, verify, log, and stop one CAP-owned controller server process."""

    def __init__(
        self,
        log_path: Path,
        server_script: Path = DEFAULT_SERVER_SCRIPT,
        startup_timeout_s: float = 20.0,
        startup_settle_s: float = 2.0,
        shutdown_settle_s: float = 0.25,
        stop_timeout_s: float = 8.0,
        existing_node_detector: Callable[[], Iterable[str]] = active_controller_nodes,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.log_path = Path(log_path).resolve()
        self.server_script = Path(server_script).resolve()
        self.startup_timeout_s = float(startup_timeout_s)
        self.startup_settle_s = float(startup_settle_s)
        self.shutdown_settle_s = float(shutdown_settle_s)
        self.stop_timeout_s = float(stop_timeout_s)
        self._existing_node_detector = existing_node_detector
        self._process_factory = process_factory
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._log_handle = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._tail = deque(maxlen=30)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
        return False

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def _validate_timings(self) -> None:
        values = {
            "startup_timeout_s": self.startup_timeout_s,
            "startup_settle_s": self.startup_settle_s,
            "shutdown_settle_s": self.shutdown_settle_s,
            "stop_timeout_s": self.stop_timeout_s,
        }
        for name, value in values.items():
            if not (value >= 0.0 and value < float("inf")):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.startup_timeout_s == 0.0 or self.stop_timeout_s == 0.0:
            raise ValueError("Controller server startup and stop timeouts must be positive")

    def start(self) -> None:
        if self._process is not None:
            raise ControllerServerError("Controller server manager was already started")
        self._validate_timings()
        if not self.server_script.is_file():
            raise ControllerServerError(
                f"Controller server script is missing: {self.server_script}"
            )

        existing_nodes = sorted(str(node) for node in self._existing_node_detector())
        if existing_nodes:
            raise ControllerServerError(
                "A controller robot server is already registered with ROS: "
                + ", ".join(existing_nodes)
                + ". Stop it before CAP execution so RTDE Control has one owner."
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        command = [sys.executable, "-u", str(self.server_script)]
        try:
            self._process = self._process_factory(
                command,
                cwd=str(REPOSITORY_ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception:
            self._close_log()
            raise

        self._reader_thread = threading.Thread(
            target=self._drain_output,
            name="cap-controller-server-log",
            daemon=True,
        )
        self._reader_thread.start()

        deadline = time.monotonic() + self.startup_timeout_s
        try:
            while time.monotonic() < deadline:
                self.assert_running("during startup")
                remaining = max(deadline - time.monotonic(), 0.0)
                try:
                    line = self._lines.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if READY_MARKER in line:
                    self._settle_after_startup()
                    return
            raise ControllerServerError(
                f"Controller server did not become ready within "
                f"{self.startup_timeout_s:.1f}s.{self._tail_text()}"
            )
        except Exception:
            self.stop()
            raise

    def _settle_after_startup(self) -> None:
        deadline = time.monotonic() + self.startup_settle_s
        while time.monotonic() < deadline:
            self.assert_running("while settling")
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))

    def assert_running(self, phase: str = "") -> None:
        if self._process is None:
            raise ControllerServerError("Controller server has not been started")
        return_code = self._process.poll()
        if return_code is not None:
            # Give the output reader a brief chance to capture the final error
            # lines before surfacing the child-process failure to the UI.
            if self._reader_thread is not None:
                self._reader_thread.join(timeout=0.2)
            suffix = f" {phase}" if phase else ""
            raise ControllerServerError(
                f"Controller server exited with code {return_code}{suffix}."
                f"{self._tail_text()}"
            )

    def stop(self) -> None:
        process = self._process
        if process is None:
            self._close_log()
            return

        if process.poll() is None and self.shutdown_settle_s:
            time.sleep(self.shutdown_settle_s)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=min(self.stop_timeout_s, 3.0))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=min(self.stop_timeout_s, 3.0))

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()
        self._close_log()
        self._process = None

    def _drain_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            self._tail.append(line)
            self._lines.put(line)
            if self._log_handle is not None:
                self._log_handle.write(raw_line)
                self._log_handle.flush()

    def _tail_text(self) -> str:
        if not self._tail:
            return ""
        return "\nRecent server output:\n" + "\n".join(self._tail)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
