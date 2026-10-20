"""Hardware-free tests for the CAP-owned controller server lifecycle."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.cap.controller_server import (
    DEFAULT_SERVER_SCRIPT,
    READY_MARKER,
    ControllerServerError,
    ManagedControllerServer,
)


READY_TEST_SERVER = f"""\
import signal
import time

running = True

def stop(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, stop)
print({READY_MARKER!r}, flush=True)
while running:
    time.sleep(0.01)
print("test server stopped", flush=True)
"""


class ManagedControllerServerTests(unittest.TestCase):
    def test_default_launcher_uses_robot_server_entry_point(self):
        self.assertEqual(
            DEFAULT_SERVER_SCRIPT,
            Path(__file__).resolve().parents[1] / "scene" / "robot" / "server.py",
        )
        self.assertTrue(DEFAULT_SERVER_SCRIPT.is_file())

    def test_starts_waits_for_marker_and_stops_with_sigint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "ready_server.py"
            log_path = root / "controller_server.log"
            script.write_text(READY_TEST_SERVER, encoding="utf-8")
            manager = ManagedControllerServer(
                log_path=log_path,
                server_script=script,
                startup_timeout_s=2.0,
                startup_settle_s=0.02,
                shutdown_settle_s=0.0,
                stop_timeout_s=1.0,
                existing_node_detector=lambda: [],
            )

            with manager as running:
                self.assertIsNotNone(running.pid)
                running.assert_running("inside test")

            self.assertIsNone(manager.pid)
            output = log_path.read_text(encoding="utf-8")
            self.assertIn(READY_MARKER, output)
            self.assertIn("test server stopped", output)

    def test_rejects_an_existing_controller_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "controller_server.log"
            manager = ManagedControllerServer(
                log_path=log_path,
                server_script=Path(__file__),
                existing_node_detector=lambda: ["/existing_robot_server"],
            )

            with self.assertRaisesRegex(
                ControllerServerError, "already registered.*existing_robot_server"
            ):
                manager.start()
            self.assertFalse(log_path.exists())

    def test_reports_output_when_server_exits_before_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "failed_server.py"
            log_path = root / "controller_server.log"
            script.write_text(
                'print("synthetic startup failure", flush=True)\nraise SystemExit(7)\n',
                encoding="utf-8",
            )
            manager = ManagedControllerServer(
                log_path=log_path,
                server_script=script,
                startup_timeout_s=2.0,
                startup_settle_s=0.0,
                shutdown_settle_s=0.0,
                stop_timeout_s=1.0,
                existing_node_detector=lambda: [],
            )

            with self.assertRaises(ControllerServerError) as raised:
                manager.start()

            self.assertIn("exited with code 7", str(raised.exception))
            self.assertIn("synthetic startup failure", str(raised.exception))
            self.assertIsNone(manager.pid)


if __name__ == "__main__":
    unittest.main()
