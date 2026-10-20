"""Hardware-free tests for the CaP-local controller robot loop."""

import concurrent.futures
import unittest

import numpy as np

from benchmarks.cap.runtime import ControllerRobotRuntime


class RecordingController:
    type = "positional"

    def __init__(self):
        self.elapsed_times = []

    def compute_error(self, elapsed_time=0.0):
        self.elapsed_times.append(elapsed_time)
        return {
            "error": np.zeros(3),
            "type": self.type,
            "axis": np.array([1.0, 0.0, 0.0]),
        }


class CapControllerRobotRuntimeTests(unittest.TestCase):
    def make_runtime(self, events, updates, times):
        def project(errors):
            events.append(("project", errors))
            return np.zeros(3), np.eye(3), None, None

        def update(*commands):
            updates.append(commands)
            return True

        return ControllerRobotRuntime(
            start_skill=lambda: events.append(("start", None)),
            end_skill=lambda: events.append(("end", None)),
            update_control_commands=update,
            error_projector=project,
            shutdown_requested=lambda: False,
            clock=lambda: next(times),
            sleeper=lambda _seconds: None,
        )

    def test_execute_subtask_runs_from_worker_thread_without_signal_registration(self):
        events = []
        updates = []
        controller = RecordingController()
        runtime = self.make_runtime(events, updates, iter([10.0, 11.0]))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(runtime.execute_subtask, [controller], 1.0).result()

        self.assertEqual(controller.elapsed_times, [1.0])
        self.assertEqual(len(updates), 1)
        self.assertEqual(events[0], ("start", None))
        self.assertEqual(events[-1], ("end", None))

    def test_end_skill_runs_when_command_publication_fails(self):
        events = []
        runtime = ControllerRobotRuntime(
            start_skill=lambda: events.append("start"),
            end_skill=lambda: events.append("end"),
            update_control_commands=lambda *_commands: False,
            error_projector=lambda _errors: (np.zeros(3), np.eye(3), None, None),
            clock=iter([0.0, 1.0]).__next__,
            sleeper=lambda _seconds: None,
        )

        with self.assertRaisesRegex(RuntimeError, "Failed to publish"):
            runtime.execute_subtask([RecordingController()], duration=1.0)
        self.assertEqual(events, ["start", "end"])


if __name__ == "__main__":
    unittest.main()
