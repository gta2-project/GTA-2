"""Tests for query-derived controller objects in the CaP policy context."""

import unittest

import numpy as np

from benchmarks.cap.context import PolicyContext
from benchmarks.cap.runtime import OfflineControllerRuntime
from controllers import GripperControl
from gta.error_projection import project_errors


class CapPolicyObjectTests(unittest.TestCase):
    def make_context(self):
        pose = np.zeros(6, dtype=float)
        context = PolicyContext(
            points={
                "tool_tip": [1.0, 0.0, 0.0],
                "tool_side": [0.0, 1.0, 0.0],
                "destination": [4.0, 5.0, 6.0],
            },
            normals={
                "tool_tip": [1.0, 0.0, 0.0],
                "tool_side": [0.0, 1.0, 0.0],
                "destination": [0.0, 0.0, 1.0],
            },
            current_pose_provider=lambda: pose.copy(),
            query_objects={
                "tool_tip": "tool",
                "tool_side": "tool",
                "destination": "tray",
            },
        )
        return context, pose

    def test_aliases_with_the_same_declared_object_share_one_handler(self):
        context, _ = self.make_context()

        self.assertIs(context.object_for("tool_tip"), context.object_for("tool_side"))
        self.assertIsNot(
            context.object_for("tool_tip"), context.object_for("destination")
        )

    def test_gripper_attachment_tracks_all_object_points_and_axes(self):
        context, pose = self.make_context()
        tool = context.object_for("tool_tip")

        close = GripperControl("close", 0.0, tool)
        project_errors([close.compute_error(0.0)])
        self.assertEqual(tool.frame, "robot")

        pose[:] = [2.0, 3.0, 4.0, 0.0, 0.0, np.pi / 2.0]
        np.testing.assert_allclose(tool.get_k("tool_tip"), [2.0, 4.0, 4.0])
        np.testing.assert_allclose(tool.get_k("tool_side"), [1.0, 3.0, 4.0])
        np.testing.assert_allclose(
            tool.get_a("tool_tip"), [0.0, 1.0, 0.0], atol=1e-12
        )

        # Frozen observations remain frozen even while the controller object moves.
        np.testing.assert_allclose(context.point("tool_tip"), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(context.normal("tool_tip"), [1.0, 0.0, 0.0])

        open_controller = GripperControl("open", 0.0, tool)
        project_errors([open_controller.compute_error(0.0)])
        self.assertEqual(tool.frame, "world")
        detached_tip = tool.get_k("tool_tip")

        pose[:] = [8.0, 9.0, 10.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(tool.get_k("tool_tip"), detached_tip)

    def test_offline_runtime_applies_explicit_object_frame_transitions(self):
        context, _ = self.make_context()
        tool = context.object_for("tool_tip")
        runtime = OfflineControllerRuntime()

        runtime.execute_subtask([GripperControl("close", 0.0, tool)])
        self.assertEqual(tool.frame, "robot")
        self.assertEqual(
            runtime.phases[-1]["object_transitions"],
            [{"object": "tool", "frame": "robot"}],
        )

        runtime.execute_subtask([GripperControl("open", 0.0, tool)])
        self.assertEqual(tool.frame, "world")

    def test_object_for_rejects_an_alias_without_controller_object_metadata(self):
        context = PolicyContext(
            points={"target": [0.0, 0.0, 0.0]},
            current_pose_provider=lambda: np.zeros(6),
        )

        with self.assertRaisesRegex(KeyError, "No attachable object"):
            context.object_for("target")


if __name__ == "__main__":
    unittest.main()
