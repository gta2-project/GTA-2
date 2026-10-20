"""Prompt-balance regressions for the CaP benchmark conditions."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from benchmarks.cap.conditions import get_condition
from benchmarks.cap.prompts import build_system_prompt


class CapPromptTests(unittest.TestCase):
    def test_all_conditions_receive_tool_axis_semantics(self):
        for name in (
            "controller_zeroshot",
            "controller_fewshot",
            "cartesian_zeroshot",
            "cartesian_fewshot",
        ):
            with self.subTest(condition=name):
                prompt = build_system_prompt(get_condition(name))
                self.assertIn("z is the\napproach axis", prompt)
                self.assertIn("x is the grasp axis", prompt)
                self.assertIn("y is the orthogonal axis", prompt)

    def test_zeroshot_prompts_do_not_receive_orientation_example(self):
        for name in ("controller_zeroshot", "cartesian_zeroshot"):
            with self.subTest(condition=name):
                prompt = build_system_prompt(get_condition(name))
                self.assertNotIn("Example: force contact on a surface", prompt)
                self.assertNotIn("center of the marked contact area", prompt)

    def test_cartesian_fewshot_has_one_generic_geometry_driven_alignment(self):
        prompt = build_system_prompt(get_condition("cartesian_fewshot"))
        self.assertEqual(prompt.count("R.align_vectors"), 1)
        self.assertIn(
            'R.align_vectors([desired_direction], [ctx.tcp_axis("x")])[0]',
            prompt,
        )
        force_example = prompt.split("Example: force contact on a surface", 1)[1]
        force_example = force_example.split("Example: point traverse", 1)[0]
        self.assertNotIn("R.align_vectors", force_example)
        self.assertIn("pose[3], pose[4], pose[5]", force_example)
        self.assertNotIn("30 degrees", prompt)
        self.assertNotIn("button", prompt.lower())
        self.assertNotIn("microwave", prompt.lower())

    def test_controller_fewshot_has_matched_direction_alignment(self):
        prompt = build_system_prompt(get_condition("controller_fewshot"))
        self.assertEqual(prompt.count("AxisAlign("), 2)  # API signature + one example.
        self.assertIn(
            'AxisAlign(lambda: ctx.tcp_axis("x"), desired_direction, rotation_axis, 0.0)',
            prompt,
        )
        force_example = prompt.split("Example: force contact on a surface", 1)[1]
        force_example = force_example.split(
            "Example: point traverse along eef x axis", 1
        )[0]
        self.assertNotIn("AxisAlign", force_example)
        self.assertNotIn("approach_axis", force_example)
        self.assertEqual(force_example.count("PosAlign("), 3)
        self.assertIn("np.array([1.0, 0.0, 0.0])", force_example)
        self.assertIn("np.array([0.0, 1.0, 0.0])", force_example)
        self.assertIn("np.array([0.0, 0.0, 1.0])", force_example)
        self.assertNotIn("button", prompt.lower())
        self.assertNotIn("microwave", prompt.lower())

    def test_controller_conditions_expose_explicit_object_attachment(self):
        zero = build_system_prompt(get_condition("controller_zeroshot"))
        few = build_system_prompt(get_condition("controller_fewshot"))
        cartesian = build_system_prompt(get_condition("cartesian_zeroshot"))

        self.assertIn("GripperControl(state, time, object=None)", zero)
        self.assertIn('ctx.object_for("alias")', zero)
        self.assertIn('lambda: object.get_k("alias")', zero)
        self.assertIn('GripperControl("close", 0.0, block)', few)
        self.assertIn('GripperControl("open", 0.0, block)', few)
        self.assertNotIn("ctx.object_for", cartesian)

    def test_cartesian_example_alignment_maps_tcp_x_to_visible_direction(self):
        current_rotation = R.from_rotvec([0.2, -0.3, 0.1])
        current_x = current_rotation.as_matrix()[:, 0]
        desired_direction = np.array([0.8, 0.4, 0.4472135955])
        desired_direction /= np.linalg.norm(desired_direction)

        alignment = R.align_vectors([desired_direction], [current_x])[0]
        target_rotation = alignment * current_rotation

        np.testing.assert_allclose(
            target_rotation.as_matrix()[:, 0],
            desired_direction,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
