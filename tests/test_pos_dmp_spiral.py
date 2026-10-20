"""Offline tests for the learned planar PosDMPSpiral controller."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

import controllers.pos_dmp as pos_dmp_module
from controllers.pos_dmp import PosDMPSpiral
from gta.dmp import (
    fit_discrete_dmp,
    generate_minimum_jerk_spiral,
    rollout_discrete_dmp,
)
from gta.error_projection import project_errors


class LivePoint:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def __call__(self):
        return self.position.copy()


class PosDMPSpiralTests(unittest.TestCase):
    def test_synthetic_demonstration_is_three_turn_minimum_jerk_spiral(self):
        time, demonstration = generate_minimum_jerk_spiral()
        velocity = np.gradient(demonstration, time, axis=0, edge_order=2)
        angle = np.unwrap(np.arctan2(demonstration[:, 1], demonstration[:, 0]))

        np.testing.assert_allclose(demonstration[0], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(demonstration[-1], [1.0, 0.0], atol=1e-12)
        self.assertAlmostEqual((angle[-1] - angle[1]) / (2.0 * np.pi), 3.0, places=6)
        self.assertLess(np.linalg.norm(velocity[0]), 1e-3)
        self.assertLess(np.linalg.norm(velocity[-1]), 1e-3)

    def test_dmp_fit_is_deterministic_and_meets_reconstruction_thresholds(self):
        _, demonstration = generate_minimum_jerk_spiral()
        first = fit_discrete_dmp(demonstration)
        second = fit_discrete_dmp(demonstration)

        registered_document = yaml.safe_load(
            pos_dmp_module._SPIRAL_MODEL_PATH.read_text()
        )
        self.assertEqual(first.keys(), second.keys())
        self.assertEqual(first.keys(), registered_document["dmp"].keys())
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(first[name], registered_document["dmp"][name])

        rollout = rollout_discrete_dmp(
            first, registered_document["execution"]["rollout_samples"]
        )
        difference = rollout - demonstration
        rmse = float(np.sqrt(np.mean(difference**2)))
        maximum_deviation = float(np.max(np.linalg.norm(difference, axis=1)))
        endpoint_error = float(np.linalg.norm(rollout[-1] - demonstration[-1]))

        self.assertLess(rmse, 0.01)
        self.assertLess(maximum_deviation, 0.03)
        self.assertLess(endpoint_error, 0.005)

    def test_radius_scaling_preserves_normalized_shape(self):
        phases = [0.0, 0.2, 0.5, 0.84, 1.0]
        origin = np.array([0.2, -0.3, 0.4])
        point_small = LivePoint(origin)
        point_large = LivePoint(origin)
        small = PosDMPSpiral(point_small, [0.0, 0.0, 1.0], 0.03)
        large = PosDMPSpiral(point_large, [0.0, 0.0, 1.0], 0.10)

        for phase in phases:
            small_delta = small.target_at(phase) - origin
            large_delta = large.target_at(phase) - origin
            np.testing.assert_allclose(
                large_delta,
                small_delta * (0.10 / 0.03),
                atol=1e-8,
            )

        self.assertAlmostEqual(np.linalg.norm(small.target_at(1.0) - origin), 0.03)
        self.assertAlmostEqual(np.linalg.norm(large.target_at(1.0) - origin), 0.10)

    def test_targets_and_errors_stay_in_tilted_plane(self):
        origin = np.array([0.4, -0.2, 0.3])
        normal = np.array([1.0, 2.0, 3.0])
        normal /= np.linalg.norm(normal)
        point = LivePoint(origin)
        controller = PosDMPSpiral(point, normal, 0.05)

        self.assertAlmostEqual(np.dot(controller.plane_u, normal), 0.0, places=12)
        self.assertAlmostEqual(np.dot(controller.plane_v, normal), 0.0, places=12)
        self.assertAlmostEqual(np.dot(controller.plane_u, controller.plane_v), 0.0, places=12)

        for phase in np.linspace(0.0, 1.0, 21):
            displacement = controller.target_at(phase) - origin
            self.assertAlmostEqual(np.dot(displacement, normal), 0.0, places=10)

        point.position = controller.target_at(0.5) + 0.02 * normal
        packet = controller.compute_error(0.7)
        self.assertAlmostEqual(np.dot(packet["error"], normal), 0.0, places=10)
        self.assertEqual(len(packet["axes"]), 2)

    def test_global_x_parallel_normal_uses_stable_fallback(self):
        point = LivePoint([0.0, 0.0, 0.0])
        controller = PosDMPSpiral(point, [1.0, 0.0, 0.0], 0.05)
        np.testing.assert_allclose(controller.plane_u, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(controller.plane_v, [0.0, 0.0, 1.0], atol=1e-12)

    def test_live_keypoint_feedback_tracks_attached_tool_point(self):
        point = LivePoint([0.1, 0.2, 0.3])
        controller = PosDMPSpiral(point, [0.0, 0.0, 1.0], 0.05)
        target = controller.target_at(0.5)
        point.position = target + np.array([0.004, -0.006, 0.02])

        packet = controller.compute_error(0.5)
        np.testing.assert_allclose(packet["error"], [-0.004, 0.006, 0.0], atol=1e-10)

    def test_normal_force_and_planar_dmp_coexist_in_projection(self):
        point = LivePoint([0.0, 0.0, 0.0])
        controller = PosDMPSpiral(point, [0.0, 0.0, 1.0], 0.05)
        dmp_packet = controller.compute_error(0.5)
        force_packet = {
            "type": "force",
            "error": np.zeros(3),
            "axis": np.array([0.0, 0.0, 1.0]),
            "magnitude": -5.0,
        }

        position_error, _, force_command, _ = project_errors([force_packet, dmp_packet])
        self.assertAlmostEqual(position_error[2], 0.0, places=12)
        np.testing.assert_allclose(force_command, [0.0, 0.0, 1.0, -5.0], atol=1e-12)

    def test_model_resolution_is_cwd_independent(self):
        self.assertEqual(
            pos_dmp_module._SPIRAL_MODEL_PATH,
            Path(pos_dmp_module.__file__).resolve().parent / "models" / "spiral_planar.yaml",
        )
        point = LivePoint([0.0, 0.0, 0.0])
        previous_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                os.chdir(temporary_directory)
                controller = PosDMPSpiral(point, [0.0, 0.0, 1.0], 0.05)
                self.assertEqual(
                    controller.model_path,
                    pos_dmp_module._SPIRAL_MODEL_PATH,
                )
        finally:
            os.chdir(previous_cwd)

    def test_invalid_radius_and_normal_fail_before_motion(self):
        point = LivePoint([0.0, 0.0, 0.0])
        for radius in (0.0, -0.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                PosDMPSpiral(point, [0.0, 0.0, 1.0], radius)

        with self.assertRaises(ValueError):
            PosDMPSpiral(point, [0.0, 0.0, 0.0], 0.05)

    def test_malformed_and_inconsistent_models_fail_before_motion(self):
        point = LivePoint([0.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            malformed_path = root / "malformed.yaml"
            malformed_path.write_text("dmp: [not: valid", encoding="utf-8")

            with patch.object(pos_dmp_module, "_SPIRAL_MODEL_PATH", malformed_path):
                with self.assertRaisesRegex(ValueError, "Invalid YAML"):
                    PosDMPSpiral(point, [0.0, 0.0, 1.0], 0.05)

            inconsistent = yaml.safe_load(pos_dmp_module._SPIRAL_MODEL_PATH.read_text())
            inconsistent["dmp"]["weights"] = inconsistent["dmp"]["weights"][:1]
            inconsistent_path = root / "inconsistent.yaml"
            inconsistent_path.write_text(
                yaml.safe_dump(inconsistent, sort_keys=False),
                encoding="utf-8",
            )

            with patch.object(pos_dmp_module, "_SPIRAL_MODEL_PATH", inconsistent_path):
                with self.assertRaisesRegex(ValueError, "weights"):
                    PosDMPSpiral(point, [0.0, 0.0, 1.0], 0.05)


if __name__ == "__main__":
    unittest.main()
