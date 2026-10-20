"""Focused regression tests for additive rotational error projection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "gta" / "error_projection.py"

# Load this module directly so the unit test does not initialize camera/robot
# dependencies imported by gta.__init__.
SPEC = importlib.util.spec_from_file_location("error_projection_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
ERROR_PROJECTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ERROR_PROJECTION)

project_errors = ERROR_PROJECTION.project_errors


def rotational_packet(error, axis):
    return {
        "type": "rotational",
        "error": np.asarray(error, dtype=float),
        "axis": None if axis is None else np.asarray(axis, dtype=float),
    }


class RotationalProjectionTests(unittest.TestCase):
    def test_single_rotational_controller_is_unchanged(self):
        error = np.array([0.25, -0.1, 0.2])
        packets = [rotational_packet(error, [0.0, 0.0, 1.0])]

        _, actual, _, _ = project_errors(packets)
        expected = Rotation.from_rotvec(error).as_matrix()

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_secondary_error_is_projected_then_added(self):
        primary_error = np.array([0.4, 0.0, 0.0])
        primary_axis = np.array([0.0, 0.0, 1.0])
        secondary_error = np.array([0.2, -0.3, 0.5])
        packets = [
            rotational_packet(primary_error, primary_axis),
            rotational_packet(secondary_error, None),
        ]

        _, actual, _, _ = project_errors(packets)

        projected_secondary = np.dot(secondary_error, primary_axis) * primary_axis
        expected_vector = primary_error + projected_secondary
        expected = Rotation.from_rotvec(expected_vector).as_matrix()
        legacy_composition = (
            Rotation.from_rotvec(projected_secondary).as_matrix()
            @ Rotation.from_rotvec(primary_error).as_matrix()
        )

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(actual - legacy_composition), 1e-3)

    def test_combined_rotvec_is_limited_before_so3_conversion(self):
        packets = [
            rotational_packet([2.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            rotational_packet([2.0, 0.0, 0.0], None),
        ]

        _, actual, _, _ = project_errors(packets)
        actual_rotvec = Rotation.from_matrix(actual).as_rotvec()

        self.assertAlmostEqual(np.linalg.norm(actual_rotvec), np.pi - 1e-6, places=7)
        np.testing.assert_allclose(
            actual_rotvec / np.linalg.norm(actual_rotvec),
            [1.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_no_rotational_packets_returns_identity(self):
        _, actual, _, _ = project_errors([])
        np.testing.assert_allclose(actual, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
