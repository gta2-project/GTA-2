"""Sweep 50 compatible button-surface angles across the AxisAlign boundary.

The surface normals are kept horizontal. This guarantees that the microwave
recipe's desired approach axis and global-Z upright axis are perpendicular, so
the sweep measures projection behavior rather than incompatible orientation
goals.

Run from the repository root:

    python tests/test_axis_align_surface_angle_sweep.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIRECTORY = Path(__file__).resolve().parent
if str(TESTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TESTS_DIRECTORY))

import simulate_axis_align_subtask_boundary as boundary_simulation


def load_error_projection_module():
    module_path = (
        REPOSITORY_ROOT / "gta" / "error_projection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "error_projection_surface_sweep", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ERROR_PROJECTION = load_error_projection_module()
ANGLE_COUNT = 50
BUG_THRESHOLD_DEG = 1.0


def production_additive_projection(error_packets) -> np.ndarray:
    return ERROR_PROJECTION.project_errors(error_packets)[1]


class AxisAlignSurfaceAngleSweepTests(unittest.TestCase):
    def test_fifty_horizontal_surface_angles(self):
        rows = []

        for angle_deg in np.linspace(0.0, 360.0, ANGLE_COUNT, endpoint=False):
            angle_rad = np.radians(angle_deg)
            surface_normal = np.array(
                [np.cos(angle_rad), np.sin(angle_rad), 0.0]
            )

            legacy, _ = boundary_simulation.run_simulation(
                "legacy",
                boundary_simulation.legacy_matrix_composition,
                surface_normal=surface_normal,
            )
            fixed, _ = boundary_simulation.run_simulation(
                "production additive",
                production_additive_projection,
                surface_normal=surface_normal,
            )

            rows.append(
                (
                    float(angle_deg),
                    legacy.next_subtask_command_deg,
                    fixed.next_subtask_command_deg,
                )
            )

        values = np.asarray(rows)
        legacy_commands = values[:, 1]
        fixed_commands = values[:, 2]
        legacy_bug_count = int(np.sum(legacy_commands > BUG_THRESHOLD_DEG))
        fixed_bug_count = int(np.sum(fixed_commands > BUG_THRESHOLD_DEG))

        print("\n50-angle AxisAlign boundary sweep (degrees)")
        print("-" * 54)
        print(f"{'surface angle':>13} {'legacy entry':>15} {'fixed entry':>15}")
        print("-" * 54)
        for angle_deg, legacy_command, fixed_command in rows:
            print(
                f"{angle_deg:>13.1f} "
                f"{legacy_command:>15.6f} "
                f"{fixed_command:>15.6f}"
            )
        print("-" * 54)
        print(
            "legacy min/median/max: "
            f"{np.min(legacy_commands):.6f} / "
            f"{np.median(legacy_commands):.6f} / "
            f"{np.max(legacy_commands):.6f}"
        )
        print(
            "fixed  min/median/max: "
            f"{np.min(fixed_commands):.6f} / "
            f"{np.median(fixed_commands):.6f} / "
            f"{np.max(fixed_commands):.6f}"
        )
        print(
            f"> {BUG_THRESHOLD_DEG:.1f} deg boundary command: "
            f"legacy={legacy_bug_count}/{ANGLE_COUNT}, "
            f"fixed={fixed_bug_count}/{ANGLE_COUNT}"
        )

        self.assertEqual(len(rows), ANGLE_COUNT)
        self.assertGreaterEqual(legacy_bug_count, 45)
        self.assertEqual(fixed_bug_count, 0)
        self.assertTrue(np.all(fixed_commands < legacy_commands))
        self.assertLess(np.max(fixed_commands), BUG_THRESHOLD_DEG)


if __name__ == "__main__":
    unittest.main()
