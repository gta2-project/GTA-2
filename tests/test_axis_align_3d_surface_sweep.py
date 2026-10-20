"""Stress AxisAlign projection across 200 compatible 3D surface directions.

The primary goal points EEF Z opposite each surface normal. The secondary goal
uses world-up projected into that surface's tangent plane, so the requested EEF
Z and EEF X axes remain perpendicular and therefore physically achievable.

The direction set combines a near-uniform Fibonacci sphere with explicit
parallel, anti-parallel, pole, and cross-product fallback-threshold cases.

Run from the repository root:

    python tests/test_axis_align_3d_surface_sweep.py
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


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
        "error_projection_3d_surface_sweep", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ERROR_PROJECTION = load_error_projection_module()
DIRECTION_COUNT = 200
BUG_THRESHOLD_DEG = 1.0
SETTLED_COMMAND_RAD = 1e-10
SETTLED_TICKS_REQUIRED = 5


@dataclass(frozen=True)
class DirectionCase:
    name: str
    surface_normal: np.ndarray
    upright_target: np.ndarray


@dataclass(frozen=True)
class BoundaryMeasurement:
    boundary_command_deg: float
    approach_error_deg: float
    upright_error_deg: float
    control_ticks: int


def production_additive_projection(error_packets) -> np.ndarray:
    return ERROR_PROJECTION.project_errors(error_packets)[1]


def tangent_upright(surface_normal: np.ndarray) -> np.ndarray:
    """Choose a deterministic, compatible gripper-X target for a surface."""
    normal = boundary_simulation.unit(surface_normal)
    tangent = boundary_simulation.GLOBAL_Z - (
        np.dot(boundary_simulation.GLOBAL_Z, normal) * normal
    )

    # World-up has no tangent component at the poles. Use world Y there; this
    # is a real choice of roll, not a relaxation of the orientation objective.
    if np.linalg.norm(tangent) < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0])
        tangent = fallback - np.dot(fallback, normal) * normal

    return boundary_simulation.unit(tangent)


def _add_unique(
    directions: list[tuple[str, np.ndarray]], name: str, vector: np.ndarray
) -> None:
    vector = boundary_simulation.unit(vector)
    if any(np.dot(vector, existing) > 1.0 - 1e-12 for _, existing in directions):
        return
    directions.append((name, vector))


def make_direction_cases() -> list[DirectionCase]:
    """Return exactly 200 deterministic directions including hard edge cases."""
    directions: list[tuple[str, np.ndarray]] = []

    cardinal_directions = {
        "+X": [1.0, 0.0, 0.0],
        "-X": [-1.0, 0.0, 0.0],
        "+Y": [0.0, 1.0, 0.0],
        "-Y": [0.0, -1.0, 0.0],
        "+Z pole": [0.0, 0.0, 1.0],
        "-Z pole": [0.0, 0.0, -1.0],
    }
    for name, vector in cardinal_directions.items():
        _add_unique(directions, name, np.asarray(vector, dtype=float))

    reset_matrix = Rotation.from_rotvec(
        boundary_simulation.RESET_TCP_ROTVEC
    ).as_matrix()
    initial_z = reset_matrix @ boundary_simulation.EEF_Z_LOCAL
    initial_x = reset_matrix @ boundary_simulation.EEF_X_LOCAL
    for name, vector in (
        ("initial EEF Z parallel", initial_z),
        ("initial EEF Z anti-parallel", -initial_z),
        ("initial EEF X parallel", initial_x),
        ("initial EEF X anti-parallel", -initial_x),
    ):
        _add_unique(directions, name, vector)

    # Exercise both sides of cross_product_axis's 1e-6 fallback threshold.
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(initial_z, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    perturbation = boundary_simulation.unit(np.cross(initial_z, reference))
    for sign in (-1.0, 1.0):
        for delta in (0.5e-6, 2.0e-6):
            _add_unique(
                directions,
                f"near initial Z sign={sign:+.0f} delta={delta:g}",
                initial_z + sign * delta * perturbation,
            )
            _add_unique(
                directions,
                f"near anti-initial Z sign={sign:+.0f} delta={delta:g}",
                -initial_z + sign * delta * perturbation,
            )

    # Near-pole cases retain a tiny but nonzero world-up tangent projection.
    for pole_sign in (-1.0, 1.0):
        for azimuth_index, azimuth_deg in enumerate((0.0, 90.0, 180.0, 270.0)):
            azimuth = np.radians(azimuth_deg)
            _add_unique(
                directions,
                f"near {pole_sign:+.0f}Z azimuth={azimuth_index * 90}",
                np.array(
                    [
                        5e-7 * np.cos(azimuth),
                        5e-7 * np.sin(azimuth),
                        pole_sign,
                    ]
                ),
            )

    fibonacci_count = DIRECTION_COUNT - len(directions)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(fibonacci_count):
        z = 1.0 - 2.0 * (index + 0.5) / fibonacci_count
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        azimuth = index * golden_angle
        _add_unique(
            directions,
            f"fibonacci {index:03d}",
            np.array(
                [radius * np.cos(azimuth), radius * np.sin(azimuth), z]
            ),
        )

    if len(directions) != DIRECTION_COUNT:
        raise AssertionError(
            f"Expected {DIRECTION_COUNT} unique directions, got {len(directions)}"
        )

    return [
        DirectionCase(name, normal, tangent_upright(normal))
        for name, normal in directions
    ]


def simulate_boundary_case(projector, case: DirectionCase) -> BoundaryMeasurement:
    """Run to a fixed point, then reconstruct the same controller pair."""
    state = boundary_simulation.OrientationState(
        Rotation.from_rotvec(
            boundary_simulation.RESET_TCP_ROTVEC
        ).as_matrix()
    )
    controllers = boundary_simulation.make_controllers(
        state, case.surface_normal, case.upright_target
    )
    settled_ticks = 0
    ticks_run = 0

    for ticks_run in range(1, boundary_simulation.CONTROL_STEPS + 1):
        command = projector(boundary_simulation.controller_packets(controllers))
        command_angle = Rotation.from_matrix(command).magnitude()
        if command_angle < SETTLED_COMMAND_RAD:
            settled_ticks += 1
            if settled_ticks >= SETTLED_TICKS_REQUIRED:
                break
        else:
            settled_ticks = 0

        state.matrix = (
            boundary_simulation.limited_rotation_step(command) @ state.matrix
        )

    desired_approach = controllers[0].target
    approach_error = boundary_simulation.angle_between_degrees(
        state.eef_axis(boundary_simulation.EEF_Z_LOCAL), desired_approach
    )
    upright_error = boundary_simulation.angle_between_degrees(
        state.eef_axis(boundary_simulation.EEF_X_LOCAL), case.upright_target
    )

    reconstructed = boundary_simulation.make_controllers(
        state, case.surface_normal, case.upright_target
    )
    boundary_command = projector(
        boundary_simulation.controller_packets(reconstructed)
    )

    return BoundaryMeasurement(
        boundary_command_deg=boundary_simulation.rotation_magnitude_degrees(
            boundary_command
        ),
        approach_error_deg=approach_error,
        upright_error_deg=upright_error,
        control_ticks=ticks_run,
    )


class AxisAlignThreeDimensionalSweepTests(unittest.TestCase):
    def test_two_hundred_compatible_surface_directions(self):
        cases = make_direction_cases()
        rows = []

        for case in cases:
            legacy = simulate_boundary_case(
                boundary_simulation.legacy_matrix_composition, case
            )
            fixed = simulate_boundary_case(production_additive_projection, case)
            rows.append((case, legacy, fixed))

        legacy_commands = np.array(
            [legacy.boundary_command_deg for _, legacy, _ in rows]
        )
        fixed_commands = np.array(
            [fixed.boundary_command_deg for _, _, fixed in rows]
        )
        fixed_approach_errors = np.array(
            [fixed.approach_error_deg for _, _, fixed in rows]
        )
        fixed_upright_errors = np.array(
            [fixed.upright_error_deg for _, _, fixed in rows]
        )
        legacy_bug_count = int(np.sum(legacy_commands > BUG_THRESHOLD_DEG))
        fixed_bug_count = int(np.sum(fixed_commands > BUG_THRESHOLD_DEG))
        fixed_bug_names = {
            rows[index][0].name
            for index in np.flatnonzero(fixed_commands > BUG_THRESHOLD_DEG)
        }
        expected_frozen_axis_singularities = {
            "-Z pole",
            "initial EEF Z parallel",
        }
        regular_mask = np.array(
            [
                case.name not in expected_frozen_axis_singularities
                for case, _, _ in rows
            ]
        )

        worst_indices = np.argsort(fixed_commands)[-10:][::-1]
        print("\n200-direction compatible 3D AxisAlign sweep (degrees)")
        print("-" * 92)
        print(
            f"{'case':<39} {'legacy':>10} {'fixed':>10} "
            f"{'approach':>10} {'upright':>10}"
        )
        print("-" * 92)
        for index in worst_indices:
            case, legacy, fixed = rows[index]
            print(
                f"{case.name:<39} "
                f"{legacy.boundary_command_deg:>10.4f} "
                f"{fixed.boundary_command_deg:>10.4f} "
                f"{fixed.approach_error_deg:>10.4f} "
                f"{fixed.upright_error_deg:>10.4f}"
            )
        print("-" * 92)
        print(
            "legacy boundary min/median/max: "
            f"{np.min(legacy_commands):.6f} / "
            f"{np.median(legacy_commands):.6f} / "
            f"{np.max(legacy_commands):.6f}"
        )
        print(
            "fixed  boundary min/median/max: "
            f"{np.min(fixed_commands):.6f} / "
            f"{np.median(fixed_commands):.6f} / "
            f"{np.max(fixed_commands):.6f}"
        )
        print(
            f"> {BUG_THRESHOLD_DEG:.1f} deg boundary command: "
            f"legacy={legacy_bug_count}/{DIRECTION_COUNT}, "
            f"fixed={fixed_bug_count}/{DIRECTION_COUNT}"
        )
        print(
            "fixed > 1 deg cases (frozen-a3 180 deg singularity): "
            f"{sorted(fixed_bug_names)}"
        )

        self.assertEqual(len(cases), DIRECTION_COUNT)
        for case in cases:
            self.assertAlmostEqual(
                np.linalg.norm(case.surface_normal), 1.0, places=12
            )
            self.assertAlmostEqual(
                np.linalg.norm(case.upright_target), 1.0, places=12
            )
            self.assertLess(
                abs(np.dot(case.surface_normal, case.upright_target)), 1e-10
            )

        self.assertTrue(np.all(np.isfinite(legacy_commands)))
        self.assertTrue(np.all(np.isfinite(fixed_commands)))
        self.assertGreaterEqual(legacy_bug_count, 150)
        self.assertEqual(fixed_bug_names, expected_frozen_axis_singularities)
        self.assertLess(
            np.max(fixed_commands[regular_mask]), BUG_THRESHOLD_DEG
        )
        self.assertLess(
            np.max(fixed_approach_errors[regular_mask]), BUG_THRESHOLD_DEG
        )
        self.assertLess(
            np.max(fixed_upright_errors[regular_mask]), BUG_THRESHOLD_DEG
        )
        self.assertLess(np.max(fixed_commands), 2.0)
        self.assertLess(np.max(fixed_approach_errors), 2.0)
        self.assertLess(np.max(fixed_upright_errors), 2.0)
        self.assertTrue(np.all(fixed_commands <= legacy_commands + 1e-9))


if __name__ == "__main__":
    unittest.main()
