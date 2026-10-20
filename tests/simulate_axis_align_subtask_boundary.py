"""Reproduce the AxisAlign subtask-boundary rotation and compare projection modes.

This is a standalone, robot-free regression simulation for the microwave-button
recipe in ``robot_script.py``.  It uses the same two rotational objectives:

1. Point EEF Z opposite the button surface normal (180 degree offset).
2. Keep EEF X aligned with global Z.

The simulation runs a 30 second subtask at 100 Hz, reconstructs the exact same
controllers as the next subtask would, and measures the first new rotation
command.  Two projection implementations are compared:

* ``legacy`` reproduces the current finite rotation-matrix composition.
* ``additive`` sums the projected rotation vectors before one SO(3) conversion.

The additive-only version intentionally keeps AxisAlign's current frozen ``a3``
behavior.  It should substantially reduce the boundary correction, but this
test does not claim that additive projection alone completely fixes the issue.

Run from the repository root:

    python tests/simulate_axis_align_subtask_boundary.py

Use ``--no-animation`` for a text-only regression run, or save the animation:

    python tests/simulate_axis_align_subtask_boundary.py --save axis_boundary.gif
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from controllers.axis_align import AxisAlign


CONTROL_HZ = 100
SUBTASK_DURATION_S = 30
CONTROL_STEPS = CONTROL_HZ * SUBTASK_DURATION_S
MAX_ANGLE_STEP_RAD = 0.03
PRE_BOUNDARY_RECORD_STRIDE = 20
BOUNDARY_PAUSE_FRAMES = 20
POST_BOUNDARY_STEPS = 50
DEFAULT_ANIMATION_INTERVAL_MS = 40

# Matches the EEF axis mapping embedded in the current robot_script.py.
EEF_X_LOCAL = np.array([0.0, -1.0, 0.0])
EEF_Z_LOCAL = np.array([0.0, 0.0, 1.0])
GLOBAL_Z = np.array([0.0, 0.0, 1.0])

# Deterministic horizontal normal representative of the microwave front panel.
# Runtime perception will produce a different numeric normal, so correction
# magnitudes on the physical robot need not match this simulation exactly.
BUTTON_SURFACE_NORMAL = np.array([0.0, -1.0, 0.0])

# The reset TCP orientation recorded alongside scripts/ur5e/reset_robot.py.
RESET_TCP_ROTVEC = np.array([-2.221, -2.221, 0.0])

EPS = 1e-9


@dataclass
class OrientationState:
    matrix: np.ndarray

    def eef_axis(self, local_axis: np.ndarray) -> np.ndarray:
        return self.matrix @ local_axis


@dataclass(frozen=True)
class SimulationResult:
    name: str
    exit_command_deg: float
    approach_axis_error_deg: float
    upright_axis_error_deg: float
    next_subtask_command_deg: float
    next_primary_packet_deg: float
    next_secondary_packet_deg: float


@dataclass(frozen=True)
class SimulationTrajectory:
    name: str
    matrices: np.ndarray
    times_s: np.ndarray
    phases: tuple[str, ...]
    command_degrees: np.ndarray


Projector = Callable[[Iterable[dict]], np.ndarray]


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def cross_product_axis(a1: np.ndarray, a2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Local copy of the production parallel/anti-parallel fallback."""
    a1 = unit(a1)
    a2 = unit(a2)
    cross = np.cross(a1, a2)
    if np.linalg.norm(cross) > eps:
        return unit(cross)

    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a1, candidate)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    return unit(np.cross(a1, candidate))


def make_controllers(
    state: OrientationState,
    surface_normal: np.ndarray = BUTTON_SURFACE_NORMAL,
    upright_target: np.ndarray = GLOBAL_Z,
) -> list[AxisAlign]:
    """Construct the same rotational controller pair as microwave subtasks 2/3."""
    surface_normal = unit(surface_normal)
    upright_target = unit(upright_target)
    current_z = state.eef_axis(EEF_Z_LOCAL)
    current_x = state.eef_axis(EEF_X_LOCAL)

    # AxisAlign currently prints its non-zero-offset target during construction.
    # Suppress that unrelated output so the regression table stays readable.
    with contextlib.redirect_stdout(io.StringIO()):
        return [
            AxisAlign(
                lambda: state.eef_axis(EEF_Z_LOCAL),
                surface_normal,
                cross_product_axis(current_z, surface_normal),
                180.0,
            ),
            AxisAlign(
                lambda: state.eef_axis(EEF_X_LOCAL),
                upright_target,
                cross_product_axis(current_x, upright_target),
                0.0,
            ),
        ]


def controller_packets(controllers: list[AxisAlign]) -> list[dict]:
    """Match execute_subtask: first rotation constrained, remaining ones free."""
    return [
        controllers[0].compute_error(free_rotation=False),
        controllers[1].compute_error(free_rotation=True),
    ]


def legacy_matrix_composition(error_packets: Iterable[dict]) -> np.ndarray:
    """Reproduce the current rotational branch of project_errors()."""
    rotation_error_matrix = np.eye(3)
    primary_rotation_axis = None

    for packet in error_packets:
        error = np.asarray(packet["error"], dtype=float)
        axis = packet.get("axis")

        if primary_rotation_axis is None:
            if axis is not None:
                primary_rotation_axis = unit(np.asarray(axis, dtype=float))
        else:
            error = np.dot(error, primary_rotation_axis) * primary_rotation_axis

        rotation_error_matrix = (
            Rotation.from_rotvec(error).as_matrix() @ rotation_error_matrix
        )

    return rotation_error_matrix


def additive_projection(error_packets: Iterable[dict]) -> np.ndarray:
    """Candidate core fix: add projected rotvecs, then exponentiate once."""
    rotation_error_vector = np.zeros(3)
    primary_rotation_axis = None
    rotation_seen = False

    for packet in error_packets:
        error = np.asarray(packet["error"], dtype=float)
        axis = packet.get("axis")

        if not rotation_seen:
            rotation_seen = True
            if axis is not None:
                primary_rotation_axis = unit(np.asarray(axis, dtype=float))
        elif primary_rotation_axis is not None:
            error = np.dot(error, primary_rotation_axis) * primary_rotation_axis

        rotation_error_vector += error

    # Avoid converting a vector longer than pi into an aliased SO(3) target.
    angle = np.linalg.norm(rotation_error_vector)
    max_error = np.pi - 1e-6
    if angle > max_error:
        rotation_error_vector *= max_error / angle

    return Rotation.from_rotvec(rotation_error_vector).as_matrix()


def limited_rotation_step(rotation_error_matrix: np.ndarray) -> np.ndarray:
    """Match the robot server's 0.03 radian matrix-space step limit."""
    rotvec = Rotation.from_matrix(rotation_error_matrix).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle > MAX_ANGLE_STEP_RAD:
        rotvec *= MAX_ANGLE_STEP_RAD / angle
    return Rotation.from_rotvec(rotvec).as_matrix()


def angle_between_degrees(a1: np.ndarray, a2: np.ndarray) -> float:
    cosine = np.clip(np.dot(unit(a1), unit(a2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_magnitude_degrees(rotation_matrix: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(rotation_matrix).magnitude()))


def run_simulation(
    name: str,
    projector: Projector,
    surface_normal: np.ndarray = BUTTON_SURFACE_NORMAL,
    upright_target: np.ndarray = GLOBAL_Z,
) -> tuple[SimulationResult, SimulationTrajectory]:
    state = OrientationState(Rotation.from_rotvec(RESET_TCP_ROTVEC).as_matrix())
    upright_target = unit(upright_target)
    controllers = make_controllers(state, surface_normal, upright_target)

    matrices = [state.matrix.copy()]
    times_s = [0.0]
    phases = ["subtask 2: approach"]
    command_degrees = [
        rotation_magnitude_degrees(projector(controller_packets(controllers)))
    ]

    for step_index in range(CONTROL_STEPS):
        rotation_error = projector(controller_packets(controllers))
        state.matrix = limited_rotation_step(rotation_error) @ state.matrix

        if (step_index + 1) % PRE_BOUNDARY_RECORD_STRIDE == 0:
            current_command = projector(controller_packets(controllers))
            matrices.append(state.matrix.copy())
            times_s.append((step_index + 1) / CONTROL_HZ)
            phases.append("subtask 2: approach")
            command_degrees.append(
                rotation_magnitude_degrees(current_command)
            )

    exit_command = projector(controller_packets(controllers))
    desired_approach_axis = controllers[0].target

    approach_error = angle_between_degrees(
        state.eef_axis(EEF_Z_LOCAL), desired_approach_axis
    )
    upright_error = angle_between_degrees(
        state.eef_axis(EEF_X_LOCAL), upright_target
    )

    # This is the subtask boundary: reconstruct the exact same controllers from
    # the actual final orientation, just as robot_script.py does for subtask 3.
    next_controllers = make_controllers(
        state, surface_normal, upright_target
    )
    next_packets = controller_packets(next_controllers)
    next_command = projector(next_packets)

    result = SimulationResult(
        name=name,
        exit_command_deg=rotation_magnitude_degrees(exit_command),
        approach_axis_error_deg=approach_error,
        upright_axis_error_deg=upright_error,
        next_subtask_command_deg=rotation_magnitude_degrees(next_command),
        next_primary_packet_deg=float(
            np.degrees(np.linalg.norm(next_packets[0]["error"]))
        ),
        next_secondary_packet_deg=float(
            np.degrees(np.linalg.norm(next_packets[1]["error"]))
        ),
    )

    # Pause on the exact boundary so the hidden geometric error and newly
    # reconstructed command are readable before the correction starts.
    for _ in range(BOUNDARY_PAUSE_FRAMES):
        matrices.append(state.matrix.copy())
        times_s.append(float(SUBTASK_DURATION_S))
        phases.append("boundary: controllers reconstructed")
        command_degrees.append(result.next_subtask_command_deg)

    # Animate a short subtask-3 phase using the newly constructed controllers.
    # The legacy path visibly corrects its hidden error here; additive projection
    # has only the much smaller residual left by frozen a3.
    for step_index in range(POST_BOUNDARY_STEPS):
        rotation_error = projector(controller_packets(next_controllers))
        state.matrix = limited_rotation_step(rotation_error) @ state.matrix
        following_command = projector(controller_packets(next_controllers))

        matrices.append(state.matrix.copy())
        times_s.append(
            SUBTASK_DURATION_S + (step_index + 1) / CONTROL_HZ
        )
        phases.append("subtask 3: same controllers, reconstructed")
        command_degrees.append(
            rotation_magnitude_degrees(following_command)
        )

    trajectory = SimulationTrajectory(
        name=name,
        matrices=np.stack(matrices),
        times_s=np.asarray(times_s, dtype=float),
        phases=tuple(phases),
        command_degrees=np.asarray(command_degrees, dtype=float),
    )
    return result, trajectory


def _draw_vector(
    axis,
    vector: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 3.0,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    vector = unit(vector)
    axis.plot(
        [0.0, vector[0]],
        [0.0, vector[1]],
        [0.0, vector[2]],
        color=color,
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )


def _configure_3d_axis(axis) -> None:
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_zlim(-1.05, 1.05)
    axis.set_xlabel("Base X")
    axis.set_ylabel("Base Y")
    axis.set_zlabel("Base Z")
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=24, azim=38)
    axis.grid(True, alpha=0.25)


def animate_comparison(
    trajectories: Sequence[SimulationTrajectory],
    *,
    interval_ms: int = DEFAULT_ANIMATION_INTERVAL_MS,
    save_path: Path | None = None,
    show: bool = True,
):
    """Render synchronized legacy/additive EEF frames in two 3D panels."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(trajectories) != 2:
        raise ValueError("The comparison animation requires exactly two trajectories")

    frame_count = min(len(trajectory.matrices) for trajectory in trajectories)
    desired_approach = -unit(BUTTON_SURFACE_NORMAL)

    figure = plt.figure(figsize=(14, 7))
    axes = [
        figure.add_subplot(1, 2, 1, projection="3d"),
        figure.add_subplot(1, 2, 2, projection="3d"),
    ]

    def update(frame_index: int):
        phase = trajectories[0].phases[frame_index]
        phase_time = trajectories[0].times_s[frame_index]

        for axis, trajectory in zip(axes, trajectories):
            axis.clear()
            _configure_3d_axis(axis)

            matrix = trajectory.matrices[frame_index]
            current_x = matrix @ EEF_X_LOCAL
            current_z = matrix @ EEF_Z_LOCAL
            approach_error = angle_between_degrees(current_z, desired_approach)
            upright_error = angle_between_degrees(current_x, GLOBAL_Z)

            _draw_vector(
                axis,
                current_x,
                color="#d62728",
                label="current EEF X",
                linewidth=4.0,
            )
            _draw_vector(
                axis,
                current_z,
                color="#1f77b4",
                label="current EEF Z",
                linewidth=4.0,
            )
            _draw_vector(
                axis,
                GLOBAL_Z,
                color="#2ca02c",
                label="target EEF X (upright)",
                linewidth=2.5,
                linestyle="--",
                alpha=0.8,
            )
            _draw_vector(
                axis,
                desired_approach,
                color="#9467bd",
                label="target EEF Z (button)",
                linewidth=2.5,
                linestyle="--",
                alpha=0.8,
            )

            axis.set_title(
                f"{trajectory.name.upper()}\n"
                f"approach error: {approach_error:6.2f} deg | "
                f"upright error: {upright_error:6.2f} deg\n"
                f"published command: "
                f"{trajectory.command_degrees[frame_index]:6.2f} deg"
            )
            axis.legend(loc="upper left", fontsize=8)

        figure.suptitle(
            f"AxisAlign boundary reproduction | {phase} | t={phase_time:5.2f}s",
            fontsize=14,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        return axes

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        frames_per_second = max(1, round(1000 / interval_ms))
        animation.save(str(save_path), fps=frames_per_second)
        print(f"Saved animation to {save_path}")

    if show:
        plt.show()

    return animation


def print_results(results: list[SimulationResult]) -> None:
    headers = (
        "mode",
        "exit cmd",
        "actual Z err",
        "actual X err",
        "next entry cmd",
        "next primary",
        "next secondary",
    )
    print("\nAxisAlign subtask-boundary simulation (degrees)")
    print("-" * 105)
    print(
        f"{headers[0]:<12} {headers[1]:>11} {headers[2]:>14} "
        f"{headers[3]:>14} {headers[4]:>15} {headers[5]:>14} {headers[6]:>16}"
    )
    print("-" * 105)
    for result in results:
        print(
            f"{result.name:<12} "
            f"{result.exit_command_deg:>11.6f} "
            f"{result.approach_axis_error_deg:>14.6f} "
            f"{result.upright_axis_error_deg:>14.6f} "
            f"{result.next_subtask_command_deg:>15.6f} "
            f"{result.next_primary_packet_deg:>14.6f} "
            f"{result.next_secondary_packet_deg:>16.6f}"
        )
    print("-" * 105)


def assert_regression(legacy: SimulationResult, additive: SimulationResult) -> None:
    """Fail loudly if the reproduction or the additive-only comparison changes."""
    assert legacy.exit_command_deg < 1e-6, legacy
    assert legacy.approach_axis_error_deg > 10.0, legacy
    assert legacy.next_subtask_command_deg > 10.0, legacy

    # Additive projection should materially reduce the false-equilibrium error.
    assert additive.approach_axis_error_deg < legacy.approach_axis_error_deg * 0.1, (
        legacy,
        additive,
    )
    assert additive.next_subtask_command_deg < legacy.next_subtask_command_deg * 0.1, (
        legacy,
        additive,
    )

    # Frozen a3 remains intentionally unchanged in this comparison, so additive
    # projection alone is not expected to make the boundary correction zero.
    assert additive.next_subtask_command_deg > 0.1, additive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Run assertions and print results without opening a Matplotlib window.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optionally save the comparison animation (for example, comparison.gif).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_ANIMATION_INTERVAL_MS,
        help="Milliseconds between rendered animation frames (default: 40).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy, legacy_trajectory = run_simulation(
        "legacy", legacy_matrix_composition
    )
    additive, additive_trajectory = run_simulation(
        "additive", additive_projection
    )
    print_results([legacy, additive])
    assert_regression(legacy, additive)

    reduction = 100.0 * (
        1.0 - additive.next_subtask_command_deg / legacy.next_subtask_command_deg
    )
    print(
        f"PASS: additive projection reduced the reconstructed boundary command "
        f"by {reduction:.2f}%."
    )
    print(
        "NOTE: the remaining additive-mode correction demonstrates that frozen a3 "
        "is a separate issue."
    )

    if not args.no_animation or args.save is not None:
        animate_comparison(
            [legacy_trajectory, additive_trajectory],
            interval_ms=args.interval_ms,
            save_path=args.save,
            show=not args.no_animation,
        )


if __name__ == "__main__":
    main()
