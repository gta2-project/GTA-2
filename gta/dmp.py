"""Utilities for fitting and rolling out discrete positional DMPs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np


def generate_minimum_jerk_spiral(
    samples: int = 2001,
    turns: float = 3.0,
    direction: str = "ccw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a normalized Archimedean spiral with minimum-jerk timing."""
    if not isinstance(samples, int) or samples < 3:
        raise ValueError("samples must be an integer greater than or equal to 3")
    if not np.isfinite(turns) or turns <= 0:
        raise ValueError("turns must be finite and positive")
    if direction not in {"cw", "ccw"}:
        raise ValueError("direction must be either 'cw' or 'ccw'")

    time = np.linspace(0.0, 1.0, samples)
    phase = 10.0 * time**3 - 15.0 * time**4 + 6.0 * time**5
    direction_sign = 1.0 if direction == "ccw" else -1.0
    angle = direction_sign * 2.0 * np.pi * float(turns) * phase
    trajectory = np.column_stack(
        (phase * np.cos(angle), phase * np.sin(angle))
    )
    trajectory[0] = np.zeros(2)

    # Integer-turn demonstrations have an exact normalized endpoint. Setting it
    # explicitly avoids a tiny floating-point residual in the learned goal.
    if np.isclose(turns, round(turns)):
        trajectory[-1] = np.array([1.0, 0.0])

    return time, trajectory


def _basis_parameters(alpha_x: float, basis_functions: int) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.exp(-alpha_x * np.linspace(0.0, 1.0, basis_functions))
    center_steps = np.diff(centers)
    widths = np.empty_like(centers)
    widths[:-1] = 1.0 / (center_steps**2 + 1e-12)
    widths[-1] = widths[-2]
    return centers, widths


def fit_discrete_dmp(
    demonstration: np.ndarray,
    *,
    alpha_x: float = 4.0,
    alpha_z: float = 25.0,
    beta_z: float = 6.25,
    basis_functions: int = 25,
    ridge: float = 1e-8,
) -> Dict[str, Any]:
    """Fit an unscaled multidimensional discrete DMP to a normalized demonstration."""
    demonstration = np.asarray(demonstration, dtype=float)
    if demonstration.ndim != 2 or demonstration.shape[0] < 3:
        raise ValueError("demonstration must have shape (samples, dimensions)")
    if not np.all(np.isfinite(demonstration)):
        raise ValueError("demonstration contains non-finite values")
    if not isinstance(basis_functions, int) or basis_functions < 2:
        raise ValueError("basis_functions must be an integer greater than or equal to 2")

    scalar_parameters = {
        "alpha_x": alpha_x,
        "alpha_z": alpha_z,
        "beta_z": beta_z,
        "ridge": ridge,
    }
    for name, value in scalar_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    sample_count = demonstration.shape[0]
    dt = 1.0 / (sample_count - 1)
    velocity = np.gradient(demonstration, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)

    canonical_phase = np.exp(-alpha_x * np.linspace(0.0, 1.0, sample_count))
    centers, widths = _basis_parameters(alpha_x, basis_functions)
    activations = np.exp(
        -widths[None, :] * (canonical_phase[:, None] - centers[None, :]) ** 2
    )
    features = (
        canonical_phase[:, None]
        * activations
        / (np.sum(activations, axis=1, keepdims=True) + 1e-12)
    )

    y0 = demonstration[0].copy()
    goal = demonstration[-1].copy()
    forcing_target = acceleration - alpha_z * (
        beta_z * (goal - demonstration) - velocity
    )

    normal_matrix = features.T @ features + ridge * np.eye(basis_functions)
    weights = np.linalg.solve(normal_matrix, features.T @ forcing_target).T

    return {
        "alpha_x": float(alpha_x),
        "alpha_z": float(alpha_z),
        "beta_z": float(beta_z),
        "basis_functions": int(basis_functions),
        "ridge": float(ridge),
        "y0": y0,
        "goal": goal,
        "centers": centers,
        "widths": widths,
        "weights": weights,
    }


def rollout_discrete_dmp(model: Mapping[str, Any], samples: int) -> np.ndarray:
    """Integrate a fitted discrete DMP over normalized time [0, 1]."""
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer greater than or equal to 2")

    alpha_x = float(model["alpha_x"])
    alpha_z = float(model["alpha_z"])
    beta_z = float(model["beta_z"])
    centers = np.asarray(model["centers"], dtype=float)
    widths = np.asarray(model["widths"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    position = np.asarray(model["y0"], dtype=float).copy()
    goal = np.asarray(model["goal"], dtype=float)

    if centers.ndim != 1 or widths.shape != centers.shape:
        raise ValueError("DMP centers and widths must be matching one-dimensional arrays")
    if weights.ndim != 2 or weights.shape[1] != centers.size:
        raise ValueError("DMP weights must have shape (dimensions, basis_functions)")
    if position.shape != goal.shape or weights.shape[0] != position.size:
        raise ValueError("DMP y0, goal, and weight dimensions do not match")

    dt = 1.0 / (samples - 1)
    velocity = np.zeros_like(position)
    canonical_phase = 1.0
    trajectory = np.empty((samples, position.size), dtype=float)
    trajectory[0] = position

    for index in range(1, samples):
        activations = np.exp(-widths * (canonical_phase - centers) ** 2)
        forcing = (
            canonical_phase
            * (activations @ weights.T)
            / (np.sum(activations) + 1e-12)
        )
        acceleration = alpha_z * (beta_z * (goal - position) - velocity) + forcing
        velocity += acceleration * dt
        position += velocity * dt
        canonical_phase += -alpha_x * canonical_phase * dt
        trajectory[index] = position

    if not np.all(np.isfinite(trajectory)):
        raise ValueError("DMP rollout produced non-finite values")
    return trajectory
