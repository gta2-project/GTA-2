"""Learned planar positional Dynamic Motor Primitive controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from .base_controller import BaseController
from gta.dmp import rollout_discrete_dmp


_SPIRAL_MODEL_PATH = Path(__file__).resolve().parent / "models" / "spiral_planar.yaml"


def _finite_vector(value: Any, shape: tuple, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def _load_spiral_model() -> Dict[str, Any]:
    model_path = _SPIRAL_MODEL_PATH
    try:
        with model_path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise ValueError(f"Unable to load PosDMPSpiral model: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML for PosDMPSpiral model: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError("PosDMPSpiral model must contain a YAML mapping")
    if document.get("version") != 1:
        raise ValueError("PosDMPSpiral model version must be 1")
    if document.get("primitive") != "discrete_planar_dmp":
        raise ValueError("PosDMPSpiral model primitive must be 'discrete_planar_dmp'")

    dmp = document.get("dmp")
    execution = document.get("execution")
    if not isinstance(dmp, dict) or not isinstance(execution, dict):
        raise ValueError("PosDMPSpiral model requires 'dmp' and 'execution' mappings")

    try:
        basis_functions = int(dmp["basis_functions"])
        rollout_samples = int(execution["rollout_samples"])
        settle_fraction = float(execution["settle_fraction"])
        alpha_x = float(dmp["alpha_x"])
        alpha_z = float(dmp["alpha_z"])
        beta_z = float(dmp["beta_z"])
        ridge = float(dmp["ridge"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"PosDMPSpiral model has invalid scalar parameters: {exc}"
        ) from exc

    if basis_functions < 2:
        raise ValueError("PosDMPSpiral basis_functions must be at least 2")
    if rollout_samples < 2:
        raise ValueError("PosDMPSpiral rollout_samples must be at least 2")
    if not np.isfinite(settle_fraction) or not 0.0 <= settle_fraction < 1.0:
        raise ValueError("PosDMPSpiral settle_fraction must be in [0, 1)")
    for name, value in {
        "alpha_x": alpha_x,
        "alpha_z": alpha_z,
        "beta_z": beta_z,
        "ridge": ridge,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"PosDMPSpiral {name} must be finite and positive")

    centers = _finite_vector(dmp.get("centers"), (basis_functions,), "centers")
    widths = _finite_vector(dmp.get("widths"), (basis_functions,), "widths")
    weights = _finite_vector(
        dmp.get("weights"), (2, basis_functions), "weights"
    )
    y0 = _finite_vector(dmp.get("y0"), (2,), "y0")
    goal = _finite_vector(dmp.get("goal"), (2,), "goal")

    if np.any(widths <= 0):
        raise ValueError("PosDMPSpiral widths must be positive")
    if not np.allclose(y0, np.zeros(2), atol=1e-9):
        raise ValueError("PosDMPSpiral model must start at the local origin")
    if not np.isclose(np.linalg.norm(goal), 1.0, atol=1e-6):
        raise ValueError("PosDMPSpiral model goal must have normalized radius 1")

    return {
        "model_path": model_path,
        "rollout_samples": rollout_samples,
        "settle_fraction": settle_fraction,
        "dmp": {
            "alpha_x": alpha_x,
            "alpha_z": alpha_z,
            "beta_z": beta_z,
            "basis_functions": basis_functions,
            "ridge": ridge,
            "y0": y0,
            "goal": goal,
            "centers": centers,
            "widths": widths,
            "weights": weights,
        },
    }


class PosDMPSpiral(BaseController):
    """Replay a learned planar positional DMP from a live keypoint's current position."""

    def __init__(self, g1, a1, o1):
        self.type = "positional"

        if not callable(g1):
            raise TypeError("PosDMPSpiral g1 must be a callable returning a live 3D position")
        self.g1 = g1

        self.radius = float(o1)
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("PosDMPSpiral radius o1 must be finite and positive")

        normal = _finite_vector(a1, (3,), "a1")
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            raise ValueError("PosDMPSpiral plane normal a1 must be non-zero")
        self.normal = normal / normal_norm

        self.origin = _finite_vector(self.g1(), (3,), "g1()").copy()
        self.plane_u, self.plane_v = self._make_plane_basis(self.normal)

        primitive = _load_spiral_model()
        self.model_path = primitive["model_path"]
        self.settle_fraction = primitive["settle_fraction"]
        self.local_goal = primitive["dmp"]["goal"].copy()
        self.rollout = rollout_discrete_dmp(
            primitive["dmp"], primitive["rollout_samples"]
        )

        print(
            f"[PosDMPSpiral] Initialized origin={self.origin}, "
            f"radius={self.radius:.3f} m, normal={self.normal}"
        )

    @staticmethod
    def _make_plane_basis(normal: np.ndarray):
        reference = np.array([1.0, 0.0, 0.0])
        plane_u = reference - np.dot(reference, normal) * normal
        if np.linalg.norm(plane_u) < 1e-6:
            reference = np.array([0.0, 1.0, 0.0])
            plane_u = reference - np.dot(reference, normal) * normal

        plane_u /= np.linalg.norm(plane_u)
        plane_v = np.cross(normal, plane_u)
        plane_v /= np.linalg.norm(plane_v)
        return plane_u, plane_v

    def _local_target(self, elapsed_time: float) -> np.ndarray:
        elapsed_time = float(elapsed_time)
        if not np.isfinite(elapsed_time):
            raise ValueError("PosDMPSpiral elapsed_time must be finite")
        elapsed_time = float(np.clip(elapsed_time, 0.0, 1.0))

        motion_fraction = 1.0 - self.settle_fraction
        if elapsed_time >= motion_fraction:
            return self.local_goal.copy()

        phase = elapsed_time / motion_fraction
        sample_position = phase * (len(self.rollout) - 1)
        lower = int(np.floor(sample_position))
        upper = min(lower + 1, len(self.rollout) - 1)
        blend = sample_position - lower
        return (1.0 - blend) * self.rollout[lower] + blend * self.rollout[upper]

    def target_at(self, elapsed_time: float) -> np.ndarray:
        """Return the world-frame target for a normalized subtask time."""
        local_target = self._local_target(elapsed_time)
        return self.origin + self.radius * (
            local_target[0] * self.plane_u + local_target[1] * self.plane_v
        )

    def compute_error(self, elapsed_time=0.0):
        current = _finite_vector(self.g1(), (3,), "g1()")
        error = self.target_at(elapsed_time) - current
        error -= np.dot(error, self.normal) * self.normal

        return {
            "error": error,
            "type": self.type,
            "axes": [self.plane_u.copy(), self.plane_v.copy()],
        }

    def __repr__(self):
        return (
            f"PosDMPSpiral(g1={self.g1}, a1={self.normal}, o1={self.radius})"
        )


# Import compatibility for previously generated scripts. New prompts and script
# generation expose only the dedicated PosDMPSpiral controller.
PosDMP = PosDMPSpiral
