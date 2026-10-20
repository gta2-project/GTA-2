"""Frozen perception and live robot-state access for generated policies."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return vector


def _pose_transform(pose) -> np.ndarray:
    value = _finite_vector(pose, 6, "object pose")
    transform = np.eye(4)
    transform[:3, :3] = R.from_rotvec(value[3:]).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


class PolicyObject:
    """Query-derived object whose geometry can attach to the live TCP frame.

    This mirrors the ``ObjectHandler`` contract used by GTA controller scripts,
    while reading the pose from this policy's configured provider so deterministic
    offline contexts do not open the robot connection.
    """

    def __init__(
        self,
        name: str,
        current_pose_provider: Callable[[], Sequence[float]],
    ):
        self.name = str(name)
        self.keypoints: Dict[str, np.ndarray] = {}
        self.axes: Dict[str, np.ndarray] = {}
        self.frame = "world"
        self._current_pose_provider = current_pose_provider

    def add_keypoint(self, alias: str, position) -> None:
        self.keypoints[str(alias)] = _finite_vector(
            position, 3, f"{self.name}.keypoint({alias!r})"
        ).copy()

    def add_axis(self, alias: str, direction) -> None:
        axis = _finite_vector(direction, 3, f"{self.name}.axis({alias!r})")
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            raise ValueError(f"{self.name}.axis({alias!r}) must be non-zero")
        self.axes[str(alias)] = (axis / length).copy()

    def _current_transform(self) -> np.ndarray:
        return _pose_transform(self._current_pose_provider())

    def get_k(self, alias: str) -> np.ndarray:
        try:
            point = self.keypoints[str(alias)]
        except KeyError as exc:
            raise KeyError(
                f"Object {self.name!r} has no keypoint alias {alias!r}"
            ) from exc
        if self.frame == "robot":
            homogeneous = np.concatenate([point, [1.0]])
            return (self._current_transform() @ homogeneous)[:3]
        return point.copy()

    def get_a(self, alias: str) -> np.ndarray:
        try:
            axis = self.axes[str(alias)]
        except KeyError as exc:
            raise KeyError(f"Object {self.name!r} has no axis alias {alias!r}") from exc
        if self.frame == "robot":
            return self._current_transform()[:3, :3] @ axis
        return axis.copy()

    def set_frame(self, frame: str) -> None:
        target_frame = str(frame).lower()
        if target_frame not in {"world", "robot"}:
            raise ValueError("PolicyObject frame must be 'world' or 'robot'")
        if self.frame == target_frame:
            return

        transform = self._current_transform()
        if target_frame == "robot":
            transform = np.linalg.inv(transform)

        for alias, point in self.keypoints.items():
            homogeneous = np.concatenate([point, [1.0]])
            self.keypoints[alias] = (transform @ homogeneous)[:3]
        for alias, axis in self.axes.items():
            self.axes[alias] = transform[:3, :3] @ axis
        self.frame = target_frame

    def __repr__(self) -> str:
        return (
            f"PolicyObject(name={self.name!r}, keypoints={len(self.keypoints)}, "
            f"axes={len(self.axes)}, frame={self.frame!r})"
        )


class PolicyContext:
    """Read-only observations exposed to a validated CaP policy.

    Points and normals are frozen for an entire trial. The TCP pose provider is
    live in robot mode and may be a deterministic fixture in offline mode.
    """

    def __init__(
        self,
        points: Mapping[str, Sequence[float]],
        normals: Optional[Mapping[str, Sequence[float]]] = None,
        current_pose_provider: Optional[Callable[[], Sequence[float]]] = None,
        query_objects: Optional[Mapping[str, str]] = None,
    ):
        self._points: Dict[str, np.ndarray] = {
            str(name): _finite_vector(value, 3, f"point({name!r})").copy()
            for name, value in points.items()
        }
        self._normals: Dict[str, np.ndarray] = {}
        for name, value in (normals or {}).items():
            normal = _finite_vector(value, 3, f"normal({name!r})")
            length = float(np.linalg.norm(normal))
            if length < 1e-9:
                raise ValueError(f"normal({name!r}) must be non-zero")
            self._normals[str(name)] = normal / length

        self._current_pose_provider = current_pose_provider
        self._objects_by_alias: Dict[str, PolicyObject] = {}
        if query_objects:
            if self._current_pose_provider is None:
                raise ValueError("Attachable policy objects require a TCP pose provider")
            objects_by_name: Dict[str, PolicyObject] = {}
            for raw_alias, raw_object_name in query_objects.items():
                alias = str(raw_alias)
                if alias not in self._points:
                    raise KeyError(
                        f"Object metadata references unknown perception point {alias!r}"
                    )
                object_name = str(raw_object_name).strip()
                if not object_name:
                    raise ValueError(f"Object name for query {alias!r} must be non-empty")
                policy_object = objects_by_name.setdefault(
                    object_name,
                    PolicyObject(object_name, self.current_pose),
                )
                policy_object.add_keypoint(alias, self._points[alias])
                if alias in self._normals:
                    policy_object.add_axis(alias, self._normals[alias])
                self._objects_by_alias[alias] = policy_object

    @property
    def aliases(self):
        return tuple(sorted(self._points))

    def point(self, name: str) -> np.ndarray:
        try:
            return self._points[str(name)].copy()
        except KeyError as exc:
            raise KeyError(f"Unknown perception point {name!r}") from exc

    def normal(self, name: str) -> np.ndarray:
        try:
            return self._normals[str(name)].copy()
        except KeyError as exc:
            raise KeyError(f"No surface normal was localized for {name!r}") from exc

    def object_for(self, alias: str) -> PolicyObject:
        """Return the attachable object declared by a keypoint query's object field."""
        try:
            return self._objects_by_alias[str(alias)]
        except KeyError as exc:
            raise KeyError(f"No attachable object is registered for query {alias!r}") from exc

    def current_pose(self) -> np.ndarray:
        if self._current_pose_provider is None:
            raise RuntimeError("Current TCP pose is unavailable in this execution mode")
        return _finite_vector(self._current_pose_provider(), 6, "current_pose()").copy()

    def tcp_position(self) -> np.ndarray:
        return self.current_pose()[:3]

    def tcp_axis(self, axis: str) -> np.ndarray:
        column = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower())
        if column is None:
            raise ValueError("tcp_axis() accepts only 'x', 'y', or 'z'")
        rotation = R.from_rotvec(self.current_pose()[3:]).as_matrix()
        return rotation[:, column].copy()


def norm(value) -> float:
    vector = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError("norm() input must be finite")
    return float(np.linalg.norm(vector))


def normalize(value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("normalize() input must be a finite 3D vector")
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("normalize() input must be non-zero")
    return vector / length


def dot(left, right) -> float:
    left_vector = _finite_vector(left, 3, "dot() left input")
    right_vector = _finite_vector(right, 3, "dot() right input")
    return float(np.dot(left_vector, right_vector))


def cross(left, right) -> np.ndarray:
    left_vector = _finite_vector(left, 3, "cross() left input")
    right_vector = _finite_vector(right, 3, "cross() right input")
    return np.cross(left_vector, right_vector)
