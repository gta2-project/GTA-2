"""Execution environments for parsed CaP policies."""

from __future__ import annotations

import builtins
import math
import time
from typing import Callable, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .conditions import ApiKind
from .context import PolicyContext, cross, dot, norm, normalize
from .validation import ValidatedPolicy


class OfflineControllerRuntime:
    """Records controller phases without importing or commanding ROS."""

    def __init__(self, event_sink: Optional[Callable[[dict], None]] = None):
        self.phases = []
        self._event_sink = event_sink or (lambda event: None)

    def execute_subtask(self, controllers, duration=10.0):
        controllers = list(controllers)
        object_transitions = []
        for controller in controllers:
            if getattr(controller, "type", None) != "gripper":
                continue
            packet = controller.compute_error(elapsed_time=1.0)
            state = packet.get("state")
            policy_object = packet.get("object")
            if state is None or policy_object is None:
                continue
            frame = "robot" if float(state) == 0.0 else "world"
            policy_object.set_frame(frame)
            object_transitions.append(
                {"object": policy_object.name, "frame": frame}
            )
        phase = {
            "controller_types": [controller.__class__.__name__ for controller in controllers],
            "duration_s": float(duration),
            "object_transitions": object_transitions,
        }
        self.phases.append(phase)
        self._event_sink({"event": "offline_controller_phase", **phase})


class ControllerRobotRuntime:
    """CAP-local controller loop suitable for a web worker thread.

    GTA's script executor installs a process-level SIGINT handler, which Python
    permits only in the main interpreter thread. The CaP UI deliberately uses
    this equivalent loop without changing or monkey-patching the GTA executor.
    """

    def __init__(
        self,
        start_skill: Callable[[], None],
        end_skill: Callable[[], None],
        update_control_commands: Callable[..., bool],
        error_projector: Callable,
        shutdown_requested: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self._start_skill = start_skill
        self._end_skill = end_skill
        self._update_control_commands = update_control_commands
        self._error_projector = error_projector
        self._shutdown_requested = shutdown_requested or (lambda: False)
        self._clock = clock
        self._sleeper = sleeper

    def execute_subtask(self, controllers, duration=10.0):
        controllers = list(controllers)
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("Controller subtask duration must be finite and positive")

        start_time = self._clock()
        self._start_skill()
        try:
            while not self._shutdown_requested():
                elapsed_time = min(
                    max((self._clock() - start_time) / duration, 0.0),
                    1.0,
                )
                errors = []
                free_rotation = False
                for controller in controllers:
                    if controller.type == "rotational":
                        errors.append(
                            controller.compute_error(
                                elapsed_time=elapsed_time,
                                free_rotation=free_rotation,
                            )
                        )
                        free_rotation = True
                    else:
                        errors.append(
                            controller.compute_error(elapsed_time=elapsed_time)
                        )

                commands = self._error_projector(errors)
                if not self._update_control_commands(*commands):
                    raise RuntimeError("Failed to publish control commands")

                try:
                    self._sleeper(0.01)
                except KeyboardInterrupt:
                    break
                if elapsed_time >= 1.0:
                    break
        finally:
            self._end_skill()


class OfflineCartesianExecutor:
    """Dry-run Cartesian API that records calls and never imports RTDE."""

    def __init__(self, initial_pose, event_sink: Optional[Callable[[dict], None]] = None):
        import numpy as np

        pose = np.asarray(initial_pose, dtype=float)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ValueError("Offline Cartesian initial_pose must be a finite 6D pose")
        self._pose = pose.copy()
        self.calls = []
        self._event_sink = event_sink or (lambda event: None)

    def current_pose(self):
        return self._pose.copy()

    def move_to(self, x, y, z, rx, ry, rz):
        target = self._target([x, y, z, rx, ry, rz])
        self._record("move_to", target_pose=target.tolist())
        self._pose = target

    def gripper(self, state):
        if state not in {"open", "close"}:
            raise ValueError("gripper(state) accepts only 'open' or 'close'")
        self._record("gripper", state=state)

    def move_force(self, x, y, z, rx, ry, rz, fx, fy, fz, fn):
        import numpy as np

        target = self._target([x, y, z, rx, ry, rz])
        axis = np.asarray([fx, fy, fz], dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-9:
            raise ValueError("Force axis must be a finite non-zero 3D vector")
        magnitude = float(fn)
        if not np.isfinite(magnitude) or magnitude == 0.0:
            raise ValueError("fn must be finite and non-zero")
        self._record(
            "move_force",
            target_pose=target.tolist(),
            force_axis=(axis / np.linalg.norm(axis)).tolist(),
            force_n=magnitude,
        )
        # This is an API dry run, not a contact simulator. Advancing to the
        # requested pose only lets later generated calls inspect deterministic state.
        self._pose = target

    @staticmethod
    def _target(values):
        import numpy as np

        target = np.asarray(values, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("Cartesian target must contain six finite values")
        return target

    def _record(self, primitive, **payload):
        event = {"event": "offline_cartesian_call", "primitive": primitive, **payload}
        self.calls.append(event)
        self._event_sink(event)


def _controller_namespace(execute_subtask):
    from controllers import (
        AxisAlign,
        AxisWaypoint,
        ForceConstant,
        GripperControl,
        PosAlign,
        PosDMPSpiral,
        PosWaypoint,
    )
    from gta.utils import a_along_keypoints, a_crossprod_axes

    return {
        "AxisAlign": AxisAlign,
        "AxisWaypoint": AxisWaypoint,
        "ForceConstant": ForceConstant,
        "GripperControl": GripperControl,
        "PosAlign": PosAlign,
        "PosDMPSpiral": PosDMPSpiral,
        "PosWaypoint": PosWaypoint,
        "a_along_keypoints": a_along_keypoints,
        "a_crossprod_axes": a_crossprod_axes,
        "execute_subtask": execute_subtask,
    }


def execute_validated_policy(
    policy: ValidatedPolicy,
    api_kind: ApiKind,
    context: PolicyContext,
    cartesian_executor=None,
    controller_runtime=None,
    event_sink: Optional[Callable[[dict], None]] = None,
):
    """Compile and execute parsed policy source in the CaP API namespace."""
    emit = event_sink or (lambda event: None)
    namespace: Dict[str, object] = {
        "__builtins__": builtins.__dict__,
        "cross": cross,
        "dot": dot,
        "math": math,
        "norm": norm,
        "normalize": normalize,
        "np": np,
        "R": R,
    }

    if api_kind == ApiKind.CARTESIAN:
        if cartesian_executor is None:
            raise ValueError("Cartesian policy execution requires a CartesianExecutor")
        namespace.update(
            {
                "move_to": cartesian_executor.move_to,
                "gripper": cartesian_executor.gripper,
                "move_force": cartesian_executor.move_force,
            }
        )
    else:
        if controller_runtime is None:
            controller_runtime = OfflineControllerRuntime(event_sink=emit)
        execute_subtask = getattr(controller_runtime, "execute_subtask", controller_runtime)
        if not callable(execute_subtask):
            raise TypeError("controller_runtime must provide callable execute_subtask")
        namespace.update(_controller_namespace(execute_subtask))

    code = compile(policy.tree, "<cap-policy>", "exec")
    exec(code, namespace, namespace)
    generated_policy = namespace.get("policy")
    if not callable(generated_policy):
        raise RuntimeError("Program did not create callable policy(ctx)")

    emit({"event": "policy_started", "api_kind": api_kind.value})
    try:
        result = generated_policy(context)
    except Exception as exc:
        emit({"event": "policy_failed", "api_kind": api_kind.value, "error": str(exc)})
        raise
    emit({"event": "policy_completed", "api_kind": api_kind.value})
    return result
