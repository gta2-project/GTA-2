"""Dedicated RTDE executor for the Cartesian CaP benchmark condition.

The module deliberately imports robot-only dependencies lazily. Unit tests and
offline benchmark runs can therefore use injected interfaces without ROS, RTDE,
or a connected robot.
"""

from __future__ import annotations

import fcntl
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


class CartesianExecutionError(RuntimeError):
    """Raised for validation, connection, RTDE, gripper, or safety failures."""


class CartesianOwnershipError(CartesianExecutionError):
    """Raised when another Cartesian executor owns the process lock."""


@dataclass(frozen=True)
class CartesianConfig:
    robot_ip: str = "192.168.1.10"
    move_speed: float = 0.1
    move_acceleration: float = 0.1
    control_hz: float = 100.0
    force_duration_s: float = 5.0
    lookahead_s: float = 0.10
    servo_gain: float = 400.0
    position_speed_mps: float = 0.8
    position_tolerance_m: float = 0.002
    max_angle_step_rad: float = 0.03
    orientation_deadband_rad: float = np.deg2rad(0.05)
    force_type: int = 2
    force_limits: tuple = (2.0, 2.0, 2.0, 1.0, 1.0, 1.0)
    reconnect_sleep_s: float = 0.75
    reconnect_initialize_s: float = 0.35
    reconnect_settle_s: float = 0.02
    gripper_timeout_s: float = 10.0
    workspace_min: tuple = (-1.0, -1.5, -0.10)
    workspace_max: tuple = (1.0, 0.50, 1.50)
    max_abs_force_n: float = 30.0
    ownership_lock_path: str = "/tmp/gta2_cap_rtde.lock"

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @classmethod
    def from_environment(cls):
        return cls(robot_ip=os.getenv("ROBOT_IP", cls.robot_ip))


def _safe_call(obj, method_name: str, *args):
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _default_control_factory(robot_ip: str):
    from rtde_control import RTDEControlInterface

    return RTDEControlInterface(robot_ip)


def _default_receive_factory(robot_ip: str):
    from rtde_receive import RTDEReceiveInterface

    return RTDEReceiveInterface(robot_ip)


def get_safe_target_position(
    current_position,
    position_error,
    dt: float,
    speed_mps: float,
    tolerance_m: float,
):
    """Match the current robot server's norm-limited Cartesian step."""
    current = np.asarray(current_position, dtype=float).reshape(3)
    error = np.asarray(position_error, dtype=float).reshape(3)
    distance = float(np.linalg.norm(error))
    if distance <= tolerance_m:
        return current + error
    max_step = float(speed_mps) * float(dt)
    return current + error / distance * min(distance, max_step)


def get_safe_target_rotvec(
    current_rotation,
    rotation_error,
    max_angle_step: float,
    deadband: float,
):
    """Match the current robot server's step-limited SO(3) correction."""
    current_rotation = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    rotation_error = np.asarray(rotation_error, dtype=float).reshape(3, 3)
    error_rotation = R.from_matrix(rotation_error)
    angle = float(error_rotation.magnitude())
    if angle <= deadband:
        return R.from_matrix(rotation_error @ current_rotation).as_rotvec()
    rotvec = error_rotation.as_rotvec()
    axis = rotvec / (angle + 1e-12)
    step = R.from_rotvec(axis * min(angle, max_angle_step)).as_matrix()
    return R.from_matrix(step @ current_rotation).as_rotvec()


class RobotiqGripperDriver:
    """Small blocking adapter around the repository's Robotiq ROS action."""

    def __init__(self, timeout_s: float = 10.0):
        import actionlib
        import rospy
        import robotiq_mm_ros.msg

        self._rospy = rospy
        self._messages = robotiq_mm_ros.msg
        self._timeout_s = float(timeout_s)
        self._client = actionlib.SimpleActionClient(
            "/ur5/move_gripper",
            robotiq_mm_ros.msg.GripperCommandAction,
        )
        if not self._client.wait_for_server(rospy.Duration(self._timeout_s)):
            raise CartesianExecutionError("Timed out waiting for Robotiq gripper action server")

    def command(self, state: str):
        position = 0.085 if state == "open" else 0.0
        goal = self._messages.GripperCommandGoal(
            position=position,
            velocity=0.05,
            force=165,
        )
        self._client.send_goal(goal)
        if not self._client.wait_for_result(self._rospy.Duration(self._timeout_s)):
            self._client.cancel_goal()
            raise CartesianExecutionError(f"Gripper {state!r} command timed out")


class CartesianExecutor:
    """Exclusive, blocking implementation of the three Cartesian primitives."""

    def __init__(
        self,
        config: Optional[CartesianConfig] = None,
        control_factory: Optional[Callable[[str], object]] = None,
        receive_factory: Optional[Callable[[str], object]] = None,
        gripper_driver=None,
        event_sink: Optional[Callable[[dict], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config or CartesianConfig.from_environment()
        self._control_factory = control_factory or _default_control_factory
        self._receive_factory = receive_factory or _default_receive_factory
        self._gripper_driver = gripper_driver
        self._event_sink = event_sink or (lambda event: None)
        self._monotonic = monotonic
        self._sleep = sleep
        self._control = None
        self._receive = None
        self._lock_handle = None

        self._validate_config()

    def _validate_config(self):
        numeric_positive = {
            "move_speed": self.config.move_speed,
            "move_acceleration": self.config.move_acceleration,
            "control_hz": self.config.control_hz,
            "force_duration_s": self.config.force_duration_s,
            "lookahead_s": self.config.lookahead_s,
            "servo_gain": self.config.servo_gain,
            "position_speed_mps": self.config.position_speed_mps,
            "max_angle_step_rad": self.config.max_angle_step_rad,
            "max_abs_force_n": self.config.max_abs_force_n,
        }
        for name, value in numeric_positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"CartesianConfig.{name} must be finite and positive")
        lower = np.asarray(self.config.workspace_min, dtype=float)
        upper = np.asarray(self.config.workspace_max, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("Cartesian workspace bounds must be valid 3D min/max vectors")

    @property
    def connected(self) -> bool:
        return self._control is not None and self._receive is not None

    def connect(self):
        if self.connected:
            return self
        lock_path = Path(self.config.ownership_lock_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise CartesianOwnershipError(
                "Another Cartesian CaP executor owns the RTDE lock. Stop it and the "
                "existing robot server before starting a Cartesian robot trial."
            ) from exc

        self._lock_handle = handle
        try:
            self._control = self._control_factory(self.config.robot_ip)
            self._receive = self._receive_factory(self.config.robot_ip)
            zero_ft_sensor = getattr(self._control, "zeroFtSensor", None)
            if not callable(zero_ft_sensor):
                raise CartesianExecutionError(
                    "RTDE Control does not provide zeroFtSensor()"
                )
            try:
                zeroed = zero_ft_sensor()
            except Exception as exc:
                raise CartesianExecutionError(
                    f"Unable to zero the F/T sensor at Cartesian startup: {exc}"
                ) from exc
            if zeroed is False:
                raise CartesianExecutionError(
                    "zeroFtSensor() returned failure at Cartesian startup"
                )
            self._emit("ft_sensor_zeroed")
        except Exception:
            self.close()
            raise
        self._emit("executor_connected", robot_ip=self.config.robot_ip)
        return self

    def close(self):
        if self._control is not None:
            _safe_call(self._control, "servoStop")
            _safe_call(self._control, "stopScript")
            _safe_call(self._control, "disconnect")
            self._control = None
        if self._receive is not None:
            _safe_call(self._receive, "disconnect")
            self._receive = None
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None
        self._emit("executor_closed")

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def current_pose(self) -> np.ndarray:
        self._ensure_connected()
        try:
            pose = np.asarray(self._receive.getActualTCPPose(), dtype=float)
        except Exception as exc:
            raise CartesianExecutionError(f"Unable to read TCP pose: {exc}") from exc
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise CartesianExecutionError(f"RTDE returned invalid TCP pose: {pose}")
        return pose

    def move_to(self, x, y, z, rx, ry, rz):
        target = self._validate_pose([x, y, z, rx, ry, rz])
        self._ensure_connected()
        started = self._monotonic()
        self._emit("move_to_started", target_pose=target.tolist())
        try:
            result = self._control.moveL(
                target.tolist(),
                self.config.move_speed,
                self.config.move_acceleration,
            )
        except Exception as exc:
            raise CartesianExecutionError(f"RTDE moveL failed: {exc}") from exc
        if result is False:
            raise CartesianExecutionError("RTDE moveL returned failure")
        final_pose = self.current_pose()
        self._emit_pose_result(
            "move_to_completed",
            target,
            final_pose,
            self._monotonic() - started,
        )

    def gripper(self, state):
        if state not in {"open", "close"}:
            raise ValueError("gripper(state) accepts only 'open' or 'close'")
        self._ensure_connected()
        if self._gripper_driver is None:
            self._gripper_driver = RobotiqGripperDriver(self.config.gripper_timeout_s)
        started = self._monotonic()
        self._emit("gripper_started", state=state)
        try:
            self._gripper_driver.command(state)
        except CartesianExecutionError:
            raise
        except Exception as exc:
            raise CartesianExecutionError(f"Gripper {state!r} command failed: {exc}") from exc
        self._emit(
            "gripper_completed",
            state=state,
            elapsed_s=self._monotonic() - started,
        )

    def move_force(self, x, y, z, rx, ry, rz, fx, fy, fz, fn):
        target = self._validate_pose([x, y, z, rx, ry, rz])
        force_axis = self._validate_force_axis([fx, fy, fz])
        magnitude = self._validate_force_magnitude(fn)
        self._ensure_connected()
        task_frame = self._make_force_task_frame(force_axis)
        selection_vector = [0, 0, 1, 0, 0, 0]
        wrench = [0.0, 0.0, magnitude, 0.0, 0.0, 0.0]
        cycles = int(round(self.config.force_duration_s * self.config.control_hz))
        started = self._monotonic()
        self._emit(
            "move_force_started",
            target_pose=target.tolist(),
            force_axis=force_axis.tolist(),
            force_n=magnitude,
            duration_s=self.config.force_duration_s,
            expected_cycles=cycles,
        )
        operation_error = None
        final_pose = None
        actual_wrench = None
        try:
            target_rotation = R.from_rotvec(target[3:]).as_matrix()
            for _ in range(cycles):
                cycle_started = self._monotonic()
                period = self._begin_period()
                current = self.current_pose()
                current_rotation = R.from_rotvec(current[3:]).as_matrix()

                # Deliberately send the complete position error. RTDE force mode's
                # selection vector, not CaP-side projection, owns the compliant DOF.
                position_error = target[:3] - current[:3]
                next_position = get_safe_target_position(
                    current[:3],
                    position_error,
                    self.config.dt,
                    self.config.position_speed_mps,
                    self.config.position_tolerance_m,
                )
                rotation_error = target_rotation @ current_rotation.T
                next_rotvec = get_safe_target_rotvec(
                    current_rotation,
                    rotation_error,
                    self.config.max_angle_step_rad,
                    self.config.orientation_deadband_rad,
                )

                try:
                    force_ok = self._control.forceMode(
                        task_frame,
                        selection_vector,
                        wrench,
                        self.config.force_type,
                        list(self.config.force_limits),
                    )
                    if force_ok is False:
                        raise CartesianExecutionError("RTDE forceMode returned failure")
                    servo_ok = self._control.servoL(
                        np.concatenate([next_position, next_rotvec]).tolist(),
                        0,
                        0,
                        self.config.dt,
                        self.config.lookahead_s,
                        self.config.servo_gain,
                    )
                    if servo_ok is False:
                        raise CartesianExecutionError("RTDE servoL returned failure")
                finally:
                    self._end_period(period, cycle_started)

            final_pose = self.current_pose()
            actual_wrench = self._read_actual_wrench()
            self._emit_pose_result(
                "move_force_completed",
                target,
                final_pose,
                self._monotonic() - started,
                force_axis=force_axis,
                actual_wrench=actual_wrench,
                cycles=cycles,
            )
        except Exception as exc:
            operation_error = exc
            self._emit("move_force_failed", error=str(exc))
        finally:
            try:
                self._restart_control()
            except Exception as cleanup_exc:
                self._emit("force_cleanup_failed", error=str(cleanup_exc))
                if operation_error is None:
                    operation_error = cleanup_exc

        if operation_error is not None:
            if isinstance(operation_error, CartesianExecutionError):
                raise operation_error
            raise CartesianExecutionError(f"RTDE force move failed: {operation_error}") from operation_error

    @staticmethod
    def _make_force_task_frame(force_axis: np.ndarray):
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(force_axis, arbitrary))) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        axis_one = np.cross(arbitrary, force_axis)
        axis_one /= np.linalg.norm(axis_one)
        axis_two = np.cross(force_axis, axis_one)
        task_rotation = np.column_stack([axis_one, axis_two, force_axis])
        return [0.0, 0.0, 0.0] + R.from_matrix(task_rotation).as_rotvec().tolist()

    def _restart_control(self):
        old_control = self._control
        self._control = None
        if old_control is not None:
            _safe_call(old_control, "servoStop")
            _safe_call(old_control, "stopScript")
            _safe_call(old_control, "disconnect")
        self._sleep(self.config.reconnect_sleep_s)
        try:
            self._control = self._control_factory(self.config.robot_ip)
        except Exception as exc:
            raise CartesianExecutionError(f"Unable to reconnect RTDE Control: {exc}") from exc
        self._sleep(self.config.reconnect_initialize_s)
        self._sleep(self.config.reconnect_settle_s)
        self._emit("force_control_reconnected")

    def _begin_period(self):
        method = getattr(self._control, "initPeriod", None)
        return method() if callable(method) else None

    def _end_period(self, period, cycle_started):
        method = getattr(self._control, "waitPeriod", None)
        if callable(method):
            method(period)

        # RTDE waitPeriod() follows the robot's native control frequency (about
        # 500 Hz on an e-Series robot), not this executor's configured 100 Hz.
        # Preserve RTDE synchronization, then pace the remainder of our cycle.
        remaining = self.config.dt - (self._monotonic() - cycle_started)
        if remaining > 0.0:
            self._sleep(remaining)

    def _read_actual_wrench(self):
        method = getattr(self._receive, "getActualTCPForce", None)
        if not callable(method):
            return None
        try:
            wrench = np.asarray(method(), dtype=float)
        except Exception:
            return None
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            return None
        return wrench

    def _validate_pose(self, pose: Sequence[float]) -> np.ndarray:
        value = np.asarray(pose, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("Cartesian target must contain six finite values")
        lower = np.asarray(self.config.workspace_min, dtype=float)
        upper = np.asarray(self.config.workspace_max, dtype=float)
        if np.any(value[:3] < lower) or np.any(value[:3] > upper):
            raise CartesianExecutionError(
                f"Target position {value[:3].tolist()} is outside configured workspace "
                f"{lower.tolist()} to {upper.tolist()}"
            )
        return value

    @staticmethod
    def _validate_force_axis(axis: Sequence[float]) -> np.ndarray:
        value = np.asarray(axis, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("Force axis must contain three finite values")
        length = float(np.linalg.norm(value))
        if length < 1e-9:
            raise ValueError("Force axis must be non-zero")
        return value / length

    def _validate_force_magnitude(self, magnitude) -> float:
        value = float(magnitude)
        if not np.isfinite(value) or value == 0.0:
            raise ValueError("fn must be finite and non-zero")
        if abs(value) > self.config.max_abs_force_n:
            raise CartesianExecutionError(
                f"Requested force {value} N exceeds configured limit "
                f"{self.config.max_abs_force_n} N"
            )
        return value

    def _ensure_connected(self):
        if not self.connected:
            self.connect()

    def _emit_pose_result(
        self,
        event_type: str,
        target_pose: np.ndarray,
        final_pose: np.ndarray,
        elapsed_s: float,
        force_axis: Optional[np.ndarray] = None,
        actual_wrench: Optional[np.ndarray] = None,
        cycles: Optional[int] = None,
    ):
        position_error = target_pose[:3] - final_pose[:3]
        rotation_error = R.from_rotvec(target_pose[3:]).as_matrix() @ R.from_rotvec(
            final_pose[3:]
        ).as_matrix().T
        payload = {
            "target_pose": target_pose.tolist(),
            "final_pose": final_pose.tolist(),
            "elapsed_s": float(elapsed_s),
            "position_residual": position_error.tolist(),
            "position_residual_norm_m": float(np.linalg.norm(position_error)),
            "orientation_residual_rad": float(R.from_matrix(rotation_error).magnitude()),
        }
        if force_axis is not None:
            axial = float(np.dot(position_error, force_axis))
            tangential = position_error - axial * force_axis
            payload.update(
                {
                    "force_axis": force_axis.tolist(),
                    "compliant_axis_residual_m": axial,
                    "controlled_tangential_residual_norm_m": float(np.linalg.norm(tangential)),
                    "actual_wrench": actual_wrench.tolist() if actual_wrench is not None else None,
                    "cycles": cycles,
                    "target_reached": bool(
                        np.linalg.norm(tangential) <= self.config.position_tolerance_m
                    ),
                }
            )
        self._emit(event_type, **payload)

    def _emit(self, event_type: str, **payload):
        event = {"event": event_type, **payload}
        self._event_sink(event)
