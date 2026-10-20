"""ROS 2 command transport for the direct-RTDE UR5e control server."""

import os
import signal
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import Config
from scene.gripper import Gripper
from gta.managed_run import get_runtime


def interruptible_sleep(seconds):
    runtime = get_runtime()
    if runtime:
        runtime.cancelled.wait(seconds)
        runtime.check_cancel()
    else:
        time.sleep(seconds)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32, Float32MultiArray, String

    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    Float32 = None
    Float32MultiArray = None
    String = None
    ROS2_AVAILABLE = False

try:
    from rtde_control import RTDEControlInterface
except ImportError:
    RTDEControlInterface = None


CTRL_HZ = 100.0
DT = 1.0 / CTRL_HZ
TOTAL_TIME = 1000.0
LOOKAHEAD = 0.10
GAIN = 400
MAX_ANGLE_STEP = 0.03
DEADBAND = np.deg2rad(0.05)
V_MPS = 0.8
POS_TOL = 0.002
TOGETHER = True
POST_RECONNECT_SETTLE_TICKS = 2
CONTROL_RECONNECT_SLEEP_S = 0.75
FORCE_TYPE = 2
FORCE_LIMITS = [2, 2, 2, 1, 1, 1]


def get_safe_target_rotvec(
    R_cur: np.ndarray,
    R_err: np.ndarray,
    max_angle_step: float = 0.04,
    deadband: float = np.deg2rad(0.1),
):
    """Step-limited orientation servo. Return the UR rotation-vector command."""
    R_cur = np.asarray(R_cur, dtype=np.float64).reshape(3, 3)
    R_err = np.asarray(R_err, dtype=np.float64).reshape(3, 3)
    rot_err = R.from_matrix(R_err)
    theta = rot_err.magnitude()

    if theta <= deadband:
        return R.from_matrix(R_err @ R_cur).as_rotvec().tolist()

    theta_step = min(theta, max_angle_step)
    rotvec = rot_err.as_rotvec()
    axis = rotvec / (theta + 1e-12)
    R_step = R.from_rotvec(axis * theta_step).as_matrix()
    return R.from_matrix(R_step @ R_cur).as_rotvec().tolist()


def get_safe_target_position(
    current_position,
    pos_error,
    dt,
    v_mps,
    pos_tol=0.002,
    together=True,
):
    cur = np.asarray(current_position, dtype=float).reshape(3)
    err = np.asarray(pos_error, dtype=float).reshape(3)
    dist = float(np.linalg.norm(err))
    if dist <= pos_tol:
        return (cur + err).tolist()

    max_step = float(v_mps) * float(dt)
    if together:
        step = err / dist * min(dist, max_step)
        return (cur + step).tolist()

    return (cur + np.clip(err, -max_step, +max_step)).tolist()


def _safe_call(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            return True, method(*args, **kwargs)
        except Exception as exc:
            return False, exc
    return False, None


def restart_rtde_control(rtde_c, sleep_s=0.75):
    """Reset RTDE Control only; the direct RTDE Receive interface is retained."""
    _safe_call(rtde_c, "servoStop")
    _safe_call(rtde_c, "stopScript")
    _safe_call(rtde_c, "disconnect")
    interruptible_sleep(float(sleep_s))
    if RTDEControlInterface is None:
        raise RuntimeError("rtde_control is unavailable")
    replacement = RTDEControlInterface(Config.ROBOT_IP)
    try:
        interruptible_sleep(0.35)
    except BaseException:
        _safe_call(replacement, 'servoStop')
        _safe_call(replacement, 'stopScript')
        _safe_call(replacement, 'disconnect')
        raise
    return replacement


def _finite_array(data, size):
    try:
        array = np.asarray(data, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        return None
    return array


class UR5eROS2Server(Node):
    """ROS 2 subscriptions and state for the existing 100 Hz RTDE loop."""

    def __init__(self):
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable; source ROS 2 and the robot workspace"
            )
        super().__init__("ur5e_robot_server_reconnect")

        self.latest_position_error = np.zeros(3)
        self.latest_rotation_error = np.eye(3)
        self.latest_force_command = None
        self.latest_gripper_command = None
        self.pending_gripper_command = None
        self.skill_active = False
        self.skill_start_time = None
        self.force_mode_was_active = False
        self.reconnect_requested = False
        self.zero_ft_requested = False
        self.previous_gripper_command = None
        self.reconnect_settle_ticks = 0
        self.rtde_c = None
        self.robot = None

        try:
            continuous_qos = self._qos(1)
            event_qos = self._qos(10)
            self._command_subscriptions = [
                self.create_subscription(
                    Float32MultiArray,
                    "/robot/position_error",
                    self.position_error_callback,
                    continuous_qos,
                ),
                self.create_subscription(
                    Float32MultiArray,
                    "/robot/rotation_error",
                    self.rotation_error_callback,
                    continuous_qos,
                ),
                self.create_subscription(
                    Float32MultiArray,
                    "/robot/force_command",
                    self.force_command_callback,
                    continuous_qos,
                ),
                self.create_subscription(
                    Float32,
                    "/robot/gripper_command",
                    self.gripper_command_callback,
                    event_qos,
                ),
                self.create_subscription(
                    String,
                    "/robot/skill_status",
                    self.skill_status_callback,
                    event_qos,
                ),
            ]
            runtime = get_runtime()
            self.gripper = Gripper(
                gta_backend='ros2',
                node=self,
                spin_until_complete=lambda future, seconds: rclpy.spin_until_future_complete(
                    self, future, timeout_sec=seconds),
                check_cancel=runtime.check_cancel if runtime else None,
            )
        except BaseException:
            # main() cannot own a constructor that did not return successfully.
            self.destroy_node()
            raise

    @staticmethod
    def _qos(depth):
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

    def wait_for_gripper_server(self, timeout_s=None):
        self.gripper.wait_for_server(timeout_s)

    def attach_hardware(self, rtde_c, robot):
        self.rtde_c = rtde_c
        self.robot = robot

    def position_error_callback(self, msg):
        value = _finite_array(msg.data, 3)
        if value is None:
            print(
                "[UR5e Server] Warning: position error must contain "
                "exactly 3 finite values"
            )
            return
        self.latest_position_error = value

    def rotation_error_callback(self, msg):
        value = _finite_array(msg.data, 9)
        if value is None:
            print(
                "[UR5e Server] Warning: rotation error must contain "
                "exactly 9 finite values"
            )
            return
        self.latest_rotation_error = value.reshape(3, 3)

    def force_command_callback(self, msg):
        if len(msg.data) == 0:
            self.latest_force_command = None
            return
        value = _finite_array(msg.data, 4)
        if value is None:
            print(
                "[UR5e Server] Warning: force command must be empty or contain "
                "exactly 4 finite values"
            )
            return
        self.latest_force_command = value

    def gripper_command_callback(self, msg):
        try:
            command = float(msg.data)
        except (TypeError, ValueError):
            print("[UR5e Server] Warning: gripper command must be finite")
            return
        if not np.isfinite(command):
            print("[UR5e Server] Warning: gripper command must be finite")
            return
        self.latest_gripper_command = command
        self.pending_gripper_command = command

    def skill_status_callback(self, msg):
        if msg.data == "skill_started":
            if self.skill_active:
                duration = (
                    time.time() - self.skill_start_time if self.skill_start_time else 0.0
                )
                print(
                    "[UR5e Server] Warning: received skill_started before skill_ended "
                    f"after {duration:.3f}s; closing previous skill implicitly"
                )
                if self.force_mode_was_active:
                    self.reconnect_requested = True

            self.skill_active = True
            self.skill_start_time = time.time()
            print(f"\n[UR5e Server] ✓ Skill started at t={self.skill_start_time:.3f}")
            self._clear_commands()
            self.zero_ft_requested = True
        elif msg.data == "skill_ended":
            if self.skill_active and self.skill_start_time:
                duration = time.time() - self.skill_start_time
                print(f"[UR5e Server] ✓ Skill ended after {duration:.3f} seconds\n")
            if self.force_mode_was_active:
                self.reconnect_requested = True
            self.skill_active = False
            self.skill_start_time = None
            self._clear_commands()
        else:
            print(f"[UR5e Server] Warning: unknown skill status {msg.data!r}")

    def _clear_commands(self):
        self.latest_position_error = np.zeros(3)
        self.latest_rotation_error = np.eye(3)
        self.latest_force_command = None
        self.latest_gripper_command = None

    def _stop_active_skill(self, reason):
        print(f"[UR5e Server] Gripper failure; stopping active skill: {reason}")
        self.skill_active = False
        self.skill_start_time = None
        self._clear_commands()
        self.pending_gripper_command = None
        if self.force_mode_was_active:
            self.reconnect_requested = True
        _safe_call(self.rtde_c, "servoStop")

    def send_gripper_command(self, command, timeout_s=None):
        operation = self.gripper.open if float(command) >= 0.5 else self.gripper.close
        return operation(timeout_s)

    def _run_pending_gripper_command(self):
        if self.pending_gripper_command is None:
            return True
        command = self.pending_gripper_command
        self.pending_gripper_command = None
        if command == self.previous_gripper_command:
            return True
        print(f"Got gripper_command: {command}")
        try:
            self.send_gripper_command(command)
        except Exception as exc:
            self._stop_active_skill(exc)
            return False
        self.previous_gripper_command = command
        return True

    def control_step(self):
        """Apply one set of callback state to RTDE, preserving ROS 1 motion behavior."""
        runtime = get_runtime()
        if runtime:
            runtime.check_cancel()
        if self.rtde_c is None or self.robot is None:
            raise RuntimeError("RTDE Control and Receive must be attached before control")

        if (
            self.skill_active
            and self.force_mode_was_active
            and self.latest_force_command is None
            and not self.reconnect_requested
        ):
            print(
                "[UR5e Server] Force mode ended without a reconnect; "
                "reconnecting before continuing"
            )
            self.reconnect_requested = True

        if self.reconnect_requested:
            print("[UR5e Server] Reconnect requested -> resetting RTDE control")
            self.rtde_c = restart_rtde_control(
                self.rtde_c,
                sleep_s=CONTROL_RECONNECT_SLEEP_S,
            )
            self.force_mode_was_active = False
            self.reconnect_requested = False
            self.reconnect_settle_ticks = POST_RECONNECT_SETTLE_TICKS

        if self.zero_ft_requested:
            # Preserve the ROS 1 server's current no-op zeroing behavior.
            self.zero_ft_requested = False
            print("[UR5e Server] Zeroed RTDE F/T sensor")

        if self.skill_active:
            try:
                current_pose = self.robot.get_tcp_pose()
            except Exception as exc:
                print(f"[UR5e Server] RTDE receive failed; stopping active skill: {exc}")
                self.skill_active = False
                self.skill_start_time = None
                self._clear_commands()
                _safe_call(self.rtde_c, "servoStop")
                return False

            current_position = current_pose[:3]
            current_rotation = R.from_rotvec(
                np.asarray(current_pose[3:6], dtype=float)
            ).as_matrix()
            position_error = self.latest_position_error.copy()
            rotation_error = self.latest_rotation_error.copy()
            force_command = (
                self.latest_force_command.copy()
                if self.latest_force_command is not None
                else None
            )

            if force_command is not None:
                force_axis = force_command[:3]
                magnitude = force_command[3]
                arbitrary = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(force_axis, arbitrary)) > 0.9:
                    arbitrary = np.array([0.0, 1.0, 0.0])
                axis_one = np.cross(arbitrary, force_axis)
                axis_one /= np.linalg.norm(axis_one)
                axis_two = np.cross(force_axis, axis_one)
                task_rotation = np.column_stack([axis_one, axis_two, force_axis])
                task_frame = [0.0, 0.0, 0.0] + R.from_matrix(
                    task_rotation
                ).as_rotvec().tolist()
                selection_vector = [0, 0, 1, 0, 0, 0]
                wrench = [0.0, 0.0, magnitude, 0.0, 0.0, 0.0]
                success = self.rtde_c.forceMode(
                    task_frame,
                    selection_vector,
                    wrench,
                    FORCE_TYPE,
                    FORCE_LIMITS,
                )
                if runtime and success is False:
                    raise RuntimeError('RTDE forceMode command failed')
                self.force_mode_was_active = True

            should_send_servo = self.reconnect_settle_ticks <= 0
            if self.reconnect_settle_ticks > 0:
                self.reconnect_settle_ticks -= 1

            if should_send_servo:
                next_position = get_safe_target_position(
                    current_position=current_position,
                    pos_error=position_error,
                    dt=DT,
                    v_mps=V_MPS,
                    pos_tol=POS_TOL,
                    together=TOGETHER,
                )
                next_rotation = get_safe_target_rotvec(
                    R_cur=current_rotation,
                    R_err=rotation_error,
                    max_angle_step=MAX_ANGLE_STEP,
                    deadband=DEADBAND,
                )
                success = self.rtde_c.servoL(
                    list(next_position) + list(next_rotation),
                    0,
                    0,
                    DT,
                    LOOKAHEAD,
                    GAIN,
                )
                if runtime and success is False:
                    raise RuntimeError('RTDE servoL command failed')

        return self._run_pending_gripper_command()

    def close(self):
        errors = []
        # Stop robot motion before waiting for asynchronous gripper cleanup.
        for method in ('forceModeStop', 'servoStop'):
            if self.rtde_c is not None and hasattr(self.rtde_c, method):
                ok, result = _safe_call(self.rtde_c, method)
                if not ok or result is False:
                    errors.append(f'{method}: {result}')
        try:
            errors.extend(self.gripper.cancel())
        except Exception as exc:
            errors.append(f'Gripper cancellation: {exc}')
        for obj, method in ((self.rtde_c, 'stopScript'), (self.rtde_c, 'disconnect'),
                            (self.robot, 'close')):
            if obj is not None and hasattr(obj, method):
                ok, result = _safe_call(obj, method)
                if not ok or result is False:
                    errors.append(f'{method}: {result}')
        try:
            errors.extend(self.gripper.shutdown())
        except Exception as exc:
            errors.append(f'Gripper cleanup: {exc}')
        return errors


def main():
    runtime = get_runtime()
    if not ROS2_AVAILABLE:
        raise RuntimeError(
            "ROS 2 dependencies are unavailable; source ROS 2 Humble and the robot workspace"
        )
    if RTDEControlInterface is None:
        raise RuntimeError("rtde_control is unavailable")

    from scene.robot import RobotState

    rclpy.init(args=None)
    node = None
    previous_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    running = True

    def stop(_sig, _frame):
        nonlocal running
        running = False
        if runtime:
            runtime.cancelled.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        node = UR5eROS2Server()
        print(
            "[UR5e Server] Subscribed to /robot/position_error, /robot/rotation_error, "
            "/robot/force_command, /robot/gripper_command, and /robot/skill_status"
        )
        node.wait_for_gripper_server()

        rtde_c = RTDEControlInterface(Config.ROBOT_IP)
        node.rtde_c = rtde_c
        robot = RobotState()
        node.robot = robot
        robot.get_tcp_pose()
        _safe_call(rtde_c, "zeroFtSensor")

        print("[UR5e Server] Waiting for skill execution commands...")
        if runtime:
            runtime.check_cancel()
            runtime.emit('server_ready')
        start_time = time.time()
        while running and rclpy.ok() and (runtime or time.time() - start_time < TOTAL_TIME):
            if runtime:
                runtime.check_cancel()
            period_start = node.rtde_c.initPeriod()
            rclpy.spin_once(node, timeout_sec=0.0)
            success = node.control_step()
            if runtime and not success:
                raise RuntimeError('Robot control step failed; terminating the managed run')
            node.rtde_c.waitPeriod(period_start)
    except BaseException as exc:
        if runtime and not runtime.cancelled.is_set():
            runtime.emit('server_failed', error=str(exc))
        raise
    finally:
        # Cleanup cannot be interrupted by a repeated Cancel click/signal.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        errors = []
        if node is not None:
            errors = node.close()
            try:
                node.destroy_node()
            except Exception as exc:
                errors.append(f'ROS node cleanup: {exc}')
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:
            errors.append(f'ROS shutdown: {exc}')
        if runtime:
            runtime.emit('server_cleaned', errors=errors)
        try:
            signal.signal(signal.SIGINT, previous_handler)
            signal.signal(signal.SIGTERM, previous_term_handler)
        except Exception:
            pass


if __name__ == "__main__":
    main()
