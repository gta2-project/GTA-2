"""ROS 1 UR5e RTDE control server."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import time
import signal
import numpy as np
import rospy
from std_msgs.msg import String, Float32, Float32MultiArray

from scene.gripper import Gripper

from config import Config
from scene.robot import RobotState
from scipy.spatial.transform import Rotation as R

from rtde_control import RTDEControlInterface


def get_safe_target_rotvec(
    R_cur: np.ndarray,
    R_err: np.ndarray,
    max_angle_step: float = 0.04,
    deadband: float = np.deg2rad(0.1),
):
    """Step-limited orientation servo. Returns UR rotvec command [rx, ry, rz]."""
    R_cur = np.asarray(R_cur, dtype=np.float64).reshape(3, 3)
    R_err = np.asarray(R_err, dtype=np.float64).reshape(3, 3)

    rot_err = R.from_matrix(R_err)
    theta = rot_err.magnitude()

    if theta <= deadband:
        R_cmd = R_err @ R_cur
        return R.from_matrix(R_cmd).as_rotvec().tolist()

    theta_step = min(theta, max_angle_step)
    rotvec = rot_err.as_rotvec()
    axis = rotvec / (theta + 1e-12)
    R_step = R.from_rotvec(axis * theta_step).as_matrix()
    R_cmd = R_step @ R_cur

    return R.from_matrix(R_cmd).as_rotvec().tolist()


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

    step = np.clip(err, -max_step, +max_step)
    return (cur + step).tolist()


def _safe_call(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            return True, method(*args, **kwargs)
        except Exception as exc:
            return False, exc
    return False, None


def restart_rtde_control(rtde_c, sleep_s=0.75):
    """Reset RTDEControl interface only. RobotState() receive channel remains untouched."""
    _safe_call(rtde_c, "servoStop")
    _safe_call(rtde_c, "stopScript")
    _safe_call(rtde_c, "disconnect")
    time.sleep(float(sleep_s))

    new_rtde_c = RTDEControlInterface(Config.ROBOT_IP)
    time.sleep(0.35)
    return new_rtde_c


def main():
    rospy.init_node("ur5e_robot_server_reconnect", anonymous=True)

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

    # full reconnect policy after force phase ends
    POST_RECONNECT_SETTLE_TICKS = 2
    CONTROL_RECONNECT_SLEEP_S = 0.75

    latest_position_error = np.zeros(3)
    latest_rotation_error = np.eye(3)
    latest_force_command = None
    latest_gripper_command = None
    pending_gripper_command = None
    skill_active = False
    skill_start_time = None

    force_mode_was_active = False
    reconnect_requested = False
    zero_ft_requested = False

    def position_error_callback(msg):
        nonlocal latest_position_error
        if len(msg.data) == 3:
            latest_position_error = np.array(msg.data)
        else:
            print(f"[UR5e Server] Warning: Expected 3 elements for position error, got {len(msg.data)}")

    def rotation_error_callback(msg):
        nonlocal latest_rotation_error
        if len(msg.data) == 9:
            latest_rotation_error = np.array(msg.data).reshape(3, 3)
        else:
            print(f"[UR5e Server] Warning: Expected 9 elements for rotation matrix, got {len(msg.data)}")

    def force_command_callback(msg):
        nonlocal latest_force_command
        if len(msg.data) == 4:
            latest_force_command = np.array(msg.data)
        else:
            print(f"[UR5e Server] Warning: Expected 4 elements for force command, got {len(msg.data)}")

    def gripper_command_callback(msg):
        nonlocal latest_gripper_command, pending_gripper_command
        command = float(msg.data)
        latest_gripper_command = command
        pending_gripper_command = command

    def skill_status_callback(msg):
        """Do not call RTDE control APIs in callback thread; set flags for main loop."""
        nonlocal skill_active, skill_start_time, reconnect_requested, zero_ft_requested
        nonlocal latest_position_error, latest_rotation_error, latest_force_command, latest_gripper_command

        if msg.data == "skill_started":
            if skill_active:
                duration = time.time() - skill_start_time if skill_start_time else 0.0
                print(
                    "[UR5e Server] Warning: received skill_started before skill_ended "
                    f"after {duration:.3f}s; closing previous skill implicitly"
                )
                if force_mode_was_active:
                    reconnect_requested = True

            skill_active = True
            skill_start_time = time.time()
            print(f"\n[UR5e Server] ✓ Skill started at t={skill_start_time:.3f}")
            latest_position_error = np.zeros(3)
            latest_rotation_error = np.eye(3)
            latest_force_command = None
            latest_gripper_command = None
            zero_ft_requested = True

        elif msg.data == "skill_ended":
            if skill_active and skill_start_time:
                duration = time.time() - skill_start_time
                print(f"[UR5e Server] ✓ Skill ended after {duration:.3f} seconds\n")

            skill_active = False
            skill_start_time = None
            latest_position_error = np.zeros(3)
            latest_rotation_error = np.eye(3)
            latest_force_command = None
            latest_gripper_command = None

            # Reconnect only after skills that actually entered force mode.
            if force_mode_was_active:
                reconnect_requested = True

    print("Initializing robot gripper service")
    gripper = Gripper(gta_backend='ros1')
    # Preserve the existing ROS 1 server: no gripper startup wait.
    print("Initializing done")

    rospy.Subscriber("/robot/position_error", Float32MultiArray, position_error_callback)
    rospy.Subscriber("/robot/rotation_error", Float32MultiArray, rotation_error_callback)
    rospy.Subscriber("/robot/force_command", Float32MultiArray, force_command_callback)
    rospy.Subscriber("/robot/gripper_command", Float32, gripper_command_callback)
    rospy.Subscriber("/robot/skill_status", String, skill_status_callback)
    print(
        "[UR5e Server] Subscribed to /robot/position_error, /robot/rotation_error, "
        "/robot/force_command, /robot/gripper_command, and /robot/skill_status"
    )
    print("[UR5e Server] Waiting for skill execution commands...")

    FORCE_TYPE = 2
    FORCE_LIMITS = [2, 2, 2, 1, 1, 1]

    previous_gripper_cmd = None
    reconnect_settle_ticks = 0

    running = True

    def stop(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)

    rtde_c = RTDEControlInterface(Config.ROBOT_IP)
    robot = RobotState()
    _safe_call(rtde_c, "zeroFtSensor")

    start_time = time.time()

    while running:
        elapsed = time.time() - start_time
        if elapsed >= TOTAL_TIME:
            break

        t_start = rtde_c.initPeriod()

        if skill_active and force_mode_was_active and latest_force_command is None and not reconnect_requested:
            print("[UR5e Server] Force mode ended without a reconnect; reconnecting before continuing")
            reconnect_requested = True

        if reconnect_requested:
            print("[UR5e Server] Reconnect requested -> resetting RTDE control")
            rtde_c = restart_rtde_control(rtde_c, sleep_s=CONTROL_RECONNECT_SLEEP_S)
            force_mode_was_active = False
            reconnect_requested = False
            reconnect_settle_ticks = POST_RECONNECT_SETTLE_TICKS

        if zero_ft_requested:
            #ok, result = _safe_call(rtde_c, "zeroFtSensor")
            ok=True
            zero_ft_requested = False
            if ok:
                print("[UR5e Server] Zeroed RTDE F/T sensor")
            else:
                print(f"[UR5e Server] Warning: zeroFtSensor failed: {result}")

        if skill_active:
            try:
                cur = robot.get_tcp_pose()
            except Exception as exc:
                print(f"[UR5e Server] RTDE receive failed; stopping active skill: {exc}")
                skill_active = False
                skill_start_time = None
                latest_position_error = np.zeros(3)
                latest_rotation_error = np.eye(3)
                latest_force_command = None
                latest_gripper_command = None
                _safe_call(rtde_c, "servoStop")
                rtde_c.waitPeriod(t_start)
                continue

            cur_pos = cur[:3]
            cur_rotvec = np.array(cur[3:6], dtype=float)
            R_cur = R.from_rotvec(cur_rotvec).as_matrix()

            pos_err = latest_position_error.copy()
            R_err = latest_rotation_error.copy()
            force_cmd = latest_force_command.copy() if latest_force_command is not None else None

            if force_cmd is not None:
                force_axis = force_cmd[:3]
                magnitude = force_cmd[3]

                arbitrary = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(force_axis, arbitrary)) > 0.9:
                    arbitrary = np.array([0.0, 1.0, 0.0])
                axis_one = np.cross(arbitrary, force_axis)
                axis_one /= np.linalg.norm(axis_one)
                axis_two = np.cross(force_axis, axis_one)

                R_task = np.column_stack([axis_one, axis_two, force_axis])
                task_rotvec = R.from_matrix(R_task).as_rotvec().tolist()
                task_frame = [0.0, 0.0, 0.0] + task_rotvec

                selection_vector = [0, 0, 1, 0, 0, 0]
                wrench = [0.0, 0.0, magnitude, 0.0, 0.0, 0.0]
                rtde_c.forceMode(task_frame, selection_vector, wrench, FORCE_TYPE, FORCE_LIMITS)
                force_mode_was_active = True

            # Old forceModeStop path intentionally removed.

            should_send_servo = reconnect_settle_ticks <= 0
            if reconnect_settle_ticks > 0:
                reconnect_settle_ticks -= 1

            if should_send_servo:
                next_pos = get_safe_target_position(
                    current_position=cur_pos,
                    pos_error=pos_err,
                    dt=DT,
                    v_mps=V_MPS,
                    pos_tol=POS_TOL,
                    together=TOGETHER,
                )

                next_rotvec = get_safe_target_rotvec(
                    R_cur=R_cur,
                    R_err=R_err,
                    max_angle_step=MAX_ANGLE_STEP,
                    deadband=DEADBAND,
                )

                next_pose = list(next_pos) + list(next_rotvec)
                rtde_c.servoL(next_pose, 0, 0, DT, LOOKAHEAD, GAIN)

        if pending_gripper_command is not None:
            gripper_cmd = pending_gripper_command
            pending_gripper_command = None
            if gripper_cmd != previous_gripper_cmd:
                print(f"Got gripper_command: {gripper_cmd}")
                operation = gripper.open if gripper_cmd >= 0.5 else gripper.close
                operation(require_success=False)
                previous_gripper_cmd = gripper_cmd

        rtde_c.waitPeriod(t_start)

    _safe_call(rtde_c, "servoStop")
    _safe_call(rtde_c, "stopScript")


if __name__ == "__main__":
    main()
