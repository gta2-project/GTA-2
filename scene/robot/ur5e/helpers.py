"""UR5e joint reset. Settings are code-defined and specific to the workcell."""

import math
import time

from config import Config
from scene.robot.base import BaseRobotHelpers


# Same recorded pose and motion settings as scripts/ur5e/reset_joints.py (2026-07-20).
RESET_JOINTS = [-1.2959921995746058, -1.5255071830800553, -1.6169557571411133,
                -1.5701475674034846, 1.5704625844955444, 0.29003363847732544]
JOINT_SPEED = 0.2
JOINT_ACCELERATION = 0.2
RESET_TIMEOUT_S = 90.0
JOINT_TOLERANCE_RAD = 0.02


def cleanup_call(errors, label, fn, *args):
    try:
        if fn(*args) is False:
            errors.append(f'{label} returned False')
    except Exception as exc:
        errors.append(f'{label}: {exc}')


def joint_values(values):
    values = list(values)
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise RuntimeError('RTDE returned invalid joint positions')
    return values


class Helpers(BaseRobotHelpers):
    def reset_joints(self, *, check_cancel, on_started, errors, wait,
                     control_factory=None, receive_factory=None):
        if control_factory is None:
            from rtde_control import RTDEControlInterface
            control_factory = RTDEControlInterface
        if receive_factory is None:
            from rtde_receive import RTDEReceiveInterface
            receive_factory = RTDEReceiveInterface
        control = receive = None
        move_attempted = False
        try:
            check_cancel()
            control = control_factory(Config.ROBOT_IP)
            check_cancel()
            receive = receive_factory(Config.ROBOT_IP)
            current = joint_values(receive.getActualQ())
            print('Current robot joint positions:', current, flush=True)
            print('Moving to recorded reset joint positions:', RESET_JOINTS, flush=True)
            baseline = control.getAsyncOperationProgress()
            if baseline >= 0:
                raise RuntimeError('An RTDE asynchronous operation is already active')
            check_cancel()
            on_started()
            check_cancel()
            move_attempted = True
            # Asynchronous moveJ allows Cancel to reach stopJ while the robot moves.
            if not control.moveJ(RESET_JOINTS, JOINT_SPEED, JOINT_ACCELERATION, True):
                raise RuntimeError('Robot rejected the joint reset move')
            deadline = time.monotonic() + RESET_TIMEOUT_S
            seen_running = False
            while True:
                check_cancel()
                progress = control.getAsyncOperationProgress()
                seen_running = seen_running or progress >= 0
                if progress < 0 and (seen_running or progress != baseline):
                    final = joint_values(receive.getActualQ())
                    if max(abs(actual - target) for actual, target in zip(final, RESET_JOINTS)) > JOINT_TOLERANCE_RAD:
                        raise RuntimeError('Joint reset ended before reaching the recorded pose')
                    print('Final robot joint positions:', final, flush=True)
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError('Joint reset timed out')
                wait(0.02)
        finally:
            if control is not None:
                if move_attempted:
                    cleanup_call(errors, 'stopJ', control.stopJ, 0.5)
                cleanup_call(errors, 'stopScript', control.stopScript)
                cleanup_call(errors, 'RTDE Control disconnect', control.disconnect)
            if receive is not None:
                cleanup_call(errors, 'RTDE Receive disconnect', receive.disconnect)
