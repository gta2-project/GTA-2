"""Robotiq ROS 2 action client; no ROS lifecycle or robot-server ownership."""

import math
import time

from config import Config
from scene.gripper.base import BaseGripper

# Deployed normalized controller positions, NOT metres. GTA's topic uses
# the opposite convention: 1=open, 0=closed.
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = 0.85
GRIPPER_MAX_EFFORT = 165.0


class Gripper(BaseGripper):
    def __init__(self, *, node, spin_until_complete, check_cancel=None,
                 poll_interval_s=0.05, cancel_on_timeout=True):
        from action_msgs.msg import GoalStatus
        from control_msgs.action import GripperCommand
        from rclpy.action import ActionClient

        self._goal_type = GripperCommand.Goal
        self._succeeded = GoalStatus.STATUS_SUCCEEDED
        self._spin = spin_until_complete
        self._check_cancel = check_cancel
        self._poll_interval = poll_interval_s
        self._cancel_on_timeout = cancel_on_timeout
        self._pending = self._active = None
        self._destroyed = False
        self._timeout(None)  # Validate before creating a client.
        self.client = ActionClient(node, GripperCommand, Config.ROS2_GRIPPER_ACTION_NAME)

    @staticmethod
    def _timeout(value):
        timeout = float(Config.ROS2_GRIPPER_WAIT_TIMEOUT_S if value is None else value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('ROS2_GRIPPER_WAIT_TIMEOUT_S must be finite and greater than zero')
        return timeout

    def _check(self):
        if self._check_cancel:
            self._check_cancel()

    def wait_for_server(self, timeout_s=None):
        timeout = self._timeout(timeout_s)
        self._check()
        if self._check_cancel is None:
            available = self.client.wait_for_server(timeout_sec=timeout)
        else:
            deadline = time.monotonic() + timeout
            available = False
            while time.monotonic() < deadline:
                self._check()
                if self.client.wait_for_server(timeout_sec=min(0.1, max(0, deadline - time.monotonic()))):
                    available = True
                    break
            self._check()
        if not available:
            raise RuntimeError(f'ROS 2 gripper action {Config.ROS2_GRIPPER_ACTION_NAME!r} '
                               f'was unavailable after {timeout:.1f}s')

    def _wait(self, future, timeout):
        if self._check_cancel is None:
            self._spin(future, timeout)
        else:
            deadline = time.monotonic() + timeout
            while not future.done() and time.monotonic() < deadline:
                self._check()
                self._spin(future, min(self._poll_interval, max(0, deadline - time.monotonic())))
            self._check()
        if not future.done():
            raise TimeoutError(f'gripper action timed out after {timeout:.1f}s')
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(str(exception)) from exception
        return future.result()

    def _send(self, position, timeout_s):
        timeout = self._timeout(timeout_s)
        self._check()
        goal = self._goal_type()
        goal.command.position = position
        goal.command.max_effort = GRIPPER_MAX_EFFORT
        self._pending = self.client.send_goal_async(goal)
        handle = self._wait(self._pending, timeout)
        self._pending = None
        if handle is None or not handle.accepted:
            raise RuntimeError('gripper goal was rejected')
        self._active = handle
        try:
            wrapped = self._wait(handle.get_result_async(), timeout)
        except TimeoutError:
            # Preserve the server's immediate best-effort timeout cancellation.
            # Its final cleanup still verifies/reports cancellation separately.
            if self._cancel_on_timeout:
                self._spin(handle.cancel_goal_async(), 1.0)
            raise
        if wrapped.status != self._succeeded:
            raise RuntimeError(f'gripper action failed with status {wrapped.status}')
        self._active = None
        result = wrapped.result
        if hasattr(result, 'reached_goal') and not result.reached_goal:
            print('[Gripper] Warning: gripper action succeeded with reached_goal=False '
                  f'(stalled={bool(getattr(result, "stalled", False))}); continuing', flush=True)
        return result

    def open(self, timeout_s=None):
        return self._send(GRIPPER_OPEN_POSITION, timeout_s)

    def close(self, timeout_s=None):
        return self._send(GRIPPER_CLOSED_POSITION, timeout_s)

    def half_open(self, timeout_s=None):
        return self._send((GRIPPER_OPEN_POSITION + GRIPPER_CLOSED_POSITION) / 2, timeout_s)

    def cancel(self, *, ack_timeout_s=0.5, timeout_s=0.5):
        errors = []
        goal = self._active
        if self._pending is not None:
            try:
                self._spin(self._pending, ack_timeout_s)
                if not self._pending.done():
                    errors.append('Gripper goal acknowledgement is pending; cancellation could not be verified')
                elif self._pending.exception() is not None:
                    errors.append(f'Gripper goal acknowledgement: {self._pending.exception()}')
                else:
                    accepted = self._pending.result()
                    if accepted is not None and accepted.accepted:
                        goal = self._active = accepted
                    self._pending = None
            except Exception as exc:
                errors.append(f'Gripper goal acknowledgement: {exc}')
        if goal is not None:
            try:
                future = goal.cancel_goal_async()
                self._spin(future, timeout_s)
                if not future.done():
                    errors.append('Gripper goal cancellation was not acknowledged')
                elif future.exception() is not None:
                    errors.append(f'Gripper goal cancellation failed: {future.exception()}')
                elif getattr(future.result(), 'return_code', 0) != 0:
                    errors.append(f'Gripper cancellation rejected: {future.result().return_code}')
                else:
                    self._active = None
            except Exception as exc:
                errors.append(f'Gripper cancellation: {exc}')
        return errors

    def shutdown(self):
        if self._destroyed:
            return []
        self._destroyed = True
        try:
            if self.client.destroy() is False:
                return ['Action client cleanup returned False']
        except Exception as exc:
            return [f'Action client cleanup: {exc}']
        return []
