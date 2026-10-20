"""Robotiq ROS 1 action client in a caller-owned rospy environment."""

import time

from scene.gripper.base import BaseGripper

POSITIONS = {'open': 0.085, 'close': 0.0, 'half_open': 0.06}
VELOCITY = 0.05
FORCE = 165
ACTION_NAME = '/ur5/move_gripper'


class Gripper(BaseGripper):
    def __init__(self, *, check_cancel=None):
        import actionlib
        import rospy
        from actionlib_msgs.msg import GoalStatus
        from robotiq_mm_ros.msg import GripperCommandAction, GripperCommandGoal
        self._rospy = rospy
        self._goal_type = GripperCommandGoal
        self._succeeded = GoalStatus.SUCCEEDED
        self._check_cancel = check_cancel or (lambda: None)
        self._outstanding = False
        self.client = actionlib.SimpleActionClient(ACTION_NAME, GripperCommandAction)

    def _wait(self, method, timeout_s):
        self._check_cancel()
        if timeout_s is None:
            method()  # Preserve the ROS 1 server's unbounded wait policy.
            return
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_cancel()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'ROS 1 gripper action timed out after {timeout_s:.1f}s')
            if method(self._rospy.Duration(min(0.1, remaining))):
                self._check_cancel()
                return

    def wait_for_server(self, timeout_s=None):
        self._wait(self.client.wait_for_server, timeout_s)

    def _send(self, operation, timeout_s, require_success):
        goal = self._goal_type(position=POSITIONS[operation], velocity=VELOCITY, force=FORCE)
        self._check_cancel()
        self._outstanding = True
        self.client.send_goal(goal)
        self._wait(self.client.wait_for_result, timeout_s)
        if require_success:
            state = self.client.get_state()
            if state != self._succeeded:
                raise RuntimeError(f'ROS 1 gripper action failed with status {state}')
        self._outstanding = False
        return self.client.get_result() if require_success else None

    def open(self, timeout_s=None, *, require_success=True):
        return self._send('open', timeout_s, require_success)

    def close(self, timeout_s=None, *, require_success=True):
        return self._send('close', timeout_s, require_success)

    def half_open(self, timeout_s=None, *, require_success=True):
        return self._send('half_open', timeout_s, require_success)

    def cancel(self, *, ack_timeout_s=0.5, timeout_s=0.5):
        if not self._outstanding:
            return []
        try:
            self.client.cancel_goal()
            if not self.client.wait_for_result(self._rospy.Duration(timeout_s)):
                return ['Gripper stop could not be verified']
            self._outstanding = False
        except Exception as exc:
            return [f'Gripper cancellation: {exc}']
        return []

    def shutdown(self):
        # SimpleActionClient has no public destroy; rospy belongs to the caller.
        return []
