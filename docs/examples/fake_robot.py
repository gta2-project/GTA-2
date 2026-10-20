"""Memory-only robot state/helper example. No robot SDK or ROS imports."""

import time

import numpy as np

from scene.robot.base import BaseRobotHelpers, BaseRobotState, RobotStateError


class FakeArm:
    """A test double, not a driver. Calls only append to a Python list."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.pose = np.array([0.2, -0.3, 0.4, 0.0, 0.0, 0.0])
        self.sample_time = clock()
        self.connected = True
        self.calls = []
        self.polls = 0

    def read(self):
        return self.pose.copy(), self.sample_time

    def start_reset(self):
        self.calls.append("start_reset")

    def reset_done(self):
        self.polls += 1
        return self.polls >= 2

    def stop_joints(self):
        self.calls.append("stop_joints")

    def disconnect(self):
        self.calls.append("disconnect")
        self.connected = False


class RobotState(BaseRobotState):
    def __init__(self, arm=None, clock=time.monotonic, max_age_s=0.25):
        self.clock = clock
        self.arm = arm if arm is not None else FakeArm(clock)
        self.max_age_s = max_age_s

    def get_tcp_pose(self):
        if not self.arm.connected:
            raise RobotStateError("State connection is closed")
        pose, sample_time = self.arm.read()
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RobotStateError("Invalid pose")
        if self.clock() - sample_time > self.max_age_s:
            raise RobotStateError("State sample is stale")
        return pose.copy()

    def close(self):
        self.arm.disconnect()


class Helpers(BaseRobotHelpers):
    def __init__(self, factory=FakeArm, clock=time.monotonic):
        self.factory = factory
        self.clock = clock

    def reset_joints(self, *, check_cancel, on_started, errors, wait):
        check_cancel()  # Before even constructing a device client.
        arm = self.factory()
        try:
            check_cancel()
            on_started()
            arm.start_reset()  # Only a fake operation in this example.
            deadline = self.clock() + 5.0
            while True:
                check_cancel()
                if arm.reset_done():
                    break
                if self.clock() >= deadline:
                    raise TimeoutError("Reset timed out")
                wait(0.01)
            check_cancel()
        finally:
            # Attempt every cleanup step, even if an earlier one fails.
            for name, cleanup in (
                ("Stop joints", arm.stop_joints),
                ("Disconnect", arm.disconnect),
            ):
                try:
                    if cleanup() is False:
                        errors.append(f"{name} returned False")
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
