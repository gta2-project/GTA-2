"""A borrowed-client gripper adapter with a deterministic memory-only client."""

import math
from types import SimpleNamespace

from scene.gripper.base import BaseGripper


class Future:
    def __init__(self, value):
        self.value = value
        self.ready = False

    def done(self):
        return self.ready

    def result(self):
        if not self.ready:
            raise RuntimeError("Future not complete")
        return self.value


class Goal:
    def __init__(self, accepted=True, succeeded=True):
        self.accepted = accepted
        self.succeeded = succeeded
        self.cancelled = False

    def result_future(self):
        return Future(SimpleNamespace(succeeded=self.succeeded))

    def cancel_future(self):
        self.cancelled = True
        return Future(True)


class FakeClient:
    """The example's positions are synthetic; they are not Robotiq values."""

    def __init__(self, available=True, accepted=True, succeeded=True):
        self.available = available
        self.goal = Goal(accepted, succeeded)
        self.positions = []
        self.destroyed = False

    def wait_for_server(self, timeout_s):
        return self.available

    def send(self, position):
        self.positions.append(position)
        return Future(self.goal)

    def destroy(self):
        self.destroyed = True


def complete_future(future, timeout_s):
    """A fake executor: complete immediately, without threads or ROS."""
    future.ready = True


class Gripper(BaseGripper):
    def __init__(self, *, node=None, spin_until_complete=complete_future,
                 check_cancel=None, client=None):
        self.client = client if client is not None else FakeClient()
        self.spin = spin_until_complete  # Borrowed; never own ROS lifecycle.
        self.check_cancel = check_cancel or (lambda: None)
        self.pending = self.active = None

    @staticmethod
    def _timeout(value):
        value = 2.0 if value is None else float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timeout must be finite and positive")
        return value

    def _wait(self, future, seconds, *, check_cancel=True):
        if check_cancel:
            self.check_cancel()
        self.spin(future, seconds)
        if check_cancel:
            self.check_cancel()
        if not future.done():
            raise TimeoutError("Goal did not complete")
        return future.result()

    def wait_for_server(self, timeout_s=None):
        self.check_cancel()
        if not self.client.wait_for_server(self._timeout(timeout_s)):
            raise TimeoutError("Gripper unavailable")

    def _send(self, position, timeout_s):
        self.check_cancel()
        seconds = self._timeout(timeout_s)
        self.pending = self.client.send(position)
        goal = self._wait(self.pending, seconds)
        self.pending = None
        if not goal.accepted:
            raise RuntimeError("Goal rejected")
        self.active = goal
        result = self._wait(goal.result_future(), seconds)
        if not result.succeeded:
            raise RuntimeError("Goal failed")
        self.active = None
        return result

    def open(self, timeout_s=None):
        return self._send(1.0, timeout_s)

    def close(self, timeout_s=None):
        return self._send(0.0, timeout_s)

    def cancel(self, *, ack_timeout_s=0.5, timeout_s=0.5):
        errors = []
        try:
            if self.pending is not None:
                goal = self._wait(self.pending, ack_timeout_s, check_cancel=False)
                self.pending = None
                if goal.accepted:
                    self.active = goal
            if self.active is not None:
                confirmed = self._wait(
                    self.active.cancel_future(), timeout_s, check_cancel=False
                )
                if not confirmed:
                    raise RuntimeError("Cancellation not confirmed")
                self.active = None
        except Exception as exc:
            errors.append(str(exc))
        return errors

    def shutdown(self):
        try:
            if self.client.destroy() is False:
                return ["Client destruction failed"]
        except Exception as exc:
            return [str(exc)]
        return []
