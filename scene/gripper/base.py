"""Gripper contract. ROS lifecycle belongs to the caller, not the device."""

from abc import ABC, abstractmethod


class BaseGripper(ABC):
    """Operations may raise; cancellation/cleanup return lists of error strings.

    Implementations own only their action/SDK client and its outstanding goals.
    They must not initialize ROS, subscribe to GTA commands, or connect the arm.
    Shutdown releases client resources; close physically closes the jaws.
    """

    @abstractmethod
    def wait_for_server(self, timeout_s=None):
        raise NotImplementedError

    @abstractmethod
    def open(self, timeout_s=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def close(self, timeout_s=None, **kwargs):
        raise NotImplementedError

    def half_open(self, timeout_s=None, **kwargs):
        raise NotImplementedError('This gripper does not support half_open')

    @abstractmethod
    def cancel(self, *, ack_timeout_s=0.5, timeout_s=0.5):
        """Cancel this client's outstanding goal, ignoring operation cancellation."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        """Release the client only, after cancel(); return cleanup errors."""
        raise NotImplementedError
