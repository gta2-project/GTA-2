"""Robot-independent contracts. Importing them never connects to hardware."""

from abc import ABC, abstractmethod


class RobotStateError(RuntimeError):
    """Raised when the robot state stream cannot provide a fresh TCP pose."""


class BaseRobotState(ABC):
    """Read robot state without commanding motion or applying tool geometry.

    Readers must use the robot-base coordinates used by GTA's world-frame
    keypoints. The pose origin/orientation must match the frame in which the
    deployment defines its local tool keypoints and axes. The UR5e reader
    preserves the configured TCP pose returned by RTDE; it does not substitute
    a flange pose, set the controller TCP, or apply GTA's additional tool offset.

    A backend owns its receive connection and its connection-sharing policy.
    Importing a backend must not connect to hardware.
    """

    @abstractmethod
    def get_tcp_pose(self):
        """Return six finite floats as a NumPy array: [x, y, z, rx, ry, rz].

        Position is in metres; orientation is a rotation vector in radians,
        not Euler angles. Raise RobotStateError if usable state is unavailable.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Release this reader's resources, not motion-control or shared ROS resources."""
        raise NotImplementedError


class BaseRobotHelpers(ABC):
    """Supporting robot operations, independent of GUI events and grippers."""

    @abstractmethod
    def reset_joints(self, *, check_cancel, on_started, errors, wait):
        """Move to this deployment's reviewed reset pose.

        Check cancellation before connecting and throughout motion. Call
        on_started only when ready to move. Always stop/release owned hardware,
        appending cleanup failures to errors even when the operation raises.
        wait(seconds) is an interruptible pause supplied by the caller.
        """
        raise NotImplementedError
