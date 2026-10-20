"""Minimal camera contract, independent of ROS and camera hardware."""

from abc import ABC, abstractmethod


class BaseCamera(ABC):
    """Synchronous camera operations used by agents and robot utilities.

    Implementations retain their existing image formats, point-cloud units and
    ordering. This interface deliberately does not convert or normalize data.
    """

    @abstractmethod
    def get_rgb_image(self, use_new_frame: bool = True):
        """Return an RGB NumPy array, or a copy of the cached frame if requested."""
        raise NotImplementedError

    @abstractmethod
    def get_point_cloud(self, save_points: bool = False):
        """Return XYZ data, or None for an empty cloud.

        RGB-only backends raise RuntimeError. Supported backends retain their
        existing point-cloud representation and optional file-saving behavior.
        """
        raise NotImplementedError

    def close(self):
        """Release backend-owned resources, never a shared ROS runtime.

        Backends without owned resources inherit this no-op.
        """
