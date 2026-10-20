"""A memory-only camera example; no files, ROS, or SDK connections."""

import numpy as np

from scene.camera.base import BaseCamera


class ArrayCamera(BaseCamera):
    def __init__(self, rgb, xyz=None):
        self._rgb = np.asarray(rgb).copy()
        if (self._rgb.ndim != 3 or self._rgb.shape[2] != 3
                or self._rgb.dtype != np.uint8):
            raise ValueError("RGB must be an H x W x 3 uint8 array")
        self._xyz = None if xyz is None else np.asarray(xyz, dtype=float).copy()
        if self._xyz is not None:
            if self._xyz.shape != self._rgb.shape or not np.isfinite(self._xyz).all():
                raise ValueError("XYZ must be finite H x W x 3, aligned with RGB")
        self._cached = None
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Camera is closed")

    def get_rgb_image(self, use_new_frame=True):
        self._check_open()
        if use_new_frame or self._cached is None:
            self._cached = self._rgb.copy()
        return self._cached.copy()

    def get_point_cloud(self, save_points=False):
        self._check_open()
        if save_points:
            raise NotImplementedError("This memory camera does not save point clouds")
        if self._xyz is None:
            raise RuntimeError("This recording has no point cloud")
        # Already in base coordinates and metres. Preserve pixel order.
        return self._xyz.reshape(-1, 3).copy()

    def close(self):
        self._cached = None
        self._closed = True
