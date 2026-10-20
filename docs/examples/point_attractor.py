"""Educational position controller. Computes errors; never commands hardware."""

import numpy as np

from controllers.base_controller import BaseController


class PointAttractor(BaseController):
    """Track a sampled 3D target using a live point callback, in metres."""

    def __init__(self, current_point, target_point):
        super().__init__()
        if not callable(current_point):
            raise TypeError("current_point must be callable")
        self.type = "positional"
        self.current_point = current_point
        self.target = self._point(target_point).copy()

    @staticmethod
    def _point(value):
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Expected three finite coordinates")
        return point

    def compute_error(self, elapsed_time=0.0):
        current = self._point(self.current_point())
        return {
            "type": self.type,
            "error": self.target - current,
            # This controller claims all three translation directions.
            "axes": np.eye(3),
        }
