"""PosAlign Controller - Move tool keypoint to target along an axis."""

import numpy as np
from .base_controller import BaseController


class PosAlign(BaseController):
    """
    Move tool keypoint g1 to g2 along a1, applying offset o1 along the a1 axis.
    
    a_along_keypoints(g1,g2) is a common axis choice for moving the points directly to each other.
    """
        
    def __init__(self, g1, g2, a1, o1):
        """
        Initialize PosAlign controller.
        
        Args:
            g1: Tool keypoint position function (lambda function)
            g2: Target keypoint position (numpy array)
            a1: Axis along which to align (numpy array)
            o1: Offset value along the axis (float)
        """
        self.type = "positional"

        self.g1 = g1
        self.g2 = np.asarray(g2)
        self.a1 = np.asarray(a1) / np.linalg.norm(np.asarray(a1))  # Normalize
        self.o1 = o1

        self.target = self.g2 + self.a1 * self.o1
    
        print(f"[PosAlign] Initialized: moving from {self.g1()} to {self.g2} along axis {self.a1} with offset {self.o1}")

    def compute_error(self, elapsed_time=0.0):
        """Execute the position alignment controller."""
        error = self.target - self.g1()
        # Project the error on a1 axis to ensure movement is along the desired direction
        error = np.dot(error, self.a1) * self.a1

        return {'error': error, 'type': self.type, 'axis': self.a1}

    def __repr__(self):
        return f"PosAlign(g1={self.g1()}, g2={self.g2}, a1={self.a1}, o1={self.o1}, target={self.target})"
