"""ForceConstant Controller - Apply or maintain a constant force."""

import numpy as np
from .base_controller import BaseController


class ForceConstant(BaseController):
    """
    Apply or maintain a force along a1 with magnitude o1.
    
    For a surface normal axis, a negative o1 value will push into the surface.
    """
    
    def __init__(self, a1, o1):
        """
        Initialize ForceConstant controller.
        
        Args:
            a1: Axis along which to apply force (numpy array)
            o1: Force magnitude in Newtons (float)
        """
        self.type = "force"

        self.a1 = np.asarray(a1) / np.linalg.norm(np.asarray(a1))  # Normalize
        self.o1 = o1
        
        direction = "into surface" if self.o1 < 0 else "away from surface"
        print(f"[ForceConstant] Initialized: applying {abs(self.o1)}N {direction} along {self.a1}")
    
    def compute_error(self, elapsed_time=0.0):
        """
        Basic constant force component. Returns the axis and the magnitude of the force error to apply.
        Returns:
            dict: {'error': force error vector, 'type': 'force'}
        """

        error = np.zeros(3)
        magnitude = self.o1
        axis = self.a1
        return {'error': error, 'type': self.type, 'magnitude': magnitude, 'axis': axis}

    def __repr__(self):
        return f"ForceConstant(a1={self.a1}, o1={self.o1})"
