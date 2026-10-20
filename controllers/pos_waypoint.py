"""PosWaypoint Controller - Move through sequential offsets along an axis."""

import numpy as np
from .base_controller import BaseController


class PosWaypoint(BaseController):
    """
    Move g1 through sequential offsets [o1, o2, o3, ...] along a2 from g2.
    """
    
    def __init__(self, g1, g2, a2, offsets):
        """
        Initialize PosWaypoint controller.
        
        Args:
            g1: Tool keypoint position function (lambda function)
            g2: Reference keypoint position (numpy array)
            a2: Axis along which to move (numpy array)
            offsets: List of offset values (list of floats)
        """
        self.type = "positional"

        self.g1 = g1
        self.g2 = np.asarray(g2)
        self.a2 = np.asarray(a2) / np.linalg.norm(np.asarray(a2))  # Normalize
        self.offsets = offsets if isinstance(offsets, list) else [offsets]

        self.targets = [self.g2 + self.a2 * offset for offset in self.offsets]
        
        print(f"[PosWaypoint] Initialized: moving through {len(self.offsets)} waypoints along axis {self.a2}")
        print(f"[PosWaypoint] Offsets: {self.offsets}")
    
    def compute_error(self, elapsed_time=0.0):
        """
        Compute error for waypoint movement. Waypoint index is determined by elapsed time uniformly.
        For example, if there are 3 waypoints, then:
        - elapsed_time in [0, 0.33) -> target = g2 + a2 * o1
        - elapsed_time in [0.33, 0.66) -> target = g2 + a2 * o2
        - elapsed_time in [0.66, 1.0] -> target = g2 + a2 * o3
        Args:
            elapsed_time: Time elapsed since skill started (float), normalized to [0, 1] for waypoint progression
        
        Returns:
            dict: {
                'error': position error to current waypoint, 
                'type': 'positional', 
                'axis': self.a2
            }
        """
        # Determine current waypoint index based on elapsed time
        num_waypoints = len(self.offsets)
        idx = int(elapsed_time * num_waypoints)
        idx = np.clip(idx, 0, num_waypoints - 1)
        
        error = self.targets[idx] - self.g1()
        # Project the error on a2 axis to ensure movement is along the desired direction
        error = np.dot(error, self.a2) * self.a2
        
        return {'error': error, 'type': self.type, 'axis': self.a2}
    
    def __repr__(self):
        return f"PosWaypoint(g1={self.g1}, g2={self.g2}, a2={self.a2}, offsets={self.offsets})"
