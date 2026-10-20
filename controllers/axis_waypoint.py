"""AxisWaypoint Controller - Rotate through sequential offset angles."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from .base_controller import BaseController


class AxisWaypoint(BaseController):
    """
    Rotate tool axis a1 about a3 to align with a2 through sequential offsets [o1, o2, o3, ...].
    
    Setting a3 to a_crossprod_axes(a1,a2) will rotate a1 directly to a2.
    """
    
    def __init__(self, a1, a2, a3, offsets):
        """
        Initialize AxisWaypoint controller.
        
        Args:
            a1: Tool axis to rotate function (lambda function)
            a2: Target axis (numpy array)
            a3: Rotation axis (numpy array)
            offsets: List of offset rotation angles in degrees (list of floats)
        """
        self.type = "rotational"

        self.a1 = a1
        self.a2 = np.asarray(a2) / np.linalg.norm(np.asarray(a2))  # Normalize
        self.a3 = np.asarray(a3) / np.linalg.norm(np.asarray(a3))  # Normalize
        self.offsets = offsets if isinstance(offsets, list) else [offsets]
        
        # Pre-compute target orientations for each waypoint offset
        self.targets = []
        for offset_deg in self.offsets:
            offset_rad = np.radians(offset_deg)
            if abs(offset_rad) < 1e-12:
                # No offset - target is exactly a2
                target = self.a2.copy()
            else:
                rot = R.from_rotvec(self.a3 * offset_rad)
                target = rot.apply(self.a2)
            self.targets.append(target)
        
        print(f"[AxisWaypoint] Initialized: rotating through {len(self.offsets)} waypoints")
        print(f"[AxisWaypoint] Offsets (degrees): {self.offsets}")
    
    def compute_error(self, elapsed_time=0.0, free_rotation=False):
        """
        Compute error for axis waypoint movement. Waypoint index is determined by elapsed time uniformly.
        
        By default (free_rotation=False) the error is constrained to rotations about
        `a3` only (fast analytic method). If `free_rotation=True` the error is the
        unconstrained shortest-path rotation (cross-product) suitable for nullspace
        controllers.
        
        Args:
            elapsed_time: Time elapsed since skill started (float), normalized to [0, 1] for waypoint progression
            free_rotation: If True, compute an unconstrained cross-product error
        
        Returns:
            dict: {'error': rotation error as axis-angle vector, 'type': 'rotational', 'axis': axis_or_none}
        """
        eps = 1e-9
        
        # Determine current waypoint index based on elapsed time
        num_waypoints = len(self.offsets)
        idx = int(elapsed_time * num_waypoints)
        idx = np.clip(idx, 0, num_waypoints - 1)
        
        target = self.targets[idx]
        
        # Normalize current axis
        current_a1 = np.asarray(self.a1(), dtype=float)
        norm = np.linalg.norm(current_a1)
        if norm < eps:
            return {'error': np.zeros(3), 'type': self.type, 'axis': (None if free_rotation else current_a1)}
        current_a1 /= norm

        if free_rotation:
            cross = np.cross(current_a1, target)
            sin_theta = np.linalg.norm(cross)
            cos_theta = np.dot(current_a1, target)
            angle = np.arctan2(sin_theta, cos_theta)

            if sin_theta < eps:
                if cos_theta > 0:
                    error = np.zeros(3)
                else:
                    fallback = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(current_a1, fallback)) > 0.9:
                        fallback = np.array([0.0, 1.0, 0.0])
                    axis = np.cross(current_a1, fallback)
                    axis /= np.linalg.norm(axis)
                    error = axis * np.pi
            else:
                axis = cross / sin_theta          # unit rotation axis
                error = axis * angle              # true axis-angle vector  (magnitude = θ)

            return {'error': error, 'type': self.type, 'axis': None}

        # Constrained rotation about a3 only (original optimized algorithm)
        # Fast singularity check using Pythagorean identity (sin^2 = 1 - cos^2)
        # Calculates the squared magnitude of the perpendicular components without square roots
        cur_perp2 = 1.0 - (np.dot(current_a1, self.a3) ** 2)
        tgt_perp2 = 1.0 - (np.dot(target, self.a3) ** 2)

        # If either vector is practically parallel to a3, rotation about a3 is undefined
        if cur_perp2 < eps or tgt_perp2 < eps:
            return {'error': np.zeros(3), 'type': self.type, 'axis': current_a1}

        # Extract sine (y) and cosine (x) components relative to the a3 plane
        # using dot/cross product identities (avoids expensive projections)
        y = np.dot(self.a3, np.cross(current_a1, target))
        x = np.dot(current_a1, target) - (np.dot(current_a1, self.a3) * np.dot(target, self.a3))

        # Compute the signed angle in (-pi, pi]
        angle = np.arctan2(y, x)

        # Error vector strictly along a3
        error = self.a3 * angle
        return {'error': error, 'type': self.type, 'axis': current_a1}
    
    
    def __repr__(self):
        return f"AxisWaypoint(a1={self.a1}, a2={self.a2}, a3={self.a3}, offsets={self.offsets})"
