"""AxisAlign Controller - Rotate tool axis to align with target axis."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from .base_controller import BaseController


class AxisAlign(BaseController):
    """
    Rotate tool axis a1 about a3 to align with a2 by an offset of o1 radians.
    
    Setting a3 to a_crossprod_axes(a1,a2) will rotate a1 directly to a2.
    """
    
    def __init__(self, a1, a2, a3, o1):
        """
        Initialize AxisAlign controller.
        
        Args:
            a1: Tool axis to rotate function (lambda function)
            a2: Target axis (numpy array)
            a3: Rotation axis (numpy array)
            o1: Offset rotation angle in degrees (float)
        """
        self.type = "rotational"

        self.a1 = a1
        self.a2 = np.asarray(a2) / np.linalg.norm(np.asarray(a2))  # Normalize
        self.a3 = np.asarray(a3) / np.linalg.norm(np.asarray(a3))  # Normalize
        self.o1 = np.radians(o1)

        # Calculate the target axis after applying the offset rotation to a2
        if abs(self.o1) < 1e-12:
            # No offset - target is exactly a2 (avoid numerical issues from zero rotation)
            self.target = self.a2.copy()
        else:
            rot = R.from_rotvec(self.a3 * self.o1)
            self.target = rot.apply(self.a2)
            print(self.a1, self.a2, self.a3, self.o1, self.target)
        
        #print(f"[AxisAlign] Initialized: rotating axis {self.a1} to align with {self.a2} about {self.a3} with offset {np.degrees(self.o1):.1f}°")
    
    def compute_error(self, elapsed_time=0.0, free_rotation=False):
        """
        Compute rotational error for axis alignment.

        By default (free_rotation=False) the error is constrained to rotations about
        `a3` only (fast analytic method). If `free_rotation=True` the error is the
        unconstrained shortest-path rotation (cross-product) suitable for nullspace
        controllers.

        Args:
            elapsed_time: Time elapsed since skill started (float)
            free_rotation: If True, compute an unconstrained cross-product error

        Returns:
            dict: {'error': rotation error as axis-angle vector, 'type': 'rotational', 'axis': axis_or_none}
        """
        eps = 1e-9

        # Normalize current axis
        current_a1 = np.asarray(self.a1(), dtype=float)
        norm = np.linalg.norm(current_a1)
        if norm < eps:
            return {'error': np.zeros(3), 'type': self.type, 'axis': (None if free_rotation else current_a1)}
        current_a1 /= norm

        if free_rotation:
            cross = np.cross(current_a1, self.target)
            sin_theta = np.linalg.norm(cross)
            cos_theta = np.dot(current_a1, self.target)
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
        tgt_perp2 = 1.0 - (np.dot(self.target, self.a3) ** 2)

        # If either vector is practically parallel to a3, rotation about a3 is undefined
        if cur_perp2 < eps or tgt_perp2 < eps:
            return {'error': np.zeros(3), 'type': self.type, 'axis': current_a1}

        # Extract sine (y) and cosine (x) components relative to the a3 plane
        # using dot/cross product identities (avoids expensive projections)
        y = np.dot(self.a3, np.cross(current_a1, self.target))
        x = np.dot(current_a1, self.target) - (np.dot(current_a1, self.a3) * np.dot(self.target, self.a3))

        # Compute the signed angle in (-pi, pi]
        angle = np.arctan2(y, x)

        # Error vector strictly along a3
        error = self.a3 * angle
        return {'error': error, 'type': self.type, 'axis': current_a1}
        
    def __repr__(self):
        return f"AxisAlign(a1={self.a1}, a2={self.a2}, a3={self.a3}, o1={self.o1})"
