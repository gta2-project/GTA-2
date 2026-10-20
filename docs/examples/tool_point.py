"""Pure geometry example: a reported TCP pose plus an additional local point."""

import numpy as np
from scipy.spatial.transform import Rotation


def point_in_base(pose, local_offset):
    pose = np.asarray(pose, dtype=float)
    offset = np.asarray(local_offset, dtype=float)
    if pose.shape != (6,) or offset.shape != (3,):
        raise ValueError("Expected a six-value pose and three-value local offset")
    if not np.isfinite(pose).all() or not np.isfinite(offset).all():
        raise ValueError("Geometry must be finite")
    return pose[:3] + Rotation.from_rotvec(pose[3:]).apply(offset)


def example():
    # Hypothetical TCP: translation in m; +90 degrees about base Y in radians.
    pose = [0.2, -0.3, 0.4, 0, np.pi / 2, 0]
    # A point 10 cm along the TCP-local Z direction.
    result = point_in_base(pose, [0, 0, 0.10])
    np.testing.assert_allclose(result, [0.3, -0.3, 0.4])
    return result
