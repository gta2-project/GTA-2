"""Error Projection Module - Project controller errors to robot command space."""

import numpy as np
from scipy.spatial.transform import Rotation as R

eps = 1e-9

def get_nullspace_matrix(axes, eps=eps):
    """
    axes: list of 3D direction vectors (each shape (3,))
    returns: 3x3 projector onto the orthogonal complement of span(axes)
    """
    if len(axes) == 0:
        return np.eye(3)

    A = np.stack(axes, axis=1)   # (3, k)

    keep = np.linalg.norm(A, axis=0) > eps
    A = A[:, keep]
    if A.shape[1] == 0:
        return np.eye(3)

    A = A / np.linalg.norm(A, axis=0, keepdims=True)

    P = np.eye(3) - A @ np.linalg.pinv(A.T @ A) @ A.T
    return P


def project_errors(error_packets):
    """
        Project a list of controller error packets into robot command space.

        Contract:
            - input: error_packets: iterable of dicts with keys:
                    - 'type': one of 'positional', 'rotational', 'force', 'gripper'
                    - 'error': numpy array-like (length depends on type)
                    - 'axis': optional for one-axis positional controllers (3,)
                    - 'axes': optional for multi-axis positional controllers (k, 3)
            - output:
                    - position_error: (3,) numpy array
                    - rotation_error: (3,3) numpy rotation matrix (identity = no rotation)
                    - force_command: (4,) numpy array [axis_x, axis_y, axis_z, magnitude], or None if no force controller
                    - gripper_command: float in [0, 1] (0=closed, 1=open), or None if no gripper command

        Notes:
            - Rotational packet errors are expected as a 3-vector rotvec (axis-angle).
                They will be accumulated and converted to a 3x3 rotation matrix at the end.
            - GripperControl packets always arrive (never None); a 'state' of None means
                the controller has not reached its trigger_time yet and is skipped.
            - Multiple gripper packets are supported: the one with the highest
                'trigger_time' (most recently scheduled) wins.
    """
    # Initialize command vectors
    position_error = np.zeros(3)
    rotation_error_vector = np.zeros(3)  # accumulated in the SO(3) tangent space
    force_command = None  # None means no force controller this cycle
    gripper_command = None  # None means no gripper command this cycle

    position_axes = []
    primary_rotation_axis = None
    rotation_seen = False
    last_gripper_command_time = -1.0  # tracks which gripper packet has the highest trigger_time

    for packet in error_packets:
        error_type = packet.get('type', 'none')
        error = np.asarray(packet.get('error', np.zeros(3)), dtype=float)

        if error_type == 'positional':
            declared_axes = packet.get('axes', None)
            if declared_axes is None:
                declared_axes = [packet.get('axis', np.zeros(3))]

            declared_axes = np.asarray(declared_axes, dtype=float)
            if declared_axes.shape == (3,):
                declared_axes = declared_axes.reshape(1, 3)
            if declared_axes.ndim != 2 or declared_axes.shape[1] != 3:
                raise ValueError(
                    "Positional controller axes must have shape (3,) or (k, 3)"
                )

            P = get_nullspace_matrix(position_axes)

            projected_error = P @ error
            position_error += projected_error

            # Register every independent DOF claimed by this controller. Recompute
            # the projector after each axis so duplicate or linearly dependent axes
            # do not consume additional null-space dimensions.
            for axis in declared_axes:
                axis_projector = get_nullspace_matrix(position_axes)
                projected_axis = axis_projector @ axis
                n = np.linalg.norm(projected_axis)
                if n > eps:
                    position_axes.append(projected_axis / n)

        elif error_type == 'rotational':
            axis = packet.get('axis', None)  # constrained rotation axis (None for free rotation)

            if not rotation_seen:
                # First rotational controller: take error as-is and set the primary axis.
                rotation_seen = True
                if axis is not None:
                    primary_rotation_axis = np.asarray(axis, dtype=float)
                    n = np.linalg.norm(primary_rotation_axis)
                    if n > eps:
                        primary_rotation_axis /= n
            elif primary_rotation_axis is not None:
                # Subsequent controllers: project error onto the primary rotation axis
                # so it only adds the component that is consistent with the primary DOF.
                error = np.dot(error, primary_rotation_axis) * primary_rotation_axis

            # Combine projected instantaneous angular corrections before converting
            # to a finite rotation. Composing a finite matrix per controller would
            # apply lower-priority rotations after higher-priority corrections and
            # can therefore violate the intended rotational hierarchy.
            rotation_error_vector += error

        elif error_type == 'force':
            axis = np.asarray(packet.get('axis', np.zeros(3)), dtype=float)
            magnitude = packet.get('magnitude', 0.0)

            # NOTE: Only one force controller is supported at a time.
            # Multiple force controllers would require wrench composition, which is not implemented yet.
            # For now, the first force controller in the list is used and subsequent ones are skipped.
            if force_command is not None:
                continue

            # Project using the same nullspace as positional (force must not fight position)
            P = get_nullspace_matrix(position_axes)
            projected_axis = P @ axis
            n = np.linalg.norm(projected_axis)
            if n > eps:
                projected_axis_normalized = projected_axis / n
                force_command = np.zeros(4)  # [axis_x, axis_y, axis_z, magnitude]
                force_command[:3] = projected_axis_normalized
                force_command[3] = magnitude
                # Append to position_axes so subsequent controllers respect this direction
                position_axes.append(projected_axis_normalized)

        elif error_type == 'gripper':
            # Gripper is an independent DOF - no projection needed.
            state = packet.get('state')
            if state is None:
                continue  # not yet triggered
            command_time = packet.get('command_time', 0.0)
            if gripper_command is None or command_time > last_gripper_command_time:
                gripper_command = float(state)
                last_gripper_command_time = command_time
                obj = packet.get('object', None)
                if obj is not None:
                    obj.set_frame('robot' if gripper_command == 0 else 'world')

    # A rotation vector longer than pi aliases to a different principal SO(3)
    # representation when converted to a matrix. Preserve the combined direction
    # while keeping the target just below that ambiguity boundary. The robot server
    # still applies its smaller per-tick angular step limit.
    rotation_angle = np.linalg.norm(rotation_error_vector)
    max_rotation_error = np.pi - 1e-6
    if rotation_angle > max_rotation_error:
        rotation_error_vector *= max_rotation_error / rotation_angle

    rotation_error_matrix = R.from_rotvec(rotation_error_vector).as_matrix()

    return position_error, rotation_error_matrix, force_command, gripper_command
