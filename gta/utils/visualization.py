"""Open3D previews for robot scripts."""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .geometry import unit


def visualize_point_cloud(
    pcd_np,
    colors=None,
    show_coordinate_frame=True,
    frame_size=0.1,
    camera_pose=None,
    min_z=-0.05,
    objects=None,
    keypoint_radius=0.01,
    axis_length=0.12,
    axis_radius=0.004,
):
    """
    Visualize a point cloud using Open3D.

    Args:
        pcd_np: Point cloud as (N, 3) or (H, W, 3) numpy array.
        colors: Optional per-point RGB colors as (N, 3) or (H, W, 3) in [0, 1].
                If not provided, a default gray color is used.
        show_coordinate_frame: If True, draw a coordinate frame at the origin.
        frame_size: Size of the coordinate frame when enabled.
        camera_pose: Optional 4x4 camera pose matrix in world frame
                 (T_world_cam). If provided, the Open3D view is initialized
                 from this pose.
        min_z: Minimum z value to visualize. Points with z < min_z are removed.
        objects: Optional list of objects containing keypoints and axes.
             Compatible with ObjectHandler and similar structures.
        keypoint_radius: Radius of rendered keypoint spheres.
        axis_length: Length of rendered object axes.
           axis_radius: Radius of object axis arrows.
    """
    pcd_np = np.asarray(pcd_np, dtype=float)

    if pcd_np.ndim == 3 and pcd_np.shape[-1] == 3:
        points = pcd_np.reshape(-1, 3)
    elif pcd_np.ndim == 2 and pcd_np.shape[1] == 3:
        points = pcd_np
    else:
        raise ValueError("pcd_np must have shape (N, 3) or (H, W, 3)")

    if min_z is not None:
        z_mask = (points[:, 2] >= float(min_z)) & (points[:, 1] < 0) & (points[:, 0] < 0.5)
        points = points[z_mask]
    else:
        z_mask = None

    if points.size == 0:
        raise ValueError("No points remain after z filtering")

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        colors = np.asarray(colors, dtype=float)
        if colors.ndim == 3 and colors.shape[-1] == 3:
            colors = colors.reshape(-1, 3)
        elif colors.ndim == 2 and colors.shape[1] == 3:
            pass
        else:
            raise ValueError("colors must have shape (N, 3) or (H, W, 3)")

        if colors.shape[0] != points.shape[0]:
            if z_mask is not None and colors.shape[0] == z_mask.shape[0]:
                colors = colors[z_mask]
            else:
                raise ValueError("colors must contain exactly one RGB triplet per point")

        if np.any(colors < 0.0) or np.any(colors > 1.0):
            colors /= 255.0  # assume colors are in [0, 255], convert to [0, 1]
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd_o3d.paint_uniform_color([0.7, 0.7, 0.7])

    geometries = [pcd_o3d]
    if show_coordinate_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(frame_size),
            origin=[0.0, 0.0, 0.0],
        )
        geometries.append(frame)

    if objects:
        def _rotation_from_z_to_axis(axis_vector):
            z_axis = np.array([0.0, 0.0, 1.0])
            target = unit(axis_vector)
            dot = np.clip(np.dot(z_axis, target), -1.0, 1.0)

            if dot > 1.0 - 1e-8:
                return np.eye(3)
            if dot < -1.0 + 1e-8:
                return R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()

            rot_axis = np.cross(z_axis, target)
            rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-12)
            rot_angle = np.arccos(dot)
            return R.from_rotvec(rot_axis * rot_angle).as_matrix()

        axis_colors = {
            "x": np.array([1.0, 0.2, 0.2]),
            "y": np.array([0.2, 1.0, 0.2]),
            "z": np.array([0.2, 0.4, 1.0]),
        }

        for obj_idx, obj in enumerate(objects):
            keypoints = getattr(obj, "keypoints", {}) or {}
            axes = getattr(obj, "axes", {}) or {}

            keypoint_names = list(keypoints.keys())
            keypoint_positions = []
            for kp_name in keypoint_names:
                if hasattr(obj, "get_k") and callable(obj.get_k):
                    kp = obj.get_k(kp_name)
                else:
                    kp = keypoints[kp_name]

                if kp is None:
                    continue

                kp = np.asarray(kp, dtype=float).reshape(-1)
                if kp.size < 3:
                    continue

                keypoint_positions.append(kp[:3])

            if len(keypoint_positions) == 0:
                axis_origin = np.zeros(3)
            else:
                keypoint_positions = np.asarray(keypoint_positions)
                axis_origin = keypoint_positions.mean(axis=0)

                for kp in keypoint_positions:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(keypoint_radius))
                    sphere.translate(kp)
                    sphere.paint_uniform_color([1.0, 0.65, 0.0])
                    geometries.append(sphere)

            for axis_name in axes.keys():
                if hasattr(obj, "get_a") and callable(obj.get_a):
                    axis_vec = obj.get_a(axis_name)
                else:
                    axis_vec = axes[axis_name]

                if axis_vec is None:
                    continue

                axis_vec = np.asarray(axis_vec, dtype=float).reshape(-1)
                if axis_vec.size < 3:
                    continue

                axis_vec = axis_vec[:3]
                axis_norm = np.linalg.norm(axis_vec)
                if axis_norm < 1e-8:
                    continue

                axis_key = axis_name.lower().strip()
                axis_color = axis_colors.get(axis_key, np.array([0.9, 0.9, 0.2]))

                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=float(axis_radius),
                    cone_radius=float(axis_radius) * 2.2,
                    cylinder_height=float(axis_length) * 0.75,
                    cone_height=float(axis_length) * 0.25,
                )
                arrow.paint_uniform_color(axis_color)
                arrow.rotate(_rotation_from_z_to_axis(axis_vec), center=np.zeros(3))
                arrow.translate(axis_origin)
                geometries.append(arrow)

    if camera_pose is None:
        camera_pose = np.array([[ 0.70857228, -0.49764763,  0.50027208, -0.48682316],
                                [-0.70322773, -0.43945916,  0.5588796,  -0.97660094],
                                [-0.05827596, -0.74781179, -0.66134819,  0.5947388],
                                [ 0.,          0.,          0.,          1.        ]])

    camera_pose = np.asarray(camera_pose, dtype=float)
    if camera_pose.shape != (4, 4):
        raise ValueError("camera_pose must have shape (4, 4)")

    from gta.managed_run import get_runtime
    runtime = get_runtime()
    vis = o3d.visualization.Visualizer()
    if vis.create_window(window_name="Point Cloud") is False and runtime:
        vis.destroy_window()
        raise RuntimeError('Could not open the required 3D preview; execution will not continue')
    for geom in geometries:
        vis.add_geometry(geom)

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    # Open3D expects world-to-camera extrinsic; invert pose (world transform of camera).
    camera_params.extrinsic = np.linalg.inv(camera_pose)
    view_control.convert_from_pinhole_camera_parameters(
        camera_params,
        allow_arbitrary=True,
    )

    try:
        if runtime is None:
            vis.run()
        else:
            # Keep Python responsive to Cancel while the native preview is open.
            while vis.poll_events():
                runtime.check_cancel()
                vis.update_renderer()
                runtime.cancelled.wait(0.02)
            runtime.check_cancel()
    finally:
        vis.destroy_window()


def visualize_point_cloud_normals(pcd_np, normal, origin=None, normal_length=0.5):
    """
    Visualize a point cloud together with a single normal vector.

    Args:
        pcd_np: (N, 3) numpy array
        normal: (3,) normal vector
        origin: (3,) starting point of the normal line.
                If None, uses the point cloud center.
        normal_length: displayed length of the normal
    """
    pcd_np = np.asarray(pcd_np, dtype=float)
    normal = unit(normal)

    if pcd_np.ndim != 2 or pcd_np.shape[1] != 3:
        raise ValueError("pcd_np must have shape (N, 3)")

    if origin is None:
        origin = pcd_np.mean(axis=0)
    else:
        origin = np.asarray(origin, dtype=float).reshape(3,)

    # point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
    pcd_o3d.paint_uniform_color([0.7, 0.7, 0.7])

    # line for normal
    p0 = origin
    p1 = origin + normal_length * normal

    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p0, p1])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])

    # small sphere at origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=normal_length * 0.08)
    sphere.translate(origin)
    sphere.paint_uniform_color([0.0, 1.0, 0.0])

    o3d.visualization.draw_geometries([pcd_o3d, line, sphere])
