"""Offline regression checks for the script-helper package; no camera or GUI IO."""

import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

from config import Config
import gta as components
from gta import managed_run, utils
from gta.utils import geometry, sensors, visualization


def test_public_helpers_are_direct_exports_from_their_modules():
    groups = {
        sensors: ("get_point_cloud_data", "get_rgb_image"),
        geometry: (
            "get_transformation_matrix_from_pose", "get_pose_from_transformation_matrix",
            "get_3d_position", "a_surface_normal", "a_crossprod_axes", "a_along_keypoints",
            "unit", "choose_perp_axis",
        ),
        visualization: ("visualize_point_cloud", "visualize_point_cloud_normals"),
    }
    assert set(utils.__all__) == {name for names in groups.values() for name in names}
    for module, names in groups.items():
        for name in names:
            assert getattr(utils, name) is getattr(module, name)
            assert getattr(utils, name).__module__ == module.__name__
    assert components.get_3d_position is utils.get_3d_position
    assert visualization.unit is geometry.unit
    root = Path(__file__).resolve().parents[1] / "gta"
    assert Path(utils.__file__).resolve() == root / "utils" / "__init__.py"
    assert not (root / "utils.py").exists()


def test_public_import_does_not_initialize_camera_or_robot():
    probe = """
import builtins
import sys
original_import = builtins.__import__
def no_robot_import(name, *args, **kwargs):
    if name.split('.')[0] in {'rospy', 'rclpy', 'rtde_control', 'rtde_receive'}:
        raise AssertionError(f'Unexpected hardware dependency: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_robot_import
from gta.utils import get_rgb_image, get_3d_position, visualize_point_cloud
from scene.camera import Camera
assert Camera._instance is None
for name in ('scene.camera.ros1', 'scene.camera.ros2', 'scene.robot.ur5e.state',
             'gta.skill_execution'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("helper,method", [
    ("get_rgb_image", "get_rgb_image"),
    ("get_point_cloud_data", "get_point_cloud"),
])
def test_sensor_helpers_forward_settings_results_and_failures(helper, method):
    camera = Mock()
    with patch.object(sensors, "Camera", return_value=camera) as factory:
        assert getattr(utils, helper)() is getattr(camera, method).return_value
        factory.assert_called_once_with(
            camera_node=Config.CAMERA_NODE,
            camera_model=Config.CAMERA_MODEL,
            rosmaster_ip=Config.ROSMASTER_IP,
            rosmaster_port=Config.ROSMASTER_PORT,
            camera_backend=Config.CAMERA_BACKEND,
            camera_image_path=Config.CAMERA_IMAGE_PATH,
        )
        getattr(camera, method).assert_called_once_with()
        getattr(camera, method).side_effect = RuntimeError("sensor unavailable")
        with pytest.raises(RuntimeError, match="sensor unavailable"):
            getattr(utils, helper)()


@pytest.mark.parametrize("pose", [
    [0, 0, 0, 0, 0, 0],
    [0.1, -0.2, 0.3, 0.2, 0.4, -0.3],
    [-1, 2, 0.5, 0, 0, np.pi / 2],
])
def test_pose_transform_round_trip(pose):
    matrix = utils.get_transformation_matrix_from_pose(pose)
    np.testing.assert_allclose(matrix[:3, 3], pose[:3])
    np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(utils.get_pose_from_transformation_matrix(matrix), pose, atol=1e-14)


def test_pixel_projection_preserves_workspace_paths_padding_and_cloud_order(tmp_path, monkeypatch):
    for name, size in (("first", (8, 4)), ("second", (12, 8))):
        workspace = tmp_path / name
        workspace.mkdir()
        monkeypatch.chdir(workspace)
        Image.new("RGB", size).save("environment.jpg")
        Image.new("RGB", (4, 4)).save("resized_img.png")
        width, height = size
        cloud = np.arange(height * width * 3).reshape(-1, 3)
        x = width // 4
        y = width // 2 - (width - height) // 2
        position = utils.get_3d_position(cloud, [1, 2])
        np.testing.assert_array_equal(position, cloud.reshape(height, width, 3)[y, x])
        assert np.shares_memory(position, cloud)


@pytest.mark.parametrize("first,second,expected", [
    ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
    ([1, 0, 0], [1, 0, 0], [0, 0, 1]),
    ([1, 0, 0], [-1, 0, 0], [0, 0, 1]),
    ([0, 0, 0], [1, 0, 0], [0, 0, 0]),
])
def test_cross_product_axis_preserves_parallel_and_zero_cases(first, second, expected):
    np.testing.assert_allclose(utils.a_crossprod_axes(first, second), expected, atol=1e-10)


def test_direction_and_normalization_helpers():
    vector = np.array([3., 4., 0.])
    np.testing.assert_array_equal(utils.unit(vector), vector / (5.0 + 1e-12))
    np.testing.assert_array_equal(utils.unit(np.zeros(3)), np.zeros(3))
    np.testing.assert_allclose(utils.a_along_keypoints([1, 2, 3], [1, 5, 7]), [0, 0.6, 0.8])
    with pytest.raises(ValueError, match="same location"):
        utils.a_along_keypoints([1, 2, 3], [1, 2, 3])
    for axis in (np.array([1., 0, 0]), np.array([0., 0, 1])):
        perpendicular = utils.choose_perp_axis(axis)
        assert np.dot(perpendicular, axis) == pytest.approx(0)
        assert np.linalg.norm(perpendicular) == pytest.approx(1)


@pytest.mark.parametrize("camera_z", [-1, 0, 1])
def test_surface_normal_keeps_nearest_point_search_and_camera_orientation(monkeypatch, camera_z):
    cloud = np.array([[0., 0, 0], [1., 0, 0]])
    fake_cloud = SimpleNamespace(normals=np.array([[1., 0, 0], [0., 0, -1]]),
                                 estimate_normals=Mock())
    transform = np.eye(4)
    transform[:3, 3] = [1, 0, camera_z]
    monkeypatch.setattr(Config, "T_BASE_CAM", transform)
    with patch.object(geometry.o3d.geometry, "PointCloud", return_value=fake_cloud), \
         patch.object(geometry.o3d.geometry, "KDTreeSearchParamKNN") as search:
        normal = utils.a_surface_normal(np.array([1.01, 0, 0]), cloud)
    search.assert_called_once_with(knn=100)
    fake_cloud.estimate_normals.assert_called_once_with(search_param=search.return_value)
    np.testing.assert_array_equal(np.asarray(fake_cloud.points), cloud)
    np.testing.assert_allclose(normal, [0, 0, camera_z], atol=1e-10)


@pytest.fixture
def preview(monkeypatch):
    visualizer = Mock()
    visualizer.create_window.return_value = True
    visualizer.poll_events.side_effect = [True, False]
    parameters = SimpleNamespace(extrinsic=None)
    visualizer.get_view_control.return_value.convert_to_pinhole_camera_parameters.return_value = parameters
    factory = Mock(return_value=visualizer)
    monkeypatch.setattr(visualization.o3d.visualization, "Visualizer", factory)
    monkeypatch.setattr(managed_run, "get_runtime", lambda: None)
    return SimpleNamespace(visualizer=visualizer, parameters=parameters, factory=factory)


def test_preview_preserves_filters_colors_and_camera_transform(preview):
    cloud = np.array([[[0.1, -0.2, 0.3], [0.6, -0.2, 0.3],
                       [0.2, 0.1, 0.3], [0.2, -0.2, -0.06]]])
    colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]])
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.2, 0.6]
    utils.visualize_point_cloud(cloud, colors=colors, camera_pose=pose, show_coordinate_frame=False)
    shown = preview.visualizer.add_geometry.call_args.args[0]
    np.testing.assert_allclose(np.asarray(shown.points), [[0.1, -0.2, 0.3]])
    np.testing.assert_allclose(np.asarray(shown.colors), [[1, 0, 0]])
    np.testing.assert_allclose(preview.parameters.extrinsic, np.linalg.inv(pose))
    preview.visualizer.create_window.assert_called_once_with(window_name="Point Cloud")
    preview.visualizer.run.assert_called_once_with()
    preview.visualizer.poll_events.assert_not_called()
    preview.visualizer.destroy_window.assert_called_once_with()


def test_preview_defaults_are_unchanged():
    expected = {
        "colors": None, "show_coordinate_frame": True, "frame_size": 0.1,
        "camera_pose": None, "min_z": -0.05, "objects": None,
        "keypoint_radius": 0.01, "axis_length": 0.12, "axis_radius": 0.004,
    }
    signature = inspect.signature(utils.visualize_point_cloud)
    assert {name: param.default for name, param in signature.parameters.items()
            if name != "pcd_np"} == expected


def test_managed_preview_keeps_polling_and_cancellation_checks(preview, monkeypatch):
    runtime = Mock()
    monkeypatch.setattr(managed_run, "get_runtime", lambda: runtime)
    utils.visualize_point_cloud([[0, -0.1, 0.2]], show_coordinate_frame=False)
    assert preview.visualizer.poll_events.call_count == 2
    assert runtime.check_cancel.call_count == 2
    runtime.cancelled.wait.assert_called_once_with(0.02)
    preview.visualizer.update_renderer.assert_called_once_with()
    preview.visualizer.run.assert_not_called()
    preview.visualizer.destroy_window.assert_called_once_with()


def test_failed_managed_preview_stops_before_display_or_motion(preview, monkeypatch):
    monkeypatch.setattr(managed_run, "get_runtime", lambda: Mock())
    preview.visualizer.create_window.return_value = False
    with pytest.raises(RuntimeError, match="required 3D preview"):
        utils.visualize_point_cloud([[0, -0.1, 0.2]], show_coordinate_frame=False)
    preview.visualizer.add_geometry.assert_not_called()
    preview.visualizer.run.assert_not_called()
    preview.visualizer.destroy_window.assert_called_once_with()


@pytest.mark.parametrize("points,kwargs,error", [
    ([1, 2, 3], {}, "pcd_np must have shape"),
    ([[0, 1, 1]], {}, "No points remain"),
    ([[0, -0.1, 0.2]], {"colors": [[1, 0]]}, "colors must have shape"),
    ([[0, -0.1, 0.2]], {"colors": [[1, 0, 0], [0, 1, 0]]}, "exactly one RGB triplet"),
    ([[0, -0.1, 0.2]], {"camera_pose": np.eye(3)}, "camera_pose must have shape"),
])
def test_invalid_preview_inputs_fail_before_opening_window(preview, points, kwargs, error):
    with pytest.raises(ValueError, match=error):
        utils.visualize_point_cloud(points, show_coordinate_frame=False, **kwargs)
    preview.factory.assert_not_called()


def test_normal_visualization_preserves_origin_and_length():
    cloud = np.array([[0., 0, 0], [2., 0, 0]])
    with patch.object(visualization.o3d.visualization, "draw_geometries") as draw:
        utils.visualize_point_cloud_normals(cloud, [0, 0, 2])
    drawn_cloud, line, sphere = draw.call_args.args[0]
    np.testing.assert_array_equal(np.asarray(drawn_cloud.points), cloud)
    np.testing.assert_allclose(np.asarray(line.points), [[1, 0, 0], [1, 0, 0.5]])
    np.testing.assert_array_equal(np.asarray(line.lines), [[0, 1]])
    np.testing.assert_array_equal(np.asarray(line.colors), [[1, 0, 0]])
    np.testing.assert_allclose(sphere.get_center(), [1, 0, 0], atol=1e-12)
