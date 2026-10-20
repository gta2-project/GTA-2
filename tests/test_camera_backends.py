"""Behavioral checks for the camera split; all ROS and sensor access is fake."""

import builtins
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from config import Config
from scene.camera import Camera
from scene.camera.base import BaseCamera
from scene.camera.file import CameraFile
from scene.camera.ros1 import CameraROS1
from scene.camera.ros2 import CameraROS2
from test_camera_ros2 import (
    FakeImage,
    FakeNode,
    FakePointCloud2,
    fake_ros2_modules,
    image_message,
    point_cloud_message,
)


@pytest.fixture(autouse=True)
def isolated_cameras():
    Camera._instance = None
    if CameraROS2._instance is not None:
        CameraROS2._instance.close()
    with (
        patch.object(Config, "CAMERA_COLOR_TOPIC", ""),
        patch.object(Config, "CAMERA_POINT_CLOUD_TOPIC", ""),
    ):
        yield
    Camera._instance = None
    if CameraROS2._instance is not None:
        CameraROS2._instance.close()


def fake_ros1_modules(messages, initialized=False):
    queues = {kind: list(values) for kind, values in messages.items()}

    def wait_for_message(topic, kind, timeout):
        return queues[kind].pop(0)

    rospy = SimpleNamespace(
        core=SimpleNamespace(is_initialized=Mock(return_value=initialized)),
        init_node=Mock(),
        get_master=Mock(return_value=SimpleNamespace(getSystemState=Mock())),
        wait_for_message=Mock(side_effect=wait_for_message),
        signal_shutdown=Mock(),
    )
    ros2_modules, _state = fake_ros2_modules(FakeNode({}))
    modules = {key: value for key, value in ros2_modules.items() if key.startswith("sensor_msgs")}
    modules["rospy"] = rospy
    return modules, rospy


@pytest.fixture(params=["ros1", "ros2"])
def camera_backend(request):
    messages = {FakeImage: [image_message([[[1, 2, 3]]])]}
    if request.param == "ros1":
        modules, _state = fake_ros1_modules(messages)
    else:
        modules, _state = fake_ros2_modules(FakeNode(messages))
    with patch.dict(sys.modules, modules):
        yield Camera(camera_node="/zed_node", camera_model="zed2i", camera_backend=request.param)


def private(camera, name):
    # The ROS 1 class is renamed in the split; test the same decoder either way.
    return getattr(camera, f"_{type(camera).__name__}__{name}")


def test_file_image_cache_and_fresh_reads_without_ros(tmp_path):
    path = tmp_path / "image.png"
    original = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    assert cv2.imwrite(str(path), original)
    original_import = builtins.__import__

    def no_ros(name, *args, **kwargs):
        if name.split(".")[0] in {"rospy", "rclpy", "sensor_msgs"}:
            raise AssertionError(f"File mode imported {name}")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=no_ros):
        camera = Camera(camera_backend="file", camera_image_path=str(path))
        first = camera.get_rgb_image(use_new_frame=False)
        np.testing.assert_array_equal(first, original[:, :, ::-1])
        first[:] = 0
        np.testing.assert_array_equal(camera.get_rgb_image(False), original[:, :, ::-1])
        assert cv2.imwrite(str(path), original + 1)
        np.testing.assert_array_equal(camera.get_rgb_image(False), original[:, :, ::-1])
        np.testing.assert_array_equal(camera.get_rgb_image(), (original + 1)[:, :, ::-1])
        assert Camera(camera_backend="file", camera_image_path="ignored.png") is camera
        # Preserve the original shared file/ROS 1 instance, not dynamic switching.
        assert Camera(camera_backend="ros1") is camera
        with pytest.raises(RuntimeError, match="File camera mode only provides RGB images"):
            camera.get_point_cloud()


def test_file_missing_and_unreadable_images(tmp_path):
    path = tmp_path / "image.png"
    camera = Camera(camera_backend="file", camera_image_path=str(path))
    with pytest.raises(FileNotFoundError, match="Camera image file not found"):
        camera.get_rgb_image()
    path.touch()
    with pytest.raises(ValueError, match="Could not read camera image file"):
        camera.get_rgb_image()


@pytest.mark.parametrize("initialized", [False, True])
def test_ros1_initialization_topics_cache_and_timeout(monkeypatch, initialized):
    monkeypatch.delenv("ROS_MASTER_URI", raising=False)
    modules, rospy = fake_ros1_modules({
        FakeImage: [image_message([[[1, 2, 3]]]), image_message([[[4, 5, 6]]])],
        FakePointCloud2: [point_cloud_message()],
    }, initialized)
    with patch.dict(sys.modules, modules):
        camera = Camera(camera_backend="ros1", camera_node="/camera/realsense2_camera",
                        camera_model="d405", rosmaster_ip="test-host", rosmaster_port=11311)
        assert rospy.init_node.call_count == (0 if initialized else 1)
        if not initialized:
            rospy.init_node.assert_called_once_with("gta_camera", anonymous=True, disable_signals=True)
        rospy.get_master.return_value.getSystemState.assert_called_once()
        assert len(rospy.wait_for_message.call_args_list) == 1
        cached = camera.get_rgb_image(False)
        cached[:] = 0
        np.testing.assert_array_equal(camera.get_rgb_image(False), [[[3, 2, 1]]])
        np.testing.assert_array_equal(camera.get_rgb_image(), [[[6, 5, 4]]])
        np.testing.assert_allclose(camera.get_point_cloud(), [[0.1, 0.2, 0.3]], rtol=1e-6)
        calls = rospy.wait_for_message.call_args_list
        assert [call.args[0] for call in calls] == [
            "/camera/color/image_raw", "/camera/color/image_raw", "/camera/transform_igev_points",
        ]
        timeout = calls[0].kwargs["timeout"]
        assert all(call.kwargs["timeout"] == timeout for call in calls)
        assert Camera(camera_backend="ros1", camera_node="/ignored") is camera
        rospy.wait_for_message.side_effect = TimeoutError("no frame")
        with pytest.raises(RuntimeError, match=f"after {timeout:.1f}s"):
            camera.get_rgb_image()
        rospy.signal_shutdown.assert_not_called()


def test_failed_ros1_initialization_can_be_retried():
    modules, rospy = fake_ros1_modules({FakeImage: [image_message([[[1, 2, 3]]])]})
    rospy.get_master.return_value.getSystemState.side_effect = RuntimeError("offline")
    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match="Could not contact ROS master"):
            Camera(camera_backend="ros1", camera_node="/zed_node", camera_model="zed2i")
        assert Camera._instance is None
        rospy.get_master.return_value.getSystemState.side_effect = None
        camera = Camera(camera_backend="ros1", camera_node="/zed_node", camera_model="zed2i")
        np.testing.assert_array_equal(camera.get_rgb_image(False), [[[3, 2, 1]]])


@pytest.mark.parametrize("encoding,values,expected", [
    ("rgb8", [[[1, 2, 3]]], [[[1, 2, 3]]]),
    ("bgr8", [[[1, 2, 3]]], [[[3, 2, 1]]]),
    ("rgba8", [[[1, 2, 3, 4]]], [[[1, 2, 3]]]),
    ("bgra8", [[[1, 2, 3, 4]]], [[[3, 2, 1]]]),
    ("mono8", [[7]], [[[7, 7, 7]]]),
])
def test_image_conversion_preserves_channels_and_row_padding(camera_backend, encoding, values, expected):
    pixels = np.asarray(values, dtype=np.uint8)
    padded = np.concatenate([pixels, np.zeros_like(pixels)], axis=1)
    message = SimpleNamespace(encoding=encoding, is_bigendian=False, height=1,
                              width=1, step=padded.nbytes, data=padded.tobytes())
    np.testing.assert_array_equal(private(camera_backend, "image_msg_to_rgb")(message), expected)


@pytest.mark.parametrize("big_endian", [False, True])
def test_image_conversion_preserves_endianness(camera_backend, big_endian):
    pixels = np.array([[300, 500]], dtype=">u2" if big_endian else "<u2")
    message = SimpleNamespace(encoding="mono16", is_bigendian=big_endian, height=1,
                              width=2, step=4, data=pixels.tobytes())
    if big_endian == (sys.byteorder == "little") and not hasattr(pixels, "newbyteorder"):
        # Pre-existing NumPy 2 incompatibility, verified before the split.
        # Record the behavior without changing decoding in a structural refactor.
        with pytest.raises(AttributeError, match="newbyteorder"):
            private(camera_backend, "image_msg_to_rgb")(message)
        return
    result = private(camera_backend, "image_msg_to_rgb")(message)
    assert result.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(result, [[[300, 300, 300], [500, 500, 500]]])


@pytest.mark.parametrize("rgb_type", [None, 6, 7])
def test_cloud_conversion_keeps_existing_scale_order_nan_and_stride_behavior(camera_backend, rgb_type):
    message = point_cloud_message()
    fmt = ">fff"
    points = [(1., 2., 3.), (float("nan"), 0., 0.), (4., 5., 6.), (7., 8., 9.)]
    if rgb_type:
        message.fields.append(SimpleNamespace(name="rgb", offset=12, datatype=rgb_type, count=1))
        fmt += "I" if rgb_type == 6 else "f"
        rgb = 0x123456 if rgb_type == 6 else struct.unpack(">f", struct.pack(">I", 0x123456))[0]
        points = [(*point, rgb) for point in points]
    packed = [struct.pack(fmt, *point) for point in points]
    message.is_bigendian = True
    message.height, message.width = 2, 2
    message.point_step = struct.calcsize(fmt)
    message.row_step = 2 * message.point_step + 4
    message.data = packed[0] + packed[1] + b"\0" * 4 + packed[2] + packed[3] + b"\0" * 4
    with patch.object(camera_backend, f"_{type(camera_backend).__name__}__wait_for_message", return_value=message):
        cloud = camera_backend.get_point_cloud()
    expected = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    # This existing RGB-dependent scaling is intentionally NOT fixed in a structural refactor.
    np.testing.assert_array_equal(cloud, expected * (1000 if rgb_type else 1))


def test_empty_cloud_still_returns_none(camera_backend):
    message = point_cloud_message()
    message.width = message.height = 0
    with patch.object(camera_backend, f"_{type(camera_backend).__name__}__wait_for_message", return_value=message):
        assert camera_backend.get_point_cloud() is None


def test_compressed_image_conversion(camera_backend):
    bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    ok, data = cv2.imencode(".png", bgr)
    assert ok
    with patch.object(camera_backend, f"_{type(camera_backend).__name__}__wait_for_message",
                      return_value=SimpleNamespace(data=data.tobytes())):
        result = private(camera_backend, "get_compressedimage_from_rostopic")("/compressed")
    np.testing.assert_array_equal(result, bgr[:, :, ::-1])


@pytest.mark.parametrize("source", ["unsupported", "local", "ros"])
def test_invalid_backend_does_not_cache_an_instance(source):
    with pytest.raises(ValueError, match="Unsupported camera backend"):
        Camera(camera_backend=source)
    assert Camera._instance is None


def test_camera_interface_requires_image_and_cloud_methods():
    with pytest.raises(TypeError, match="abstract"):
        BaseCamera()

    class IncompleteCamera(BaseCamera):
        def get_rgb_image(self, use_new_frame=True):
            return None

    with pytest.raises(TypeError, match="abstract"):
        IncompleteCamera()
    for backend in (CameraFile, CameraROS1, CameraROS2):
        assert issubclass(backend, BaseCamera)
        assert not backend.__abstractmethods__


def test_factory_returns_interface_implementation(camera_backend):
    assert isinstance(camera_backend, BaseCamera)
    expected = CameraROS2 if camera_backend.camera_backend == "ros2" else CameraROS1
    assert type(camera_backend) is expected


def test_file_and_ros1_close_do_not_change_shared_resources():
    file_camera = Camera(camera_backend="file")
    assert isinstance(file_camera, BaseCamera)
    file_camera.close()
    assert Camera(camera_backend="file") is file_camera
    Camera._instance = None
    modules, rospy = fake_ros1_modules({FakeImage: [image_message([[[1, 2, 3]]])]})
    with patch.dict(sys.modules, modules):
        camera = Camera(camera_backend="ros1")
        camera.close()
        assert Camera(camera_backend="ros1") is camera
        rospy.signal_shutdown.assert_not_called()


def test_factory_forwards_all_constructor_arguments():
    for source, backend in [("file", "file.CameraFile"), ("ros1", "ros1.CameraROS1"),
                            ("ros2", "ros2.CameraROS2")]:
        Camera._instance = None
        settings = dict(camera_node="/example", camera_model="example", rosmaster_ip="host",
                        rosmaster_port=1234, camera_backend=source, camera_image_path="image.png", debug=True)
        with patch(f"scene.camera.{backend}") as factory:
            assert Camera(**settings) is factory.return_value
            factory.assert_called_once_with(**settings)


def test_ros2_singleton_cache_close_and_reinitialization():
    messages = {FakeImage: [image_message([[[1, 2, 3]]]), image_message([[[4, 5, 6]]])]}
    node = FakeNode(messages)
    modules, state = fake_ros2_modules(node)
    modules["rclpy"].shutdown = Mock()
    with patch.dict(sys.modules, modules):
        camera = Camera(camera_backend="ros2")
        assert Camera(camera_backend="ros2", camera_node="/ignored") is camera
        cached = camera.get_rgb_image(False)
        cached[:] = 0
        np.testing.assert_array_equal(camera.get_rgb_image(False), [[[3, 2, 1]]])
        assert state.spin_calls == 1
        np.testing.assert_array_equal(camera.get_rgb_image(), [[[6, 5, 4]]])
        assert state.spin_calls == 2
        assert len(node.subscriptions) == 1
        camera.close()
        camera.close()
        assert len(node.destroyed_subscriptions) == 1
        assert CameraROS2._instance is None
        assert state.ok
        modules["rclpy"].shutdown.assert_not_called()
        new_node = FakeNode({FakeImage: [image_message([[[7, 8, 9]]])]})
        modules["rclpy"].create_node = Mock(return_value=new_node)
        replacement = Camera(camera_backend="ros2")
        assert replacement is not camera
        np.testing.assert_array_equal(replacement.get_rgb_image(False), [[[9, 8, 7]]])


@pytest.mark.parametrize("publishers,detail", [
    (0, "no publishers were discovered"),
    (1, "1 publisher(s) were discovered, but none delivered a message"),
])
def test_ros2_startup_timeout_cleans_up_and_preserves_diagnostics(publishers, detail):
    from scene.camera.ros2 import ROS2_TOPIC_TIMEOUT_S

    node = FakeNode({})
    node.count_publishers = Mock(return_value=publishers)
    clock_start = SimpleNamespace()
    clock_end = Mock()
    clock_end.__sub__ = Mock(return_value=SimpleNamespace(nanoseconds=(ROS2_TOPIC_TIMEOUT_S + 1) * 1e9))
    node.get_clock = Mock(return_value=SimpleNamespace(now=Mock(side_effect=[clock_start, clock_end])))
    modules, state = fake_ros2_modules(node)
    with patch.dict(sys.modules, modules):
        with pytest.raises(TimeoutError) as error:
            Camera(camera_backend="ros2", camera_node="/zed_node", camera_model="zed2i")
        assert f"after {ROS2_TOPIC_TIMEOUT_S:.1f}s; {detail}" in str(error.value)
        assert state.last_spin == (node, 0.1)
        assert node.destroyed
        assert len(node.destroyed_subscriptions) == 1
        assert CameraROS2._instance is None


def test_ros2_unavailable_dependency_keeps_clear_error_and_allows_retry():
    with patch.dict(sys.modules, {"rclpy": None}):
        with pytest.raises(RuntimeError, match="CAMERA_BACKEND=ros2 requires rclpy"):
            Camera(camera_backend="ros2")
    assert CameraROS2._instance is None
    node = FakeNode({FakeImage: [image_message([[[1, 2, 3]]])]})
    modules, _state = fake_ros2_modules(node)
    with patch.dict(sys.modules, modules):
        assert isinstance(Camera(camera_backend="ros2"), CameraROS2)


def test_ros2_path_has_moved_without_a_compatibility_wrapper():
    scene_dir = Path(__file__).resolve().parents[1] / "scene"
    assert not (scene_dir / "cameraROS2.py").exists()
    assert (scene_dir / "camera" / "ros2.py").is_file()


def test_file_backend_has_no_old_name_or_wrapper():
    scene_dir = Path(__file__).resolve().parents[1] / "scene"
    assert not (scene_dir / "camera" / "local.py").exists()
    assert (scene_dir / "camera" / "file.py").is_file()
    assert CameraFile.__name__ == "CameraFile"


def test_factory_and_backends_share_one_package_without_old_paths():
    scene_dir = Path(__file__).resolve().parents[1] / "scene"
    assert not (scene_dir / "camera.py").exists()
    assert not (scene_dir / "cameras").exists()
    for cls, filename in (
        (Camera, "__init__.py"),
        (BaseCamera, "base.py"),
        (CameraFile, "file.py"),
        (CameraROS1, "ros1.py"),
        (CameraROS2, "ros2.py"),
    ):
        module = sys.modules[cls.__module__]
        assert Path(module.__file__).resolve() == scene_dir / "camera" / filename


@pytest.mark.parametrize("configured_backend", ["file", "ros1", "ros2"])
def test_package_import_stays_lazy_and_file_mode_needs_no_ros(configured_backend):
    probe = """
import builtins
import sys
from config import Config

Config.CAMERA_BACKEND = sys.argv[1]
original_import = builtins.__import__
def no_hardware_import(name, *args, **kwargs):
    if name.split('.')[0] in {'rospy', 'rclpy', 'sensor_msgs', 'open3d', 'rtde_receive', 'rtde_control'}:
        raise AssertionError(f'Unexpected hardware dependency: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_hardware_import

from scene.camera import Camera
from scene.camera.base import BaseCamera
for name in ('scene.camera.file', 'scene.camera.ros1', 'scene.camera.ros2', 'cv2', 'open3d'):
    assert name not in sys.modules, name
assert Camera._instance is None

# Construction in file mode neither reads an image nor starts ROS.
camera = Camera(camera_backend='file', camera_image_path='unused.png')
assert isinstance(camera, BaseCamera)
assert type(camera).__module__ == 'scene.camera.file'
for name in ('scene.camera.ros1', 'scene.camera.ros2', 'rospy', 'rclpy', 'sensor_msgs', 'open3d'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, configured_backend],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("old_keyword", ["camera_source", "camera_type", "local_image_path"])
def test_old_camera_constructor_keywords_are_not_aliases(old_keyword):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Camera(**{old_keyword: "file"})
    assert Camera._instance is None


def test_renamed_camera_attributes_and_cached_backend_validation():
    camera = Camera(camera_backend="file", camera_model="d405", camera_image_path="scene.png")
    assert camera.camera_backend == "file"
    assert camera.camera_image_path == "scene.png"
    assert camera.args.camera_seperation == 0.018
    assert not hasattr(camera, "camera_source")
    assert not hasattr(camera, "local_image_path")
    # Invalid/retired names are rejected even when a valid instance is cached.
    with pytest.raises(ValueError, match="Unsupported camera backend: ros"):
        Camera(camera_backend="ros")
    assert Camera(camera_backend="file") is camera
