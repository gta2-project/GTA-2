"""Documentation contracts and fixtures: no ROS, SDK, provider, or robot execution."""

import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from docs.examples.array_camera import ArrayCamera
from docs.examples.fake_gripper import FakeClient, Gripper, complete_future
from docs.examples.fake_robot import FakeArm, Helpers, RobotState
from docs.examples.point_attractor import PointAttractor
from scene.robot.base import RobotStateError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"


def test_example_sources_parse_without_execution():
    for path in EXAMPLES.glob("*.py"):
        ast.parse(path.read_text(), filename=path.name)


def test_camera_preserves_pixel_order_and_owns_copies():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    xyz = np.arange(18, dtype=float).reshape(2, 3, 3) / 100
    camera = ArrayCamera(rgb, xyz)
    rgb[:] = 255
    assert not camera.get_rgb_image().any()
    frame = camera.get_rgb_image(False)
    frame[:] = 255
    assert not camera.get_rgb_image(False).any()
    cloud = camera.get_point_cloud()
    np.testing.assert_array_equal(cloud[4], xyz[1, 1])
    cloud[:] = 100
    assert camera.get_point_cloud().max() < 1
    camera.close()
    with pytest.raises(RuntimeError, match="closed"):
        camera.get_rgb_image()


def test_camera_missing_cloud_is_not_simulation():
    camera = ArrayCamera(np.zeros((2, 3, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="no point cloud"):
        camera.get_point_cloud()
    with pytest.raises(NotImplementedError):
        camera.get_point_cloud(save_points=True)


@pytest.mark.parametrize("rgb,xyz", [
    (np.zeros((2, 3)), None),
    (np.zeros((2, 3, 3), dtype=float), None),
    (np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((2, 2, 3))),
    (np.zeros((2, 3, 3), dtype=np.uint8), np.full((2, 3, 3), np.nan)),
])
def test_camera_rejects_invalid_recordings(rgb, xyz):
    with pytest.raises(ValueError):
        ArrayCamera(rgb, xyz)


def test_controller_packet_and_priority_projection():
    from gta.error_projection import project_errors
    current = np.zeros(3)
    controller = PointAttractor(lambda: current, [0.1, 0.2, 0.3])
    packet = controller.compute_error(0.5)
    position, rotation, force, gripper = project_errors([packet])
    np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(rotation, np.eye(3))
    assert force is gripper is None
    first = {"type": "positional", "error": [0.04, 0, 0], "axis": [1, 0, 0]}
    position, *_ = project_errors([first, packet])
    np.testing.assert_allclose(position, [0.04, 0.2, 0.3])
    current[:] = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(controller.compute_error(1)["error"], 0)


@pytest.mark.parametrize("value", [[1, 2], [1, 2, np.nan], [1, np.inf, 3]])
def test_controller_rejects_invalid_points(value):
    with pytest.raises(ValueError):
        PointAttractor(lambda: np.zeros(3), value)
    with pytest.raises(ValueError):
        PointAttractor(lambda: value, np.zeros(3)).compute_error()


def test_robot_state_is_finite_fresh_and_independent():
    clock = [1.0]
    arm = FakeArm(clock=lambda: clock[0])
    state = RobotState(arm, clock=lambda: clock[0])
    pose = state.get_tcp_pose()
    pose[:] = 100
    assert state.get_tcp_pose()[0] == 0.2
    clock[0] = 2.0
    with pytest.raises(RobotStateError, match="stale"):
        state.get_tcp_pose()
    arm.sample_time = 2.0
    arm.pose[0] = np.nan
    with pytest.raises(RobotStateError, match="Invalid"):
        state.get_tcp_pose()
    state.close()
    with pytest.raises(RobotStateError, match="closed"):
        state.get_tcp_pose()


def test_reset_cleanup_and_callbacks():
    arm = FakeArm()
    events, errors = [], []
    Helpers(factory=lambda: arm).reset_joints(
        check_cancel=lambda: None, on_started=lambda: events.append("started"),
        errors=errors, wait=lambda seconds: None)
    assert events == ["started"] and not errors
    assert arm.calls == ["start_reset", "stop_joints", "disconnect"]


def test_reset_cancel_before_connection():
    def cancel():
        raise RuntimeError("cancelled")
    def forbidden():
        pytest.fail("Client must not be constructed after cancellation")
    with pytest.raises(RuntimeError, match="cancelled"):
        Helpers(factory=forbidden).reset_joints(
            check_cancel=cancel, on_started=forbidden, errors=[], wait=lambda _: None)


def test_reset_cancel_during_wait_still_stops_and_disconnects():
    arm = FakeArm()
    errors = []
    def cancel_wait(seconds):
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        Helpers(factory=lambda: arm).reset_joints(
            check_cancel=lambda: None, on_started=lambda: None,
            errors=errors, wait=cancel_wait)
    assert arm.calls[-2:] == ["stop_joints", "disconnect"]
    assert not errors


def test_reset_timeout_and_cleanup_error_collection():
    arm = FakeArm()
    arm.reset_done = lambda: False
    arm.stop_joints = lambda: False
    ticks = iter([0.0, 6.0])
    errors = []
    with pytest.raises(TimeoutError):
        Helpers(factory=lambda: arm, clock=lambda: next(ticks)).reset_joints(
            check_cancel=lambda: None, on_started=lambda: None,
            errors=errors, wait=lambda _: None)
    assert errors == ["Stop joints returned False"]
    assert not arm.connected


def test_gripper_open_close_and_shutdown_are_distinct():
    client = FakeClient()
    gripper = Gripper(client=client)
    gripper.wait_for_server()
    gripper.open()
    gripper.close()
    assert client.positions == [1.0, 0.0]
    with pytest.raises(NotImplementedError):
        gripper.half_open()
    assert gripper.cancel() == []
    assert gripper.shutdown() == []
    assert client.destroyed and client.positions == [1.0, 0.0]


@pytest.mark.parametrize("options,method,error", [
    ({"available": False}, "wait_for_server", "unavailable"),
    ({"accepted": False}, "open", "rejected"),
    ({"succeeded": False}, "close", "failed"),
])
def test_gripper_rejects_unavailable_or_failed_operations(options, method, error):
    client = FakeClient(**options)
    gripper = Gripper(client=client)
    with pytest.raises((RuntimeError, TimeoutError), match=error):
        getattr(gripper, method)()
    assert gripper.cancel() == []
    assert gripper.shutdown() == []


def test_gripper_late_ack_cancellation_ignores_operation_cancel_flag():
    client = FakeClient()
    gripper = Gripper(client=client, spin_until_complete=lambda *_: None)
    with pytest.raises(TimeoutError):
        gripper.open()
    assert gripper.pending is not None
    def cancelled():
        raise RuntimeError("cancelled")
    gripper.check_cancel = cancelled
    gripper.spin = complete_future
    assert gripper.cancel() == []
    assert client.goal.cancelled
    assert gripper.shutdown() == []


def test_gripper_reports_unconfirmed_cleanup():
    gripper = Gripper(spin_until_complete=lambda *_: None)
    with pytest.raises(TimeoutError):
        gripper.open()
    assert gripper.cancel() == ["Goal did not complete"]
    gripper.client.destroy = lambda: False
    assert gripper.shutdown() == ["Client destruction failed"]


def test_cube_fixture_validates_and_generates_without_running(tmp_path):
    from gta.plan_validation import validate_plan
    from gta.script_generator import generate_robot_script
    plan = json.loads((EXAMPLES / "cube-plan.json").read_text())
    validate_plan(plan)
    output = tmp_path / "robot_script.py"
    generate_robot_script(EXAMPLES / "cube-keypoints.json",
                          EXAMPLES / "cube-plan.json", output, strict=True)
    tree = ast.parse(output.read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "world_origin" not in names
    imports = [node for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module == "controllers"]
    allowed = {alias.name for node in imports for alias in node.names}
    for skill in plan:
        for item in skill["controllers"]:
            assert item["controller"].split("(")[0] in allowed
    assert "lambda: eef.get_k('TCP')" in output.read_text()
    assert output.parent == tmp_path  # Never create a live root robot_script.py.


def test_diagrams_are_local_editable_svg():
    paths = list((ROOT / "docs/_static/diagrams").glob("*.svg"))
    assert len(paths) == 5
    for path in paths:
        root = ET.parse(path).getroot()
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
        assert "http" not in path.read_text().replace("http://www.w3.org/2000/svg", "")


def test_tool_offset_rotates_before_translation():
    from docs.examples.tool_point import example, point_in_base
    np.testing.assert_allclose(example(), [0.3, -0.3, 0.4])
    with pytest.raises(ValueError):
        point_in_base([0] * 6, [0, 0, np.nan])


def test_photo_is_metadata_reviewed_copy():
    from PIL import Image
    copy = ROOT / "docs/_static/images/cube-scene.jpg"
    with Image.open(copy) as image:
        assert image.size == (1920, 1080)
        assert not image.getexif()
        assert not set(image.info) - {"jfif", "jfif_version", "jfif_unit", "jfif_density"}
    original = ROOT / "environment.jpg"
    if original.is_file():
        # The approved source photograph must not be edited by docs generation.
        # Runtime images may legitimately differ after users start the application,
        # so equality is checked during release validation, not enforced here.
        assert original.stat().st_size > 0


def test_local_site_checker_detects_missing_anchors(tmp_path):
    from docs.tools.check_site import check
    (tmp_path / "index.html").write_text(
        '<a href="page.html#ok">valid</a><img src="missing.png">'
        '<a href="page.html#absent">invalid</a>')
    (tmp_path / "page.html").write_text('<h1 id="ok">Page</h1>')
    _, count, errors = check(tmp_path)
    assert count == 3 and len(errors) == 2
    assert any("missing anchor" in error for error in errors)
