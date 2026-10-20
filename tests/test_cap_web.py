"""Tests for the CaP perception-review-and-run web interface."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from benchmarks.cap.perception import PerceptionSnapshot
from benchmarks.cap.visualization import show_interactive_point_cloud
from benchmarks.cap.web import CapWebService, create_cap_web_app


CARTESIAN_POLICY = """KEYPOINT_QUERIES = {}

def policy(ctx):
    move_to(0.1, -0.4, 0.3, 0.0, 3.14, 0.0)
"""

EDITED_POLICY = """KEYPOINT_QUERIES = {}

def policy(ctx):
    move_to(0.2, -0.4, 0.3, 0.0, 3.14, 0.0)
"""

QUERY_POLICY = """KEYPOINT_QUERIES = {
    "target": {"object": "block", "description": "center of the block"},
}

def policy(ctx):
    point = ctx.point("target")
    move_to(point[0], point[1], point[2], 0.0, 3.14, 0.0)
"""

CONTROLLER_POLICY = """KEYPOINT_QUERIES = {}

def policy(ctx):
    execute_subtask([])
"""


class FakeTrialRunner:
    def __init__(self, generated_policy=CARTESIAN_POLICY):
        self.generated_policy = generated_policy
        self.calls = []

    def __call__(self, **kwargs):
        call = dict(kwargs)
        policy_file = call.get("policy_file")
        if policy_file is not None:
            call["policy_source"] = Path(policy_file).read_text(encoding="utf-8")
        self.calls.append(call)

        index = len(self.calls)
        artifact_dir = Path(kwargs["output_root"]) / f"fake_trial_{index}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        if kwargs["mode"] == "generate":
            (artifact_dir / "policy.py").write_text(
                self.generated_policy, encoding="utf-8"
            )
            return {
                "status": "generated",
                "artifact_dir": str(artifact_dir),
                "model": {"served_model": "fake-model"},
            }
        return {"status": "robot_completed", "artifact_dir": str(artifact_dir)}


class FailingRobotTrialRunner(FakeTrialRunner):
    def __call__(self, **kwargs):
        if kwargs["mode"] == "robot":
            raise RuntimeError("synthetic robot execution failure")
        return super().__call__(**kwargs)


class FakeControllerServer:
    def __init__(self, factory, log_path):
        self.factory = factory
        self.log_path = Path(log_path)

    def __enter__(self):
        self.factory.events.append("start")
        self.log_path.write_text("fake controller server started\n", encoding="utf-8")
        return self

    def assert_running(self, phase=""):
        self.factory.events.append(f"assert_running:{phase}")

    def __exit__(self, exc_type, exc, traceback):
        self.factory.events.append("stop")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("fake controller server stopped\n")
        return False


class FakeControllerServerFactory:
    def __init__(self):
        self.events = []
        self.log_paths = []

    def __call__(self, log_path):
        self.log_paths.append(Path(log_path))
        return FakeControllerServer(self, log_path)


class BlockingTrialRunner(FakeTrialRunner):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, **kwargs):
        if kwargs["mode"] == "generate":
            self.started.set()
            if not self.release.wait(timeout=3.0):
                raise RuntimeError("Test did not release blocked generation")
        return super().__call__(**kwargs)


class FakeImageCapturer:
    def __init__(self):
        self.calls = []

    def __call__(self, destination: Path) -> Path:
        self.calls.append(Path(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        color = (20 * len(self.calls), 40, 60)
        Image.new("RGB", (160, 120), color).save(destination)
        return destination


class FakePointCloudCapturer:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        x, y = np.meshgrid(
            np.linspace(0.0, 0.14, 160), np.linspace(-0.5, -0.4, 120)
        )
        z = np.full_like(x, 0.3)
        return np.stack([x, y, z], axis=-1)


class FakeLocalizer:
    def __init__(self):
        self.calls = []

    def localize(self, **kwargs):
        self.calls.append(kwargs)
        aliases = list(kwargs["queries"])
        return PerceptionSnapshot(
            points={name: np.array([0.08, -0.45, 0.3]) for name in aliases},
            normals={name: np.array([0.0, 0.0, 1.0]) for name in aliases},
            pixels={name: (80, 60) for name in aliases},
        )


class FakeInteractiveVisualizer:
    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("Open3D window could not be reviewed")


class CapWebTests(unittest.TestCase):
    def make_client(
        self,
        root: Path,
        generated_policy=CARTESIAN_POLICY,
        interactive_visualizer=None,
        runner=None,
        controller_server_factory=None,
    ):
        runner = runner or FakeTrialRunner(generated_policy)
        image_capturer = FakeImageCapturer()
        cloud_capturer = FakePointCloudCapturer()
        localizer = FakeLocalizer()
        interactive_visualizer = (
            interactive_visualizer or FakeInteractiveVisualizer()
        )
        service = CapWebService(
            output_root=root / "runs",
            trial_runner=runner,
            image_capturer=image_capturer,
            point_cloud_capturer=cloud_capturer,
            perception_localizer=localizer,
            interactive_visualizer=interactive_visualizer,
            controller_server_factory=controller_server_factory,
        )
        client = TestClient(create_cap_web_app(service))
        return (
            client,
            runner,
            image_capturer,
            cloud_capturer,
            localizer,
            interactive_visualizer,
        )

    def test_config_and_selector_offer_only_standard_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, image_capturer, cloud_capturer, _, _ = self.make_client(
                Path(tmpdir)
            )
            expected = {
                "controller_zeroshot",
                "controller_fewshot",
                "cartesian_zeroshot",
                "cartesian_fewshot",
            }
            response = client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["conditions"], sorted(expected))
            self.assertEqual(response.json()["default_condition"], "cartesian_zeroshot")

            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            selector = page.text.split('<select id="condition">', 1)[1].split(
                "</select>", 1
            )[0]
            self.assertEqual(selector.count("<option "), len(expected))
            for name in expected:
                self.assertIn(f'value="{name}"', selector)
            self.assertEqual(runner.calls, [])
            self.assertEqual(image_capturer.calls, [])
            self.assertEqual(cloud_capturer.calls, 0)

    @staticmethod
    def generate(client, policy_condition="cartesian_zeroshot"):
        response = client.post(
            "/api/generate",
            json={"task": "Move to the visible target", "condition": policy_condition},
        )
        if response.status_code != 202:
            raise AssertionError(response.text)
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status = client.get(f"/api/generate/{job_id}")
            if status.status_code != 200:
                raise AssertionError(status.text)
            body = status.json()
            if body["status"] == "completed":
                return body["result"]
            if body["status"] == "failed":
                raise AssertionError(body.get("error", "Generation failed"))
            time.sleep(0.01)
        raise AssertionError("Generation job did not complete in time")

    @staticmethod
    def prepare(client, generated, fresh_scene=False):
        response = client.post(
            "/api/prepare",
            json={
                "session_id": generated["session_id"],
                "code": generated["code"],
                "fresh_scene": fresh_scene,
            },
        )
        if response.status_code != 200:
            raise AssertionError(response.text)
        return response.json()

    def test_page_exposes_prepare_previews_and_explicit_execute_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, *_ = self.make_client(Path(tmpdir))
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Generate code", response.text)
            self.assertIn("Prepare current scene", response.text)
            self.assertIn("Prepare fresh-scene rerun", response.text)
            self.assertIn("Annotated frozen RGB", response.text)
            self.assertIn("interactive Open3D", response.text)
            self.assertIn("Execute prepared policy", response.text)
            self.assertIn("CAP will start and stop it automatically", response.text)

    def test_interactive_adapter_calls_existing_gta_visualize_point_cloud(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "scene.jpg"
            Image.new("RGB", (2, 2), (10, 20, 30)).save(image_path)
            cloud = np.array(
                [
                    [[0.0, -0.5, 0.3], [0.1, -0.5, 0.3]],
                    [[0.0, -0.4, 0.3], [0.1, -0.4, 0.3]],
                ]
            )
            with patch(
                "gta.utils.visualize_point_cloud"
            ) as gta_visualizer:
                show_interactive_point_cloud(
                    cloud,
                    image_path,
                    {"target": np.array([0.1, -0.45, 0.3])},
                    {"target": np.array([0.0, 0.0, 1.0])},
                )
            gta_visualizer.assert_called_once()
            kwargs = gta_visualizer.call_args.kwargs
            np.testing.assert_allclose(kwargs["pcd_np"], cloud)
            self.assertEqual(kwargs["colors"].shape, (2, 2, 3))
            self.assertEqual(len(kwargs["objects"]), 1)
            np.testing.assert_allclose(
                kwargs["objects"][0].keypoints["target"],
                [0.1, -0.45, 0.3],
            )

    def test_generate_does_not_capture_point_cloud_or_open_robot_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                client,
                runner,
                image_capturer,
                cloud_capturer,
                localizer,
                interactive_visualizer,
            ) = self.make_client(Path(tmpdir))
            generated = self.generate(client)
            self.assertEqual(generated["code"], CARTESIAN_POLICY)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0]["mode"], "generate")
            self.assertEqual(len(image_capturer.calls), 1)
            self.assertEqual(cloud_capturer.calls, 0)
            self.assertEqual(localizer.calls, [])
            self.assertEqual(interactive_visualizer.calls, [])

    def test_config_remains_responsive_during_blocking_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = BlockingTrialRunner()
            service = CapWebService(
                output_root=root / "runs",
                trial_runner=runner,
                image_capturer=FakeImageCapturer(),
                point_cloud_capturer=FakePointCloudCapturer(),
                perception_localizer=FakeLocalizer(),
                interactive_visualizer=FakeInteractiveVisualizer(),
            )
            with TestClient(create_cap_web_app(service)) as client:
                started = time.monotonic()
                response = client.post(
                    "/api/generate",
                    json={
                        "task": "Move to the visible target",
                        "condition": "cartesian_zeroshot",
                    },
                )
                start_latency = time.monotonic() - started
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]
                self.assertTrue(runner.started.wait(timeout=1.0))
                started = time.monotonic()
                config_response = client.get("/api/config")
                config_latency = time.monotonic() - started
                running = client.get(f"/api/generate/{job_id}").json()
                runner.release.set()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    completed = client.get(f"/api/generate/{job_id}").json()
                    if completed["status"] == "completed":
                        break
                    time.sleep(0.01)

            self.assertEqual(config_response.status_code, 200)
            self.assertLess(start_latency, 1.0)
            self.assertLess(config_latency, 1.0)
            self.assertEqual(running["status"], "running")
            self.assertEqual(completed["status"], "completed")
            self.assertIn("session_id", completed["result"])

    def test_prepare_localizes_and_serves_both_previews_without_robot_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                client,
                runner,
                _,
                cloud_capturer,
                localizer,
                interactive_visualizer,
            ) = self.make_client(Path(tmpdir), QUERY_POLICY)
            generated = self.generate(client)
            prepared = self.prepare(client, generated)

            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(cloud_capturer.calls, 1)
            self.assertEqual(len(localizer.calls), 1)
            self.assertEqual(len(interactive_visualizer.calls), 1)
            self.assertIn("target", interactive_visualizer.calls[0]["points"])
            self.assertEqual(prepared["points"]["target"], [0.08, -0.45, 0.3])
            self.assertEqual(prepared["normals"]["target"], [0.0, 0.0, 1.0])
            for key in (
                "annotated_image_url",
                "point_cloud_preview_url",
                "scene_image_url",
            ):
                response = client.get(prepared[key])
                self.assertEqual(response.status_code, 200, key)
                self.assertTrue(response.headers["content-type"].startswith("image/"))

    def test_execute_requires_approval_after_preparation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, *_ = self.make_client(Path(tmpdir))
            generated = self.generate(client)
            prepared = self.prepare(client, generated)
            response = client.post(
                "/api/execute",
                json={
                    "session_id": generated["session_id"],
                    "preparation_id": prepared["preparation_id"],
                    "code": generated["code"],
                    "operator_approved": False,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("previews", response.json()["detail"])
            self.assertEqual(len(runner.calls), 1)

    def test_execute_uses_exact_prepared_policy_and_perception_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, *_ = self.make_client(Path(tmpdir), QUERY_POLICY)
            generated = self.generate(client)
            prepared = self.prepare(client, generated)
            response = client.post(
                "/api/execute",
                json={
                    "session_id": generated["session_id"],
                    "preparation_id": prepared["preparation_id"],
                    "code": generated["code"],
                    "operator_approved": True,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            call = runner.calls[1]
            self.assertEqual(call["mode"], "robot")
            self.assertTrue(call["approve_robot"])
            self.assertIn("KEYPOINT_QUERIES", call["policy_source"])
            self.assertTrue(Path(call["perception_path"]).is_file())
            self.assertEqual(
                Path(call["perception_path"]).parent,
                Path(call["policy_file"]).parent,
            )
            artifact_dir = Path(response.json()["artifact_dir"])
            self.assertTrue((artifact_dir / "annotated_keypoints.png").is_file())
            self.assertTrue((artifact_dir / "point_cloud_preview.png").is_file())
            self.assertTrue((artifact_dir / "point_cloud.npy").is_file())
            self.assertTrue((artifact_dir / "preparation.json").is_file())

    def test_controller_execute_manages_server_and_copies_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server_factory = FakeControllerServerFactory()
            client, runner, *_ = self.make_client(
                Path(tmpdir),
                CONTROLLER_POLICY,
                controller_server_factory=server_factory,
            )
            generated = self.generate(client, "controller_zeroshot")
            prepared = self.prepare(client, generated)
            response = client.post(
                "/api/execute",
                json={
                    "session_id": generated["session_id"],
                    "preparation_id": prepared["preparation_id"],
                    "code": generated["code"],
                    "operator_approved": True,
                },
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                server_factory.events,
                ["start", "assert_running:after policy execution", "stop"],
            )
            self.assertEqual(runner.calls[-1]["mode"], "robot")
            artifact_dir = Path(response.json()["artifact_dir"])
            server_log = artifact_dir / "controller_server.log"
            self.assertTrue(server_log.is_file())
            self.assertIn("stopped", server_log.read_text(encoding="utf-8"))
            self.assertIn("server was stopped", response.json()["message"])

    def test_controller_server_stops_when_policy_execution_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server_factory = FakeControllerServerFactory()
            runner = FailingRobotTrialRunner(CONTROLLER_POLICY)
            client, *_ = self.make_client(
                Path(tmpdir),
                runner=runner,
                controller_server_factory=server_factory,
            )
            generated = self.generate(client, "controller_zeroshot")
            prepared = self.prepare(client, generated)
            response = client.post(
                "/api/execute",
                json={
                    "session_id": generated["session_id"],
                    "preparation_id": prepared["preparation_id"],
                    "code": generated["code"],
                    "operator_approved": True,
                },
            )

            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("synthetic robot execution failure", response.json()["detail"])
            self.assertEqual(server_factory.events, ["start", "stop"])

    def test_policy_edit_after_preparation_invalidates_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, *_ = self.make_client(Path(tmpdir))
            generated = self.generate(client)
            prepared = self.prepare(client, generated)
            response = client.post(
                "/api/execute",
                json={
                    "session_id": generated["session_id"],
                    "preparation_id": prepared["preparation_id"],
                    "code": EDITED_POLICY,
                    "operator_approved": True,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Policy changed", response.json()["detail"])
            self.assertEqual(len(runner.calls), 1)

    def test_preparation_can_authorize_only_one_physical_attempt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, *_ = self.make_client(Path(tmpdir))
            generated = self.generate(client)
            prepared = self.prepare(client, generated)
            request = {
                "session_id": generated["session_id"],
                "preparation_id": prepared["preparation_id"],
                "code": generated["code"],
                "operator_approved": True,
            }
            self.assertEqual(client.post("/api/execute", json=request).status_code, 200)
            second = client.post("/api/execute", json=request)
            self.assertEqual(second.status_code, 400)
            self.assertIn("already been used", second.json()["detail"])
            self.assertEqual(len(runner.calls), 2)

    def test_new_preparation_invalidates_older_preview_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, *_ = self.make_client(Path(tmpdir))
            generated = self.generate(client)
            older = self.prepare(client, generated)
            self.prepare(client, generated, fresh_scene=True)
            response = client.post(
                "/api/execute",
                json={
                    "session_id": generated["session_id"],
                    "preparation_id": older["preparation_id"],
                    "code": generated["code"],
                    "operator_approved": True,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("newer preparation", response.json()["detail"])
            self.assertEqual(len(runner.calls), 1)

    def test_fresh_scene_preparation_recaptures_image_without_model_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                client,
                runner,
                image_capturer,
                cloud_capturer,
                _,
                interactive_visualizer,
            ) = self.make_client(Path(tmpdir))
            generated = self.generate(client)
            generation_image = Path(runner.calls[0]["image_path"])
            prepared = self.prepare(client, generated, fresh_scene=True)

            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(len(image_capturer.calls), 2)
            self.assertEqual(cloud_capturer.calls, 1)
            self.assertEqual(len(interactive_visualizer.calls), 1)
            prepared_scene = Path(prepared["preparation_dir"]) / "scene_image.jpg"
            self.assertNotEqual(prepared_scene, generation_image)
            self.assertTrue(prepared_scene.is_file())

    def test_invalid_python_fails_before_point_cloud_capture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client, runner, _, cloud_capturer, _, interactive_visualizer = (
                self.make_client(Path(tmpdir))
            )
            generated = self.generate(client)
            response = client.post(
                "/api/prepare",
                json={
                    "session_id": generated["session_id"],
                    "code": "KEYPOINT_QUERIES = {}\ndef policy(ctx)\n    pass\n",
                    "fresh_scene": False,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(cloud_capturer.calls, 0)
            self.assertEqual(interactive_visualizer.calls, [])
            self.assertEqual(len(runner.calls), 1)

    def test_failed_interactive_review_does_not_create_executable_preparation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = FakeInteractiveVisualizer()
            visualizer.fail = True
            client, runner, *_ = self.make_client(
                Path(tmpdir), interactive_visualizer=visualizer
            )
            generated = self.generate(client)
            response = client.post(
                "/api/prepare",
                json={
                    "session_id": generated["session_id"],
                    "code": generated["code"],
                    "fresh_scene": False,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Open3D window", response.json()["detail"])
            self.assertEqual(len(visualizer.calls), 1)
            self.assertEqual(len(runner.calls), 1)


if __name__ == "__main__":
    unittest.main()
