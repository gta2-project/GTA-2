"""Tests for the FastAPI dashboard API."""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from api.service import GtaWebService
from config import Config


class ApiTests(unittest.TestCase):
    def make_client(self, root: Path) -> TestClient:
        service = GtaWebService(root)
        app = create_app(service=service, initialize_on_startup=False)
        return TestClient(app)

    def write_workflow_artifacts(self, root: Path, environment_path: Path = None) -> None:
        artifacts = {
            "annotated_keypoints.png": b"annotated",
            "keypoint_locations.json": b"{}",
            "resized_img.png": b"resized",
            "robot_script.py": b"print('robot')\n",
            "skills_output.json": b"[]",
        }
        for name, content in artifacts.items():
            (root / name).write_bytes(content)

        environment_path = environment_path or root / "environment.jpg"
        environment_path.parent.mkdir(parents=True, exist_ok=True)
        environment_path.write_bytes(b"environment")

    def test_health_and_state_are_available_without_agent_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(Path(tmpdir))

            health = client.get("/api/health")
            state = client.get("/api/state")

            self.assertEqual(health.status_code, 200)
            self.assertIn("ready", health.json())
            self.assertEqual(state.status_code, 200)
            self.assertIn("artifacts", state.json())

    def test_artifact_allowlist_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(Path(tmpdir))

            traversal = client.get("/api/artifacts/%2E%2E%2Fconfig.py")
            missing = client.get("/api/artifacts/annotated_keypoints.png")

            self.assertEqual(traversal.status_code, 400)
            self.assertEqual(missing.status_code, 404)

    def test_script_generation_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keypoint_locations.json").write_text(
                json.dumps({"g_Obox_Pside_Lcenter": {"point": [100, 200]}}),
                encoding="utf-8",
            )
            (root / "skills_output.json").write_text(
                json.dumps(
                    [
                        {
                            "subtask_id": 1,
                            "task": "Press",
                            "controllers": [
                                {
                                    "controller": "ForceConstant(a_surface_normal(g_Obox_Pside_Lcenter), -5.0)",
                                    "offset_parameters": [],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            client = self.make_client(root)

            response = client.post("/api/script/generate")

            self.assertEqual(response.status_code, 200)
            self.assertTrue((root / "robot_script.py").exists())
            self.assertIn("script", response.json())

    def test_script_generation_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(Path(tmpdir))

            response = client.post("/api/script/generate")

            self.assertEqual(response.status_code, 400)

    def test_workflow_save_success(self):
        previous_camera_image_path = Config.CAMERA_IMAGE_PATH
        try:
            Config.CAMERA_IMAGE_PATH = "environment.jpg"
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                self.write_workflow_artifacts(root)
                client_service = GtaWebService(root)
                client_service._state["task"] = "Pick up the apple"
                app = create_app(service=client_service, initialize_on_startup=False)
                client = TestClient(app)

                response = client.post("/api/workflow/save")

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertTrue(payload["folder"].startswith("generated_plans/pick_up_the_apple_"))
                self.assertEqual(
                    sorted(payload["files"]),
                    sorted(
                        [
                            "annotated_keypoints.png",
                            "environment.jpg",
                            "keypoint_locations.json",
                            "resized_img.png",
                            "robot_script.py",
                            "skills_output.json",
                        ]
                    ),
                )
                save_dir = root / payload["folder"]
                self.assertTrue(save_dir.is_dir())
                for name in payload["files"]:
                    self.assertTrue((save_dir / name).is_file())
        finally:
            Config.CAMERA_IMAGE_PATH = previous_camera_image_path

    def test_workflow_save_missing_artifact_fails_without_partial_folder(self):
        previous_camera_image_path = Config.CAMERA_IMAGE_PATH
        try:
            Config.CAMERA_IMAGE_PATH = "environment.jpg"
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                self.write_workflow_artifacts(root)
                (root / "resized_img.png").unlink()
                client = self.make_client(root)

                response = client.post("/api/workflow/save")

                self.assertEqual(response.status_code, 400)
                self.assertIn("Missing workflow artifacts: resized_img.png", response.json()["detail"])
                self.assertFalse((root / "generated_plans").exists())
        finally:
            Config.CAMERA_IMAGE_PATH = previous_camera_image_path

    def test_workflow_save_uses_configured_environment_path(self):
        previous_camera_image_path = Config.CAMERA_IMAGE_PATH
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                Config.CAMERA_IMAGE_PATH = "captures/current_environment.jpg"
                environment_path = root / Config.CAMERA_IMAGE_PATH
                self.write_workflow_artifacts(root, environment_path=environment_path)
                client = self.make_client(root)

                response = client.post("/api/workflow/save")

                self.assertEqual(response.status_code, 200)
                save_dir = root / response.json()["folder"]
                self.assertEqual((save_dir / "environment.jpg").read_bytes(), b"environment")
        finally:
            Config.CAMERA_IMAGE_PATH = previous_camera_image_path


if __name__ == "__main__":
    unittest.main()
