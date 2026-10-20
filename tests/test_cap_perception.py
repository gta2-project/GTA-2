"""Offline frozen-image/point-cloud tests for the CaP perception adapter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.cap.perception import SURFACE_NORMAL_KNN, VisionPerceptionLocalizer


class CapPerceptionTests(unittest.TestCase):
    def test_surface_normals_use_30_point_knn(self):
        self.assertEqual(SURFACE_NORMAL_KNN, 30)

    def test_localizes_point_and_normal_from_same_frozen_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "scene.png"
            Image.new("RGB", (4, 2), color=(20, 30, 40)).save(image_path)
            cloud = np.zeros((2, 4, 3), dtype=float)
            for y in range(2):
                for x in range(4):
                    cloud[y, x] = [x * 0.01, y * 0.01, 0.30]

            calls = []

            def detector(prompt, image):
                calls.append((prompt, image.copy()))
                return '[{"label": "mark", "point": [500, 500]}]'

            localizer = VisionPerceptionLocalizer(detector=detector)
            snapshot = localizer.localize(
                queries={
                    "mark": {
                        "object": "whiteboard",
                        "description": "center of the red mark",
                    }
                },
                image_path=image_path,
                point_cloud=cloud,
                point_cloud_frame="base",
                base_from_camera=np.eye(4),
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1].shape, (1600, 1600, 3))
            self.assertIn("center of the red mark", calls[0][0])
            self.assertEqual(snapshot.pixels["mark"], (2, 1))
            np.testing.assert_allclose(snapshot.points["mark"], [0.02, 0.01, 0.30])
            np.testing.assert_allclose(snapshot.normals["mark"], [0.0, 0.0, -1.0])

    def test_camera_frame_cloud_is_transformed_to_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "scene.png"
            Image.new("RGB", (3, 3), color=(0, 0, 0)).save(image_path)
            cloud = np.zeros((3, 3, 3), dtype=float)
            for y in range(3):
                for x in range(3):
                    cloud[y, x] = [x * 0.01, y * 0.01, 0.2]
            transform = np.eye(4)
            transform[:3, 3] = [0.1, -0.2, 0.3]
            localizer = VisionPerceptionLocalizer(
                detector=lambda _prompt, _image: '[{"label": "p", "point": [500, 500]}]'
            )
            snapshot = localizer.localize(
                {"p": {"object": "plane", "description": "center"}},
                image_path,
                cloud,
                point_cloud_frame="camera",
                base_from_camera=transform,
            )
            np.testing.assert_allclose(snapshot.points["p"], [0.12, -0.18, 0.5])


if __name__ == "__main__":
    unittest.main()
