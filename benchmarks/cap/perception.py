"""Frozen-image semantic localization for CaP-selected keypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

import numpy as np
import open3d as o3d
import yaml
from PIL import Image, ImageOps


SURFACE_NORMAL_KNN = 30

class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerceptionSnapshot:
    points: Mapping[str, np.ndarray]
    normals: Mapping[str, np.ndarray]
    pixels: Mapping[str, tuple]

    def as_dict(self):
        return {
            "points": {name: value.tolist() for name, value in self.points.items()},
            "normals": {name: value.tolist() for name, value in self.normals.items()},
            "pixels": {name: list(value) for name, value in self.pixels.items()},
        }

    @classmethod
    def from_file(cls, path: Path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() in {".yaml", ".yml"}:
                document = yaml.safe_load(handle)
            else:
                document = json.load(handle)
        if not isinstance(document, dict):
            raise PerceptionError("Perception fixture must contain a mapping")
        points = _vector_mapping(document.get("points", {}), "points", normalize=False)
        normals = _vector_mapping(document.get("normals", {}), "normals", normalize=True)
        pixels = {}
        for name, pixel in document.get("pixels", {}).items():
            value = np.asarray(pixel, dtype=int)
            if value.shape != (2,):
                raise PerceptionError(f"Pixel {name!r} must contain [x, y]")
            pixels[str(name)] = (int(value[0]), int(value[1]))
        return cls(points=points, normals=normals, pixels=pixels)


def _vector_mapping(raw, name: str, normalize: bool):
    if not isinstance(raw, dict):
        raise PerceptionError(f"{name} must be a mapping")
    result = {}
    for key, item in raw.items():
        value = np.asarray(item, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise PerceptionError(f"{name}.{key} must be a finite 3D vector")
        if normalize:
            length = float(np.linalg.norm(value))
            if length < 1e-9:
                raise PerceptionError(f"{name}.{key} must be non-zero")
            value = value / length
        result[str(key)] = value
    return result


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```json"):
        value = value[len("```json") :].strip()
    elif value.startswith("```"):
        value = value[len("```") :].strip()
    if value.endswith("```"):
        value = value[: -len("```")].strip()
    return value


class VisionPerceptionLocalizer:
    """Use the existing Vision Module model/prompt on one frozen image."""

    def __init__(self, detector: Optional[Callable[[str, np.ndarray], str]] = None):
        self._detector = detector
        self._agent = None

    def _call_detector(self, prompt: str, image: np.ndarray) -> str:
        if self._detector is not None:
            return self._detector(prompt, image)
        if self._agent is None:
            from agents.vision_module import VisionModuleAgent
            from communication import MessageBus

            self._agent = VisionModuleAgent(MessageBus())
        return self._agent.call_keypoint_detector(prompt, image)

    @staticmethod
    def _build_detection_prompt(query_items):
        # Kept byte-for-byte equivalent to
        # VisionModuleAgent.build_keypoint_detection_prompt so injected/offline
        # runs do not have to import the configured LLM clients.
        return f"""
Detect these requested keypoints in the current image using their descriptions:
{json.dumps(query_items, indent=2)}

Return one point for each requested keypoint, preserving these exact labels.
"""

    def localize(
        self,
        queries: Mapping[str, Mapping[str, str]],
        image_path: Path,
        point_cloud,
        point_cloud_frame: str = "base",
        base_from_camera=None,
    ) -> PerceptionSnapshot:
        image_path = Path(image_path)
        try:
            original = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise PerceptionError(f"Unable to open frozen image {image_path}: {exc}") from exc
        padded = ImageOps.pad(original, (1600, 1600), color=(255, 255, 255))
        image_np = np.asarray(padded)
        query_items = [
            {
                "label": alias,
                "description": (
                    f"Object: {query['object']}. Keypoint: {query['description']}"
                ),
            }
            for alias, query in queries.items()
        ]
        prompt = self._build_detection_prompt(query_items)
        try:
            print(
                f"[CaP Perception] Image loaded; waiting for Gemini localization "
                f"response for {len(query_items)} keypoint(s)...",
                flush=True,
            )
            raw_response = self._call_detector(prompt, image_np)
            print("[CaP Perception] Gemini localization response received.", flush=True)
            response_points = json.loads(_strip_json_fence(raw_response))
        except Exception as exc:
            raise PerceptionError(f"Vision Module localization failed: {exc}") from exc
        if not isinstance(response_points, list):
            raise PerceptionError("Vision Module response must be a JSON array")

        ordered = self._match_response(query_items, response_points)
        organized = self._organized_cloud(point_cloud, original.height, original.width)
        base_cloud, camera_position = self._to_base_cloud(
            organized,
            point_cloud_frame,
            base_from_camera,
        )
        valid_flat = base_cloud.reshape(-1, 3)
        valid_mask = np.all(np.isfinite(valid_flat), axis=1)
        valid_mask &= np.linalg.norm(valid_flat, axis=1) > 1e-9
        valid_points = valid_flat[valid_mask]
        if len(valid_points) < 3:
            raise PerceptionError("Point cloud has too few valid points")
        estimated_normals = self._estimate_cloud_normals(valid_points)

        points: Dict[str, np.ndarray] = {}
        normals: Dict[str, np.ndarray] = {}
        pixels: Dict[str, tuple] = {}
        for alias, response_point in ordered.items():
            padded_x, padded_y = self._normalized_response_pixel(response_point, padded.size)
            pixel = self._unpad_pixel(
                padded_x,
                padded_y,
                original.width,
                original.height,
                padded.width,
                padded.height,
            )
            x, y = pixel
            point = np.asarray(base_cloud[y, x], dtype=float)
            if not np.all(np.isfinite(point)) or np.linalg.norm(point) < 1e-9:
                raise PerceptionError(f"Point cloud is invalid at localized pixel for {alias!r}")
            normal = self._estimate_normal(
                point,
                valid_points,
                estimated_normals,
                camera_position,
            )
            points[alias] = point
            normals[alias] = normal
            pixels[alias] = pixel
        return PerceptionSnapshot(points=points, normals=normals, pixels=pixels)

    @staticmethod
    def _match_response(query_items, response_points):
        result = {}
        used = set()
        for index, query in enumerate(query_items):
            alias = query["label"]
            match = next(
                (
                    candidate_index
                    for candidate_index, point in enumerate(response_points)
                    if candidate_index not in used
                    and isinstance(point, dict)
                    and point.get("label") == alias
                ),
                None,
            )
            # Preserve the current Vision Module's response-order fallback.
            if match is None and index < len(response_points) and index not in used:
                match = index
            if match is None:
                raise PerceptionError(f"Vision Module did not return keypoint {alias!r}")
            point = response_points[match]
            if not isinstance(point, dict) or "point" not in point:
                raise PerceptionError(f"Vision response for {alias!r} has no point")
            used.add(match)
            result[alias] = point
        return result

    @staticmethod
    def _normalized_response_pixel(response_point, padded_size):
        value = np.asarray(response_point["point"], dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise PerceptionError("Vision response point must be finite [y, x]")
        width, height = padded_size
        x = int(np.clip(value[1] * width / 1000.0, 0, width - 1))
        y = int(np.clip(value[0] * height / 1000.0, 0, height - 1))
        return x, y

    @staticmethod
    def _unpad_pixel(x, y, original_width, original_height, padded_width, padded_height):
        scale = min(padded_width / original_width, padded_height / original_height)
        resized_width = original_width * scale
        resized_height = original_height * scale
        offset_x = (padded_width - resized_width) / 2.0
        offset_y = (padded_height - resized_height) / 2.0
        original_x = int(round((x - offset_x) / scale))
        original_y = int(round((y - offset_y) / scale))
        if not (0 <= original_x < original_width and 0 <= original_y < original_height):
            raise PerceptionError("Localized keypoint lies in padded image area")
        return original_x, original_y

    @staticmethod
    def _organized_cloud(point_cloud, height, width):
        cloud = np.asarray(point_cloud, dtype=float)
        if cloud.shape == (height, width, 3):
            return cloud.copy()
        if cloud.shape == (height * width, 3):
            return cloud.reshape(height, width, 3).copy()
        raise PerceptionError(
            f"Point cloud shape {cloud.shape} does not match image {width}x{height}"
        )

    @staticmethod
    def _to_base_cloud(cloud, frame: str, base_from_camera):
        frame = str(frame).lower()
        if base_from_camera is None:
            from config import Config

            base_from_camera = Config.T_BASE_CAM
        transform = np.asarray(
            base_from_camera,
            dtype=float,
        )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise PerceptionError("base_from_camera must be a finite 4x4 transform")
        camera_position = transform[:3, 3]
        if frame == "base":
            return cloud, camera_position
        if frame != "camera":
            raise PerceptionError("point_cloud_frame must be 'base' or 'camera'")
        flat = cloud.reshape(-1, 3)
        transformed = (transform[:3, :3] @ flat.T).T + transform[:3, 3]
        return transformed.reshape(cloud.shape), camera_position

    @staticmethod
    def _estimate_cloud_normals(valid_points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(valid_points)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=SURFACE_NORMAL_KNN)
        )
        normals = np.asarray(cloud.normals, dtype=float).copy()
        if normals.shape != valid_points.shape or not np.all(np.isfinite(normals)):
            raise PerceptionError("Open3D returned invalid surface normals")
        return normals

    @staticmethod
    def _estimate_normal(point, valid_points, estimated_normals, camera_position):
        # Match GTA's a_surface_normal(): select the Open3D normal belonging to
        # the point-cloud sample nearest to the localized 3D keypoint.
        distances = np.linalg.norm(valid_points - point, axis=1)
        nearest_index = int(np.argmin(distances))
        normal = estimated_normals[nearest_index]
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            raise PerceptionError("Could not estimate a stable surface normal")
        normal = normal / length
        if np.dot(normal, camera_position - point) < 0:
            normal = -normal
        return normal
