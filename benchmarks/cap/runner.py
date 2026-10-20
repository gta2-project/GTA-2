"""End-to-end orchestration for Code-as-Policies benchmark trials."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .artifacts import TrialArtifacts
from .cartesian import CartesianConfig, CartesianExecutor
from .conditions import ApiKind, get_condition
from .context import PolicyContext
from .model import CapModelClient
from .perception import PerceptionSnapshot, VisionPerceptionLocalizer
from .prompts import build_system_prompt, build_user_prompt
from .runtime import (
    ControllerRobotRuntime,
    OfflineCartesianExecutor,
    OfflineControllerRuntime,
    execute_validated_policy,
)
from .validation import validate_policy


class BenchmarkConfigurationError(ValueError):
    pass


def load_task_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            document = yaml.safe_load(handle)
        else:
            document = json.load(handle)
    if not isinstance(document, dict):
        raise BenchmarkConfigurationError("Task manifest must contain a mapping")
    task = document.get("task")
    if not isinstance(task, str) or not task.strip():
        raise BenchmarkConfigurationError("Task manifest requires non-empty 'task'")
    task_id = document.get("id") or "cap_task"
    if not isinstance(task_id, str) or not task_id.strip():
        raise BenchmarkConfigurationError("Task manifest 'id' must be a non-empty string")
    document = dict(document)
    document["id"] = task_id.strip()
    document["task"] = task.strip()
    document["_manifest_path"] = str(path)
    document["_manifest_dir"] = str(path.parent)
    return document


def _resolve_path(value, manifest, required=False):
    if value is None:
        if required:
            raise BenchmarkConfigurationError("Required artifact path is missing")
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(manifest["_manifest_dir"]) / path
    path = path.resolve()
    if required and not path.is_file():
        raise BenchmarkConfigurationError(f"Artifact does not exist: {path}")
    return path


def _load_point_cloud(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.keys())
        if len(keys) != 1:
            raise BenchmarkConfigurationError("Point-cloud NPZ must contain exactly one array")
        return archive[keys[0]]
    raise BenchmarkConfigurationError("Point cloud must be a .npy or single-array .npz file")


def _initial_pose(manifest):
    pose = np.asarray(manifest.get("initial_pose", [0.0, -0.491, 0.487, -2.221, -2.221, 0.0]))
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise BenchmarkConfigurationError("initial_pose must contain six finite values")
    return pose.astype(float)


def _cartesian_config(manifest):
    config = CartesianConfig.from_environment()
    safety = manifest.get("safety", {}) or {}
    if not isinstance(safety, dict):
        raise BenchmarkConfigurationError("safety must be a mapping")
    changes = {}
    if "workspace_min" in safety:
        changes["workspace_min"] = tuple(float(value) for value in safety["workspace_min"])
    if "workspace_max" in safety:
        changes["workspace_max"] = tuple(float(value) for value in safety["workspace_max"])
    if "max_abs_force_n" in safety:
        changes["max_abs_force_n"] = float(safety["max_abs_force_n"])
    return dataclasses.replace(config, **changes) if changes else config


def _snapshot_for_execution(
    manifest,
    queries,
    image_path,
    perception_path=None,
    point_cloud_path=None,
    localizer=None,
):
    if perception_path is not None:
        fixture_path = Path(perception_path).resolve()
        if not fixture_path.is_file():
            raise BenchmarkConfigurationError(f"Artifact does not exist: {fixture_path}")
        snapshot = PerceptionSnapshot.from_file(fixture_path)
    elif manifest.get("perception"):
        snapshot = PerceptionSnapshot.from_file(
            _resolve_path(manifest["perception"], manifest, required=True)
        )
    elif not queries:
        snapshot = PerceptionSnapshot(points={}, normals={}, pixels={})
    else:
        if point_cloud_path is not None:
            cloud_path = Path(point_cloud_path).resolve()
            if not cloud_path.is_file():
                raise BenchmarkConfigurationError(f"Artifact does not exist: {cloud_path}")
        elif manifest.get("point_cloud"):
            cloud_path = _resolve_path(manifest["point_cloud"], manifest, required=True)
        else:
            raise BenchmarkConfigurationError(
                "Execution with semantic queries requires a perception fixture or point cloud"
            )
        cloud = _load_point_cloud(cloud_path)
        localizer = localizer or VisionPerceptionLocalizer()
        snapshot = localizer.localize(
            queries=queries,
            image_path=image_path,
            point_cloud=cloud,
            point_cloud_frame=manifest.get("point_cloud_frame", "base"),
            base_from_camera=manifest.get("base_from_camera"),
        )
    missing = sorted(set(queries) - set(snapshot.points))
    if missing:
        raise BenchmarkConfigurationError(
            f"Perception snapshot is missing policy queries: {', '.join(missing)}"
        )
    return snapshot


def _robot_controller_runtime():
    try:
        import rospy
        from gta.error_projection import project_errors
        from gta.skill_execution import (
            end_skill,
            initialize_ros_publishers,
            start_skill,
            update_control_commands,
        )
    except ImportError as exc:
        raise RuntimeError("Controller robot mode requires ROS") from exc
    if not rospy.core.is_initialized():
        rospy.init_node("cap_controller_benchmark", anonymous=True)
    if not initialize_ros_publishers():
        raise RuntimeError("Could not initialize controller benchmark ROS publishers")
    return ControllerRobotRuntime(
        start_skill=start_skill,
        end_skill=end_skill,
        update_control_commands=update_control_commands,
        error_projector=project_errors,
        shutdown_requested=rospy.is_shutdown,
    )


def _capture_live_point_cloud():
    from config import Config
    from scene.camera import Camera

    camera = Camera(
        camera_node=Config.CAMERA_NODE,
        camera_model=Config.CAMERA_MODEL,
        rosmaster_ip=Config.ROSMASTER_IP,
        rosmaster_port=Config.ROSMASTER_PORT,
        camera_backend=Config.CAMERA_BACKEND,
        camera_image_path=Config.CAMERA_IMAGE_PATH,
    )
    cloud = np.asarray(camera.get_point_cloud(), dtype=float)
    if cloud.ndim not in {2, 3} or cloud.shape[-1] != 3:
        raise RuntimeError(f"Camera returned invalid point cloud shape {cloud.shape}")
    return cloud


def run_trial(
    condition_name: str,
    task_manifest: Path,
    image_path: Optional[Path],
    mode: str,
    output_root: Path,
    policy_file: Optional[Path] = None,
    perception_path: Optional[Path] = None,
    point_cloud_path: Optional[Path] = None,
    approve_robot: bool = False,
    model_client=None,
    localizer=None,
):
    if mode not in {"generate", "offline", "robot"}:
        raise BenchmarkConfigurationError("mode must be generate, offline, or robot")
    if mode == "robot" and not approve_robot:
        raise BenchmarkConfigurationError(
            "Robot mode requires --approve-robot after checking the cell and correct server state"
        )

    condition = get_condition(condition_name)
    manifest = load_task_manifest(task_manifest)
    if image_path is not None:
        frozen_image = Path(image_path).resolve()
        if not frozen_image.is_file():
            raise BenchmarkConfigurationError(f"Artifact does not exist: {frozen_image}")
    else:
        frozen_image = _resolve_path(manifest.get("image"), manifest, required=True)
    artifacts = TrialArtifacts(output_root, manifest["id"], condition.name.value)
    artifacts.copy(frozen_image, "scene_image" + frozen_image.suffix.lower())
    artifacts.write_json(
        "task_manifest.json",
        {key: value for key, value in manifest.items() if not key.startswith("_")},
    )

    result = {
        "task_id": manifest["id"],
        "condition": condition.name.value,
        "api_kind": condition.api_kind.value,
        "few_shot": condition.few_shot,
        "mode": mode,
        "status": "running",
        "artifact_dir": str(artifacts.path),
    }
    artifacts.event({"event": "trial_started", **result})
    try:
        system_prompt = build_system_prompt(condition)
        user_prompt = build_user_prompt(manifest["task"], manifest.get("success_criteria"))
        artifacts.write_text("system_prompt.txt", system_prompt + "\n")
        artifacts.write_text("user_prompt.txt", user_prompt + "\n")

        if policy_file is not None:
            source = Path(policy_file).read_text(encoding="utf-8")
            model_metadata = {"source": "policy_file", "path": str(Path(policy_file).resolve())}
        else:
            model_client = model_client or CapModelClient()
            generation_started = time.monotonic()
            response = model_client.generate(system_prompt, user_prompt, frozen_image)
            source = response.text
            model_metadata = dict(response.metadata)
            model_metadata["latency_s"] = time.monotonic() - generation_started
        artifacts.write_text("raw_response.txt", source)
        artifacts.write_json("model_metadata.json", model_metadata)

        validated = validate_policy(source, condition.api_kind)
        artifacts.write_text("policy.py", validated.source + "\n")
        artifacts.write_json("keypoint_queries.json", validated.keypoint_queries)
        artifacts.event(
            {
                "event": "policy_parsed",
                "keypoint_query_count": len(validated.keypoint_queries),
            }
        )

        if mode == "generate":
            result.update(
                {
                    "status": "generated",
                    "keypoint_query_count": len(validated.keypoint_queries),
                    "model": model_metadata,
                }
            )
            artifacts.write_json("result.json", result)
            artifacts.event({"event": "trial_completed", "status": result["status"]})
            return result

        effective_point_cloud_path = point_cloud_path
        if (
            mode == "robot"
            and validated.keypoint_queries
            and perception_path is None
            and not manifest.get("perception")
            and point_cloud_path is None
            and not manifest.get("point_cloud")
        ):
            live_cloud = _capture_live_point_cloud()
            effective_point_cloud_path = artifacts.path / "point_cloud.npy"
            np.save(effective_point_cloud_path, live_cloud, allow_pickle=False)
            artifacts.event(
                {
                    "event": "point_cloud_frozen",
                    "path": str(effective_point_cloud_path),
                    "shape": list(live_cloud.shape),
                }
            )

        snapshot = _snapshot_for_execution(
            manifest=manifest,
            queries=validated.keypoint_queries,
            image_path=frozen_image,
            perception_path=perception_path,
            point_cloud_path=effective_point_cloud_path,
            localizer=localizer,
        )
        artifacts.write_json("perception_snapshot.json", snapshot.as_dict())
        artifacts.event(
            {"event": "perception_frozen", "aliases": sorted(snapshot.points)}
        )

        if mode == "offline":
            if condition.api_kind == ApiKind.CARTESIAN:
                runtime = OfflineCartesianExecutor(
                    _initial_pose(manifest), event_sink=artifacts.event
                )
                context = PolicyContext(
                    snapshot.points,
                    snapshot.normals,
                    current_pose_provider=runtime.current_pose,
                )
                execute_validated_policy(
                    validated,
                    condition.api_kind,
                    context,
                    cartesian_executor=runtime,
                    event_sink=artifacts.event,
                )
                result["primitive_calls"] = len(runtime.calls)
            else:
                pose = _initial_pose(manifest)
                runtime = OfflineControllerRuntime(event_sink=artifacts.event)
                context = PolicyContext(
                    snapshot.points,
                    snapshot.normals,
                    current_pose_provider=lambda: pose.copy(),
                    query_objects={
                        alias: query["object"]
                        for alias, query in validated.keypoint_queries.items()
                    },
                )
                execute_validated_policy(
                    validated,
                    condition.api_kind,
                    context,
                    controller_runtime=runtime,
                    event_sink=artifacts.event,
                )
                result["controller_phases"] = len(runtime.phases)
        elif condition.api_kind == ApiKind.CARTESIAN:
            # The operator approval also certifies that the ROS RTDE robot server
            # has been stopped; RTDE Control must have one owner.
            with CartesianExecutor(
                config=_cartesian_config(manifest), event_sink=artifacts.event
            ) as runtime:
                context = PolicyContext(
                    snapshot.points,
                    snapshot.normals,
                    current_pose_provider=runtime.current_pose,
                )
                execute_validated_policy(
                    validated,
                    condition.api_kind,
                    context,
                    cartesian_executor=runtime,
                    event_sink=artifacts.event,
                )
        else:
            from scene.robot import RobotState

            context = PolicyContext(
                snapshot.points,
                snapshot.normals,
                current_pose_provider=RobotState().get_tcp_pose,
                query_objects={
                    alias: query["object"]
                    for alias, query in validated.keypoint_queries.items()
                },
            )
            execute_validated_policy(
                validated,
                condition.api_kind,
                context,
                controller_runtime=_robot_controller_runtime(),
                event_sink=artifacts.event,
            )

        result.update(
            {
                "status": "offline_completed" if mode == "offline" else "robot_completed",
                "model": model_metadata,
                "keypoint_query_count": len(validated.keypoint_queries),
            }
        )
        artifacts.write_json("result.json", result)
        artifacts.event({"event": "trial_completed", "status": result["status"]})
        return result
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})
        artifacts.write_json("result.json", result)
        artifacts.event({"event": "trial_failed", "error": str(exc)})
        raise
