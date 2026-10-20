"""Hardware-free integration tests for the CaP benchmark conditions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SciPyRotation

from benchmarks.cap.conditions import CONDITIONS
from benchmarks.cap.model import ModelResponse
from benchmarks.cap.prompts import build_system_prompt
from benchmarks.cap.runner import BenchmarkConfigurationError, run_trial


CARTESIAN_POLICY = '''
KEYPOINT_QUERIES = {
    "target": {"object": "target", "description": "center of the target"}
}
def policy(ctx):
    point = ctx.point("target")
    pose = ctx.current_pose()
    move_to(point[0], point[1], point[2], pose[3], pose[4], pose[5])
    gripper("close")
'''

CONTROLLER_POLICY = '''
KEYPOINT_QUERIES = {
    "target": {"object": "target", "description": "center of the target"}
}
def policy(ctx):
    point = ctx.point("target")
    execute_subtask([PosAlign(ctx.tcp_position, point, normalize(point - ctx.tcp_position()), 0.0)])
'''

ATTACHMENT_CONTROLLER_POLICY = '''
KEYPOINT_QUERIES = {
    "target": {"object": "tool", "description": "tracked point on the tool"}
}
def policy(ctx):
    tool = ctx.object_for("target")
    execute_subtask([GripperControl("close", 0.0, tool)])
    execute_subtask([GripperControl("open", 0.0, tool)])
'''

ROTATION_CARTESIAN_POLICY = '''
KEYPOINT_QUERIES = {
    "target": {"object": "target", "description": "center of the target"}
}
def policy(ctx):
    point = ctx.point("target")
    pose = ctx.current_pose()
    yaw = R.from_rotvec([0.0, 0.0, math.radians(30.0)])
    target_rotation = (yaw * R.from_rotvec(pose[3:6])).as_rotvec()
    move_to(point[0], point[1], point[2], target_rotation[0], target_rotation[1], target_rotation[2])
'''

FULL_BUILTINS_CARTESIAN_POLICY = '''
import itertools

KEYPOINT_QUERIES = {
    "target": {"object": "target", "description": "center of the target"}
}
def policy(ctx):
    point = ctx.point("target")
    pose = ctx.current_pose()
    for index, scale in enumerate(itertools.repeat(1.0, 2)):
        target = point * scale
        move_to(target[0], target[1], target[2], pose[3], pose[4], pose[5])
'''


class FakeModel:
    def __init__(self, source):
        self.source = source
        self.calls = []

    def generate(self, system_prompt, user_prompt, image_path):
        self.calls.append((system_prompt, user_prompt, Path(image_path)))
        return ModelResponse(
            text=self.source,
            metadata={"provider": "fake", "requested_model": "same-model"},
        )


class CapRunnerTests(unittest.TestCase):
    def make_files(self, root):
        image = root / "scene.png"
        Image.new("RGB", (8, 8), color=(100, 100, 100)).save(image)
        perception = root / "perception.json"
        perception.write_text(
            json.dumps(
                {
                    "points": {"target": [0.1, -0.5, 0.35]},
                    "normals": {"target": [0.0, 0.0, 1.0]},
                    "pixels": {"target": [4, 4]},
                }
            ),
            encoding="utf-8",
        )
        manifest = root / "task.yaml"
        manifest.write_text(
            "id: unit_task\n"
            "task: Touch the target.\n"
            "image: scene.png\n"
            "perception: perception.json\n"
            "initial_pose: [0.0, -0.5, 0.4, 0.0, 0.0, 0.0]\n",
            encoding="utf-8",
        )
        return image, perception, manifest

    def test_all_conditions_have_distinct_prompt_settings(self):
        self.assertEqual(set(CONDITIONS), {
            "controller_zeroshot",
            "controller_fewshot",
            "cartesian_zeroshot",
            "cartesian_fewshot",
        })
        zero = build_system_prompt(CONDITIONS["cartesian_zeroshot"])
        few = build_system_prompt(CONDITIONS["cartesian_fewshot"])
        self.assertNotIn("Three independent examples follow", zero)
        self.assertIn("Three independent examples follow", few)
        self.assertIn("move_force(x, y, z, rx, ry, rz, fx, fy, fz, fn)", zero)
        self.assertIn("Rotation as R are available without imports", zero)
        self.assertNotIn("R.align_vectors", zero)
        self.assertIn(
            'R.align_vectors([desired_direction], [ctx.tcp_axis("x")])[0]',
            few,
        )
        self.assertNotIn("math.radians(30.0)", few)

    def test_cartesian_offline_trial_records_primitives_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, manifest = self.make_files(root)
            policy = root / "policy.py"
            policy.write_text(CARTESIAN_POLICY, encoding="utf-8")
            result = run_trial(
                "cartesian_zeroshot",
                manifest,
                image_path=None,
                mode="offline",
                output_root=root / "runs",
                policy_file=policy,
            )
            self.assertEqual(result["status"], "offline_completed")
            self.assertEqual(result["primitive_calls"], 2)
            artifact_dir = Path(result["artifact_dir"])
            for name in (
                "policy.py",
                "keypoint_queries.json",
                "perception_snapshot.json",
                "trace.jsonl",
                "result.json",
            ):
                self.assertTrue((artifact_dir / name).is_file(), name)

    def test_cartesian_policy_executes_with_full_python_builtins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, manifest = self.make_files(root)
            policy = root / "full_builtins_policy.py"
            policy.write_text(FULL_BUILTINS_CARTESIAN_POLICY, encoding="utf-8")

            result = run_trial(
                "cartesian_zeroshot",
                manifest,
                image_path=None,
                mode="offline",
                output_root=root / "runs",
                policy_file=policy,
            )

            self.assertEqual(result["status"], "offline_completed")
            self.assertEqual(result["primitive_calls"], 2)

    def test_cartesian_runtime_composes_base_yaw(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, manifest = self.make_files(root)
            initial_rotation = np.array([-0.02, -3.14, 0.0])
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "initial_pose: [0.0, -0.5, 0.4, 0.0, 0.0, 0.0]",
                    "initial_pose: [0.0, -0.5, 0.4, -0.02, -3.14, 0.0]",
                ),
                encoding="utf-8",
            )
            policy = root / "rotation_policy.py"
            policy.write_text(ROTATION_CARTESIAN_POLICY, encoding="utf-8")
            result = run_trial(
                "cartesian_zeroshot",
                manifest,
                image_path=None,
                mode="offline",
                output_root=root / "runs",
                policy_file=policy,
            )
            events = [
                json.loads(line)
                for line in (Path(result["artifact_dir"]) / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            move = next(
                event
                for event in events
                if event.get("event") == "offline_cartesian_call"
                and event.get("primitive") == "move_to"
            )
            expected = (
                SciPyRotation.from_rotvec([0.0, 0.0, np.deg2rad(30.0)])
                * SciPyRotation.from_rotvec(initial_rotation)
            ).as_rotvec()
            np.testing.assert_allclose(move["target_pose"][3:], expected, atol=1e-12)

    def test_controller_offline_trial_uses_controller_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, manifest = self.make_files(root)
            policy = root / "policy.py"
            policy.write_text(CONTROLLER_POLICY, encoding="utf-8")
            result = run_trial(
                "controller_zeroshot",
                manifest,
                image_path=None,
                mode="offline",
                output_root=root / "runs",
                policy_file=policy,
            )
            self.assertEqual(result["status"], "offline_completed")
            self.assertEqual(result["controller_phases"], 1)

    def test_controller_trial_builds_objects_from_its_own_query_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, manifest = self.make_files(root)
            policy = root / "attachment_policy.py"
            policy.write_text(ATTACHMENT_CONTROLLER_POLICY, encoding="utf-8")

            result = run_trial(
                "controller_zeroshot",
                manifest,
                image_path=None,
                mode="offline",
                output_root=root / "runs",
                policy_file=policy,
            )

            self.assertEqual(result["status"], "offline_completed")
            self.assertEqual(result["controller_phases"], 2)
            events = [
                json.loads(line)
                for line in (Path(result["artifact_dir"]) / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            transitions = [
                event["object_transitions"]
                for event in events
                if event.get("event") == "offline_controller_phase"
            ]
            self.assertEqual(
                transitions,
                [
                    [{"object": "tool", "frame": "robot"}],
                    [{"object": "tool", "frame": "world"}],
                ],
            )

    def test_generation_uses_frozen_image_once_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image, _, manifest = self.make_files(root)
            model = FakeModel(CARTESIAN_POLICY)
            result = run_trial(
                "cartesian_fewshot",
                manifest,
                image_path=None,
                mode="generate",
                output_root=root / "runs",
                model_client=model,
            )
            self.assertEqual(result["status"], "generated")
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(model.calls[0][2], image.resolve())
            metadata = json.loads(
                (Path(result["artifact_dir"]) / "model_metadata.json").read_text()
            )
            self.assertEqual(metadata["requested_model"], "same-model")

    def test_robot_mode_requires_explicit_approval_before_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, manifest = self.make_files(root)
            with self.assertRaises(BenchmarkConfigurationError):
                run_trial(
                    "cartesian_zeroshot",
                    manifest,
                    image_path=None,
                    mode="robot",
                    output_root=root / "runs",
                    policy_file=root / "missing.py",
                    approve_robot=False,
                )

    def test_cartesian_executor_does_not_use_gta_projection(self):
        package = Path(__file__).resolve().parents[1] / "benchmarks" / "cap"
        source = (package / "cartesian.py").read_text(encoding="utf-8")
        self.assertNotIn("project_errors", source)
        self.assertNotIn("error_projection", source)


if __name__ == "__main__":
    unittest.main()
