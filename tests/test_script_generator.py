"""Tests for backend robot script generation."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from gta.script_generator import generate_robot_script


class ScriptGeneratorTests(unittest.TestCase):
    def test_generates_surface_normal_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keypoints_path = root / "keypoint_locations.json"
            skills_path = root / "skills_output.json"
            output_path = root / "robot_script.py"

            keypoints_path.write_text(
                json.dumps(
                    {
                        "g_Obox_Pside_Lcenter": {"point": [100, 200]},
                        "g_Obox_Ptop_Lcenter": {"point": [150, 180]},
                    }
                ),
                encoding="utf-8",
            )
            skills_path.write_text(
                json.dumps(
                    [
                        {
                            "subtask_id": 1,
                            "task": "Press on box side",
                            "controllers": [
                                {
                                    "priority": 1,
                                    "controller": "ForceConstant(a_surface_normal(g_Obox_Pside_Lcenter), -5.0)",
                                    "offset_parameters": [],
                                }
                            ],
                        },
                        {
                            "subtask_id": 2,
                            "task": "Move above box",
                            "controllers": [
                                {
                                    "priority": 1,
                                    "controller": "PosAlign(g_eef_TCP, g_Obox_Ptop_Lcenter, a_surface_normal(g_Obox_Ptop_Lcenter), offset_height)",
                                    "offset_parameters": [
                                        {"name": "offset_height", "value": 0.05}
                                    ],
                                }
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result = generate_robot_script(keypoints_path, skills_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(result["keypoint_count"], 2)
            self.assertIn(
                "box.axes['side_center_surface_normal'] = a_surface_normal(box.keypoints['side_center'], pcd)",
                result["script"],
            )
            self.assertIn("ForceConstant(box.get_a('side_center_surface_normal'), -5.0)", result["script"])
            self.assertIn("'height': 0.05", result["script"])

    def test_generates_dynamic_pos_dmp_spiral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            keypoints_path = root / "keypoint_locations.json"
            skills_path = root / "skills_output.json"
            output_path = root / "robot_script.py"

            keypoints_path.write_text(
                json.dumps(
                    {
                        "g_Oplate_Psurface_Lcenter": {"point": [100, 200]},
                    }
                ),
                encoding="utf-8",
            )
            skills_path.write_text(
                json.dumps(
                    [
                        {
                            "subtask_id": 1,
                            "task": "Trace a learned spiral",
                            "controllers": [
                                {
                                    "priority": 1,
                                    "controller": (
                                        "PosDMPSpiral(g_eef_TCP, "
                                        "a_surface_normal(g_Oplate_Psurface_Lcenter), "
                                        "offset_spiral_radius)"
                                    ),
                                    "offset_parameters": [
                                        {
                                            "name": "offset_spiral_radius",
                                            "value": 0.10,
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = generate_robot_script(keypoints_path, skills_path, output_path)

            self.assertIn(
                "plate.axes['surface_center_surface_normal'] = "
                "a_surface_normal(plate.keypoints['surface_center'], pcd)",
                result["script"],
            )
            self.assertIn(
                "PosDMPSpiral(lambda: eef.get_k('TCP'), "
                "plate.get_a('surface_center_surface_normal'), "
                "offsets['spiral_radius'])",
                result["script"],
            )
            self.assertIn("'spiral_radius': 0.1", result["script"])


@pytest.mark.parametrize('camera_backend', ['file', 'ros1', 'ros2'])
@pytest.mark.parametrize('gta_backend', ['ros1', 'ros2'])
def test_generated_script_initializes_gta_independently_of_camera(tmp_path, monkeypatch, camera_backend, gta_backend):
    keypoints = tmp_path / 'keypoints.json'
    skills = tmp_path / 'skills.json'
    keypoints.write_text('{}')
    skills.write_text('[]')
    script = generate_robot_script(keypoints, skills, tmp_path / 'script.py')['script']
    assert 'CAMERA_SOURCE' not in script
    assert 'CAMERA_BACKEND' not in script
    tree = ast.parse(script)
    initialize_branch = next(node for node in tree.body if isinstance(node, ast.If))
    assert isinstance(initialize_branch.test, ast.Compare)
    assert initialize_branch.test.left.attr == 'GTA_BACKEND'
    # Execute only transport initialization with fakes, never the generated motion.
    rospy = ModuleType('rospy')
    rospy.init_node = Mock()
    monkeypatch.setitem(sys.modules, 'rospy', rospy)
    initialize = Mock(return_value=True)
    namespace = {'Config': SimpleNamespace(GTA_BACKEND=gta_backend, CAMERA_BACKEND=camera_backend),
                 'initialize_ros_publishers': initialize}
    exec(compile(ast.Module(body=[initialize_branch], type_ignores=[]), '<initialization>', 'exec'), namespace)
    initialize.assert_called_once_with()
    assert namespace['ROS_AVAILABLE'] is True
    if gta_backend == 'ros1':
        rospy.init_node.assert_called_once_with('skill_executor', anonymous=True)
    else:
        rospy.init_node.assert_not_called()


if __name__ == "__main__":
    unittest.main()
