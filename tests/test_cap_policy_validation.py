"""Tests for the minimal trusted-code CaP parser."""

import unittest

from benchmarks.cap.conditions import ApiKind
from benchmarks.cap.validation import PolicyValidationError, validate_policy


class PolicyParsingTests(unittest.TestCase):
    def test_extracts_literal_keypoint_queries(self):
        source = '''
KEYPOINT_QUERIES = {
    "target": {"object": " button ", "description": " visible face "}
}
def policy(ctx):
    return ctx.point("target")
'''
        parsed = validate_policy(source, ApiKind.CARTESIAN)
        self.assertEqual(
            parsed.keypoint_queries,
            {"target": {"object": "button", "description": "visible face"}},
        )

    def test_accepts_starred_unpacking_for_primitive_calls(self):
        source = '''
KEYPOINT_QUERIES = {}
def policy(ctx):
    pose = ctx.current_pose()
    move_to(*pose)
    values = [*pose[:3], *pose[3:]]
    move_force(*values, 0.0, 0.0, 1.0, -5.0)
'''
        validate_policy(source, ApiKind.CARTESIAN)

    def test_accepts_unrestricted_python_syntax_without_ast_allowlist(self):
        source = '''
import os

KEYPOINT_QUERIES = {}

def helper(value):
    return {index: item for index, item in enumerate(value)}

def policy(ctx):
    pose = ctx.current_pose()
    pose[3:6] = R.from_rotvec([0.0, 0.0, 0.2]).as_rotvec()
    matrix = np.eye(3)
    matrix[:, 2] = np.array([0.0, 0.0, 1.0])
    while False:
        pass
    return helper(pose)
'''
        validate_policy(source, ApiKind.CARTESIAN)
        validate_policy(source, ApiKind.CONTROLLER)

    def test_api_kind_no_longer_restricts_calls(self):
        cartesian_source = '''
KEYPOINT_QUERIES = {}
def policy(ctx):
    execute_subtask([])
'''
        controller_source = '''
KEYPOINT_QUERIES = {}
def policy(ctx):
    move_to(0, 0, 0, 0, 0, 0)
'''
        validate_policy(cartesian_source, ApiKind.CARTESIAN)
        validate_policy(controller_source, ApiKind.CONTROLLER)

    def test_still_rejects_invalid_python(self):
        with self.assertRaises(PolicyValidationError):
            validate_policy(
                "KEYPOINT_QUERIES = {}\ndef policy(ctx)\n    pass\n",
                ApiKind.CARTESIAN,
            )

    def test_still_requires_runtime_interface(self):
        with self.assertRaises(PolicyValidationError):
            validate_policy("def policy(ctx):\n    pass\n", ApiKind.CARTESIAN)
        with self.assertRaises(PolicyValidationError):
            validate_policy(
                "KEYPOINT_QUERIES = {}\ndef policy():\n    pass\n",
                ApiKind.CARTESIAN,
            )
        with self.assertRaises(PolicyValidationError):
            validate_policy(
                "KEYPOINT_QUERIES = make_queries()\ndef policy(ctx):\n    pass\n",
                ApiKind.CARTESIAN,
            )


if __name__ == "__main__":
    unittest.main()
