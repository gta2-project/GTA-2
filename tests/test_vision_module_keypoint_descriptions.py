"""Tests for passing keypoint descriptions into VisionModule prompts."""

import unittest

from agents.vision_module import VisionModuleAgent


class VisionModuleKeypointDescriptionTests(unittest.TestCase):
    def make_agent_without_clients(self):
        agent = VisionModuleAgent.__new__(VisionModuleAgent)
        agent.name = "VisionModule"
        agent.keypoint_descriptions = {}
        return agent

    def test_extraction_preserves_first_nonempty_description(self):
        agent = self.make_agent_without_clients()
        keypoint_name = "g_Ocup_Phandle_Lcenter"
        skills = [
            {
                "keypoints": [
                    {"name": keypoint_name, "description": "  Center of the cup handle.  "},
                    {"name": "g_eef_TCP", "description": "Tool center point."},
                ]
            },
            {
                "keypoints": [
                    {"name": keypoint_name, "description": "A later duplicate description."},
                    {"name": "g_Obowl_Prim_Lcenter"},
                ]
            },
        ]

        keypoints = agent.extract_keypoints_from_skills(skills)

        self.assertIn(keypoint_name, keypoints)
        self.assertIn("g_Obowl_Prim_Lcenter", keypoints)
        self.assertNotIn("g_eef_TCP", keypoints)
        self.assertEqual(
            agent.keypoint_descriptions[keypoint_name],
            "Center of the cup handle.",
        )
        self.assertNotIn("g_Obowl_Prim_Lcenter", agent.keypoint_descriptions)

    def test_detection_prompt_keeps_labels_and_adds_descriptions(self):
        query_items = [
            {
                "label": "cup_handle_center",
                "description": "Center of the cup handle.",
            },
            {
                "label": "bowl_rim_center",
                "description": "",
            },
        ]

        prompt = VisionModuleAgent.build_keypoint_detection_prompt(query_items)

        self.assertIn('"label": "cup_handle_center"', prompt)
        self.assertIn('"description": "Center of the cup handle."', prompt)
        self.assertIn('"label": "bowl_rim_center"', prompt)
        self.assertIn("preserving these exact labels", prompt)


if __name__ == "__main__":
    unittest.main()
