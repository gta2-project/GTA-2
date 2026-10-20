"""Vision Module Agent - Detects keypoint locations in environment images using Gemini vision models."""
import json
import os
import re
from typing import Dict, List, Any

import cv2
from agents.base_agent import BaseAgent
from communication import MessageBus, Message
from config import Config
from agents.prompts.vision_module import SYSTEM_PROMPT
from PIL import Image, ImageOps
import numpy as np

from agents.vision_visualization import save_annotated_keypoints
from scene.camera import Camera


class VisionModuleAgent(BaseAgent):
    """Agent that uses Gemini vision models to detect keypoints in environment images."""
    
    def __init__(self, message_bus: MessageBus, environment_image_path: str = None):
        super().__init__(
            name="VisionModule",
            system_prompt=SYSTEM_PROMPT,
            message_bus=message_bus
        )
        self.environment_image_path = environment_image_path or str(self.artifact_file('environment.jpg'))
        self.current_skills = []
        self.keypoint_locations = {}
        self.current_keypoints = []
        self.current_queries = []
        self.keypoint_descriptions = {}
        self.last_image_np = None
        self.last_image_size = None
        self.last_resized_path = None
        self.processing = False
        self.camera_backend = Config.CAMERA_BACKEND
        print(f"[{self.name}] Vision model ready: {self.model}")
        
        # Register message handlers
        self.register_handler("parameters_completed", self.on_parameters_completed)
        self.register_handler("requery_vision", self.on_parameters_completed)
        self.register_handler("vision_completed", self.on_vision_completed)
        
    def on_parameters_completed(self, message: Message):
        """Handle parameter completion or requery - detect keypoints in environment."""
        print(f"\n[{self.name}] Received message: {message.message_type} from {message.sender}")
        if self.processing:
            print(f"\n[{self.name}] Already processing, skipping new request")
            return

        self.processing = True
        
        # Get skills from message (first time) or from file (requery)
        skills = message.content.get("skills", [])
        if not skills:
            try:
                with open(self.artifact_file('skills_output.json'), 'r') as f:
                    skills = json.load(f)
                print(f"\n🔍 VisionModule: Requerying keypoints from skills_output.json...")
            except Exception as e:
                print(f"[⚠] Could not load skills_output.json: {e}")
                self.processing = False
                raise RuntimeError(f'Cannot load parameterized skills: {e}') from e
        else:
            print(f"\n🔍 VisionModule: Detecting keypoints for {len(skills)} skills...")
        
        try:
            # Extract and process keypoints
            keypoints = self.extract_keypoints_from_skills(skills)
            
            if not keypoints:
                print(f"[{self.name}] No valid keypoints to detect")
                self.current_skills = skills
                self.keypoint_locations = {}
                self.save_skills_to_json()
                self.notify_agent(
                    agent_name="VisionModule",
                    notification_type="vision_completed",
                    data={
                        "keypoint_locations": {},
                        "annotated_image": "annotated_keypoints.png",
                        "skills_file": "skills_output.json",
                        "warning": "No valid keypoints to detect"
                    }
                )
                return
            
            # Detect keypoints in environment image
            self.detect_keypoints(keypoints)
            
            # Save skills to JSON file (update current_skills before saving)
            self.current_skills = skills
            self.save_skills_to_json()
            
            # Notify about completion
            self.notify_agent(
                agent_name="VisionModule",
                notification_type="vision_completed",
                data={
                    "keypoint_locations": self.keypoint_locations,
                    "annotated_image": "annotated_keypoints.png",
                    "skills_file": "skills_output.json"
                }
            )
            
            print(f"✓ Detected {len(self.keypoint_locations)} keypoints in environment")
        except Exception as e:
            error_message = f"Vision module failed: {e}"
            print(f"⚠ {error_message}")
            self.send_message(
                receiver="WebDashboard",
                message_type="vision_failed",
                content={"error": error_message},
            )
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False
    
    def on_vision_completed(self, message: Message):
        """Handle vision completion - placeholder for future pipeline steps."""
        # This handler is registered for future extensibility
        # e.g., triggering next agent in pipeline, validation, etc.
        print(f"[{self.name}] Vision processing completed")
    
    def extract_keypoints_from_skills(self, skills: List[Dict]) -> List[str]:
        """Extract all unique keypoints from skills, filter and clean them."""
        keypoint_set = set()
        self.keypoint_descriptions = {}

        print(f"[{self.name}] DEBUG: Extracting keypoints from {len(skills)} skills")
        print(f"[{self.name}] DEBUG: Skills type: {type(skills)}")

        # Iterate through skills and extract keypoints
        for i, skill in enumerate(skills):
            print(f"[{self.name}] DEBUG: Processing skill {i}: {type(skill)}")
            print(f"[{self.name}] DEBUG: Skill keys: {skill.keys() if isinstance(skill, dict) else 'NOT A DICT'}")

            # Check keypoints list
            if 'keypoints' in skill:
                keypoints = skill['keypoints']
                print(f"[{self.name}] DEBUG: Found keypoints field, type: {type(keypoints)}, len: {len(keypoints) if isinstance(keypoints, list) else 'N/A'}")

                # Ensure keypoints is a list
                if not isinstance(keypoints, list):
                    print(f"[{self.name}] ⚠ Skipping non-list keypoints: {type(keypoints)}")
                    continue

                for j, kp in enumerate(keypoints):
                    print(f"[{self.name}] DEBUG: Keypoint {j}: {kp} (type: {type(kp)})")
                    # Ensure kp is a dict with 'name' key
                    if not isinstance(kp, dict) or 'name' not in kp:
                        print(f"[{self.name}] ⚠ Skipping invalid keypoint: {kp}")
                        continue

                    name = kp['name']
                    print(f"[{self.name}] DEBUG: Keypoint name: {name}")
                    if name.startswith('g_'):
                        keypoint_set.add(name)
                        description = kp.get('description', '')
                        if isinstance(description, str) and description.strip():
                            self.keypoint_descriptions.setdefault(name, description.strip())
                        print(f"[{self.name}] DEBUG: Added keypoint: {name}")
            else:
                print(f"[{self.name}] DEBUG: No 'keypoints' field in skill {i}")

        # Filter out g_eef keypoints (end-effector keypoints)
        filtered_keypoints = [kp for kp in keypoint_set if not kp.startswith('g_eef')]

        print(f"[{self.name}] Extracted {len(keypoint_set)} keypoints, {len(filtered_keypoints)} after filtering g_eef")
        print(f"[{self.name}] DEBUG: All keypoints: {keypoint_set}")
        print(f"[{self.name}] DEBUG: Filtered keypoints: {filtered_keypoints}")

        return filtered_keypoints
    
    def clean_keypoint_name(self, keypoint: str) -> str:
        """Remove g_, O, and P prefixes from keypoint names."""
        # Remove g_ prefix
        cleaned = keypoint.replace('g_', '', 1) if keypoint.startswith('g_') else keypoint
        
        # Parse format: O<objectname>_P<partname>_L<location>
        # Remove O and P prefixes but keep the structure readable
        parts = cleaned.split('_')
        result_parts = []
        
        for part in parts:
            if part.startswith('O'):
                # Object name - remove O prefix
                result_parts.append(part[1:])
            elif part.startswith('P'):
                # Part name - remove P prefix
                result_parts.append(part[1:])
            elif part.startswith('L'):
                # Location - remove L prefix
                result_parts.append(part[1:])
            else:
                result_parts.append(part)
        
        return '_'.join(result_parts)
    
    def detect_keypoints(self, keypoints: List[str]):
        """Detect keypoints in environment image using Gemini vision models."""
        # Clean keypoint names for queries
        queries = [self.clean_keypoint_name(kp) for kp in keypoints]
        query_items = [
            {
                "label": query,
                "description": self.keypoint_descriptions.get(keypoint, ""),
            }
            for keypoint, query in zip(keypoints, queries)
        ]
        
        print(f"[{self.name}] Detecting keypoints: {', '.join(queries)}")

        # In ROS 1/2 mode, refresh environment.jpg from the live camera. In file
        # mode, keep using the configured environment image for offline tests.
        if self.camera_backend in {'ros1', 'ros2'}:
            rgb = Camera(
                camera_node=Config.CAMERA_NODE,
                camera_model=Config.CAMERA_MODEL,
                rosmaster_ip=Config.ROSMASTER_IP,
                rosmaster_port=Config.ROSMASTER_PORT,
                camera_backend=Config.CAMERA_BACKEND,
                camera_image_path=Config.CAMERA_IMAGE_PATH
            ).get_rgb_image(use_new_frame=True)
            if not cv2.imwrite(self.environment_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not write live camera image to {self.environment_image_path}")
        else:
            if not os.path.exists(self.environment_image_path):
                raise FileNotFoundError(f"Environment image not found at {self.environment_image_path}")
            print(f"[{self.name}] File camera mode: using {self.environment_image_path}")

        # Get file modification time to confirm we're reading current version
        file_mtime = os.path.getmtime(self.environment_image_path)
        print(f"[{self.name}] Reading environment image: {self.environment_image_path} (modified: {file_mtime})")
    
        # Prepare image
        original_image = Image.open(self.environment_image_path)
        image = ImageOps.pad(original_image, (1600, 1600), color=(255, 255, 255))
        resized_path = str(self.artifact_file('resized_img.png'))
        image.save(resized_path)
        
        prompt = self.build_keypoint_detection_prompt(query_items)
        image_np = np.array(image.convert('RGB'))
        self.current_keypoints = keypoints
        self.current_queries = queries
        self.last_image_np = image_np
        self.last_image_size = image.size
        self.last_resized_path = resized_path
        
        print(f"[{self.name}] Calling {self.provider} vision model...")
        
        try:
            response_text = self.call_keypoint_detector(prompt, image_np)
            print(f"[{self.name}] Vision model response received")
            print(f"[{self.name}] Raw response text: {response_text}")
            self.apply_keypoint_response(response_text, keypoints, queries, image.size, resized_path)
            
        except Exception as e:
            raise RuntimeError(f"Vision detection failed: {e}") from e

    @staticmethod
    def build_keypoint_detection_prompt(query_items: List[Dict]) -> str:
        """Build the per-call keypoint detection query."""
        return f"""
Detect these requested keypoints in the current image using their descriptions:
{json.dumps(query_items, indent=2)}

This is a newly captured scene. Ground every point in this image; previous
images and pixel locations are historical references, not current coordinates.
Return one point for each requested keypoint, preserving these exact labels.
"""

    def build_keypoint_refinement_prompt(self, feedback: str) -> str:
        """Build the per-call refinement query for the latest keypoint annotation."""
        return f"""
Refine the previous keypoint annotation using this user feedback:
{feedback}

Requested keypoint labels, which must stay exactly the same:
{json.dumps(self.current_queries, indent=2)}

Current keypoint locations:
{json.dumps(self.keypoint_locations, indent=2)}

Return the complete corrected keypoint array for the current image.
"""

    def call_keypoint_detector(self, prompt: str, image: np.ndarray) -> str:
        """Call the configured VisionModule model through the shared BaseAgent path."""
        return self.call_llm(prompt, response_format="json", image=image)

    def apply_keypoint_response(
        self,
        response_text: str,
        keypoints: List[str],
        queries: List[str],
        image_size,
        resized_path: str
    ):
        """Parse, save, and visualize a keypoint response."""
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-len("```")].strip()

        points = json.loads(response_text)
        print(f"[{self.name}] Raw keypoint data: {points}")
        print(f"[{self.name}] Original keypoints: {keypoints}")
        print(f"[{self.name}] Original queries: {queries}")
        print(f"Number of VLM points {len(points)}, number of Recipe points {len(keypoints)}")

        if not isinstance(points, list) or len(points) != len(queries):
            raise ValueError('Vision must return exactly one point per requested keypoint')
        if sorted(obj.get('label', '') for obj in points if isinstance(obj, dict)) != sorted(queries):
            raise ValueError('Vision response labels must match the requested keypoints')
        for obj in points:
            point = np.asarray(obj.get('point'), dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)) or np.any(point < 0) or np.any(point >= 1000):
                raise ValueError('Keypoint coordinates must be finite [y, x] values in [0, 1000)')
        width, height = image_size
        for obj in points:
            obj["point"] = [int(obj['point'][1] * width // 1000), int(obj['point'][0] * height // 1000)]

        print(f"[{self.name}] Detected {len(points)} keypoints")

        self.keypoint_locations = {}
        resulting_points = []

        used_indices = set()
        for query_index, query in enumerate(queries):
            point_index = next(
                (
                    i for i, point in enumerate(points)
                    if i not in used_indices and point.get('label') == query
                ),
                None
            )
            if point_index is None and query_index < len(points):
                point_index = query_index
                model_label = points[point_index].get('label')
                if model_label != query:
                    print(
                        f"[{self.name}] Label mismatch for query '{query}'; "
                        f"using response order label '{model_label}'"
                    )

            if point_index is None or point_index in used_indices:
                continue

            used_indices.add(point_index)
            point = points[point_index]
            kp = keypoints[query_index]
            normalized_point = {
                'point': point['point'],
                'label': query
            }
            self.keypoint_locations[kp] = {
                'point': normalized_point['point'],
                'label': normalized_point['label'],
                'query': query
            }
            resulting_points.append(normalized_point)

        output_json = self.artifact_file('keypoint_locations.json')
        with open(output_json, 'w') as f:
            json.dump(self.keypoint_locations, f, indent=2)
        print(f"[{self.name}] Keypoint locations saved to: {output_json}")

        self.visualize_keypoints(resized_path, resulting_points)

    def refine_keypoints(self, feedback: str) -> Dict[str, Any]:
        """Refine the latest keypoint annotation based on user feedback."""
        if self.last_image_np is None or not self.current_keypoints or not self.current_queries:
            return {
                "error": "No prior keypoint annotation is available to refine. Run vision detection first."
            }

        print(f"\n[{self.name}] Refining keypoints based on feedback")
        prompt = self.build_keypoint_refinement_prompt(feedback)
        response_text = self.call_keypoint_detector(prompt, self.last_image_np)
        print(f"[{self.name}] Refinement response received")
        print(f"[{self.name}] Raw refinement response text: {response_text}")

        self.apply_keypoint_response(
            response_text,
            self.current_keypoints,
            self.current_queries,
            self.last_image_size,
            self.last_resized_path
        )

        self.notify_agent(
            agent_name="VisionModule",
            notification_type="vision_completed",
            data={
                "keypoint_locations": self.keypoint_locations,
                "annotated_image": "annotated_keypoints.png",
                "skills_file": "skills_output.json"
            }
        )

        return {
            "keypoint_locations": self.keypoint_locations,
            "count": len(self.keypoint_locations)
        }
    
    def visualize_keypoints(self, image_path: str, points: List[Dict]):
        """Visualize detected keypoints on the image."""
        output_path = str(self.artifact_file('annotated_keypoints.png'))
        save_annotated_keypoints(image_path, points, output_path)
        print(f"[{self.name}] Annotated image saved to: {output_path}")
    
    def get_keypoint_locations(self) -> Dict[str, Any]:
        """Get the detected keypoint locations."""
        return self.keypoint_locations
    
    def save_skills_to_json(self):
        """Save the current skills to a JSON file."""
        output_json = self.artifact_file('skills_output.json')
        try:
            with open(output_json, 'w') as f:
                json.dump(self.current_skills, f, indent=2)
            print(f"[{self.name}] Skills saved to: {output_json}")
        except Exception as e:
            raise RuntimeError(f"Error saving skills to JSON: {e}") from e
