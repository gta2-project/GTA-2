from typing import Dict, List, Optional
import numpy as np
from agents.base_agent import BaseAgent
from communication.message_bus import MessageBus, Message
from agents.prompts.parameter_setter import SYSTEM_PROMPT
import json
from gta.plan_validation import validate_plan


class ParameterSettingAgent(BaseAgent):
    """Agent that enriches skills with physical parameter values based on scene analysis."""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__("ParameterSetter", SYSTEM_PROMPT, message_bus)
        self.current_skills = []
        self.processing = False
        
        # Register message handlers
        self.register_handler("skills_updated", self.on_skills_updated)
        self.register_handler("refinement_request", self.on_refinement_request)
    
    def on_skills_updated(self, message: Message):
        """Handle skill update - enrich with parameter values."""
        if self.processing:
            print(f"\n[{self.name}] Already processing, skipping new request")
            return
        
        self.processing = True
        skills = message.content.get("skills", [])
        print(f"\n🔧 ParameterSetter: Analyzing {len(skills)} skills and setting parameter values...")
        
        try:
            self.current_skills = self.generate_parameters(skills)
            
            # Send to VisionModule for keypoint detection
            self.notify_agent(
                agent_name="VisionModule",
                notification_type="parameters_completed",
                data={
                    "skills": self.current_skills,
                    "action": "generated"
                    }
            )
            
            print(f"✓ Parameters set for {len(self.current_skills)} skills")
        finally:
            self.processing = False
    
    def on_refinement_request(self, message: Message):
        """Handle parameter refinement requests from Error Analyzer."""
        refinement_data = message.content
        if refinement_data.get('refinement_type') == 'parameter_setting':
            print(f"\n[{self.name}] Received parameter refinement request, processing...")
            
            recommendations = refinement_data.get('recommendations', [])
            feedback = f"Parameter refinement needed: {', '.join(recommendations)}"
            self.current_skills = self.refine_parameters(feedback)
            
            # Send to VisionModule for updated keypoint detection
            self.notify_agent(
                agent_name="VisionModule",
                notification_type="parameters_completed",
                data={
                    "skills": self.current_skills,
                    "action": "refined"
                }
            )
    
    def generate_parameters(self, skills: List[Dict], image: Optional[np.ndarray] = None) -> List[Dict]:
        """Enrich skills with parameter values using LLM reasoning.
        
        Args:
            skills: List of skills to enrich with parameters
            image: Optional RGB image override (uses camera if use_vision is enabled)
        """
        try:
            # Prepare input for LLM
            user_message = f"""Analyze these skills and enrich the offset_parameters with values, units, and physical reasoning.

Skills to enrich:
{json.dumps(skills, indent=2)}

For each offset parameter:
1. Estimate object dimensions from keypoint names
2. Infer material properties from object names
3. Apply physical reasoning for distances, forces, angles
4. Provide value, unit, and detailed reasoning

Return the complete enriched skills array."""

            # Call LLM with JSON response format
            response = self.call_llm(user_message, response_format="json", image=image)
            
            # Parse response
            result = json.loads(response)
            
            if "skills" not in result:
                if getattr(self.message_bus, 'workspace', None) is not None:
                    raise RuntimeError("Parameter response is missing the skills field")
                print("⚠ Warning: LLM response missing 'skills' field, returning original skills")
                return skills
            
            if getattr(self.message_bus, 'workspace', None) is not None:
                validate_plan(result['skills'])
            return result["skills"]
            
        except json.JSONDecodeError as e:
            if getattr(self.message_bus, 'workspace', None) is not None:
                raise RuntimeError(f'Invalid parameter response: {e}') from e
            print(f"⚠ Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response[:500]}...")
            return skills
        except RuntimeError:
            # Camera failures in ROS mode must stop the workflow rather than
            # silently continuing with parameters inferred without an image.
            raise
        except Exception as e:
            if getattr(self.message_bus, 'workspace', None) is not None:
                raise RuntimeError(f'Parameter generation failed: {e}') from e
            print(f"⚠ Warning: Error enriching parameters: {e}")
            return skills
    
    def refine_parameters(self, feedback: str, image: Optional[np.ndarray] = None) -> List[Dict]:
        """Refine specific parameters based on feedback.
        
        Args:
            feedback: Feedback for refinement
            image: Optional RGB image override (uses camera if use_vision is enabled)
        """
        try:
            user_message = f"""Current skills with parameters:
{json.dumps(self.current_skills, indent=2)}

Feedback: {feedback}

Update the specified parameters based on this feedback. Keep all other parameters unchanged.
Return a JSON object with a top-level "skills" field containing the complete updated skills list."""

            print(
                f"[{self.name}][refinement-debug] "
                f"provider={self.provider}, model={self.model}, "
                f"current_skills={len(self.current_skills)}, feedback={feedback!r}",
                flush=True,
            )
            response = self.call_llm(user_message, response_format="json", image=image)
            # Log before parsing so malformed JSON and unexpected response shapes
            # remain visible. Do not dump credentials, images, or chat history.
            print(
                f"[{self.name}][refinement-debug] Raw response "
                f"(type={type(response).__name__}): {response!r}",
                flush=True,
            )
            result = json.loads(response)
            structure = (
                f"keys={list(result)}" if isinstance(result, dict)
                else f"items={len(result)}" if isinstance(result, list)
                else "not an object or array"
            )
            print(
                f"[{self.name}][refinement-debug] Parsed response: "
                f"type={type(result).__name__}, {structure}",
                flush=True,
            )
            
            if "skills" in result:
                if getattr(self.message_bus, 'workspace', None) is not None:
                    validate_plan(result['skills'])
                self.current_skills = result["skills"]
                return self.current_skills
            else:
                if getattr(self.message_bus, 'workspace', None) is not None:
                    raise RuntimeError("Parameter refinement response is missing the skills field")
                print("⚠ Warning: LLM response missing 'skills' field")
                return self.current_skills
                
        except RuntimeError:
            # Keep refinement behavior consistent with initial parameter
            # generation when the live ROS image cannot be captured.
            raise
        except Exception as e:
            if getattr(self.message_bus, 'workspace', None) is not None:
                raise RuntimeError(f'Parameter refinement failed: {e}') from e
            print(f"⚠ Error refining parameters: {e}")
            return self.current_skills
    
    def get_current_skills(self) -> List[Dict]:
        """Get the current enriched skills."""
        return self.current_skills
