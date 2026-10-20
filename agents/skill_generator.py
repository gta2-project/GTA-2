"""Skill Generator Agent - Creates GTA framework skill recipes for subtasks."""
import json
import numpy as np
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from communication import MessageBus, Message
from agents.prompts.skill_generator import SYSTEM_PROMPT

class SkillGeneratorAgent(BaseAgent):
    """Agent responsible for generating GTA framework skill recipes with controllers, keypoints and priorities."""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(
            name="SkillGenerator",
            system_prompt=SYSTEM_PROMPT,
            message_bus=message_bus
        )
        self.current_skills = []
        self.processing = False
        
        # Register message handlers
        self.register_handler("subtasks_updated", self.on_subtasks_updated)
        self.register_handler("refinement_request", self.on_refinement_request)
    
    def generate_skills(self, subtasks: List[Dict], image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate detailed skill recipes for subtasks using GTA framework.
        
        Args:
            subtasks: List of subtasks to generate skills for
            image: Optional RGB image override (uses camera if use_vision is enabled)
        """
        print(f"\n[{self.name}] Generating GTA skill recipes for {len(subtasks)} subtasks")
        
        # Format subtasks for skill recipe generation
        formatted_subtasks = []
        for subtask in subtasks:
            formatted_subtasks.append({
                "id": subtask.get('id'),
                "subtask_prompt": subtask.get('subtask_prompt'),
                "goal_check_prompt": subtask.get('goal_check_prompt'),
                "action_space_description": subtask.get('action_space_description')
            })
        
        subtasks_json = json.dumps(formatted_subtasks, indent=2)
        user_message = f"""Please generate skill recipes for these subtasks from the task decomposer.

For each subtask, create a skill recipe with:
1. Appropriate keypoints for the objects and parts mentioned in the action_space_description
2. Controllers prioritized by execution order (1 = highest priority)
3. Proper controller arguments using the available keypoints and axes
4. Explanations for why each controller is needed

Subtasks to process:
{subtasks_json}"""
        
        response = self.call_llm(user_message, response_format="json", image=image)
        
        print(f"[{self.name}] Response:\n{response}...\n")
        
        try:
            result = json.loads(response)
            self.current_skills = result.get('skills', [])
            self.update_state('current_skills', self.current_skills)
            
            print(f"[{self.name}] Successfully generated {len(self.current_skills)} skill recipes")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON Parse Error: {str(e)}")
            return {
                'skills': [],
                'error': f'Failed to parse response: {str(e)}',
                'raw_response': response[:500]
            }
    
    def refine_skills(self, feedback: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Refine skill recipes based on feedback.
        
        Args:
            feedback: Feedback for refinement
            image: Optional RGB image override (uses camera if use_vision is enabled)
        """
        print(f"\n[{self.name}] Refining skill recipes based on feedback")
        
        user_message = f"""I need you to refine the skill recipes based on this feedback.

Focus on:
- Adjusting controller selection and priorities
- Modifying keypoint specifications
- Improving controller arguments and parameters
- Ensuring proper parallel execution logic

Feedback: {feedback}

Current skill recipes:
{json.dumps(self.current_skills, indent=2)}

Please return ALL skills in the same JSON format, following the REFINEMENT RULES in your system prompt."""
        
        response = self.call_llm(user_message, response_format="json", image=image)
        
        print(f"[{self.name}] Refinement response:\n{response}...\n")
        
        try:
            result = json.loads(response)
            self.current_skills = result.get('skills', [])
            self.update_state('current_skills', self.current_skills)
            
            print(f"[{self.name}] Successfully refined {len(self.current_skills)} skill recipes")
            
            # Notify ParameterSetter about updated skills (same as initial generation)
            self.notify_agent(
                agent_name="ParameterSetter",
                notification_type="skills_updated",
                data={
                    'skills': self.current_skills,
                    'count': len(self.current_skills),
                    'type': 'skill_recipes',
                    'action': 'refined'
                }
            )
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON Parse Error during refinement: {str(e)}")
            return {
                'skills': self.current_skills,
                'error': f'Failed to parse refined response: {str(e)}'
            }
    
    def on_subtasks_updated(self, message: Message):
        """Handle subtask updates from Task Decomposer."""
        if self.processing:
            print(f"\n[{self.name}] Already processing, skipping new request")
            return
        
        print(f"\n[{self.name}] Subtasks updated, regenerating skill recipes...")
        self.processing = True
        
        try:
            subtasks_data = message.content
            subtasks = subtasks_data.get('subtasks', [])
            if subtasks:
                result = self.generate_skills(subtasks)
                if result.get('error') and getattr(self.message_bus, 'workspace', None) is not None:
                    raise RuntimeError(result['error'])
                
                # Notify ParameterSetter for parameter enrichment
                self.notify_agent(
                    agent_name="ParameterSetter",
                    notification_type="skills_updated",
                    data={
                        'skills': self.current_skills,
                        'count': len(self.current_skills),
                        'type': 'skill_recipes',
                        'action': 'generated'
                    }
                )
            elif getattr(self.message_bus, 'workspace', None) is not None:
                raise RuntimeError('No subtasks were generated')
        finally:
            self.processing = False
    
    def on_refinement_request(self, message: Message):
        """Handle refinement requests from Error Analyzer."""
        refinement_data = message.content
        if refinement_data.get('refinement_type') == 'skill_generation':
            print(f"\n[{self.name}] Received skill recipe refinement request, processing...")
            
            recommendations = refinement_data.get('recommendations', [])
            feedback = f"Skill recipe refinement needed: {', '.join(recommendations)}"
            result = self.refine_skills(feedback)
            if result.get('error') and getattr(self.message_bus, 'workspace', None) is not None:
                raise RuntimeError(result['error'])
    
    def get_current_skills(self) -> list:
        """Get the current list of skills."""
        return self.current_skills.copy()
    
    def get_skill_for_subtask(self, subtask_id: int) -> Dict[str, Any]:
        """Get skill recipe for a specific subtask."""
        for skill in self.current_skills:
            if skill.get('subtask_id') == subtask_id:
                return skill
        return {}
    
    def get_controllers_for_subtask(self, subtask_id: int) -> List[Dict]:
        """Get the controller list for a specific subtask."""
        skill = self.get_skill_for_subtask(subtask_id)
        return skill.get('controllers', [])
    
    def get_keypoints_for_subtask(self, subtask_id: int) -> List[Dict]:
        """Get the keypoints list for a specific subtask."""
        skill = self.get_skill_for_subtask(subtask_id)
        return skill.get('keypoints', [])
