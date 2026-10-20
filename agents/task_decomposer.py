"""Task Decomposer Agent - Breaks down tasks into subtasks."""
import json
import numpy as np
from typing import Dict, Any, Optional
from agents.base_agent import BaseAgent
from communication import MessageBus, Message
from agents.prompts.task_decomposer import SYSTEM_PROMPT

class TaskDecomposerAgent(BaseAgent):
    """Agent responsible for decomposing tasks into subtasks."""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(
            name="TaskDecomposer",
            system_prompt=SYSTEM_PROMPT,
            message_bus=message_bus
        )
        self.current_task = None
        self.current_subtasks = []
        
        # Register message handlers
        self.register_handler("refinement_request", self.on_refinement_request)
    
    def generate_subtasks(self, task: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Decompose a task into subtasks.
        
        Args:
            task: The task description to decompose
            image: Optional RGB image override (uses camera if use_vision is enabled)
        """
        print(f"\n[{self.name}] Decomposing task: {task}")
        
        user_message = f"Please break down this task into subtasks:\n\n{task}"
        response = self.call_llm(user_message, response_format="json", image=image)
        
        print(f"[{self.name}] Response:\n{response}\n")
        
        try:
            result = json.loads(response)
            self.current_task = task
            self.current_subtasks = result.get('subtasks', [])
            self.update_state('current_task', task)
            self.update_state('current_subtasks', self.current_subtasks)
            
            print(f"[{self.name}] Successfully parsed {len(self.current_subtasks)} subtasks")
            
            # Notify Skill Generator about new subtasks
            self.notify_agent(
                agent_name="SkillGenerator",
                notification_type="subtasks_updated",
                data={
                    'task': task,
                    'subtasks': self.current_subtasks,
                    'action': 'generated'
                }
            )
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON Parse Error: {str(e)}")
            print(f"[{self.name}] Failed to parse response: {response[:200]}")
            return {
                'task': task,
                'subtasks': [],
                'error': f'Failed to parse response: {str(e)}',
                'raw_response': response[:500]
            }
    
    def refine_subtasks(self, feedback: str, image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Refine subtasks based on feedback.
        
        Args:
            feedback: Feedback for refinement
            image: Optional RGB image override (uses camera if use_vision is enabled)
        """
        print(f"\n[{self.name}] Refining subtasks based on feedback")
        
        user_message = f"""I need you to refine the subtasks based on this feedback.

Feedback: {feedback}

Current task: {self.current_task}

Current subtasks:
{json.dumps(self.current_subtasks, indent=2)}

Please return ALL subtasks in the same JSON format."""
        
        response = self.call_llm(user_message, response_format="json", image=image)
        
        print(f"[{self.name}] Refinement response:\n{response[:200]}...\n")
        
        try:
            result = json.loads(response)
            self.current_subtasks = result.get('subtasks', [])
            self.update_state('current_subtasks', self.current_subtasks)
            
            print(f"[{self.name}] Successfully refined to {len(self.current_subtasks)} subtasks")
            
            # Notify Skill Generator about updated subtasks
            self.notify_agent(
                agent_name="SkillGenerator",
                notification_type="subtasks_updated",
                data={
                    'task': self.current_task,
                    'subtasks': self.current_subtasks,
                    'action': 'refined'
                }
            )
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"[{self.name}] JSON Parse Error during refinement: {str(e)}")
            return {
                'task': self.current_task,
                'subtasks': self.current_subtasks,
                'error': f'Failed to parse refined response: {str(e)}'
            }
    
    def on_refinement_request(self, message: Message):
        """Handle refinement requests from Error Analyzer."""
        refinement_data = message.content
        if refinement_data.get('refinement_type') == 'task_decomposition':
            print(f"\n[{self.name}] Received refinement request, processing...")
            recommendations = refinement_data.get('recommendations', [])
            feedback = f"Refinement needed: {', '.join(recommendations)}"
            result = self.refine_subtasks(feedback)
            if result.get('error') and getattr(self.message_bus, 'workspace', None) is not None:
                raise RuntimeError(result['error'])
    
    def get_current_subtasks(self) -> list:
        """Get the current list of subtasks."""
        return self.current_subtasks.copy()
