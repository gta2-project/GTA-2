"""Base agent class for all agents in the system."""
from typing import Dict, Any, Optional, Callable
from openai import OpenAI
from google import genai
import json
import threading
import base64
import io
import copy
from pathlib import Path
import numpy as np
from PIL import Image
from communication import MessageBus, Message
from config import Config
from .checkpoint import STATE_FIELDS, encode, decode, openai_input

# Shared camera instance (singleton)
_CAMERA_INSTANCE = None

class BaseAgent:
    """Base class for all agents with message handler registration system."""
    
    def __init__(self, name: str, system_prompt: str, message_bus: MessageBus, skip_client_init: bool = False):
        self.name = name
        self.system_prompt = system_prompt
        self.message_bus = message_bus
        self.state = {}
        
        # Message handler registry
        self.message_handlers: Dict[str, Callable] = {}
        
        # Get agent-specific configuration
        saved = getattr(message_bus, 'agent_snapshots', {}).get(name)
        agent_config = saved['config'] if saved else Config.get_agent_config(name)
        self.provider = agent_config['provider']
        self.model = agent_config['model']
        self.api_key = agent_config.get('api_key')
        self.use_vision = agent_config['use_vision']
        self.workspace = Path(getattr(message_bus, 'workspace', None) or '.')
        self.generation_settings = copy.deepcopy(agent_config.get('generation_settings', {
            'reasoning': {'effort': 'medium'}, 'text': {'verbosity': 'low'},
        } if self.provider == 'openai' else {}))
        self._pending_history = decode(saved['history']) if saved else None
        self._has_model_calls = bool(saved and self._pending_history)
        self._lazy_clients = bool(saved or getattr(message_bus, 'lazy_clients', False))
        if saved:
            self.system_prompt = agent_config['system_prompt']
        
        # Initialize camera if vision is enabled
        if self.use_vision and not self._lazy_clients:
            self._init_camera()
        
        # Initialize the appropriate client based on provider (unless skipped)
        if not skip_client_init and not self._lazy_clients:
            if self.provider == 'openai':
                self.client = OpenAI(api_key=self.api_key)
                self.conversation_id = None  # Will store OpenAI conversation ID
                self._create_conversation()
            elif self.provider == 'gemini':
                self.gemini_client = genai.Client(api_key=self.api_key)
                self.gemini_chat = None  # Will store Gemini chat object
                self._create_gemini_chat()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            vision_status = "with vision" if self.use_vision else "text-only"
            print(f"[{self.name}] Initialized with {self.provider} ({self.model}) - {vision_status}")
        else:
            # For agents that manage their own clients
            self.client = None
            self.gemini_client = None
            self.conversation_id = None
            self.gemini_chat = None
            vision_status = "with vision" if self.use_vision else "text-only"
            print(f"[{self.name}] Initialized (manages own API clients) - {vision_status}")
        
        # Subscribe to messages
        self.message_bus.subscribe(self.name, self.handle_message)

    def artifact_file(self, name):
        return self.workspace / name

    def _ensure_client(self):
        key = self.api_key or (Config.OPENAI_API_KEY if self.provider == 'openai' else Config.GEMINI_API_KEY)
        if not key:
            raise RuntimeError(f'{self.name}: API key is required for {self.provider}')
        if self.provider == 'openai':
            if getattr(self, 'client', None) is None:
                self.client = OpenAI(api_key=key)
            if self._pending_history is not None:
                # Never reuse the saved provider conversation: library edits fork it.
                conversation = self.client.conversations.create(metadata={'agent': self.name})
                history = [openai_input(item) for item in self._pending_history]
                for offset in range(0, len(history), 20):
                    self.client.conversations.items.create(conversation.id, items=history[offset:offset + 20])
                self.conversation_id = conversation.id
                self._pending_history = None
            elif not getattr(self, 'conversation_id', None):
                self.conversation_id = self.client.conversations.create(metadata={'agent': self.name}).id
        elif self.provider == 'gemini':
            if getattr(self, 'gemini_client', None) is None:
                self.gemini_client = genai.Client(api_key=key)
            if getattr(self, 'gemini_chat', None) is None:
                self.gemini_chat = self.gemini_client.chats.create(
                    model=self.model,
                    config={'system_instruction': self.system_prompt, **self.generation_settings},
                    history=self._pending_history or [],
                )
                self._pending_history = None
        else:
            raise RuntimeError(f'Unsupported provider: {self.provider}')

    def export_checkpoint(self):
        if self._pending_history is not None:
            history = self._pending_history
        elif self.provider == 'openai' and getattr(self, 'conversation_id', None):
            # SDK cursor iteration follows every page, rather than only .data.
            history = list(self.client.conversations.items.list(
                self.conversation_id, order='asc', limit=100,
                include=['message.input_image.image_url', 'reasoning.encrypted_content'],
            ))
        elif self.provider == 'gemini' and getattr(self, 'gemini_chat', None):
            history = self.gemini_chat.get_history(curated=False)
        elif self._has_model_calls:
            raise RuntimeError(f'{self.name}: conversation history is unavailable')
        else:
            history = []
        if self._has_model_calls and not history:
            raise RuntimeError(f'{self.name}: provider returned an incomplete empty history')
        history = encode(history)
        if self.provider == 'openai':
            for item in history:
                for part in item.get('content', []):
                    if part.get('type') == 'input_image' and not str(part.get('image_url', '')).startswith('data:image/'):
                        raise RuntimeError(f'{self.name}: image context is not embedded; cannot save a complete checkpoint')
        return {
            'config': {'provider': self.provider, 'model': self.model,
                       'use_vision': self.use_vision, 'system_prompt': self.system_prompt,
                       'generation_settings': copy.deepcopy(self.generation_settings)},
            'history': history,
            'state': encode({key: getattr(self, key) for key in STATE_FIELDS if hasattr(self, key)}),
        }

    def restore_checkpoint_state(self, saved):
        for key, value in decode(saved['state']).items():
            if key not in STATE_FIELDS:
                raise ValueError(f'Unexpected agent state field: {key}')
            setattr(self, key, value)
        if hasattr(self, 'processing'):
            self.processing = False
        if hasattr(self, 'last_resized_path'):
            self.last_resized_path = str(self.artifact_file('resized_img.png'))
    
    def _init_camera(self):
        """Initialize camera singleton if vision is enabled."""
        global _CAMERA_INSTANCE
        if _CAMERA_INSTANCE is None:
            try:
                from scene.camera import Camera
                _CAMERA_INSTANCE = Camera(
                    camera_node=Config.CAMERA_NODE,
                    camera_model=Config.CAMERA_MODEL,
                    rosmaster_ip=Config.ROSMASTER_IP,
                    rosmaster_port=Config.ROSMASTER_PORT,
                    camera_backend=Config.CAMERA_BACKEND,
                    camera_image_path=Config.CAMERA_IMAGE_PATH
                )
                print(f"[{self.name}] Camera initialized successfully")
            except Exception as e:
                if Config.CAMERA_BACKEND in {'ros1', 'ros2'}:
                    raise RuntimeError(
                        f"Camera initialization failed while CAMERA_BACKEND={Config.CAMERA_BACKEND}: {e}"
                    ) from e
                print(f"[{self.name}] Warning: Could not initialize camera: {e}")
                print(f"[{self.name}] Vision will be disabled")
                self.use_vision = False
        self.camera = _CAMERA_INSTANCE
    
    def register_handler(self, message_type: str, handler: Callable[[Message], None]):
        """Register a handler function for a specific message type.
        
        Args:
            message_type: The type of message to handle (e.g., 'subtasks_updated')
            handler: Callable that accepts a Message object
        """
        self.message_handlers[message_type] = handler
        print(f"[{self.name}] Registered handler for '{message_type}'")
    
    def handle_message(self, message: Message):
        """Route incoming messages to registered handlers (runs in background thread)."""
        print(f"\n[{self.name}] Received message from {message.sender}: {message.message_type}")
        
        # Store message in state
        if 'received_messages' not in self.state:
            self.state['received_messages'] = []
        self.state['received_messages'].append(message)
        
        # Route to registered handler if exists
        handler = self.message_handlers.get(message.message_type)
        if handler:
            # Run handler in background thread to keep system responsive
            self.message_bus.run_background(self._safe_handler_execution, handler, message)
        else:
            # No handler registered for this message type
            self.on_unhandled_message(message)
    
    def _safe_handler_execution(self, handler: Callable, message: Message):
        """Safely execute a handler with error handling."""
        try:
            handler(message)
        except Exception as e:
            print(f"[{self.name}] Error in handler for '{message.message_type}': {str(e)}")
            try:
                self.send_message(
                    receiver="WebDashboard",
                    message_type="agent_error",
                    content={
                        "agent": self.name,
                        "source_message_type": message.message_type,
                        "error": str(e),
                    },
                )
            except Exception:
                pass
            import traceback
            traceback.print_exc()
    
    def on_unhandled_message(self, message: Message):
        """Override in child classes to handle unregistered message types."""
        print(f"[{self.name}] No handler registered for message type: {message.message_type}")
    
    def on_subtasks_updated(self, subtasks_data: Dict):
        """Override in child classes to handle subtask updates."""
        pass
    
    def on_refinement_request(self, refinement_data: Dict):
        """Override in child classes to handle refinement requests."""
        pass
    
    def _create_conversation(self):
        """Create a new OpenAI conversation for this agent."""
        try:
            conversation = self.client.conversations.create(
                metadata={"agent": self.name}
            )
            self.conversation_id = conversation.id

        except Exception as e:
            print(f"[{self.name}] Error creating OpenAI conversation: {str(e)}")
            self.conversation_id = None
    
    def _create_gemini_chat(self):
        """Create a new Gemini chat for this agent."""
        try:
            self.gemini_chat = self.gemini_client.chats.create(
                model=self.model,
                config={"system_instruction": self.system_prompt}
            )
        except Exception as e:
            print(f"[{self.name}] Error creating Gemini chat: {str(e)}")
            self.gemini_chat = None
    
    def send_message(self, receiver: str, message_type: str, content: Any):
        """Send a message to a specific agent."""
        message = Message(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content
        )
        print(f"[{self.name}] Sending message to {receiver}: {message_type}")
        self.message_bus.publish(message)
    
    def notify_agent(self, agent_name: str, notification_type: str, data: Dict):
        """Send notification to a specific agent (standardized helper)."""
        self.send_message(
            receiver=agent_name,
            message_type=notification_type,
            content=data
        )
    
    def call_llm(self, user_message: str, response_format: Optional[str] = None, image: Optional[np.ndarray] = None) -> str:
        """Call the LLM API (OpenAI or Gemini) based on provider configuration.
        
        Args:
            user_message: Text message to send to the LLM
            response_format: Optional format specifier (e.g., 'json')
            image: Optional RGB image as numpy array (H, W, 3) with values 0-255
                   If None and use_vision is True, will automatically fetch from camera
        """
        self._ensure_client()
        if self.use_vision and image is None and not hasattr(self, 'camera'):
            self._init_camera()
        # Automatically fetch image if vision is enabled and no image provided
        if self.use_vision and image is None and hasattr(self, 'camera') and self.camera is not None:
            try:
                image = self.camera.get_rgb_image()
            except Exception as e:
                if Config.CAMERA_BACKEND in {'ros1', 'ros2'}:
                    raise RuntimeError(
                        f"Camera image fetch failed while CAMERA_BACKEND={Config.CAMERA_BACKEND}: {e}"
                    ) from e
                print(f"[{self.name}] Warning: Failed to get image from camera: {e}")
                image = None
        
        if self.provider == 'openai':
            return self._call_openai(user_message, response_format, image)
        elif self.provider == 'gemini':
            return self._call_gemini(user_message, response_format, image)
        else:
            return f"Error: Unsupported provider {self.provider}"
    
    def _call_openai(self, user_message: str, response_format: Optional[str] = None, image: Optional[np.ndarray] = None) -> str:
        """Call the OpenAI API using Responses API with Conversations."""
        try:
            # Create response configuration
            response_config = {
                "conversation": self.conversation_id,
                "model": self.model,
                "instructions": self.system_prompt,
                **copy.deepcopy(self.generation_settings),
            }
            
            if response_format == "json":
                # For Responses API, request JSON in instructions instead
                response_config["instructions"] = self.system_prompt + "\n\nIMPORTANT: You MUST respond with valid JSON only, no other text."
            
            # Handle image input
            if image is not None:
                # Convert numpy array to base64
                image_base64 = self._encode_image_to_base64(image)
                # Responses API multimodal input expects a message item with
                # nested input_text/input_image content parts.
                response_config["input"] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_message},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        ],
                    }
                ]
            else:
                response_config["input"] = user_message
            
            # Create response using Responses API
            response = self.client.responses.create(**response_config)
            self._has_model_calls = True
            
            # Extract the assistant's message
            if response.output and len(response.output) > 0:
                # Iterate through output items to find the message
                for output_item in response.output:
                    if output_item.type == "message" and output_item.content:
                        for content_part in output_item.content:
                            if content_part.type == "output_text":
                                return content_part.text
            
            return "No response generated"
        
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"
    
    def _call_gemini(self, user_message: str, response_format: Optional[str] = None, image: Optional[np.ndarray] = None) -> str:
        """Call the Google Gemini API using chat mode."""
        try:
            if not self.gemini_chat:
                self._create_gemini_chat()
            
            # Prepare the message
            if response_format == "json":
                user_message += "\n\nIMPORTANT: You MUST respond with valid JSON only, no other text."
            
            # Handle image input
            if image is not None:
                # Convert numpy array to PIL Image for Gemini
                pil_image = self._numpy_to_pil(image)
                # Send message with image to Gemini chat
                response = self.gemini_chat.send_message([user_message, pil_image])
            else:
                # Send text-only message to Gemini chat
                response = self.gemini_chat.send_message(user_message)
            self._has_model_calls = True

            # Confirm the model endpoint and whether Gemini reports image tokens
            # in the processed prompt. This verifies transport, not that the model
            # semantically relied on the image in its reasoning.
            usage = getattr(response, "usage_metadata", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None) or []
            image_tokens = sum(
                int(getattr(detail, "token_count", 0) or 0)
                for detail in prompt_details
                if getattr(getattr(detail, "modality", None), "value", None) == "IMAGE"
            )
            if image is None:
                image_status = "not_attached"
            elif image_tokens > 0:
                image_status = f"confirmed ({image_tokens} prompt tokens)"
            else:
                image_status = "attached, but usage metadata did not report image tokens"
            print(
                f"[{self.name}] Gemini response metadata: "
                f"requested_model={self.model}, "
                f"served_model={getattr(response, 'model_version', None) or 'unknown'}, "
                f"response_id={getattr(response, 'response_id', None) or 'unknown'}, "
                f"image_input={image_status}"
            )
            
            response_text = response.text
            
            # Clean up markdown code blocks if present (for JSON responses)
            if response_format == "json" and response_text:
                import re
                # Remove ```json ... ``` or ``` ... ```
                cleaned = re.sub(r'^```(?:json)?\s*\n', '', response_text.strip())
                cleaned = re.sub(r'\n```\s*$', '', cleaned)
                return cleaned.strip()
            
            return response_text
        
        except Exception as e:
            return f"Error calling Gemini: {str(e)}"
    
    def reset_conversation(self):
        """Reset the conversation by creating a new one."""
        self._pending_history = None
        self._has_model_calls = False
        if self._lazy_clients:
            self.conversation_id = None
            self.gemini_chat = None
            return
        if self.provider == 'openai':
            self._create_conversation()
        elif self.provider == 'gemini':
            self._create_gemini_chat()
    
    def get_state(self) -> Dict:
        """Get the current state of the agent."""
        return self.state.copy()
    
    def get_state_value(self, key: str, default=None):
        """Safely retrieve a state value."""
        return self.state.get(key, default)
    
    def has_state(self, key: str) -> bool:
        """Check if a state key exists."""
        return key in self.state
    
    def update_state(self, key: str, value: Any):
        """Update the agent's state."""
        self.state[key] = value
    
    @staticmethod
    def _encode_image_to_base64(image: np.ndarray) -> str:
        """Encode numpy image array to base64 string.
        
        Args:
            image: RGB image as numpy array (H, W, 3) with values 0-255
            
        Returns:
            Base64 encoded string
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    @staticmethod
    def _numpy_to_pil(image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image.
        
        Args:
            image: RGB image as numpy array (H, W, 3) with values 0-255
            
        Returns:
            PIL Image object
        """
        return Image.fromarray(image.astype('uint8'), 'RGB')
