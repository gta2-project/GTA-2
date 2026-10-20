"""Message bus for inter-agent communication."""
from typing import Callable, Dict, List, Any
from datetime import datetime
import json
import threading

class Message:
    """Represents a message between agents."""
    
    def __init__(self, sender: str, receiver: str, message_type: str, content: Any):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.timestamp = datetime.now()
        self.id = f"{sender}_{receiver}_{self.timestamp.timestamp()}"
    
    def to_dict(self):
        """Convert message to dictionary."""
        return {
            'id': self.id,
            'sender': self.sender,
            'receiver': self.receiver,
            'message_type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.sender} -> {self.receiver}: {self.message_type}"


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.lifecycle_lock = threading.RLock()
        self.active_handlers = 0
        self.lazy_clients = False
        self.agent_snapshots = {}
        self.workspace = None

    def run_background(self, target, *args):
        """Register work before spawning, including recursively dispatched work."""
        with self.lifecycle_lock:
            self.active_handlers += 1
        def run():
            try:
                target(*args)
            finally:
                with self.lifecycle_lock:
                    self.active_handlers -= 1
        thread = threading.Thread(target=run, daemon=True)
        try:
            thread.start()
        except BaseException:
            with self.lifecycle_lock:
                self.active_handlers -= 1
            raise
        return thread
    
    def subscribe(self, agent_name: str, callback: Callable):
        """Subscribe an agent to receive messages."""
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        # Prevent duplicate subscriptions
        if callback not in self.subscribers[agent_name]:
            self.subscribers[agent_name].append(callback)
    
    def publish(self, message: Message):
        """Publish a message to the bus."""
        with self.lifecycle_lock:
            self._publish(message)

    def _publish(self, message: Message):
        self.message_history.append(message)
        
        # Broadcast to all subscribers if receiver is "ALL"
        if message.receiver == "ALL":
            for agent_name, callbacks in self.subscribers.items():
                if agent_name != 'broadcast' and agent_name != message.sender:
                    for callback in callbacks:
                        callback(message)
        # Deliver to specific receiver
        elif message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                callback(message)
        
        # Also deliver to broadcast subscribers (for logging/monitoring)
        if 'broadcast' in self.subscribers:
            for callback in self.subscribers['broadcast']:
                callback(message)
    
    def get_messages_for(self, agent_name: str) -> List[Message]:
        """Get all messages for a specific agent."""
        return [msg for msg in self.message_history if msg.receiver == agent_name]
    
    def get_conversation_history(self, agent1: str, agent2: str) -> List[Message]:
        """Get conversation history between two agents."""
        return [
            msg for msg in self.message_history
            if (msg.sender == agent1 and msg.receiver == agent2) or
               (msg.sender == agent2 and msg.receiver == agent1)
        ]
    
    def clear_history(self):
        """Clear message history."""
        self.message_history.clear()
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages in the bus."""
        return self.message_history.copy()
