"""GripperControl Controller - Open or close the gripper."""

from .base_controller import BaseController


class GripperControl(BaseController):
    """
    Open or close the gripper.
    
    state is 'open' or 'close'.
    command_time is a normalized value between 0 and 1 indicating when during the skill 
    the gripper action occurs (0 = beginning, 1 = end).
    object is an optional ObjectHandler instance specifying which object is being attached 
    (when closing) or detached (when opening) from the gripper.
    """

    def __init__(self, state, command_time, object=None):
        """
        Initialize GripperControl controller.
        
        Args:
            state: Gripper state - 'open' or 'close' (str)
            command_time: Normalized time during skill execution (float, 0.0-1.0)
            object: Optional ObjectHandler instance being grasped/released (ObjectHandler or None)
        """
        self.type = "gripper"

        self.state = state
        self.command_time = command_time
        self.object = object

        if self.object is not None:
            print(f"[GripperControl] Initialized: {self.state} at time {self.command_time} (object: {self.object.name})")
        else:
            print(f"[GripperControl] Initialized: {self.state} at time {self.command_time}")

    def compute_error(self, elapsed_time=0.0):
        """
        Compute gripper error.

        A GripperControl is time-scheduled: it only becomes active once
        elapsed_time >= self.command_time.  Before that trigger point, returns None

        Args:
            elapsed_time: Normalized skill time in [0, 1] (float)

        Returns:
            dict with keys {'error', 'type', 'command_time', 'state', 'object'}
        """
        # Not yet triggered - signal to project_errors to ignore this controller
        if elapsed_time < self.command_time:
            return {
                'error': 0, 
                'type': self.type, 
                'command_time': self.command_time, 
                'state': None,
                'object': self.object
            }

        # Normalized gripper targets: 1.0 = fully open, 0.0 = fully closed
        state_value = 1.0 if self.state == 'open' else 0.0

        return {
            'error': 0, 
            'type': self.type, 
            'command_time': self.command_time, 
            'state': state_value,
            'object': self.object
        }
    
    def __repr__(self):
        if self.object is not None:
            return f"GripperControl(state='{self.state}', time={self.command_time}, object={self.object.name})"
        else:
            return f"GripperControl(state='{self.state}', time={self.command_time})"

    def __repr__(self):
        return f"GripperControl(state='{self.state}', command_time={self.command_time})"
