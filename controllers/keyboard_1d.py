"""Keyboard1D Controller - 1D positional control via keyboard input."""

import numpy as np
from .base_controller import BaseController

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[Keyboard1D] Warning: pynput not available. Install with: pip install pynput")


class Keyboard1D(BaseController):
    """
    1D positional controller driven by keyboard input.
    
    Listens to two keys (positive and negative) and generates position error
    along a specified axis. Useful for teleoperation and manual control.
    
    Example:
        # Control along X-axis with 'w' (forward) and 's' (backward)
        controller = Keyboard1D(
            axis=np.array([1.0, 0.0, 0.0]),
            key_positive='w',
            key_negative='s',
            speed=0.01
        )
    """
    
    def __init__(self, axis, key_positive='w', key_negative='s', speed=0.01, deadzone=False):
        """
        Initialize Keyboard1D controller.
        
        Args:
            axis: 3D direction vector for movement (numpy array)
            key_positive: Key for positive direction movement (str)
            key_negative: Key for negative direction movement (str)
            speed: Movement speed in meters per control cycle (float)
            deadzone: If True, error is 0 when no key pressed (bool)
        """
        super().__init__()
        self.type = "positional"
        
        self.axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(self.axis)
        if axis_norm > 1e-9:
            self.axis = self.axis / axis_norm
        else:
            raise ValueError("Axis vector must have non-zero magnitude")
        
        self.key_positive = key_positive
        self.key_negative = key_negative
        self.speed = speed
        self.deadzone = deadzone
        
        # Key state tracking
        self._key_state = {
            key_positive: False,
            key_negative: False
        }
        
        # Start keyboard listener if available
        if PYNPUT_AVAILABLE:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self._listener.start()
            print(f"[Keyboard1D] Initialized: axis={self.axis}, "
                  f"keys=[{key_positive}(+), {key_negative}(-)], speed={speed}")
        else:
            self._listener = None
            print(f"[Keyboard1D] Warning: Running in dummy mode (pynput not installed)")
    
    def _on_press(self, key):
        """Callback for key press events."""
        try:
            # Handle character keys
            key_char = key.char
            if key_char in self._key_state:
                self._key_state[key_char] = True
        except AttributeError:
            # Special keys (e.g., arrow keys, ctrl, etc.)
            pass
    
    def _on_release(self, key):
        """Callback for key release events."""
        try:
            key_char = key.char
            if key_char in self._key_state:
                self._key_state[key_char] = False
        except AttributeError:
            pass
    
    def compute_error(self, elapsed_time=0.0):
        """
        Compute position error based on current key state.
        
        Args:
            elapsed_time: Time elapsed since skill started (float, unused)
        
        Returns:
            dict: {
                'error': 3D position error vector (numpy array),
                'type': 'positional',
                'axis': axis direction for nullspace projection
            }
        """
        # Compute net input: +1 if positive key, -1 if negative key, 0 if both/neither
        net_input = 0.0
        
        if self._key_state[self.key_positive]:
            net_input += 1.0
        if self._key_state[self.key_negative]:
            net_input -= 1.0
        
        # Apply deadzone if enabled
        if self.deadzone and abs(net_input) < 0.1:
            net_input = 0.0
        
        # Compute error along axis
        error = self.axis * self.speed * net_input
        
        return {
            'error': error,
            'type': self.type,
            'axis': self.axis  # Provide axis for nullspace projection
        }
    
    def stop(self):
        """Stop the keyboard listener."""
        if self._listener:
            self._listener.stop()
            print("[Keyboard1D] Keyboard listener stopped")
    
    def get_key_state(self):
        """Get current state of tracked keys (for debugging)."""
        return self._key_state.copy()
    
    def __del__(self):
        """Cleanup keyboard listener on deletion."""
        self.stop()
    
    def __repr__(self):
        return (f"Keyboard1D(axis={self.axis}, "
                f"keys=[{self.key_positive}(+), {self.key_negative}(-)], "
                f"speed={self.speed})")
