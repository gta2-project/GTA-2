"""Base Controller - Abstract base class for all robot controllers."""

from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Abstract base class for all robot controllers.
    
    All controllers must inherit from this class and implement the compute_error() method.
    This ensures a consistent interface across all controller types.
    """
    
    def __init__(self):
        """Initialize base controller with type attribute."""
        self.type = "base"  # Should be overridden by subclasses
    
    @abstractmethod
    def compute_error(self, elapsed_time=0.0):
        """
        Compute the error for this controller.
        
        Args:
            elapsed_time: Time elapsed since skill started (float)
            **kwargs: Additional controller-specific arguments
        
        Returns:
            dict: Dictionary with keys:
                - 'error': The computed error (numpy array or scalar)
                - 'type': The controller type ('positional', 'rotational', 'force', 'gripper')
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement compute_error()")
    
    def __repr__(self):
        """String representation of the controller."""
        return f"{self.__class__.__name__}()"
