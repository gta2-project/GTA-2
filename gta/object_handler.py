"""Object Handler for robot script components."""
from scene.robot import RobotState
from gta.utils import get_transformation_matrix_from_pose
import numpy as np
from scipy.spatial.transform import Rotation as R

class ObjectHandler:
    """Handles object data including keypoints, axes, and frame information."""
    
    def __init__(self, name: str, keypoints: dict = None, axes: dict = None, frame: str = None):
        """
        Initialize ObjectHandler.
        
        Args:
            name: Name of the object
            keypoints: Dictionary of keypoint names to their 3D positions
            axes: Dictionary of axis information
            frame: Reference frame identifier
        """
        self.name = name
        self.keypoints = keypoints if keypoints is not None else {}
        self.axes = axes if axes is not None else {}
        self.frame = frame if frame is not None else "world"
    
    def add_keypoint(self, keypoint_name: str, position):
        """Add or update a keypoint position."""
        self.keypoints[keypoint_name] = position
    
    def get_k(self, keypoint_name: str):
        """
        Get a keypoint position by name according to world frame.
        If the keypoint is in robot frame, the position will be transformed
        to world frame using the current frame information.
        """
            
        if self.frame == "robot":
            # Calculate the world frame position without changing the frame
            T_robot = get_transformation_matrix_from_pose(RobotState().get_tcp_pose())
            keypoint_relative_position = self.keypoints.get(keypoint_name)
            return (T_robot @ np.concatenate([keypoint_relative_position, [1]]))[:3]

        return self.keypoints.get(keypoint_name)
    
    def add_axis(self, axis_name: str, axis_data):
        """Add or update axis information."""
        self.axes[axis_name] = axis_data
    
    def get_a(self, axis_name: str):
        """
        Get axis data by name according to world frame.
        """

        if self.frame == "robot":
            # Calculate the world frame position without changing the frame
            T_robot = get_transformation_matrix_from_pose(RobotState().get_tcp_pose())
            axis_relative_direction = self.axes.get(axis_name)
            return T_robot[:3, :3] @ axis_relative_direction

        return self.axes.get(axis_name)
    
    def set_frame(self, frame: str):
        """Set the reference frame."""
        if self.frame == frame:
            return
        
        self.frame = frame

        if frame == "robot":
            # Transform keypoints and axes to robot frame
            T_robot = get_transformation_matrix_from_pose(RobotState().get_tcp_pose())
            T_robot_inv = np.linalg.inv(T_robot)
            for keypoint_name, position in self.keypoints.items():
                self.keypoints[keypoint_name] = (T_robot_inv @ np.concatenate([position, [1]]))[:3]
            for axis_name, axis_data in self.axes.items():
                self.axes[axis_name] = T_robot_inv[:3, :3] @ axis_data  # Rotate axes to robot frame

        elif frame == "world":
            # Transform keypoints and axes to world frame
            T_robot = get_transformation_matrix_from_pose(RobotState().get_tcp_pose())
            for keypoint_name, position in self.keypoints.items():
                self.keypoints[keypoint_name] = (T_robot @ np.concatenate([position, [1]]))[:3]
            for axis_name, axis_data in self.axes.items():
                self.axes[axis_name] = T_robot[:3, :3] @ axis_data  # Rotate axes to world frame

    def __repr__(self):
        return f"ObjectHandler(name='{self.name}', keypoints={len(self.keypoints)}, axes={len(self.axes)}, frame='{self.frame}')"
