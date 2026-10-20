"""Select the robot-command transport without changing generated-script imports."""

from config import Config

if Config.GTA_BACKEND == "ros2":
    from .ros2 import (
        ROS_AVAILABLE,
        end_skill,
        execute_subtask,
        initialize_ros_publishers,
        start_skill,
        update_control_commands,
    )

    BACKEND = "ros2"
elif Config.GTA_BACKEND == "ros1":
    from .ros1 import (
        ROS_AVAILABLE,
        end_skill,
        execute_subtask,
        initialize_ros_publishers,
        start_skill,
        update_control_commands,
    )

    BACKEND = "ros1"
else:
    raise ValueError("GTA_BACKEND must be ros1 or ros2.")

__all__ = [
    "BACKEND",
    "ROS_AVAILABLE",
    "initialize_ros_publishers",
    "start_skill",
    "update_control_commands",
    "end_skill",
    "execute_subtask",
]
