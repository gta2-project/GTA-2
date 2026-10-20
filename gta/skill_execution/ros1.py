"""ROS 1 skill execution backend."""

import time
import numpy as np
from ..error_projection import project_errors
import signal

try:
    import rospy
    from geometry_msgs.msg import Vector3, Twist
    from std_msgs.msg import String, Float32, Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[skill_execution] ROS not available, running in standalone mode")


# Global state for skill timing
_skill_start_time = None

# ROS publishers (initialized on first use)
_position_pub = None
_rotation_pub = None
_force_pub = None
_gripper_pub = None
_skill_status_pub = None
_ros_initialized = False
_STATUS_EVENT_SETTLE_S = 0.05


def _assert_finite_command(name, value):
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")


def initialize_ros_publishers():
    """
    Initialize ROS publishers for robot control commands.
    
    Call this once at startup if using ROS integration.
    If not called, skill_execution runs in standalone mode (prints only).
    """
    global _position_pub, _rotation_pub, _force_pub, _gripper_pub, _skill_status_pub, _ros_initialized
    
    if not ROS_AVAILABLE:
        print("[skill_execution] Cannot initialize ROS - rospy not available")
        return False
    
    try:
        # Position error: 3D translational error
        _position_pub = rospy.Publisher('/robot/position_error', Float32MultiArray, queue_size=1)
        
        # Rotation error: 3x3 rotation matrix (flattened to 9D array)
        _rotation_pub = rospy.Publisher('/robot/rotation_error', Float32MultiArray, queue_size=1)
        
        # Force command: 4D [axis_x, axis_y, axis_z, magnitude]
        _force_pub = rospy.Publisher('/robot/force_command', Float32MultiArray, queue_size=1)
        
        # Gripper command: 1D scalar. This is an event-like command, so keep
        # a small queue to avoid dropping a command at a skill boundary.
        _gripper_pub = rospy.Publisher('/robot/gripper_command', Float32, queue_size=10)
        
        # Skill status events
        # Status is an event stream, not a "latest value" command. Keep a small
        # queue so back-to-back skill_ended/skill_started events are not dropped.
        _skill_status_pub = rospy.Publisher('/robot/skill_status', String, queue_size=10)

        time.sleep(0.5)
        
        _ros_initialized = True
        print("[skill_execution] ROS publishers initialized")
        return True
        
    except Exception as e:
        print(f"[skill_execution] Error initializing ROS publishers: {e}")
        return False


def start_skill():
    """
    This function is called at the beginning of each skill (subtask).
    It prepares the robot for executing a new set of controllers.
    """
    global _skill_start_time

    _skill_start_time = time.time()
    print(f"\n[start_skill] Skill started at t={_skill_start_time:.3f}")
    
    # Publish skill start event to ROS
    if _ros_initialized and _skill_status_pub:
        try:
            _skill_status_pub.publish(String(data="skill_started"))
            time.sleep(_STATUS_EVENT_SETTLE_S)
        except Exception as e:
            print(f"[start_skill] Error publishing to ROS: {e}")

def update_control_commands(position_error, rotation_error, force_command, gripper_command):
    """
    Execute control commands on the robot.
    
    This function takes the projected errors and converts them into robot commands.
    It applies gains, rate limiting, and safety checks before sending to hardware.
    
    Args:
        position_error: numpy array (3,) - translational velocity command [m/s]
        rotation_error: numpy array (3, 3) - 3x3 rotation matrix representing orientation error
        force_command: numpy array (4,) [axis_x, axis_y, axis_z, magnitude], or None if no force controller this cycle
        gripper_command: float in [0, 1] (0=closed, 1=open), or None if no gripper command this cycle
    
    Returns:
        bool: True if commands executed successfully, False if safety violation
    """
    _assert_finite_command("position_error", position_error)
    _assert_finite_command("rotation_error", rotation_error)
    if force_command is not None:
        _assert_finite_command("force_command", force_command)
    if gripper_command is not None:
        _assert_finite_command("gripper_command", [gripper_command])

    try:
        # Position error: 3D vector as Float32MultiArray
        pos_msg = Float32MultiArray()
        pos_msg.data = position_error.astype(float).tolist()
        _position_pub.publish(pos_msg)
    
        # Rotation error: 3x3 rotation matrix (flattened to 9D array)
        rot_msg = Float32MultiArray()
        rot_msg.data = rotation_error.flatten().astype(float).tolist()
        _rotation_pub.publish(rot_msg)
        
        # Force command: 4D [axis_x, axis_y, axis_z, magnitude], or None if no force controller this cycle
        if force_command is not None:
            force_msg = Float32MultiArray()
            force_msg.data = force_command.astype(float).tolist()
            _force_pub.publish(force_msg)
        
        # Gripper command: 1D scalar, or None if no gripper controller is active this cycle
        if gripper_command is not None:
            _gripper_pub.publish(Float32(data=float(gripper_command)))
            
    except Exception as e:
        print(f"[execute_control_commands] Error publishing to ROS: {e}")
        return False
    
    return True


def end_skill():
    """
    This function is called at the end of each skill (subtask).
    """
    global _skill_start_time
    
    if _skill_start_time is not None:
        duration = time.time() - _skill_start_time
        print(f"[end_skill] Skill completed in {duration:.3f} seconds")
        _skill_start_time = None
    else:
        print("[end_skill] Skill ended")
    
    # Publish skill end event to ROS
    if _ros_initialized and _skill_status_pub:
        try:
            _skill_status_pub.publish(String(data="skill_ended"))
            time.sleep(_STATUS_EVENT_SETTLE_S)
        except Exception as e:
            print(f"[end_skill] Error publishing to ROS: {e}")


def execute_subtask(subtask_controllers, duration=15.0):
    """
    Execute a subtask with the given controllers.
    
    This method runs the control loop for a subtask by calling each
    controller's compute_error, projecting errors, and sending commands
    to the robot.
    
    Args:
        subtask_controllers: List of controller instances
        duration: Maximum duration to run the subtask (seconds)
    """
    start_time = time.time()
    start_skill()

    # Graceful Ctrl+C / SIGINT handling for interactive runs
    running = True
    def _signal_handler(sig, frame):
        nonlocal running
        print("\n[skill_execution] SIGINT received - stopping subtask...")
        running = False

    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _signal_handler)


    try:
        while running:
            # If running under ROS, also exit if ROS requests shutdown
            if ROS_AVAILABLE and getattr(__import__('rospy'), 'is_shutdown', lambda: False)():
                print("[skill_execution] ROS shutdown detected - stopping subtask")
                break

            elapsed_time = (time.time() - start_time) / duration  # Normalize to [0, 1]
            if elapsed_time > 1.0:
                elapsed_time = 1.0

            errors = []
            free_rotation = False


            for controller in subtask_controllers:
                if controller.type == "rotational":
                    errors.append(controller.compute_error(elapsed_time=elapsed_time, free_rotation=free_rotation))
                    free_rotation = True
                else:
                    errors.append(controller.compute_error(elapsed_time=elapsed_time))

            position_error, rotation_error, force_command, gripper_command = project_errors(errors)
            if not update_control_commands(position_error, rotation_error, force_command, gripper_command):
                raise RuntimeError("Failed to publish control commands")

            try:
                time.sleep(0.01)  # Control loop at 100 Hz
            except KeyboardInterrupt:
                # In some environments KeyboardInterrupt may be raised instead of SIGINT handler
                print("\n[skill_execution] KeyboardInterrupt caught - stopping subtask...")
                break
            if elapsed_time >= 1.0:
                break
    finally:
        # Restore previous SIGINT handler
        try:
            signal.signal(signal.SIGINT, prev_handler)
        except Exception:
            pass

        # Ensure we always call end_skill to clean up
        end_skill()
