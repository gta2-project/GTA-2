"""ROS 2 command transport for GTA skill execution."""

import signal
import time

import numpy as np

from config import Config
from ..error_projection import project_errors
from ..managed_run import get_runtime, should_skip

try:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32, Float32MultiArray, String

    ROS_AVAILABLE = True
except ImportError:
    rclpy = None
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    Float32 = None
    Float32MultiArray = None
    String = None
    ROS_AVAILABLE = False


_TOPICS = (
    "/robot/position_error",
    "/robot/rotation_error",
    "/robot/force_command",
    "/robot/gripper_command",
    "/robot/skill_status",
)
_STATUS_EVENT_SETTLE_S = 0.05

_skill_start_time = None
_ros_node = None
_position_pub = None
_rotation_pub = None
_force_pub = None
_gripper_pub = None
_skill_status_pub = None
_ros_initialized = False
_force_command_active = False


def _assert_finite_command(name, value):
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")


def _qos(depth):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _publish_force_clear():
    global _force_command_active
    if _ros_initialized and _force_pub is not None and _force_command_active:
        _force_pub.publish(Float32MultiArray(data=[]))
        _force_command_active = False


def _destroy_node():
    global _ros_node, _position_pub, _rotation_pub, _force_pub
    global _gripper_pub, _skill_status_pub, _ros_initialized, _force_command_active

    if _ros_node is not None:
        try:
            _ros_node.destroy_node()
        except Exception:
            pass
    _ros_node = None
    _position_pub = None
    _rotation_pub = None
    _force_pub = None
    _gripper_pub = None
    _skill_status_pub = None
    _ros_initialized = False
    _force_command_active = False


def initialize_ros_publishers():
    """Create all command publishers and wait for the UR5e server to match them."""
    global _ros_node, _position_pub, _rotation_pub, _force_pub
    global _gripper_pub, _skill_status_pub, _ros_initialized

    if _ros_initialized:
        return True
    if not ROS_AVAILABLE:
        print("[skill_executionROS2] Cannot initialize ROS 2 - rclpy/std_msgs unavailable")
        return False

    try:
        if not rclpy.ok():
            rclpy.init(args=None)

        _ros_node = rclpy.create_node("gta_skill_executor")
        continuous_qos = _qos(1)
        event_qos = _qos(10)
        _position_pub = _ros_node.create_publisher(
            Float32MultiArray, _TOPICS[0], continuous_qos
        )
        _rotation_pub = _ros_node.create_publisher(
            Float32MultiArray, _TOPICS[1], continuous_qos
        )
        _force_pub = _ros_node.create_publisher(
            Float32MultiArray, _TOPICS[2], continuous_qos
        )
        _gripper_pub = _ros_node.create_publisher(Float32, _TOPICS[3], event_qos)
        _skill_status_pub = _ros_node.create_publisher(String, _TOPICS[4], event_qos)

        publishers = (
            _position_pub,
            _rotation_pub,
            _force_pub,
            _gripper_pub,
            _skill_status_pub,
        )
        timeout_s = max(0.0, float(Config.ROS2_PUBLISHER_MATCH_TIMEOUT_S))
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not all(
            publisher.get_subscription_count() > 0 for publisher in publishers
        ):
            if time.monotonic() >= deadline:
                missing = [
                    topic
                    for topic, publisher in zip(_TOPICS, publishers)
                    if publisher.get_subscription_count() == 0
                ]
                print(
                    "[skill_executionROS2] Timed out waiting for robot-server "
                    f"subscriptions: {', '.join(missing)}"
                )
                _destroy_node()
                return False
            rclpy.spin_once(_ros_node, timeout_sec=min(0.05, timeout_s))

        if not rclpy.ok():
            print("[skill_executionROS2] ROS 2 shutdown while waiting for subscriptions")
            _destroy_node()
            return False

        _ros_initialized = True
        print("[skill_executionROS2] ROS 2 publishers initialized")
        return True
    except Exception as exc:
        print(f"[skill_executionROS2] Error initializing ROS 2 publishers: {exc}")
        _destroy_node()
        return False


def start_skill():
    """Mark the start of a skill and publish the corresponding status event."""
    global _skill_start_time, _force_command_active

    _skill_start_time = time.time()
    _force_command_active = False
    print(f"\n[start_skill] Skill started at t={_skill_start_time:.3f}")
    if _ros_initialized and _skill_status_pub is not None:
        try:
            _skill_status_pub.publish(String(data="skill_started"))
            time.sleep(_STATUS_EVENT_SETTLE_S)
        except Exception as exc:
            print(f"[start_skill] Error publishing to ROS 2: {exc}")


def update_control_commands(position_error, rotation_error, force_command, gripper_command):
    """Publish one projected GTA controller update using the ROS 2 wire contract."""
    global _force_command_active

    _assert_finite_command("position_error", position_error)
    _assert_finite_command("rotation_error", rotation_error)
    if force_command is not None:
        _assert_finite_command("force_command", force_command)
    if gripper_command is not None:
        _assert_finite_command("gripper_command", [gripper_command])

    if not _ros_initialized:
        print("[skill_executionROS2] ROS 2 publishers are not initialized")
        return False

    try:
        position = np.asarray(position_error, dtype=float).reshape(3)
        rotation = np.asarray(rotation_error, dtype=float).reshape(3, 3)
        _position_pub.publish(Float32MultiArray(data=position.tolist()))
        _rotation_pub.publish(Float32MultiArray(data=rotation.reshape(-1).tolist()))

        if force_command is not None:
            force = np.asarray(force_command, dtype=float).reshape(4)
            _force_pub.publish(Float32MultiArray(data=force.tolist()))
            _force_command_active = True
        else:
            _publish_force_clear()

        if gripper_command is not None:
            _gripper_pub.publish(Float32(data=float(gripper_command)))
    except Exception as exc:
        print(f"[update_control_commands] Error publishing to ROS 2: {exc}")
        return False
    return True


def end_skill():
    """End the current skill, explicitly clear force, and publish its status event."""
    global _skill_start_time

    if _skill_start_time is not None:
        duration = time.time() - _skill_start_time
        print(f"[end_skill] Skill completed in {duration:.3f} seconds")
        _skill_start_time = None
    else:
        print("[end_skill] Skill ended")

    if _ros_initialized and _skill_status_pub is not None:
        try:
            _publish_force_clear()
            _skill_status_pub.publish(String(data="skill_ended"))
            time.sleep(_STATUS_EVENT_SETTLE_S)
        except Exception as exc:
            print(f"[end_skill] Error publishing to ROS 2: {exc}")


def execute_subtask(subtask_controllers, duration=15.0):
    """Run the existing projected-controller loop at 100 Hz."""
    start_time = time.time()
    start_skill()
    running = True

    def _signal_handler(_sig, _frame):
        nonlocal running
        print("\n[skill_executionROS2] SIGINT received - stopping subtask...")
        running = False
        runtime = get_runtime()
        if runtime:
            runtime.skipped = True

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while running:
            if should_skip():
                break
            if ROS_AVAILABLE and _ros_initialized:
                if not rclpy.ok():
                    print("[skill_executionROS2] ROS 2 shutdown detected - stopping subtask")
                    break
                rclpy.spin_once(_ros_node, timeout_sec=0.0)

            elapsed_time = min((time.time() - start_time) / duration, 1.0)
            errors = []
            free_rotation = False
            for controller in subtask_controllers:
                if controller.type == "rotational":
                    errors.append(
                        controller.compute_error(
                            elapsed_time=elapsed_time,
                            free_rotation=free_rotation,
                        )
                    )
                    free_rotation = True
                else:
                    errors.append(controller.compute_error(elapsed_time=elapsed_time))

            commands = project_errors(errors)
            if not update_control_commands(*commands):
                raise RuntimeError("Failed to publish control commands")

            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("\n[skill_executionROS2] KeyboardInterrupt caught - stopping subtask...")
                break
            if elapsed_time >= 1.0:
                break
    finally:
        try:
            signal.signal(signal.SIGINT, previous_handler)
        except Exception:
            pass
        end_skill()
