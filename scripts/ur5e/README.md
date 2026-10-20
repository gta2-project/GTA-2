# Standalone UR5e utilities

These are personal, workcell-specific hardware utilities, **not automated
tests or portable setup examples**. Several connect to hardware or move the
arm on import. Do not import them into tests.

They do not use the dashboard's execution lock or managed cleanup supervision.
Never run them concurrently with GTA execution or another controller.
Review the robot address, recorded poses, tool settings, motion/force values,
and entire motion path before explicitly executing a script.

| Script | Purpose |
|---|---|
| `reset_joints.py` | Move to a recorded joint configuration |
| `reset_pouring.py` | Move to a recorded pouring configuration |
| `reset_robot.py`, `reset_robot_tcp.py` | Cartesian reset followed by a wrist-joint move |
| `open_gripper.py`, `half_open_gripper.py`, `close_gripper.py` | Send one goal through the selected gripper backend |
| `gripper_control.py` | Shared standalone ROS lifecycle and signal handling |
| `connection.py` | Connect RTDE Control/Receive and print state |
| `read_force_values.py` | Continuously read TCP force/torque |

The RTDE utilities have code-defined addresses and settings; they do not all
read `ROBOT_IP` from `.env`. The standalone reset settings are separate from
the GUI helper in `scene/robot/ur5e/helpers.py`. Editing one does not change
the other. Neither recorded reset pose is a generic safe home position.

## Gripper utilities

The three gripper entry points act only when executed, not imported. They
require `ROBOT_TYPE=ur5e`, select the independent `GRIPPER_TYPE`, and use
`GTA_BACKEND` for ROS 1/ROS 2. Start and source the external gripper controller
first. They do not need the GTA command server or an RTDE connection.

ROS 2 requires `rclpy`, `control_msgs`, and `action_msgs`. It uses
`ROS2_GRIPPER_ACTION_NAME` and `ROS2_GRIPPER_WAIT_TIMEOUT_S`.
ROS 1 requires `rospy`, `actionlib`, `actionlib_msgs`, and
`robotiq_mm_ros`; availability and result waits are bounded to ten seconds.

[Exact gripper mappings](../../docs/reference/ros.md) are implementation-specific:
ROS 2 normalized positions are open `0.0`, half-open `0.425`, closed `0.85`;
ROS 1 metre positions are `0.085`, `0.06`, and `0.0`. Effort/force is
`165`; ROS 1 velocity is `0.05`. Review your driver's interpretation.

Executing an open/half-open/close script moves the physical gripper.
Ctrl+C/SIGTERM requests goal cancellation before client shutdown; it is not a
hardware emergency stop. Rejection, timeout, failure, or failed cleanup returns
a nonzero exit status.

For supported dashboard operation, use the
[execution guide](../../docs/using/execution.md). For prerequisites, use the
[hardware checklist](../../docs/hardware/checklist.md).
