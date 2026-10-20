# ROS command contract

GTA clients publish control errors; the robot server turns them into
device-specific motion. ROS is not the UR5e motion interface itself: that
server uses RTDE directly. Camera topics and gripper actions are separate
interfaces.

## Skill command topics

| Topic | Message | Payload |
|---|---|---|
| `/robot/position_error` | `std_msgs/Float32MultiArray` | Three finite values `[dx,dy,dz]`, metres in base coordinates |
| `/robot/rotation_error` | `std_msgs/Float32MultiArray` | Nine finite values: row-major 3×3 rotation-error matrix |
| `/robot/force_command` | `std_msgs/Float32MultiArray` | `[axis_x,axis_y,axis_z,magnitude]`, direction and signed newtons |
| `/robot/gripper_command` | `std_msgs/Float32` | GTA convention: `1.0` open, `0.0` closed |
| `/robot/skill_status` | `std_msgs/String` | `skill_started` or `skill_ended` |

The table uses package/class shorthand. ROS 2 CLI type names include `msg`,
for example `std_msgs/msg/Float32MultiArray`; ROS 1 uses
`std_msgs/Float32MultiArray`.

The matrix is a relative correction, not an absolute pose or nine Euler
angles. The UR5e server forms a target rotation using
`R_error @ R_current`, then applies its angular step limit. Identity means
no rotation. Position errors are likewise limited by the server; they are
not Cartesian waypoints for an external planner.

ROS 2 uses reliable, keep-last, volatile QoS: depth 1 for position, rotation,
and force; depth 10 for gripper and status. Publisher initialization waits
for subscriptions on **all five** topics, up to
`ROS2_PUBLISHER_MATCH_TIMEOUT_S`. Discovery counts are a transport check,
not hardware readiness. Camera subscriptions use their own QoS.

An empty ROS 2 force array explicitly clears active force control. Ending a
skill also clears control state. ROS 1 preserves its existing status-driven
cleanup behavior; do not assume every ROS 1 client emits empty force arrays
as ROS 2 does.

The server maps gripper values at `0.5`: values at or above the threshold
mean open. Repeated identical open/close commands are suppressed.

## Selected Robotiq action mapping

| Operation | ROS 1 goal position (m) | Deployed ROS 2 goal position (normalized) |
|---|---:|---:|
| Open | 0.085 | 0.0 |
| Closed | 0.0 | 0.85 |
| Half-open utility | 0.06 | 0.425 |

ROS 1 uses `/ur5/move_gripper`,
`robotiq_mm_ros.msg.GripperCommandAction` with `GripperCommandGoal`, velocity
`0.05`, and force `165`. ROS 2 uses the configured
`ROS2_GRIPPER_ACTION_NAME`, default
`/robotiq_gripper_controller/gripper_cmd`, with
`control_msgs/action/GripperCommand` and maximum effort `165`.

:::{warning}
The deployed ROS 2 numbers are not metres, despite the standard action's
position field. They follow this controller integration's normalized
convention. Validate your driver's actual interpretation before sending a
goal; these mappings are not portable to every Robotiq controller.
:::

There is no GTA half-open skill-topic command; half-open is an optional
device/utility operation. `GripperControl` emits GTA's binary convention,
not the action positions above.

## Readiness and failure

The ROS 2 server waits for gripper availability before reporting readiness,
then requires RTDE Control and usable RTDE Receive state. The supervisor
also requires its structured readiness event; seeing a node in the ROS
graph alone is insufficient. Action rejection/failure/timeout stops an
active skill. A successful action status with `reached_goal=False` is
warning-only in the current ROS 2 client.

Use [read-only endpoint checks](../hardware/robot.md). Do not publish example
messages to these command topics as a connectivity test: they can move hardware.
