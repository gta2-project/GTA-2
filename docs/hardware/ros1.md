# ROS 1 differences

ROS 1 remains an alternative camera and GTA command backend. It is not the
dashboard-managed execution path documented in the main hardware tutorial.

| Setting or behavior | ROS 1 |
|---|---|
| Camera | `CAMERA_BACKEND=ros1`, with `rospy` and `sensor_msgs` |
| Commands and gripper | `GTA_BACKEND=ros1` |
| ROS master | `ROSMASTER_IP`, `ROSMASTER_PORT`; an existing environment URI can take precedence |
| Robotiq action | `/ur5/move_gripper`, supplied by `robotiq_mm_ros` |
| Gripper goals | Open 0.085 m, half-open 0.06 m, closed 0.0 m; velocity 0.05; force 165 |
| Managed Run / GUI helpers | Not supported for ROS 1 |

Source a compatible ROS 1 and robot workspace that supplies `rospy`,
`actionlib`, `actionlib_msgs`, and the required Robotiq messages.
Do not mix ROS 1 and ROS 2 package paths in one interpreter without a
deliberately validated environment.

The command topics and basic payload shapes match the
[ROS contract](../reference/ros.md), but lifecycle/QoS and force-deactivation
details differ. ROS 2 subscriber discovery and reliable keep-last QoS settings
do not apply to ROS 1.

Camera selection remains independent: selecting a ROS camera does not
automatically select the robot middleware. The ROS 2 full-topic override
settings are not a ROS 1 camera override mechanism.

Use the selected server launcher only in a reviewed standalone workflow.
Standalone robot/gripper utilities may move hardware immediately and do not
acquire the dashboard execution lock.

See [device interfaces](../reference/devices.md) for the implementation
boundaries and [troubleshooting](../troubleshooting/execution.md) for ownership
and shutdown issues.
