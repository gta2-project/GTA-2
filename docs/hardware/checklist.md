# Hardware prerequisites and readiness

**Prerequisites:** a completed image-only workflow and a reviewed workcell.
**Outcome:** identify what must be ready before GTA can execute a skill.

The reference integration uses **ROS 2 Humble, UR5e, Robotiq, and ZED**.
GTA provides interfaces to this workcell; it does not install every external
driver, calibrate the camera, or validate a robot's motion envelope.

## 1. Separate the responsibilities

| Supplied by GTA | Supplied by your workcell |
|---|---|
| Camera subscribers and image-file input | Camera SDK/driver and ROS publishers |
| Image keypoints and pixel-to-cloud lookup | Dense aligned point cloud in the correct frame and units |
| UR5e RTDE state/control integration | Robot network, controller setup, RTDE packages |
| Gripper action client | Compatible gripper action server |
| Configurable tool geometry | Measured TCP, tool axes, payload, and calibration |
| Managed script/server supervision | Physical safety measures and operator review |

## 2. Check the workcell before starting a run

1. Complete the [ROS environment checks](ros2.md) in the terminal used to start GTA.
2. Verify the [RGB and point-cloud contract](camera.md), including units, ordering,
   registration, and coordinate frame.
3. Verify [robot and gripper configuration](robot.md), including action semantics.
4. Review [TCP/tool geometry and calibration](tool-geometry.md).
5. Confirm only one intended system owns robot motion.
6. Review all reset poses, tool clearance, workpiece placement, and stopping procedures.
7. Requery the current scene and regenerate/review the script.

:::{warning}
A green application status or a discovered ROS topic is not a safety check.
Confirm physical clearance and keep the robot's emergency stop accessible.
Do not use a generated skill as the first test of unverified calibration.
:::

## 3. Choose one execution owner

For dashboard-managed execution, leave the camera pipeline and gripper driver
running but **do not separately start the GTA motion server**. The dashboard
starts and stops its own server.

Standalone execution is a separate operator-controlled workflow. Never mix it
with a managed run or run standalone reset/gripper utilities concurrently.

**Verify:** each prerequisite is independently established and the current
script has been reviewed. Continue with [managed execution](../using/execution.md).
