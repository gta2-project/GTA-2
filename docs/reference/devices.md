# Device interfaces

Public imports are lazy with respect to hardware. Constructing a selected
backend may connect a device; importing these contracts does not.

| Interface | Selector | Implementation |
|---|---|---|
| `scene.camera.Camera` | `CAMERA_BACKEND` | `scene/camera/file.py`, `ros1.py`, `ros2.py` |
| `scene.robot.RobotState` | `ROBOT_TYPE` | `scene/robot/<type>/state.py:RobotState` |
| `scene.robot.RobotHelpers` | `ROBOT_TYPE` | `scene/robot/<type>/helpers.py:Helpers` |
| `scene.gripper.Gripper` | `GRIPPER_TYPE` + `GTA_BACKEND` | `scene/gripper/<type>/<backend>.py:Gripper` |
| Robot server | `ROBOT_TYPE` + `GTA_BACKEND` | Backend `main()`, launched through `scene.robot.server` |

Only the camera selector has a fixed backend-name list. Robot/gripper type
names must match lowercase package-name rules. Missing implementations,
dependencies, and incomplete abstract contracts produce explicit errors.
They do not silently fall back to another device.

## Camera

```{literalinclude} ../../scene/camera/base.py
:language: python
```

The file backend supplies an RGB image but no point cloud. ROS backends own
their subscriptions and caching resources. See [adding a camera](../extend/camera.md)
for the additional capture call sites required by a new backend.

## Robot state and helpers

```{literalinclude} ../../scene/robot/base.py
:language: python
```

The UR5e reader shares a receive connection per process and checks freshness.
Do not assume `close()` on one shared reader leaves another reference
connected. No tool offset or physical TCP change is performed by the reader.

Helpers receive cancellation/start/wait callbacks and an error list. They
must clean up on both success and exception; they do not own GUI event names.
See [adding a robot](../extend/robot.md).

## Gripper

```{literalinclude} ../../scene/gripper/base.py
:language: python
```

ROS 2 callers supply a node and future-spinning callback. Optional wait-policy
arguments are implementation-specific; the current Robotiq client accepts
`check_cancel`, `poll_interval_s`, and `cancel_on_timeout`.
ROS 1 operations support caller-selected timeout and success-check policies.
The client owns goals, not the shared ROS lifecycle.

## Launcher boundaries

`python scene/robot/server.py` or `python -m scene.robot.server` launches
the selected server. This can initialize physical hardware.
`--help` parses without loading hardware implementations.

`--operation reset_joints`, `open_gripper`, and `close_gripper` are
supervisor-only launch modes. They require the existing control channel and
child role; they are not general terminal commands. Reset dispatches to the
robot helper, while gripper operations use a gripper-only ROS node and no RTDE
connection. See [execution](../using/execution.md) for supported operator controls.
