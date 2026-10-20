# Connect the UR5e and gripper

**Prerequisites:** [ROS 2 environment](ros2.md), reviewed robot network/settings,
an installed RTDE interface, and an external gripper controller.
**Outcome:** understand the supplied robot/gripper integration and ownership.

## 1. Configure the arm

```text
ROBOT_TYPE=ur5e
ROBOT_IP=192.168.1.10
GTA_BACKEND=ros2
```

The address is an example default; set your actual robot's address. Review
the robot's TCP, payload, operating mode, safety limits, and network access
using its vendor documentation. Follow the
[UR RTDE documentation](https://sdurobotics.gitlab.io/ur_rtde/) for installation
and controller prerequisites.

The state reader uses RTDE Receive. The motion server uses RTDE Control
directly for servo and force control. Do not run another trajectory controller,
GTA server, or standalone reset script against the same arm concurrently.

## 2. Configure the selected gripper

```text
GRIPPER_TYPE=robotiq
ROS2_GRIPPER_ACTION_NAME=/robotiq_gripper_controller/gripper_cmd
ROS2_GRIPPER_WAIT_TIMEOUT_S=10.0
```

Start the external gripper driver/action server using its own instructions.
The ROS 2 client requires
[`control_msgs/action/GripperCommand`](https://github.com/ros-controls/control_msgs/blob/humble/control_msgs/action/GripperCommand.action).
Use read-only graph inspection to check the name and type:

```bash
ros2 action list -t
ros2 action info /robotiq_gripper_controller/gripper_cmd
```

The bundled ROS 2 Robotiq adapter uses deployment-specific normalized
positions: **open 0.0, closed 0.85, half-open 0.425**, with maximum effort 165.
Do not interpret these as metres or assume another GripperCommand server uses
the same convention. Review/adapt the selected gripper backend before sending
any goal. The GTA command topic itself uses 1=open and 0=closed.

## 3. Let managed execution own the GTA server

Start `main.py` or `library.py` from the sourced environment. Managed Run
starts the selected server and waits for subscriptions, gripper availability,
RTDE Control, and a valid RTDE Receive pose before declaring readiness.

The server preserves this terminal readiness marker:

```text
[UR5e Server] Waiting for skill execution commands...
```

It also sends a structured readiness event to its supervisor. Merely printing
the earlier “Subscribed” line is not sufficient readiness.

For a deliberately reviewed **standalone** workflow, the launcher is:

```bash
python -m scene.robot.server
```

:::{warning}
The standalone command connects robot control, initializes the server, and
can accept motion commands. It is not an offline diagnostic. Do not run it
alongside dashboard-managed execution.
:::

Helper launch options such as `--operation reset_joints` require the GUI
supervisor's private control channel. They are not standalone terminal shortcuts.
Workcell-specific utilities are described in the repository's
`scripts/ur5e/README.md`.

**Verify:** the external gripper interface is correct, no competing controller
owns the arm, and managed startup reaches readiness. Continue with
[script review and execution](../using/execution.md).
