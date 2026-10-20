# Add a robot

**Prerequisites:** Python, NumPy, [coordinate frames](../concepts/frames.md),
and the [device contracts](../reference/devices.md).
**Outcome:** tested state and reset helpers using fake devices, followed by
the integration steps for a motion server.

## 1. Separate the three capabilities

A robot implementation has a **state reader**, optional **helpers**, and a
**server**. The state reader returns the current pose; helpers provide reset
operations. The server interprets GTA commands and controls motion.

`ROBOT_TYPE=my_robot` resolves:

```text
scene/robot/my_robot/
├── __init__.py
├── state.py       # RobotState(BaseRobotState)
├── helpers.py     # Helpers(BaseRobotHelpers)
└── ros2.py        # main(), including the managed server protocol
```

The public selectors import only the requested implementation and validate
concrete subclasses. There is no UR5e fallback. Importing a module must not
connect hardware.

## 2. Implement and exercise fake state and helpers

```{literalinclude} ../examples/fake_robot.py
:language: python
:caption: docs/examples/fake_robot.py — in-memory state and reset helpers
```

The pose is `[x, y, z, rx, ry, rz]`: metres and a rotation vector in radians,
in the same base frame as the point cloud. The pose origin must match the frame
in which you define GTA's local tool points. Return a fresh finite array or
raise `RobotStateError`; do not return stale data as a success.

Try this from the repository root:

```python
from docs.examples.fake_robot import FakeArm, Helpers, RobotState

clock = lambda: 10.0
arm = FakeArm(clock=clock)
state = RobotState(arm, clock=clock)
assert state.get_tcp_pose().shape == (6,)
events, errors = [], []
Helpers(factory=lambda: arm, clock=clock).reset_joints(
    check_cancel=lambda: None,
    on_started=lambda: events.append("started"),
    errors=errors,
    wait=lambda seconds: None,
)
assert events == ["started"] and not errors
assert not arm.connected
```

In a real helper, `check_cancel()` may raise at any point. Check before opening
connections and while waiting for motion. `wait(seconds)` is supplied by the
caller and is interruptible. Call `on_started()` when ready to move; device
code must not emit GUI-specific event names. Stop and disconnect in `finally`,
recording cleanup errors even when the primary operation fails.

## 3. Understand the UR5e implementation

Read these selected source files alongside the contract:

- `scene/robot/ur5e/state.py`: a shared RTDE Receive connection per process,
  finite pose validation, freshness checking, and reconnect behavior.
  `close()` releases receive resources, not the motion controller.
- `scene/robot/ur5e/helpers.py`: a recorded joint reset, asynchronous completion
  polling, timeout/tolerance checks, and `stopJ` cleanup. Its settings are
  workcell-specific, not a universal home pose.
- `scene/robot/ur5e/ros2.py`: position/rotation/force/gripper/status subscriptions,
  a 100 Hz loop, RTDE `servoL` and `forceMode`, step limits, and force-phase
  reconnect handling. A different robot needs its own equivalent behavior.

## 4. Implement the motion server

Expose a callable `main()` in the selected ROS backend. Consume the
[ROS command contract](../reference/ros.md); validate incoming values before
motion and maintain explicit active-skill state. Delegate gripper operations
to the independently selected `scene.gripper.Gripper`.

For dashboard-managed execution, use the existing `gta.managed_run` control
channel. The current server's protocol is:

1. Establish command subscriptions, verify gripper action availability,
   initialize motion control and receive interfaces, and obtain usable state.
2. Print the terminal readiness marker and emit `server_ready` only after
   those steps succeed. The supervisor has a 30-second startup deadline.
3. Poll cancellation and channel closure during the control loop, action
   waits, and reconnect delays. Managed execution has no fixed lifetime.
4. On control failure emit `server_failed` with an error, so the supervisor
   stops the script rather than letting it publish to a dead server.
5. Cancel pending gripper goals, stop force/servo control, stop/disconnect
   owned hardware, release ROS resources, and emit `server_cleaned` with
   an `errors` list. Do not report clean completion when cleanup failed.

The supervisor allows five seconds for graceful shutdown before escalation.
Cleanup has to be responsive to that deadline. See
[execution boundaries](../using/execution.md).

**Verify:** test unavailable/stale state, cancelled reset, failed cleanup,
startup ordering, control failures, and process termination using fakes.
For backend-selection errors see [execution troubleshooting](../troubleshooting/execution.md).

{download}`Download the offline example <../examples/fake_robot.py>`
