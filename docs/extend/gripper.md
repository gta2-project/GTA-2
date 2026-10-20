# Add a gripper

**Prerequisites:** Python and the [gripper contract](../reference/devices.md).
Read [ROS actions](../glossary.md) before implementing a ROS adapter.
**Outcome:** a fake goal-based gripper with completion and cancellation tests.

## 1. Select the gripper independently

`GRIPPER_TYPE=my_gripper` and `GTA_BACKEND=ros2` select
`scene/gripper/my_gripper/ros2.py`, exporting `Gripper(BaseGripper)`.
Provide the package's `__init__.py` too. `ROBOT_TYPE` is independent.
There is no fallback to Robotiq and no separate gripper-backend setting.

## 2. Implement the lifecycle

```{literalinclude} ../examples/fake_gripper.py
:language: python
:caption: docs/examples/fake_gripper.py — in-memory gripper example
```

The example uses `1.0` for open and `0.0` for closed. The bundled Robotiq
adapter's positions are listed in the [ROS reference](../reference/ros.md).

```python
from docs.examples.fake_gripper import FakeClient, Gripper

client = FakeClient()
gripper = Gripper(client=client)
gripper.wait_for_server()  # Raises if unavailable.
gripper.open()
gripper.close()
assert client.positions == [1.0, 0.0]
assert gripper.cancel() == []
assert gripper.shutdown() == []
assert client.destroyed
```

`close()` closes the jaws; `shutdown()` releases client resources.
Optional `half_open()` inherits a clear unsupported-operation
error unless implemented.

## 3. Keep ROS ownership with the caller

A production ROS 2 client accepts the caller's node,
`spin_until_complete(future, timeout_s)`, and optional `check_cancel`.
It must not initialize/shutdown the shared ROS context, create robot command
subscriptions, or open an RTDE connection. The supervised gripper-only
operation creates its own node and executor outside the device implementation.

Keep availability and goal/result waits bounded where the caller requests a
deadline. Rejected goals, failed action status, and timeouts are failures.
Cancellation can arrive before the goal handle: retain the pending
acknowledgement, then cancel a late accepted goal. Cancellation/cleanup must
remain callable after the normal operation cancellation flag has been set.
Return cleanup-error lists instead of hiding failed cleanup.

The fake makes future completion explicit so tests can inject rejection,
failure, timeout, and late acknowledgement. A real action implementation must
also handle ROS future exceptions and cancelled futures.

## 4. Follow the existing caller policies

The implementations in `scene/gripper/robotiq/` show the current integration:

- ROS 2 uses `control_msgs/action/GripperCommand`, tracks pending and active
  goals, and supports cancellation deadlines. A successful action status with
  `reached_goal=False` produces a warning rather than a failure.
- ROS 1 uses `robotiq_mm_ros`. Its server retains unbounded result waiting
  without strict success validation; the standalone gripper utilities request
  bounded waits and success checks.
- The robot servers keep GTA's open/close topic convention and suppress
  repeated identical commands. The gripper backend owns action details.

Use the goal units and open/closed positions defined by your gripper driver.

**Verify:** use fake futures to cover accepted/rejected/failed/timed-out goals,
unavailable servers, cancellation before acknowledgement, and cleanup errors.
Confirm a gripper-only operation creates no robot-command subscriptions or
arm connections. For action startup errors see
[execution troubleshooting](../troubleshooting/execution.md).

{download}`Download the offline example <../examples/fake_gripper.py>`
