# Execution and cleanup

:::{warning}
If hardware is moving unexpectedly, use the workcell's physical emergency
stop and established safety procedure. Browser controls, ROS messages, and
process termination are not a safety-rated stop.
:::

## Run is unavailable

Managed execution requires Linux, `GTA_BACKEND=ros2`, an idle workspace,
valid current grounding, and a current generated script. Requery and generate
again after a scene or plan change. Do not edit a frozen execution copy.
Reset/gripper controls do not need a script, but still require hardware
readiness and exclusive control.

## Another GTA server is detected

The supervisor checks local processes and the ROS graph and refuses to take
over another GTA server. Identify the owner and stop it deliberately before
using Run. Do not launch `scene/robot/server.py` separately for a managed run.
External camera and gripper drivers should remain running.

Both dashboards in the same checkout share an execution lock. Separate
checkouts and personal hardware scripts are not made safe by that lock.
Do not delete a lock file as a substitute for checking active processes.

## ROS probe fails before the server starts

An import/executor/context traceback can mean incompatible Python/ROS
libraries. Verify the application's interpreter, sourced Humble environment,
and robot workspace using [ROS 2 setup](../hardware/ros2.md).

## Publishers time out or the server never becomes ready

All five robot command subscriptions must match. Verify `GTA_BACKEND`,
domain ID, the selected robot implementation, and server logs.
“Subscribed” is only one startup step: gripper action availability, RTDE
control, and fresh robot state must also succeed.

The managed supervisor waits up to 30 seconds. The gripper wait defaults to
10 seconds and client subscription matching to 5 seconds. Inspect the first
failed startup step using the [endpoint checks](../hardware/robot.md).

## Open3D cannot open the preview

A local display and functioning OpenGL context are required. Headless shells,
remote display forwarding, and mixed graphical libraries can produce GLFW/GLX
errors. Test display/library configuration outside a robot run. A preview
startup failure stops the script. Cancel remains available while the preview
is open.

## Skip does not work during preparation

Skip applies only to an active subtask. It is disabled while the server starts,
the script prepares perception, or the 3D preview is open. One pending Skip is
allowed, tagged to the run and subtask, so repeated clicks cannot skip a later
subtask. Cancel stops the entire run instead.

## “Exited without confirming hardware cleanup”

This is a failure, not successful completion. Inspect the server log and
structured execution events for the first error. Cleanup must acknowledge
stopping force/servo control, cancelling goals, and releasing owned resources.
A crash, missing cleanup event, failed stop, or forced termination can prevent
confirmation.

The run remains **Stopping** until its processes exit. After the five-second
graceful window the supervisor may forcibly terminate owned processes, which
does not prove a hardware stop succeeded. Check the robot/controller state
using the workcell's safety procedure before trying again. Do not suppress
the message or start another server while cleanup is unresolved.

Logs are retained in the workspace's `execution_logs/<run-id>/`. Review
[storage and privacy](../reference/artifacts.md) before sharing them.
