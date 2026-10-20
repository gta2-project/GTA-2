# Set up the ROS 2 environment

**Prerequisites:** Ubuntu 22.04 for the Humble reference setup, installed GTA,
and access to the workcell's ROS workspace.
**Outcome:** the same Python interpreter used by GTA can import the ROS/device
packages it needs.

## 1. Install and source ROS

Follow the official [ROS 2 Humble installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
Then source ROS and your actual robot/camera workspace in the terminal that
will launch GTA:

```bash
source /opt/ros/humble/setup.bash
# Replace this with the real workspace path:
source /path/to/robot_ws/install/setup.bash
```

Replace `/path/to/robot_ws` with your robot/camera workspace path.

## 2. Keep Python and ROS compatible

Humble's Ubuntu binary Python packages target the system Python 3.10
environment. A separate Conda interpreter or mismatched native libraries can
import part of ROS successfully and still fail when creating an executor.

Use a compatible Python 3.10 environment with access to sourced packages.
When creating a dedicated hardware environment, `--system-site-packages`
can expose required system Python packages:

```bash
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Reuse an existing compatible environment if you have one. Install ROS and
message packages through the ROS installation, and follow the vendor/RTDE
instructions for device dependencies.

## 3. Select middleware

```text
GTA_BACKEND=ros2
CAMERA_BACKEND=ros2
ROBOT_TYPE=ur5e
GRIPPER_TYPE=robotiq
```

All participating ROS processes must use compatible DDS/domain/network
settings. `ROS_DOMAIN_ID` belongs to the sourced process environment; GTA
does not force a domain. Use the same value for camera, gripper, and
GTA terminals. Source the environment **before** launching either dashboard:
child processes inherit it.

## 4. Run read-only environment checks

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import rclpy, sensor_msgs, std_msgs, control_msgs, action_msgs; print(rclpy.__file__)"
python -c "import rtde_control, rtde_receive; print('RTDE imports available')"
printenv ROS_DISTRO ROS_DOMAIN_ID
ros2 node list
ros2 topic list
ros2 action list -t
```

Continue with [robot and gripper checks](robot.md) to verify device endpoints.

For motion control, GTA starts its server with the application's Python
interpreter and inherited sourced environment.

**Verify:** Python resolves the intended modules and the expected external
camera/gripper endpoints are visible. For import or environment errors, see
[environment troubleshooting](../troubleshooting/installation.md).
