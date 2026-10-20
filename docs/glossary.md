# Glossary

ROS
: Robot Operating System: middleware and tools for communicating between robot
  software components. ROS 1 and ROS 2 are different implementations.

Node
: A participant in the ROS communication graph, such as a camera publisher or
  GTA command server.

Topic
: A named message stream. Publishers send messages; subscribers receive them.

Action
: A goal-based ROS interface with acknowledgement, progress/result, and
  cancellation. GTA's ROS 2 gripper uses an action.

QoS
: Quality of Service: ROS 2 communication policies such as reliability and
  queue depth. Incompatible policies can prevent messages from arriving.

RTDE
: Real-Time Data Exchange, used here through UR RTDE libraries to read the
  UR5e's state and send motion/force-control commands.

TCP
: Tool Center Point: the tool pose configured in the robot controller. GTA
  can define an additional tool point relative to that reported pose; it does
  not change the robot's TCP configuration.

Pose
: Position and orientation together. GTA's robot-state pose contains three
  position values in metres and a three-value rotation vector in radians.

Rotation vector
: An axis-angle representation: direction gives the rotation axis and length
  gives the angle in radians. It is not roll/pitch/yaw.

Frame
: A coordinate system in which positions and directions are expressed.
  The robot base, reported TCP, camera, and attached object may have different
  frames; transforms relate them.

Keypoint
: A named geometric point used by a controller, such as the center of a
  cube's top face. Image keypoints must be mapped to 3D before robot execution.

Grounding
: Associating the plan's named keypoints with locations in a particular scene.
  A plan can remain useful while its old grounding becomes invalid.

Controller
: A function-like component that computes an error packet for a geometric,
  force, or gripper objective. GTA projects ordered packets into commands.

Priority
: The order in which controllers claim constrained directions. Lower-priority
  positional errors contribute only in the remaining translational space.

Point cloud
: A collection of 3D points. An organized, image-aligned cloud retains a known
  point for each image pixel; arbitrary unordered clouds do not.

Checkpoint
: A saved plan, context, history, and copied artifacts. Loading creates an
  independent working copy.

Session
: A mutable library working copy with its own workspace and execution records.

Requery
: Grounding the saved keypoint definitions again for a fresh scene image.
  Annotation feedback instead operates on the currently displayed image.

DMP
: Dynamic Movement Primitive: a parameterized trajectory representation. GTA
  supplies a planar spiral controller with a bundled learned model.
