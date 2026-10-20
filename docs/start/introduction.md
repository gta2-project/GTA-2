# What GTA does

GTA-2 turns a natural-language manipulation task into a sequence of
**subtasks**, each expressed as a prioritized composition of controllers.
Four model-backed agents handle task decomposition, skill recipes, physical
parameters, and visual keypoints. You can inspect and refine these outputs
before generating a Python robot script.

You need basic Python familiarity, not prior experience with GTA. The
[glossary](../glossary.md) introduces the robotics vocabulary used here.

## Planning first, hardware later

| Capability | Image-file workflow | Connected workcell |
|---|---|---|
| Generate subtasks, recipes, and parameters | Yes, with model access | Yes, with model access |
| Locate image keypoints and refine annotations | Yes | Yes |
| Generate a script for inspection | Yes | Yes |
| Obtain a 3D point cloud | No | Requires a compatible camera pipeline |
| Read robot pose or execute a skill | No | Requires device setup and operator review |
| Save library checkpoints | Yes, after a valid parameterized plan | Yes |

Image-file mode is **hardware-free, not provider-free**. Model calls require
network access and credentials and may incur charges. It is not a physics
simulator, offline model, or guarantee that a generated task is executable.

## Two interfaces

- **`python main.py`** opens the planning dashboard at
  `http://127.0.0.1:8000`. Start here for one task.
- **`python library.py`** opens Prepare / Library / Run at
  `http://127.0.0.1:8001/admin`. Use it to save and reuse complete agent/workflow
  checkpoints.

Both expose the robot script panel and managed execution controls. Opening
either application, loading a checkpoint, or generating a script does not
start a robot run.

## What is included

The release provides camera backends for file input, ROS 1, and ROS 2;
a UR5e state reader and ROS command servers; a Robotiq gripper client;
controllers; and the two web interfaces. Robot motion uses RTDE directly.
ROS carries GTA commands and gripper actions; this is not a MoveIt Servo setup.

Device drivers, camera-to-base calibration, the external dense point-cloud
pipeline, and vendor robot configuration are separate prerequisites.
See [hardware setup](../hardware/checklist.md).

## What to review

Check object/keypoint choices, controller order, axes, units, distances,
forces, grasp timing, and your workcell configuration. GTA does not perform
general collision checking. A controller composition is not a certified
motion plan.

Continue with [installation](installation.md).
