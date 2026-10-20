# Pipeline architecture

GTA separates **planning**, **visual grounding**, and **execution**. Planning
produces structured descriptions; grounding locates their named keypoints in
an image; generated code later combines 3D perception and robot state.

```{figure} ../_static/diagrams/pipeline.svg
:target: ../_static/diagrams/pipeline.svg
:alt: Four planning agents produce a grounded plan, followed by script generation and robot execution.
:class: diagram
```

## Agent responsibilities

1. **Task Decomposer:** turn an instruction into ordered subtasks.
2. **Skill Generator:** choose controllers, priority, axes, and semantic
   keypoint descriptions for each subtask.
3. **Parameter Setter:** supply numeric values, units, and explanations.
4. **Vision Module:** locate the required visible keypoints and produce
   annotated image and JSON artifacts.

The message bus drives the downstream chain:
`subtasks_updated` → `skills_updated` → `parameters_completed` →
`vision_completed`. These are internal stage events, not robot commands.
The application tracks background handlers as well as its visible busy stage.

## Where each responsibility lives

| Component | Responsibility |
|---|---|
| `agents/` and `communication/` | Model conversations, structured outputs, stage dispatch |
| `api/` and `frontend/` | Workspaces, user interaction, checkpoints, process supervision |
| `gta/` | Script generation, object geometry, controller projection, ROS publishing |
| `controllers/` | Compute task errors and control requests |
| `scene/` | Camera, robot state/server/helpers, and gripper device interfaces |

At runtime, robot state is read through the selected state backend. For UR5e,
this is RTDE Receive, not a ROS pose topic. Controllers compute errors using
that state, and GTA publishes commands to the selected motion server.

```{figure} ../_static/diagrams/execution-boundaries.svg
:target: ../_static/diagrams/execution-boundaries.svg
:alt: The supervisor owns server and script processes; the script reads camera and robot state, publishes ROS commands, and the server controls the arm and selected gripper.
:class: diagram
```

See [controllers and priorities](controllers.md) and
[runtime evaluation](runtime.md) for computation details, and
[execution](../using/execution.md) for the operator workflow.
