# Read the dashboard

**Prerequisites:** the [image-only tutorial](../start/first-task.md).
**Outcome:** know where each output comes from and where to send feedback.

## Follow the stages

| Panel | What it produces | What to inspect |
|---|---|---|
| Task Decomposer | Ordered subtasks | Object choice, sequence, missing steps |
| Skill Generator | Prioritized controller recipes and keypoint definitions | Constraints, axes, grasp timing |
| Parameter Setter | Numeric offsets and units | Distances, force magnitudes, angle units |
| Vision Module | Image keypoint locations and annotations | Visible surfaces and exact label placement |
| Robot Script | Generated Python | Geometry, recipe order, execution prerequisites |

1. Enter the task in the task input.
2. Wait until the current operation finishes.
3. Select stage tabs to inspect the outputs.
4. Send feedback to the stage responsible for the issue.
5. Inspect downstream changes before generating or running a script.

Messages and console events help explain progress, but the latest visible
artifact is not necessarily current. Requery and refinement invalidate
grounding/script readiness while work is in progress or after failure.

## The script panel

The script panel contains **Generate Script** and **Save Workflow**, followed
by execution controls. The library's Prepare view also provides **Save
Checkpoint**; its Run view provides the working-copy controls.

Saving, generating, and running are different actions. A checkpoint preserves
agent context; a workflow export preserves artifacts; Run starts supervised
hardware processes after checks.

## Two servers, separate state

`main.py` defaults to port 8000. `library.py` defaults to port 8001 and has
Prepare, Library, and Run navigation. They do not share an in-memory workflow.
For example, a task prepared in `main.py` does not automatically appear in the
library's Prepare workspace.

The applications coordinate managed robot ownership through a lock in the
same checkout. See [execution](execution.md) for process ownership and controls.

**Verify:** you can identify the stage that owns each part of a recipe.
Continue with [feedback](refinement.md) or [library workflows](library.md).
