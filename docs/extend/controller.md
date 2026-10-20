# Add a controller

**Prerequisites:** the application dependencies, basic Python/NumPy, and
[controller priorities](../concepts/controllers.md).
**Outcome:** a positional controller tested with in-memory inputs.

## 1. Implement the error contract

The example attracts a current point to a fixed target. It returns a positional
error in metres and claims all three translation directions. The GTA execution
client projects this packet with the other controller errors.

```{literalinclude} ../examples/point_attractor.py
:language: python
:caption: docs/examples/point_attractor.py — positional controller example
```

`current_point` is callable because the robot point changes during execution.
`target_point` is sampled at construction. Reject malformed or non-finite
inputs before producing a command packet. `elapsed_time` is normalized subtask
progress from 0 to 1, despite its name; this controller does not use it.

## 2. Test priority projection

Run from the repository root in the application environment:

```python
import numpy as np
from docs.examples.point_attractor import PointAttractor
from gta.error_projection import project_errors

controller = PointAttractor(lambda: np.zeros(3), [0.1, 0.2, 0.3])
packet = controller.compute_error(0.5)
position, rotation, force, gripper = project_errors([packet])
np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
np.testing.assert_allclose(rotation, np.eye(3))
assert force is None and gripper is None

first = {"type": "positional", "error": [0.04, 0, 0], "axis": [1, 0, 0]}
position, *_ = project_errors([first, packet])
np.testing.assert_allclose(position, [0.04, 0.2, 0.3])
```

The earlier X constraint wins. The attractor contributes only Y and Z in the
second case. Reversing the order makes the full-position attractor claim every
translation direction first. Projection consumes packet order, not the
`priority` field itself. The generator preserves the recipe's list order;
it does not sort that list by its numeric priority labels.

## 3. Register the controller

To make the example in `docs/examples/` available to the planner:

1. Put the implementation in `controllers/point_attractor.py`.
2. Export `PointAttractor` from `controllers/__init__.py`.
3. Add it to the generated `from controllers import (...)` block in
   `gta/script_generator.py`.
4. Add `"PointAttractor": [0]` to `apply_dynamic_parameters()` so the current
   point becomes a lambda in generated code. Otherwise it is evaluated once.
5. Teach `agents/prompts/skill_generator.py` its exact signature, axis
   ownership, ordering, and appropriate use. Update
   `agents/prompts/parameter_setter.py` if it introduces physical parameters,
   units, or constraints. Update task-planning guidance only if its capabilities
   actually change. Existing checkpoint prompts are preserved snapshots.
6. Test translation, priority interactions, invalid inputs, and the generated
   script's syntax with the [offline tests](../development/testing.md).

A recipe could then use
`PointAttractor(g_eef_TCP, g_Ocube_Ptop_Lcenter)`.
The existing translator already recognizes those keypoints. New argument
syntax or helper functions need their own translator and import support.

## 4. Compare with the supplied controllers

`PosAlign` constrains one axis instead of claiming a full position:

```{literalinclude} ../../controllers/pos_align.py
:language: python
:start-at: class PosAlign
```

Distances are metres; `AxisAlign` constructor offsets are degrees, while its
error packet is a rotation vector in radians. Force controllers emit force
packets with magnitudes in newtons.

**Verify:** run the [offline documentation example tests](../development/testing.md).
Then inspect the [controller catalog](../reference/controllers.md) and
[runtime evaluation rules](../concepts/runtime.md).
For unresolved names or stale generated output, see
[execution troubleshooting](../troubleshooting/execution.md).

{download}`Download the offline example <../examples/point_attractor.py>`
