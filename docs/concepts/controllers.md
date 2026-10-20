# Controllers and priorities

A controller computes an **error packet** from the current state and the
subtask's progress. It does not itself publish ROS messages or command RTDE.
A subtask contains a list of controllers evaluated together, in priority order.

The recipe list order is authoritative. The generator preserves that order;
it does not sort controllers using their displayed numeric `priority` fields.
Check that the labels and actual list order agree.

## Position: reserve directions

Each positional packet carries a 3D error and either one `axis` or multiple
`axes`. GTA projects lower-priority errors onto the space not already
claimed by higher-priority axes.

For example, if the first controller claims the vertical direction and
requests an upward correction, a later controller cannot cancel it with a
downward request. Its horizontal component can still contribute.

The controller must declare every direction it controls. A 3D position
attractor claims three axes; a planar spiral claims two. Declaring no axes
does not correctly reserve those degrees of freedom.

## Rotation: preserve the primary constraint

Rotational packets contain rotation-vector corrections. The first rotational
packet establishes the primary axis constraint. Subsequent corrections are
projected onto that axis when one is present. GTA accumulates the rotation
vectors, bounds their combined angle below pi, and converts the result to a
3 × 3 rotation matrix for the robot command.

The execution loop treats the first rotational controller differently from
later axis-alignment controllers through their `free_rotation` argument.
Do not substitute ordinary sequential matrix multiplication for this behavior.

## Force and gripper

A force packet contributes a direction and magnitude. The first usable force
controller is projected through the positional null space; multiple
simultaneous force requests are not composed into an arbitrary wrench.

Gripper packets are independent of Cartesian projection. Before their trigger
time they contain `state=None`; afterward they request open or closed.
The eligible packet with the greatest command time wins. Object attachment,
when requested, is updated through `ObjectHandler`.

## Time and units

`execute_subtask(..., duration=15.0)` uses a nominal 100 Hz loop and passes
normalized progress from 0 to 1 to controllers. A trigger of 0.5 means halfway
through the subtask, not half a second.

Position errors and offsets use metres; force magnitudes use newtons.
`AxisAlign` and `AxisWaypoint` constructor offsets use **degrees**,
although their internal rotational errors and robot pose rotation vectors
use radians. The server applies its own per-step motion limits.

See the [controller reference](../reference/controllers.md) or
[write a controller](../extend/controller.md).
