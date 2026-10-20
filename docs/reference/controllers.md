# Controllers and geometric helpers

This catalog is derived from `controllers/__init__.py`, the concrete
implementations, and the import block emitted by `gta/script_generator.py`.
A controller mentioned in a prompt is not necessarily implemented.

## Generated-script controllers

All of the following are exported and imported by the current generator.
`g1` or `a1` marked “live” is wrapped in a callable by the generator.
Other arguments are sampled when that subtask's controller is constructed.

| Controller | Arguments and units | What it contributes |
|---|---|---|
| `PosAlign(g1, g2, a1, o1)` | Live current point, reference point, direction, offset in m | Position error along one direction toward `g2 + a1 * o1` |
| `AxisAlign(a1, a2, a3, o1)` | Live current axis, reference axis, rotation axis, offset in degrees | Rotational alignment error, represented in radians |
| `PosWaypoint(g1, g2, a2, offsets)` | Live point, reference point, direction, list of offsets in m | Time-indexed positional waypoints along an axis |
| `AxisWaypoint(a1, a2, a3, offsets)` | Live axis, reference axis, rotation axis, list of offsets in degrees | Time-indexed angular waypoints |
| `ForceConstant(a1, o1)` | Direction, signed magnitude in N | One force axis; negative magnitude reverses direction |
| `GripperControl(state, command_time, object=None)` | `"open"`/`"close"`, normalized trigger time | Scheduled gripper command and optional object-frame attachment |
| `PosDMPSpiral(g1, a1, o1)` | Live point, plane normal, radius in m | Planar DMP trajectory claiming the two plane directions |

The execution loop supplies normalized progress, not wall-clock seconds, to
`compute_error(elapsed_time)`. The default subtask duration is 15 seconds
with a 100 Hz client loop. A trigger of `0.5` means halfway through the
subtask. Completion of the timed loop is not proof that a geometric goal was
reached. See [priorities](../concepts/controllers.md).

`PosDMPSpiral` loads the bundled model from
`controllers/models/spiral_planar.yaml`. Those learned trajectory coefficients
are not ParameterSetter output. The public `PosDMP` name aliases the spiral
implementation, but the generator imports `PosDMPSpiral`, not `PosDMP`.

## Python-only controller

`Keyboard1D(axis, key_positive="w", key_negative="s", speed=0.01,
deadzone=False)` is exported but not imported by generated scripts. Its
keyboard listener supplies a signed positional error along an axis.
`speed` is the error magnitude per computation, **not a calibrated robot
velocity**. It needs a graphical keyboard environment and is unsuitable for
headless examples. Stopping its listener is not a robot emergency stop.

No letter-drawing DMP implementation is bundled. Do not treat a planner's
unsupported expression as executable support; inspect the generated imports
and use the [extension workflow](../extend/controller.md) for new controllers.

## Packet and projection contract

`compute_error()` returns a dictionary. Positional packets declare `axis`
or `axes`; rotational errors are rotation vectors; force packets include
`axis` and `magnitude`; gripper packets include `state`,
`command_time`, and optionally `object`.

`gta.error_projection.project_errors(packets)` returns:

1. A three-value position error.
2. A 3×3 rotation-error matrix, identity for no rotation.
3. A four-value force command or `None`.
4. A scalar gripper command or `None`.

Only one usable force packet is applied at a time. Force and position share
the translational priority space. The latest triggered gripper time wins;
equal trigger times keep the earlier packet.

## Geometry vocabulary

Generated expressions use these names:

| Name | Meaning |
|---|---|
| `g_eef_TCP` | GTA's tool point, including the configured additional local offset |
| `g_O<object>_P<part>_L<location>` | Named scene keypoint, resolved through an ObjectHandler |
| `a_global_x/y/z` | Named world directions; see the actual [axis conventions](../concepts/frames.md) |
| `a_eef_x/y/z` | Named tool directions transformed by the current robot orientation |
| `a_surface_normal(g)` | Estimated local cloud normal, oriented using camera position |
| `a_along_keypoints(g1, g2)` | Unit direction from the first point toward the second |
| `a_crossprod_axes(a1, a2)` | Normalized cross product with fallback for near-parallel axes |
| `offset_<name>` | A named parameter resolved from the skill's offset-parameter values |

Public Python helpers in `gta.utils` also include `unit`,
`choose_perp_axis`, pose/matrix conversion, image/cloud access,
pixel-to-3D lookup, and visualization. Not every exported Python helper is
automatically available as a planner expression: the generator's emitted
imports and translator define that boundary.

Surface normals are computed during script preparation and stored as object
axes; they are not recomputed from live clouds on every control tick.
See [runtime evaluation](../concepts/runtime.md) and
[point-cloud limitations](../hardware/camera.md).
