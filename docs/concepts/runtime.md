# Runtime evaluation and attachment

Generated scripts translate semantic keypoints and axes into `ObjectHandler`
lookups. Some lookups are evaluated repeatedly; others are sampled when a
controller is constructed.

## Live arguments versus sampled targets

The generator wraps argument 0 of `PosAlign`, `AxisAlign`, `PosWaypoint`,
`AxisWaypoint`, and `PosDMPSpiral` in a callable. For example:

```python
# Recipe
PosAlign(g_eef_TCP, g_Ocube_Ptop_Lcenter, a_global_z, offset_clearance)

# Generated form
PosAlign(lambda: eef.get_k('TCP'), cube.get_k('top_center'),
         world.get_a('z'), offsets['clearance'])
```

The first argument reads the current tool keypoint during each error
calculation. The target point and axis are evaluated when the controller is
created for that subtask. They are not automatically redetected as an object
moves.

Surface normals requested by the recipe are estimated from the acquired
point cloud during script preparation and stored on their objects. The camera
position from `T_BASE_CAM` is used to orient normals toward the camera.

## What attachment means

When a `GripperControl` packet references an object:

- A close request converts that object's stored points and axes into
  robot-relative coordinates.
- An open request converts them back to world coordinates using the current
  robot pose.

An attached point can then follow the end effector through `get_k()`.
This is a geometric assumption triggered by the command, **not confirmation
that the object was successfully grasped**. It does not add perception-based
tracking.

## Frozen script geometry

`TOOL_TCP_OFFSET` and `TOOL_AXES` are copied into generated code. A saved
script retains those values even after the configuration changes. Restart the
application and regenerate/review the script after changing tool geometry.

Parameter values are inserted as supplied; labels such as `cm` do not
perform automatic unit conversion. See [controller units](../reference/controllers.md).

Implementation anchors: `gta/script_generator.py:apply_dynamic_parameters`,
`gta/object_handler.py:ObjectHandler`, and
`gta/error_projection.py:project_errors`.
