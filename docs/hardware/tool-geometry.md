# Configure TCP, tool geometry, and calibration

**Prerequisites:** understand [coordinate frames](../concepts/frames.md) and
have measured your physical tool and camera geometry.
**Outcome:** place GTA's tool keypoint and named axes in the correct frame
without double-applying a TCP offset.

## 1. Identify the pose origin

The UR5e state backend returns `getActualTCPPose()`: the robot controller's
configured TCP in base coordinates. A TCP is a physical/controller-defined
tool-center point. Setting GTA's `TOOL_TCP_OFFSET` does **not** change it.

Before editing GTA configuration, establish which TCP the controller already
reports. Configure the physical robot's TCP and payload using the robot's
own tooling and verify them independently.

```{figure} ../_static/diagrams/tool-frames.svg
:target: ../_static/diagrams/tool-frames.svg
:alt: Robot-base transform reaches the controller TCP; a separate local GTA offset reaches the tracked tool keypoint.
:class: diagram
```

## 2. Measure the additional GTA keypoint

`TOOL_TCP_OFFSET` is a three-vector in metres, expressed locally relative
to the reported pose. GTA transforms this point into the base frame.

- If the reported TCP already coincides with the point you want GTA to track,
  the additional offset can be `[0,0,0]`.
- If the desired point is 10 cm along the reported frame's positive Z axis,
  the geometric example is `[0,0,0.10]`.
- Do not apply a flange-to-tip offset twice if the controller TCP already
  incorporates it.

Set the offset from your tool measurements. The bundled default
`[0,0,0.1628]` is workcell-specific.

This pure numeric example applies a rotated local offset without connecting
to a device:

```{literalinclude} ../examples/tool_point.py
:language: python
```

## 3. Define meaningful directions

`TOOL_AXES` supplies the local directions referred to as
`a_eef_x`, `a_eef_y`, and `a_eef_z`. The planning conventions describe
Z as the approach direction and Y as the pinch direction.

Set all three nonzero vectors in single-line JSON. GTA validates finite
values but deliberately does not normalize them or enforce a right-handed
orthonormal basis. The default vectors are not a standard identity frame.
Check their physical meaning rather than changing signs to make a matrix
look conventional.

## 4. Establish camera-to-base calibration

`T_BASE_CAM` is a 4 × 4 camera-to-base transform. Its translation uses
metres. Set it from your measured camera-to-base calibration.

In the GTA execution path, the incoming cloud must already be in base
coordinates. Configuring this matrix alone does not transform that cloud.
The camera position is used to orient estimated normals. Keep the upstream
transformation and GTA calibration consistent.

Check known scene points and surface-normal directions without commanding
motion. Review the RGB/cloud registration and the named world-axis directions
as well.

## 5. Apply and verify a change

1. Stop active execution and close the applications.
2. Edit `TOOL_TCP_OFFSET`, `TOOL_AXES`, and calibration only from reviewed measurements.
3. Restart the application so configuration is reloaded.
4. Requery the scene, regenerate the script, and inspect the embedded tool values.
5. Review the 3D geometry before an operator-approved hardware rehearsal.

Generated scripts freeze tool geometry. Old scripts and saved preparation
artifacts are not silently rewritten.

**Verify:** a known local point transforms to the expected base location,
axis directions match the actual tool, and no offset is counted twice.
Run the [offline example tests](../development/testing.md) to check the
calculation above. For incorrect geometry, see
[camera and grounding troubleshooting](../troubleshooting/vision.md).
