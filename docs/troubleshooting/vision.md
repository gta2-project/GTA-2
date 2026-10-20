# Camera and grounding

## File image does not appear

Confirm `CAMERA_BACKEND=file`, the configured `CAMERA_IMAGE_PATH`, and
that the file is readable by the application. Library sessions use their own
working image; check the image in the active session's workspace.
See [first task](../start/first-task.md).

## ROS image or point cloud times out

Read the exact topic in the error. In the same sourced environment as GTA, use the
[camera readiness checks](../hardware/camera.md) to inspect publisher count,
type, QoS, and data rate separately for each topic.

Check domain ID and ROS networking, full topic overrides, the external device
driver, and the external transformed-cloud producer. The reference ZED image
and transformed IGEV cloud are different endpoints. Start the external cloud
pipeline and verify each publisher's QoS and message stream.

The ROS 2 camera spins its node while waiting for messages. If publication and
discovery are working, check the camera backend's executor errors.

## Keypoints are on the wrong object or part

Check the selected image, keypoint names/descriptions, and whether the required
surface is visible. Use **vision feedback** to adjust annotations on the current
image. Use **requery** after physically changing the scene. These operations
have different purposes; see [vision workflow](../using/vision.md).

If the skill requests the wrong keypoint, refine the skill rather than only
dragging the interpretation toward an incompatible location. Broader feedback
can regenerate downstream model output.

## The 3D point is far away or the cloud cannot reshape

Stop before execution. Check:

1. RGB and cloud resolution, pixel ordering, and alignment.
2. Cloud base frame and physical units after conversion.
3. Exact cloud field layout; the current XYZRGB and XYZ conversion branches
   scale differently.
4. Whether NaN filtering removed points from an organized image grid.
5. Image padding and the landscape-image assumption in pixel-to-cloud lookup.
6. That image and cloud correspond to the same physical scene.

Verify scale and pixel alignment using known scene points. The
[camera input guide](../hardware/camera.md) describes the current limitations.

## Generate Script or Run remains disabled

Check the most recent operation error and wait for background handlers to
finish. Requery/refinement invalidates old readiness immediately. A failed
operation can leave historical files on disk without making them current.
Obtain valid grounding, then generate and inspect a fresh script.
Loaded checkpoint coordinates are a reference, not new-scene grounding.
