# Add a camera

**Prerequisites:** [camera input requirements](../hardware/camera.md),
Python, and NumPy. ROS is needed only for the topic-adaptation route.
**Outcome:** determine whether configuration is sufficient, then exercise a
recorded-array camera without hardware.

## 1. Try topic configuration first

A camera publishing compatible ROS 2 images and point clouds often needs no
new class. Set full topic names in `.env`:

```text
CAMERA_BACKEND=ros2
CAMERA_COLOR_TOPIC=/camera/rgb/image_rect
CAMERA_POINT_CLOUD_TOPIC=/camera/points_in_robot_base
```

Replace these example names with your camera's topics and restart GTA.
Confirm the image encoding, aligned organized cloud,
base frame, field layout, and units using the
[readiness checks](../hardware/camera.md).

## 2. Learn the interface with recorded arrays

```{literalinclude} ../examples/array_camera.py
:language: python
:caption: docs/examples/array_camera.py — recorded-array camera example
```

From the repository root:

```python
import numpy as np
from docs.examples.array_camera import ArrayCamera

rgb = np.zeros((2, 3, 3), dtype=np.uint8)
xyz = np.zeros((2, 3, 3), dtype=float)  # synthetic organized points, in metres
camera = ArrayCamera(rgb, xyz)
assert camera.get_rgb_image().shape == (2, 3, 3)
assert camera.get_point_cloud().shape == (6, 3)
camera.close()
```

`get_rgb_image()` returns RGB, not OpenCV's usual BGR. Point ordering must
retain the image-pixel correspondence. This example rejects non-finite points
instead of silently deleting rows. Its cache is per instance and it raises
when a point cloud was not supplied.

## 3. Register the camera backend

Unlike the robot and gripper selectors, camera selection currently has a
fixed set of supported names. A real `recorded` backend requires more than
adding `scene/camera/recorded.py`:

1. Implement the `BaseCamera` contract and resource cleanup. Keep imports
   hardware-free. Specify caching, freshness, timeout, and error behavior.
2. Extend the `CAMERA_BACKEND` choices in `config.py` and the allowed set
   and import mapping in `scene/camera/__init__.py`.
3. Update capture paths in `agents/base_agent.py`,
   `agents/vision_module.py`, and `api/service.py`. They currently
   distinguish `file` from the explicit `ros1`/`ros2` names. A new backend
   must save captured images into the active workspace and must not silently
   fall through to file mode when capture fails.
4. Review `gta/utils/sensors.py`, backend health reporting, and error messages.
   Generated scripts must receive the same RGB/cloud geometry as planning.
5. Add fake-backend tests for fresh capture, cache reuse, conversion, timeout,
   missing dependencies, and library-workspace isolation.

See `scene/camera/file.py` and `scene/camera/ros2.py` for the existing
image-file and live-camera implementations.

## 4. Verify end-to-end geometry

An image-only adapter can support planning but cannot support GTA's 3D
execution preparation. For a full adapter, test pixel-to-point indexing with
known recorded data, not just matching array shapes. Invalid depth must not
shift every later pixel. Calibration and organized-cloud assumptions are
part of the integration, not automatic properties of `PointCloud2`.

**Verify:** run the documentation example tests and existing camera tests.
For a timeout, wrong annotation, or distorted 3D point see
[camera and grounding troubleshooting](../troubleshooting/vision.md).

{download}`Download the offline example <../examples/array_camera.py>`
