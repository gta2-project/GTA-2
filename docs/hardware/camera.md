# Connect camera and point-cloud inputs

**Prerequisites:** [ROS 2 setup](ros2.md), a calibrated camera, and its external
image/point-cloud pipeline.
**Outcome:** deliver RGB and matching base-frame XYZ data to GTA.

## 1. Start the external camera pipeline

For ZED, start with the official
[ZED ROS 2 wrapper](https://github.com/stereolabs/zed-ros2-wrapper).
Follow the driver documentation for your camera and platform.

GTA's default point-cloud topic, `/zed_node/transform_igev_points`, is a
deployment-specific processed cloud. Installing the standard ZED wrapper
does **not** establish that this topic exists. Dense stereo processing and
camera-to-base transformation must be supplied by your external pipeline.
GTA does not bundle its launch files.

## 2. Configure actual topics

```text
CAMERA_BACKEND=ros2
CAMERA_NODE=/zed_node
CAMERA_MODEL=zed2i
CAMERA_COLOR_TOPIC=/zed_node/rgb_rect/left/image
CAMERA_POINT_CLOUD_TOPIC=/zed_node/transform_igev_points
```

These explicit names match the reference setup. Replace them with the topics
your pipeline actually publishes. Blank overrides derive names from
`CAMERA_NODE` and `CAMERA_MODEL`. ROS 2 overrides allow a different camera
without creating another backend if its outputs already satisfy the contract.

The RGB input is a `sensor_msgs/msg/Image`; the cloud is
`sensor_msgs/msg/PointCloud2`. Camera QoS and spinning are handled by the
camera backend, separately from robot command publishing.

The current ROS 2 camera requests reliable QoS with depth 1. A best-effort-only
publisher does not satisfy that subscription. Check the publisher's QoS for
compatibility.

## 3. Verify geometry, not only connectivity

The execution path expects:

- An RGB image and cloud describing the same scene, crop, and pixel layout.
- XYZ values in robot-base coordinates and **metres at the GTA consumer**.
- One retained XYZ point per original image pixel in row-major order.
- The landscape-image padding/mapping assumed by `get_3d_position()`.
- Finite, valid geometry at all selected keypoints.

:::{warning}
The current camera converter behaves differently depending on cloud fields:
XYZ-only values pass through, while XYZ plus `rgb` values are multiplied by
1000. Both ROS camera backends skip NaN points. A standard metres-valued
XYZRGB cloud can therefore have the wrong scale, and dropping invalid points
can destroy pixel correspondence.

Verify the **returned** cloud's scale and point count before execution.
Prefer an external XYZ-only, finite, organized base-frame output that already
matches GTA's needs.
:::

The converter is not a general-purpose handler for arbitrary PointCloud2
field layouts. Extra fields, `rgba` instead of `rgb`, unorganized clouds,
and mismatched resolutions need integration work.

`T_BASE_CAM` does not automatically transform the incoming cloud in this
execution path. Supply the required base-frame cloud upstream. Its camera
position is also used when orienting surface normals.

## 4. Check endpoints without moving hardware

```bash
ros2 topic info /zed_node/rgb_rect/left/image --verbose
ros2 topic info /zed_node/transform_igev_points --verbose
ros2 topic hz /zed_node/rgb_rect/left/image
ros2 topic hz /zed_node/transform_igev_points
```

Run frequency checks separately and stop each with Ctrl+C. Confirm message
types, publishers, frame metadata, and QoS compatibility. Use your camera
viewer to inspect the image and processed cloud.

**Verify:** known scene distances and orientations agree with the robot base,
and the returned point count equals image width × height. For a timeout or
reshape failure, see [camera troubleshooting](../troubleshooting/vision.md).
