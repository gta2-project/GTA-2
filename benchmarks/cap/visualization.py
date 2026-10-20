"""Offline visualization helpers for reviewing prepared CaP perception."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
from PIL import Image

from agents.vision_visualization import save_annotated_keypoints


PREVIEW_COLORS = (
    "#ff3b5c",
    "#00b8ff",
    "#70db70",
    "#ffb000",
    "#c77dff",
    "#00d4aa",
    "#ff7a45",
    "#e052d1",
)


def show_interactive_point_cloud(
    point_cloud,
    image_path: Path,
    points: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
) -> None:
    """Open the repository's existing interactive Open3D visualization.

    This intentionally delegates to ``gta.utils`` so CaP
    review has the same interactive view as generated GTA robot scripts.
    The call blocks until the operator closes the Open3D window.
    """
    from config import Config
    from gta.utils import visualize_point_cloud

    cloud = np.asarray(point_cloud, dtype=float)
    with Image.open(image_path) as source_image:
        image = np.asarray(source_image.convert("RGB"), dtype=float) / 255.0
    cloud_point_count = int(np.prod(cloud.shape[:-1]))
    image_pixel_count = int(image.shape[0] * image.shape[1])
    colors = image if cloud_point_count == image_pixel_count else None

    # One lightweight object per query makes the existing renderer place each
    # normal arrow at its own keypoint rather than at a shared centroid.
    objects = []
    for label, raw_point in points.items():
        point = np.asarray(raw_point, dtype=float).reshape(3)
        normal = np.asarray(normals.get(label, np.zeros(3)), dtype=float).reshape(3)
        objects.append(
            SimpleNamespace(
                keypoints={str(label): point},
                axes={"normal": normal},
            )
        )

    visualize_point_cloud(
        pcd_np=cloud,
        colors=colors,
        camera_pose=np.asarray(Config.T_BASE_CAM, dtype=float),
        objects=objects,
    )


def render_annotated_scene(
    image_path: Path,
    pixels: Mapping[str, tuple],
    output_path: Path,
) -> Path:
    """Render using the exact shared GTA Vision Module annotation style."""
    with Image.open(image_path) as source_image:
        image_width, image_height = source_image.size
    points = []
    for label, pixel in pixels.items():
        value = np.asarray(pixel, dtype=int)
        if value.shape != (2,):
            raise ValueError(f"Pixel for {label!r} must contain [x, y]")
        x, y = int(value[0]), int(value[1])
        if not (0 <= x < image_width and 0 <= y < image_height):
            raise ValueError(f"Pixel for {label!r} lies outside the frozen image")
        points.append({"point": [x, y], "label": str(label)})
    return save_annotated_keypoints(image_path, points, output_path)


def render_point_cloud_preview(
    point_cloud,
    points: Mapping[str, np.ndarray],
    normals: Mapping[str, np.ndarray],
    output_path: Path,
    max_cloud_points: int = 20_000,
) -> Path:
    """Render a static 3D review image with detected points and normals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cloud = np.asarray(point_cloud, dtype=float)
    if cloud.ndim not in {2, 3} or cloud.shape[-1] != 3:
        raise ValueError(
            f"Point cloud must have shape (N,3) or (H,W,3), got {cloud.shape}"
        )
    cloud = cloud.reshape(-1, 3)
    valid = np.all(np.isfinite(cloud), axis=1)
    valid &= np.linalg.norm(cloud, axis=1) > 1e-12
    cloud = cloud[valid]
    if cloud.size == 0:
        raise ValueError("Point cloud preview has no valid points")
    if len(cloud) > max_cloud_points:
        indices = np.linspace(0, len(cloud) - 1, max_cloud_points, dtype=int)
        cloud = cloud[indices]

    fig = plt.figure(figsize=(9, 7), dpi=130)
    axis = fig.add_subplot(111, projection="3d")
    depth_colors = cloud[:, 2]
    axis.scatter(
        cloud[:, 0],
        cloud[:, 1],
        cloud[:, 2],
        c=depth_colors,
        cmap="viridis",
        s=0.8,
        alpha=0.35,
        linewidths=0,
    )

    scene_extent = np.ptp(cloud, axis=0)
    normal_scale = max(float(np.max(scene_extent)) * 0.08, 0.03)
    review_points = []
    for index, (label, raw_point) in enumerate(points.items()):
        point = np.asarray(raw_point, dtype=float)
        normal = np.asarray(normals.get(label, np.zeros(3)), dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Prepared point for {label!r} must be finite 3D")
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError(f"Prepared normal for {label!r} must be finite 3D")
        color = PREVIEW_COLORS[index % len(PREVIEW_COLORS)]
        axis.scatter(*point, color=color, s=65, edgecolor="white", linewidth=0.8)
        axis.text(*point, f" {label}", color=color, fontsize=8)
        normal_length = float(np.linalg.norm(normal))
        if normal_length > 1e-9:
            direction = normal / normal_length
            axis.quiver(
                *point,
                *(direction * normal_scale),
                color=color,
                linewidth=2.0,
                arrow_length_ratio=0.25,
            )
        review_points.append(point)

    bounds_cloud = cloud
    if review_points:
        bounds_cloud = np.vstack([bounds_cloud, np.asarray(review_points)])
    lower = np.min(bounds_cloud, axis=0)
    upper = np.max(bounds_cloud, axis=0)
    center = (lower + upper) / 2.0
    half_range = max(float(np.max(upper - lower)) / 2.0, 1e-3) * 1.05
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Base X")
    axis.set_ylabel("Base Y")
    axis.set_zlabel("Base Z")
    axis.set_title("Frozen point cloud with prepared keypoints and normals")
    axis.view_init(elev=28, azim=-55)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
