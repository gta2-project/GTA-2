"""Shared GTA-style keypoint annotation rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_annotated_keypoints(
    image_path: str | Path,
    points: Iterable[Dict],
    output_path: str | Path = "annotated_keypoints.png",
) -> Path:
    """Save annotations using the Vision Module's established GTA styling."""
    with Image.open(image_path) as image:
        image_np = np.array(image)

    fig, axis = plt.subplots(1, figsize=(12, 9))
    axis.imshow(image_np)
    for item in points:
        x, y = item["point"]
        label = item["label"]
        axis.plot(x, y, "ro", markersize=8)
        axis.text(
            x + 20,
            y - 20,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.7),
        )

    axis.axis("off")
    axis.set_title("Detected Keypoints from Skills")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
