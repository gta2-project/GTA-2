"""Deterministic trial artifact and trace helpers."""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class TrialArtifacts:
    def __init__(self, root: Path, task_id: str, condition: str):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        safe_task = "".join(char if char.isalnum() or char in "-_" else "_" for char in task_id)
        self.path = Path(root) / safe_task / condition / timestamp
        self.path.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.path / "trace.jsonl"
        self._started = time.monotonic()

    def event(self, event: dict):
        payload = {
            "wall_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - self._started,
            **event,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")

    def write_json(self, name: str, value):
        path = self.path / name
        path.write_text(
            json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_text(self, name: str, value: str):
        path = self.path / name
        path.write_text(value, encoding="utf-8")
        return path

    def copy(self, source: Path, name: str):
        target = self.path / name
        shutil.copy2(source, target)
        return target

