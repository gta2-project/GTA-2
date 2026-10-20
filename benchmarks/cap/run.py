"""Command-line entry point for CaP benchmark trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conditions import CONDITIONS
from .runner import run_trial


def build_parser():
    parser = argparse.ArgumentParser(description="Run one GTA-2 Code-as-Policies trial")
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--task", required=True, type=Path, help="Task manifest YAML/JSON")
    parser.add_argument("--image", type=Path, help="Frozen scene image; overrides manifest")
    parser.add_argument("--mode", choices=("generate", "offline", "robot"), default="generate")
    parser.add_argument("--output", type=Path, default=Path("benchmark_runs/cap"))
    parser.add_argument("--policy-file", type=Path, help="Use existing source instead of calling a model")
    parser.add_argument("--perception", type=Path, help="Base-frame perception fixture for offline/replay")
    parser.add_argument("--point-cloud", type=Path, help="Frozen organized .npy/.npz point cloud")
    parser.add_argument(
        "--approve-robot",
        action="store_true",
        help="Authorize physical execution after cell/server safety checks",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run_trial(
            condition_name=args.condition,
            task_manifest=args.task,
            image_path=args.image,
            mode=args.mode,
            output_root=args.output,
            policy_file=args.policy_file,
            perception_path=args.perception,
            point_cloud_path=args.point_cloud,
            approve_robot=args.approve_robot,
        )
    except Exception as exc:
        print(f"CaP benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

