"""Close the gripper using GTA_BACKEND; run only when idle."""

from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ur5e.gripper_control import main as run_gripper
    return run_gripper('close')


if __name__ == '__main__':
    raise SystemExit(main())
