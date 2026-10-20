"""Execute one generated script with GUI cancellation installed before its imports."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gta.managed_run import get_runtime, RunCancelled


def main():
    runtime = get_runtime()
    try:
        if runtime:
            runtime.check_cancel()
            runtime.emit('preparing')
        runpy.run_path(sys.argv[1], run_name='__main__')
        if runtime:
            runtime.check_cancel()
            runtime.emit('script_completed')
    except RunCancelled:
        if runtime:
            runtime.emit('script_cancelled')
        return 130
    except BaseException as exc:
        if runtime:
            runtime.emit('script_failed', error=str(exc))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
