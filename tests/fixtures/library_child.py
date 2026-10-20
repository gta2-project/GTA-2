"""Harmless IPC child for supervisor tests. Never imports ROS or robot hardware."""
import os
import time
from gta.managed_run import get_runtime, managed_preview, managed_subtask, should_skip, RunCancelled

runtime = get_runtime()
mode = os.environ.get('GTA_TEST_SCENARIO', 'complete')
print(f'fake {runtime.role} started', flush=True)
if runtime.role == 'manual':
    try:
        if mode != 'timeout':
            runtime.emit('manual_started')
        if mode == 'manualfail':
            raise RuntimeError('fake manual failure')
        while mode in {'run', 'timeout', 'hardkill'}:
            if mode != 'hardkill':
                runtime.check_cancel()
            time.sleep(0.01)
        runtime.emit('manual_completed')
    except RunCancelled:
        pass
    except Exception as exc:
        runtime.emit('manual_failed', error=str(exc))
        raise
    finally:
        if mode != 'unclean':
            runtime.emit('manual_cleaned', errors=['fake cleanup failure'] if mode == 'cleanupfail' else [])
elif runtime.role == 'server':
    if mode != 'timeout':
        time.sleep(0.03)
        runtime.emit('server_ready')
    deadline = time.monotonic() + 0.3
    while mode == 'hardkill' or not runtime.cancelled.wait(0.01):
        if mode == 'serverfail' and time.monotonic() > deadline:
            runtime.emit('server_failed', error='fake control failure')
            break
    runtime.emit('server_cleaned', errors=[])
else:
    try:
        runtime.emit('preparing')
        with managed_preview():
            if mode == 'preview':
                while True:
                    time.sleep(0.01)
            time.sleep(0.04)
        if mode == 'scriptfail':
            raise RuntimeError('fake script failure')
        for subtask in range(1, 3):
            with managed_subtask(subtask, f'Harmless subtask {subtask}'):
                deadline = time.monotonic() + (0.04 if mode in {'complete', 'hardkill'} else 4.0)
                while time.monotonic() < deadline:
                    if should_skip():
                        break
                    time.sleep(0.01)
        runtime.emit('script_completed')
    except RunCancelled:
        runtime.emit('script_cancelled')
    except BaseException as exc:
        runtime.emit('script_failed', error=str(exc))
        raise
