"""Private IPC used by GUI-owned children; standalone scripts remain ordinary scripts."""

from contextlib import contextmanager
import json
import os
import signal
import socket
import threading


class RunCancelled(SystemExit):
    pass


class ManagedRuntime:
    def __init__(self, fd, role, run_id):
        self.socket = socket.socket(fileno=fd)
        self.role = role
        self.run_id = run_id
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.subtask = None
        self.skip_requested = False
        self.skipped = False
        if role == 'script':
            signal.signal(signal.SIGTERM, self._cancel_signal)
        threading.Thread(target=self._listen, daemon=True).start()

    def _cancel_signal(self, *_args):
        self.cancelled.set()
        raise RunCancelled('Run cancelled')

    def _listen(self):
        try:
            with self.socket.makefile('r') as stream:
                for line in stream:
                    command = json.loads(line)
                    if command.get('run_id') != self.run_id:
                        continue
                    if command['type'] == 'cancel':
                        break
                    with self.lock:
                        if command['type'] == 'skip' and command.get('subtask') == self.subtask:
                            self.skip_requested = True
        except (OSError, ValueError):
            pass
        self.cancelled.set()
        if self.role == 'script':
            os.kill(os.getpid(), signal.SIGTERM)

    def emit(self, event, **payload):
        message = {'type': event, 'role': self.role, 'run_id': self.run_id, **payload}
        try:
            with self.write_lock:
                self.socket.sendall((json.dumps(message) + '\n').encode())
        except OSError:
            self.cancelled.set()

    def check_cancel(self):
        if self.cancelled.is_set():
            raise RunCancelled('Run cancelled')


_runtime = None


def get_runtime():
    global _runtime
    fd = os.getenv('GTA_CONTROL_FD')
    if _runtime is None and fd:
        _runtime = ManagedRuntime(int(fd), os.environ['GTA_CHILD_ROLE'], os.environ['GTA_RUN_ID'])
    return _runtime


def should_skip():
    runtime = get_runtime()
    if runtime is None:
        return False
    runtime.check_cancel()
    with runtime.lock:
        if runtime.skip_requested:
            runtime.skipped = True
            return True
    return False


@contextmanager
def managed_preview():
    runtime = get_runtime()
    if runtime:
        runtime.check_cancel()
        runtime.emit('preview')
    yield
    if runtime:
        runtime.check_cancel()
        runtime.emit('preparing')


@contextmanager
def managed_subtask(subtask_id, description):
    runtime = get_runtime()
    if runtime:
        runtime.check_cancel()
        with runtime.lock:
            runtime.subtask = str(subtask_id)
            runtime.skip_requested = runtime.skipped = False
        runtime.emit('subtask_started', subtask=str(subtask_id), description=description)
    try:
        yield
    except BaseException:
        if runtime:
            runtime.emit('subtask_ended', subtask=str(subtask_id), status='cancelled' if runtime.cancelled.is_set() else 'failed')
        raise
    else:
        if runtime:
            runtime.check_cancel()
            runtime.emit('subtask_ended', subtask=str(subtask_id), status='skipped' if runtime.skipped else 'completed')
    finally:
        if runtime:
            with runtime.lock:
                runtime.subtask = None
                runtime.skip_requested = False
