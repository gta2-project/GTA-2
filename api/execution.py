"""Own exactly one server/script pair, with explicit readiness and bounded cleanup."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from config import Config


ACTIVE = {'starting_server', 'preparing', 'preview', 'running', 'stopping'}
MANUAL_OPERATIONS = {'reset_joints', 'open_gripper', 'close_gripper'}


def _is_robot_server_process(directory, args):
    """Recognize current entry points and servers still running under retired names."""
    # Retain legacy-name detection for already-running processes or older checkouts.
    arguments = [arg.decode(errors='replace') for arg in args if arg]
    if any(Path(arg).name.startswith('ur5e_real_time_control_reconnect')
           or Path(arg).name == 'robot_manual_control.py' for arg in arguments):
        return True

    modules = {
        'robot_server.server', 'robot_server.ur5e.ros1', 'robot_server.ur5e.ros2',
        'robot_server.ur5e_real_time_control_reconnect',
        'robot_server.ur5e_real_time_control_reconnectROS1',
        'robot_server.ur5e_real_time_control_reconnectROS2',
        'scripts.robot_manual_control',
    }
    if any(arg == '-m' and (arguments[index + 1] in modules
                           or re.fullmatch(r'scene\.robot\.(?:server|[a-z][a-z0-9_]*\.ros[12])', arguments[index + 1]))
           for index, arg in enumerate(arguments[:-1])):
        return True

    paths = {
        ('scene', 'robot', 'server.py'),
        ('robot_server', 'server.py'),
        ('robot_server', 'ur5e', 'ros1.py'),
        ('robot_server', 'ur5e', 'ros2.py'),
    }
    def matches(path):
        return (any(path.parts[-len(suffix):] == suffix for suffix in paths)
                or (len(path.parts) >= 4 and path.parts[-4:-2] == ('scene', 'robot')
                    and re.fullmatch(r'[a-z][a-z0-9_]*', path.parts[-2])
                    and path.name in {'ros1.py', 'ros2.py'}))

    for argument in arguments:
        path = Path(os.path.normpath(argument))
        if path.name not in {'server.py', 'ros1.py', 'ros2.py'}:
            continue
        if matches(path):
            return True
        if not path.is_absolute():
            # Also recognize direct backend launches from their own directory.
            try:
                path = Path(os.path.normpath((directory / 'cwd').readlink() / path))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if matches(path):
                return True
    return False


def check_server_conflicts():
    """Inspect existing GTA processes and DDS endpoints without touching RTDE."""
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if _is_robot_server_process(directory, args):
            raise RuntimeError(f'An external GTA robot server is running (PID {directory.name}). Stop it before Run.')
    # The probe is a subprocess so the web service never owns a ROS executor.
    probe = """
import time, rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
context = Context()
rclpy.init(context=context)
node = None
executor = None
try:
    node = rclpy.create_node('gta_execution_probe', context=context)
    # spin_once otherwise uses the uninitialized default context, not this node's.
    executor = SingleThreadedExecutor(context=context)
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        rclpy.spin_once(node, executor=executor, timeout_sec=0.05)
    topics = ['/robot/position_error', '/robot/rotation_error', '/robot/force_command', '/robot/gripper_command', '/robot/skill_status']
    conflicts = {i.node_namespace + '/' + i.node_name for t in topics for i in node.get_subscriptions_info_by_topic(t)}
    if conflicts:
        raise RuntimeError('Robot command subscriptions already exist: ' + ', '.join(sorted(conflicts)))
finally:
    try:
        if executor is not None:
            executor.shutdown()
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.shutdown(context=context)
"""
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=8)
    if result.returncode:
        raise RuntimeError('Cannot start a managed server: ' + (result.stderr or result.stdout).strip()[-1800:])


class ExecutionSupervisor:
    def __init__(self, root, lock=None, popen=subprocess.Popen, conflict_check=check_server_conflicts,
                 startup_timeout=30.0, shutdown_timeout=5.0, lock_path=None):
        self.root = Path(root)
        self.lock = lock or threading.RLock()
        self.popen = popen
        self.conflict_check = conflict_check
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.lock_path = Path(lock_path) if lock_path is not None else self.root / '.gta-robot-execution.lock'
        self._hardware_lock = None
        self.worker = None
        self.service = None
        self.run_dir = None
        self.run_script = None
        self.children = {}
        self.channels = {}
        self.readers = []
        self.cancel_event = threading.Event()
        self.ready = threading.Event()
        self.server_cleaned = False
        self.script_completed = False
        self.manual_cleaned = self.manual_completed = False
        self.state = {'status': 'idle', 'run_id': None, 'session_id': None,
                      'subtask': None, 'subtasks': [], 'logs': [], 'error': None,
                      'skip_pending': False, 'operation': None}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def assert_idle(self):
        if self.state['status'] in ACTIVE or self._hardware_lock is not None:
            raise RuntimeError('Robot execution is active; wait for it to stop')

    def start(self, service, session_id):
        with self.lock:
            self.assert_idle()
            service.assert_idle()
            if Config.GTA_BACKEND != 'ros2':
                raise RuntimeError('GUI-managed execution requires GTA_BACKEND=ros2')
            if not service.state()['script_ready']:
                raise RuntimeError('Generate the script from current scene grounding before Run')
            script = service.root_dir / 'robot_script.py'
            script_bytes = script.read_bytes() if script.is_file() else None
            if script_bytes is None or hashlib.sha256(script_bytes).hexdigest() != service._state.get('script_sha256'):
                raise RuntimeError('The generated script has changed or is missing. Generate it again before Run.')
            run_id = uuid.uuid4().hex
            return self._begin(service, session_id, run_id, 'run', script_bytes=script_bytes)

    def start_manual(self, service, session_id, operation):
        with self.lock:
            self.assert_idle()
            service.assert_idle()
            if operation not in MANUAL_OPERATIONS:
                raise ValueError('Unknown manual robot operation')
            if Config.GTA_BACKEND != 'ros2':
                raise RuntimeError('GUI-managed robot controls require GTA_BACKEND=ros2')
            # Manual controls do not require a generated script or camera grounding.
            return self._begin(service, session_id, uuid.uuid4().hex, operation)

    def _begin(self, service, session_id, run_id, operation, script_bytes=None):
        self._acquire_hardware_lock()
        self.service = service
        self.run_dir = service.root_dir / 'execution_logs' / run_id
        self.run_script = None
        self.children = {}
        self.channels = {}
        self.readers = []
        self.cancel_event.clear()
        self.ready.clear()
        self.server_cleaned = self.script_completed = False
        self.manual_cleaned = self.manual_completed = False
        self.state = {'status': 'starting_server' if operation == 'run' else 'preparing',
                      'operation': operation, 'run_id': run_id,
                      'session_id': session_id, 'subtask': None, 'subtasks': [],
                      'logs': [], 'error': None, 'skip_pending': False}
        try:
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                if operation == 'run':
                    # Freeze the reviewed bytes before starting any child. The
                    # script still runs with the original workspace as its cwd.
                    self.run_script = self.run_dir / 'robot_script.py'
                    self.run_script.write_bytes(script_bytes)
            except OSError as exc:
                raise RuntimeError(f'Cannot create execution records in {self.run_dir}: {exc}') from exc
            self.worker = threading.Thread(target=self._run_owned,
                                           args=(self._run if operation == 'run' else self._run_manual,),
                                           daemon=True)
            self.worker.start()
        except Exception as exc:
            self.worker = None
            self.state.update(status='failed', error=str(exc))
            self._release_hardware_lock()
            raise
        return self.snapshot()

    def _acquire_hardware_lock(self):
        # main.py and library.py are separate processes. Hold one local lease through
        # startup AND cleanup so simultaneous requests cannot race the ROS probe.
        import fcntl
        handle = self.lock_path.open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError('Another GTA dashboard is controlling the robot; wait for it to stop') from exc
        except BaseException:
            handle.close()
            raise
        self._hardware_lock = handle

    def _release_hardware_lock(self):
        if self._hardware_lock is not None:
            self._hardware_lock.close()
            self._hardware_lock = None

    def _run_owned(self, target):
        try:
            target()
        finally:
            with self.lock:
                self._release_hardware_lock()

    def _notify(self, event_type='execution_status', **content):
        if self.service:
            event = {'type': 'execution', 'message_type': event_type,
                     'content': {'run_id': self.state['run_id'], **content}}
            self.service._emit_event(event)
            try:
                with (self.run_dir / 'events.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps({'time': time.time(), **event}) + '\n')
            except OSError as exc:
                # Logging failures cannot prevent robot/process cleanup.
                self.state['error'] = f'Execution event log failed: {exc}'

    def _status(self, status):
        with self.lock:
            if self.cancel_event.is_set() and status in ACTIVE - {'stopping'}:
                return
            self.state['status'] = status
            self._notify(status=status)

    def _send(self, role, command, **payload):
        channel = self.channels.get(role)
        if channel:
            try:
                channel.sendall((json.dumps({'type': command, 'run_id': self.state['run_id'], **payload}) + '\n').encode())
            except OSError:
                pass

    def skip(self, run_id, subtask):
        with self.lock:
            if (self.state['run_id'] != run_id or self.state['status'] != 'running' or
                    self.state['subtask'] is None or self.state['subtask'] != subtask or self.state['skip_pending']):
                raise RuntimeError('Skip is only available once for the currently active subtask')
            self.state['skip_pending'] = True
            self._send('script', 'skip', subtask=subtask)
            return self.snapshot()

    def cancel(self, run_id):
        with self.lock:
            if self.state['run_id'] != run_id:
                raise RuntimeError('This run is no longer active')
            if self.state['status'] in ACTIVE:
                self.cancel_event.set()
                self._status('stopping')
                self._send('script', 'cancel')
                self._send('server', 'cancel')
                self._send('manual', 'cancel')
            return self.snapshot()

    def _spawn(self, role, args):
        with self.lock:
            if self.cancel_event.is_set():
                return None
            return self._spawn_owned(role, args)

    def _spawn_owned(self, role, args):
        parent, child = socket.socketpair()
        environment = os.environ.copy()
        environment.update({'GTA_CONTROL_FD': str(child.fileno()), 'GTA_CHILD_ROLE': role,
                            'GTA_RUN_ID': self.state['run_id'], 'PYTHONUNBUFFERED': '1',
                            'CAMERA_IMAGE_PATH': str(self.service.root_dir / 'environment.jpg'),
                            'PYTHONPATH': str(self.root) + os.pathsep + environment.get('PYTHONPATH', '')})
        try:
            process = self.popen([sys.executable, '-u', *args], cwd=str(self.service.root_dir),
                                 env=environment, pass_fds=(child.fileno(),), start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        with self.lock:
            self.children[role] = process
            self.channels[role] = parent
            if self.cancel_event.is_set():
                self._send(role, 'cancel')
        for target, args in ((self._read_events, (role, parent)), (self._read_logs, (role, process))):
            thread = threading.Thread(target=target, args=args, daemon=True)
            self.readers.append(thread)
            thread.start()
        return process

    def _read_logs(self, role, process):
        log = self.run_dir / f'{role}.log'
        try:
            with log.open('a', encoding='utf-8') as output:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    with self.lock:
                        self.state['logs'] = (self.state['logs'] + [{'role': role, 'text': line.rstrip()}])[-250:]
                        self._notify('execution_log', role=role, text=line.rstrip())
        except (OSError, ValueError) as exc:
            with self.lock:
                self.state['error'] = f'{role} log failed: {exc}'

    def _read_events(self, role, channel):
        try:
            with channel.makefile('r') as stream:
                for line in stream:
                    self.handle_event(role, json.loads(line))
        except (OSError, ValueError):
            pass

    def handle_event(self, role, event):
        with self.lock:
            if event.get('run_id') != self.state['run_id']:
                return
            kind = event['type']
            if role == 'manual':
                if kind == 'manual_started':
                    self.ready.set()
                    if self.state['status'] == 'preparing':
                        self._status('running')
                elif kind == 'manual_completed':
                    self.manual_completed = True
                elif kind == 'manual_cleaned':
                    self.manual_cleaned = not event.get('errors')
                    if event.get('errors'):
                        self.state['error'] = '; '.join(event['errors'])
                elif kind == 'manual_failed':
                    self.state['error'] = event.get('error') or 'Manual robot operation failed'
            if role == 'server' and kind == 'server_ready':
                self.ready.set()
            elif role == 'server' and kind == 'server_cleaned':
                self.server_cleaned = not event.get('errors')
                if event.get('errors'):
                    self.state['error'] = '; '.join(event['errors'])
            elif kind in {'server_failed', 'script_failed'}:
                self.state['error'] = event.get('error') or kind
            elif role == 'script':
                if kind == 'script_completed':
                    self.script_completed = True
                if self.state['status'] != 'stopping':
                    if kind in {'preparing', 'preview'}:
                        self.state['status'] = kind
                    elif kind == 'subtask_started':
                        self.state.update(status='running', subtask=event['subtask'], skip_pending=False)
                        self.state['subtasks'].append({'id': event['subtask'], 'description': event['description'], 'status': 'running'})
                    elif kind == 'subtask_ended':
                        for subtask in self.state['subtasks']:
                            if subtask['id'] == event['subtask']:
                                subtask['status'] = event['status']
                        self.state.update(status='preparing', subtask=None, skip_pending=False)
            self._notify('execution_event', event=event)

    def _run(self):
        error = None
        try:
            self.conflict_check()
            if self.cancel_event.is_set():
                return
            server = self._spawn('server', [str(self.root / 'scene/robot/server.py')])
            if server is None:
                return
            deadline = time.monotonic() + self.startup_timeout
            while not self.ready.is_set():
                if self.cancel_event.wait(0.05):
                    return
                if server.poll() is not None:
                    raise RuntimeError('robot server exited before becoming ready; see server log')
                if self.state.get('error'):
                    raise RuntimeError(self.state['error'])
                if time.monotonic() >= deadline:
                    raise RuntimeError('robot server readiness timed out')
            if self.cancel_event.is_set():
                return
            if server.poll() is not None:
                raise RuntimeError('robot server stopped before script startup')
            self._status('preparing')
            script = self._spawn('script', [str(self.root / 'gta/runner.py'), str(self.run_script)])
            if script is None:
                return
            while script.poll() is None:
                if self.cancel_event.wait(0.05):
                    return
                if server.poll() is not None or self.state.get('error'):
                    raise RuntimeError(self.state.get('error') or 'robot server stopped during execution')
            if self.cancel_event.is_set():
                return
            if script.returncode != 0:
                raise RuntimeError(f'Robot script exited with code {script.returncode}')
        except Exception as exc:
            error = str(exc)
        finally:
            self._status('stopping')
            cleanup_error = self._cleanup()
            with self.lock:
                error = cleanup_error or self.state.get('error') or error
                if not error and not self.cancel_event.is_set() and not self.script_completed:
                    error = 'Robot script ended without a completion event'
                self.state['error'] = error
                for subtask in self.state['subtasks']:
                    if subtask['status'] == 'running':
                        subtask['status'] = 'cancelled' if self.cancel_event.is_set() else 'failed'
                self.state.update(subtask=None, skip_pending=False)
                self._status('failed' if error else 'cancelled' if self.cancel_event.is_set() else 'completed')

    def _run_manual(self):
        error = None
        try:
            self.conflict_check()
            process = self._spawn('manual', [str(self.root / 'scene/robot/server.py'),
                                             '--operation', self.state['operation']])
            if process is None:
                return
            startup_deadline = time.monotonic() + self.startup_timeout
            deadline = time.monotonic() + 120.0
            while process.poll() is None:
                if self.cancel_event.wait(0.05):
                    return
                if self.state.get('error'):
                    raise RuntimeError(self.state['error'])
                if not self.ready.is_set() and time.monotonic() >= startup_deadline:
                    raise RuntimeError('Manual robot control startup timed out')
                if time.monotonic() >= deadline:
                    raise RuntimeError('Manual robot operation timed out')
            if process.returncode != 0 and not self.cancel_event.is_set():
                raise RuntimeError(f'Manual robot control exited with code {process.returncode}')
        except Exception as exc:
            error = str(exc)
        finally:
            self._status('stopping')
            cleanup_error = self._cleanup()
            with self.lock:
                error = cleanup_error or self.state.get('error') or error
                if not error and not self.cancel_event.is_set() and not self.manual_completed:
                    error = 'Manual robot control ended without a completion event'
                self.state['error'] = error
                self._status('failed' if error else 'cancelled' if self.cancel_event.is_set() else 'completed')

    def _cleanup(self):
        for role in ('script', 'server', 'manual'):
            self._send(role, 'cancel')
        deadline = time.monotonic() + self.shutdown_timeout
        forced = []
        for role, process in list(self.children.items()):
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                forced.append(role)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for thread in self.readers:
            thread.join(timeout=1)
        for channel in self.channels.values():
            channel.close()
        for process in self.children.values():
            if process.stdout:
                process.stdout.close()
        if forced:
            return 'Forced termination of ' + ', '.join(forced) + '; hardware cleanup could not be verified'
        if 'server' in self.children and not self.server_cleaned:
            return 'robot server exited without confirming hardware cleanup'
        if 'manual' in self.children and not self.manual_cleaned:
            return 'Manual robot control exited without confirming hardware cleanup'
        return None

    def shutdown(self):
        with self.lock:
            if self.state['status'] in ACTIVE:
                self.cancel(self.state['run_id'])
        if self.worker:
            self.worker.join(timeout=self.startup_timeout + self.shutdown_timeout + 10)
            with self.lock:
                if not self.worker.is_alive():
                    self._release_hardware_lock()
