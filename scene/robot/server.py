"""Launch a selected robot server, or one GUI-supervised device operation."""

import argparse
import importlib
from pathlib import Path
import re
import signal
import sys

# Support both a direct file path and python -m scene.robot.server.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config
from gta.managed_run import RunCancelled, get_runtime


def _load_backend(robot_type, gta_backend):
    if gta_backend not in {"ros1", "ros2"}:
        raise ValueError("GTA_BACKEND must be ros1 or ros2.")
    if not isinstance(robot_type, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", robot_type):
        raise ValueError(
            "ROBOT_TYPE must be a robot directory name using lowercase letters, "
            "digits or underscores, starting with a letter (for example: ur5e)."
        )

    package_name = f"scene.robot.{robot_type}"
    module_name = f"{package_name}.{gta_backend}"
    try:
        backend = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {package_name, module_name}:
            raise RuntimeError(
                f"No {gta_backend} server is available for ROBOT_TYPE={robot_type!r}. "
                f"Expected scene/robot/{robot_type}/{gta_backend}.py. "
                "Only ur5e is bundled; no fallback server will be started."
            ) from exc
        # A missing dependency inside an installed backend is a different error.
        raise

    if not callable(getattr(backend, "main", None)):
        raise RuntimeError(f"Robot server {module_name} must define a callable main().")
    return backend


def _cleanup(errors, label, fn, *args, **kwargs):
    try:
        if fn(*args, **kwargs) is False:
            errors.append(f'{label} returned False')
    except Exception as exc:
        errors.append(f'{label}: {exc}')


def _run_gripper(operation, runtime, errors):
    from scene.gripper import Gripper
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    context = Context()
    node = executor = gripper = None
    initialized = False
    try:
        runtime.check_cancel()
        rclpy.init(args=None, context=context, signal_handler_options=SignalHandlerOptions.NO)
        initialized = True
        node = rclpy.create_node('gta_gripper_helper', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)

        def spin(future, seconds):
            if not context.ok():
                raise RuntimeError('ROS 2 context shut down before the gripper action finished')
            executor.spin_until_future_complete(future, timeout_sec=seconds)

        gripper = Gripper(node=node, spin_until_complete=spin, check_cancel=runtime.check_cancel)
        gripper.wait_for_server()
        runtime.check_cancel()
        runtime.emit('manual_started')
        print('Opening gripper' if operation == 'open_gripper' else 'Closing gripper', flush=True)
        runtime.check_cancel()
        (gripper.open if operation == 'open_gripper' else gripper.close)()
    finally:
        if gripper is not None:
            try:
                errors.extend(gripper.cancel())
            except Exception as exc:
                errors.append(f'Gripper cancellation: {exc}')
            try:
                errors.extend(gripper.shutdown())
            except Exception as exc:
                errors.append(f'Gripper cleanup: {exc}')
        if executor is not None:
            _cleanup(errors, 'Executor cleanup', executor.shutdown, timeout_sec=1.0)
        if node is not None:
            _cleanup(errors, 'ROS node cleanup', node.destroy_node)
        if initialized:
            _cleanup(errors, 'ROS shutdown', rclpy.shutdown, context=context)


def _run_operation(operation):
    runtime = get_runtime()
    if runtime is None or runtime.role != 'manual':
        raise RuntimeError('Robot helper operations must be started by the GUI supervisor')
    if Config.GTA_BACKEND != 'ros2':
        raise RuntimeError('GUI-managed robot controls require GTA_BACKEND=ros2')
    errors = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, lambda *_: runtime.cancelled.set())
    failed = False
    try:
        runtime.check_cancel()
        if operation == 'reset_joints':
            from scene.robot import RobotHelpers
            RobotHelpers().reset_joints(
                check_cancel=runtime.check_cancel,
                on_started=lambda: runtime.emit('manual_started'),
                errors=errors, wait=runtime.cancelled.wait,
            )
        else:
            _run_gripper(operation, runtime, errors)
        runtime.check_cancel()
        if not errors:
            runtime.emit('manual_completed')
    except RunCancelled:
        print('Robot helper operation cancelled', flush=True)
    except BaseException as exc:
        failed = True
        runtime.emit('manual_failed', error=str(exc))
        print(f'Robot helper operation failed: {exc}', flush=True)
    finally:
        runtime.emit('manual_cleaned', errors=errors)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 1 if failed or errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', choices=('reset_joints', 'open_gripper', 'close_gripper'),
                        help='GUI-supervised helper operation (requires its private control channel)')
    args = parser.parse_args(argv)
    if args.operation:
        return _run_operation(args.operation)
    backend = _load_backend(Config.ROBOT_TYPE, Config.GTA_BACKEND)
    return backend.main()


if __name__ == '__main__':
    raise SystemExit(main())
