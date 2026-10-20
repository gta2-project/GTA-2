"""Standalone lifecycle glue for the selected shared gripper implementation.

These utilities are not GUI-supervised: run only while robot execution is idle.
No ROS initialization or motion occurs on import.
"""

import math
import signal
import sys
import threading
import time


ROS1_WAIT_TIMEOUT_S = 10.0
CLEANUP_TIMEOUT_S = 1.0


def _cleanup(errors, label, fn, *args, **kwargs):
    try:
        if fn(*args, **kwargs) is False:
            errors.append(f'{label} returned False')
    except Exception as exc:
        errors.append(f'{label}: {exc}')


def _release(gripper, errors):
    try:
        errors.extend(gripper.cancel(ack_timeout_s=CLEANUP_TIMEOUT_S, timeout_s=CLEANUP_TIMEOUT_S))
    except Exception as exc:
        errors.append(f'Gripper cancellation: {exc}')
    try:
        errors.extend(gripper.shutdown())
    except Exception as exc:
        errors.append(f'Gripper cleanup: {exc}')


def _ros1(operation, config, check_cancel, errors):
    from scene.gripper import Gripper
    try:
        import rospy
    except ImportError as exc:
        raise RuntimeError('GTA_BACKEND=ros1 requires rospy; source the ROS 1 robot workspace') from exc
    initialized = False
    gripper = None
    try:
        check_cancel()
        rospy.init_node('gta_gripper_utility', anonymous=True, disable_signals=True)
        initialized = True
        gripper = Gripper(check_cancel=check_cancel)
        gripper.wait_for_server(ROS1_WAIT_TIMEOUT_S)
        print(f'Sending {operation} gripper goal', flush=True)
        return getattr(gripper, operation)(ROS1_WAIT_TIMEOUT_S)
    finally:
        if gripper is not None:
            _release(gripper, errors)
        if initialized:
            _cleanup(errors, 'ROS 1 shutdown', rospy.signal_shutdown, 'Gripper utility finished')


def _ros2(operation, config, check_cancel, errors):
    from scene.gripper import Gripper
    timeout = float(config.ROS2_GRIPPER_WAIT_TIMEOUT_S)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('ROS2_GRIPPER_WAIT_TIMEOUT_S must be finite and greater than zero')
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
    except ImportError as exc:
        raise RuntimeError(
            'GTA_BACKEND=ros2 requires rclpy and control_msgs. '
            'Source ROS 2 Humble and the robot workspace in this terminal.'
        ) from exc
    context = Context()
    node = executor = gripper = None
    initialized = False
    try:
        check_cancel()
        rclpy.init(args=None, context=context, signal_handler_options=SignalHandlerOptions.NO)
        initialized = True
        node = rclpy.create_node('gta_gripper_utility', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)

        def spin(future, seconds):
            deadline = time.monotonic() + seconds
            while not future.done() and time.monotonic() < deadline:
                if not context.ok():
                    raise RuntimeError('ROS 2 context shut down before the gripper action finished')
                executor.spin_once(timeout_sec=min(0.1, max(0, deadline - time.monotonic())))

        gripper = Gripper(node=node, spin_until_complete=spin, check_cancel=check_cancel,
                          poll_interval_s=0.1, cancel_on_timeout=False)
        gripper.wait_for_server()
        print(f'Sending {operation} gripper goal', flush=True)
        return getattr(gripper, operation)()
    finally:
        if gripper is not None:
            _release(gripper, errors)
        if executor is not None:
            _cleanup(errors, 'Executor cleanup', executor.shutdown, timeout_sec=CLEANUP_TIMEOUT_S)
        if node is not None:
            _cleanup(errors, 'Node cleanup', node.destroy_node)
        if initialized:
            _cleanup(errors, 'ROS 2 shutdown', rclpy.shutdown, context=context)


def main(operation):
    errors = []
    previous = {}
    cancelled = threading.Event()

    def check_cancel():
        if cancelled.is_set():
            raise KeyboardInterrupt

    code = 0
    try:
        from config import Config
        if operation not in {'open', 'close', 'half_open'}:
            raise ValueError(f'Unknown gripper operation: {operation}')
        if Config.ROBOT_TYPE != 'ur5e':
            raise ValueError('These UR5e utilities require ROBOT_TYPE=ur5e')
        backends = {'ros1': _ros1, 'ros2': _ros2}
        if Config.GTA_BACKEND not in backends:
            raise ValueError('GTA_BACKEND must be ros1 or ros2')
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: cancelled.set())
        print('Standalone gripper control: do not run alongside an active robot skill.', flush=True)
        backends[Config.GTA_BACKEND](operation, Config, check_cancel, errors)
        check_cancel()
    except KeyboardInterrupt:
        print('Gripper operation interrupted', file=sys.stderr, flush=True)
        code = 130
    except Exception as exc:
        print(f'Gripper operation failed: {exc}', file=sys.stderr, flush=True)
        code = 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if errors:
        for error in errors:
            print(f'Cleanup failed: {error}', file=sys.stderr, flush=True)
        return 1
    if code == 0:
        print(f'Gripper action completed: {operation}', flush=True)
    return code
