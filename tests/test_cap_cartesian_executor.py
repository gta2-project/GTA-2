"""Mock-only tests for the dedicated CaP RTDE Cartesian executor."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from benchmarks.cap.cartesian import (
    CartesianConfig,
    CartesianExecutionError,
    CartesianExecutor,
    CartesianOwnershipError,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += float(duration)


class FakeReceive:
    def __init__(self, pose=None):
        self.pose = np.asarray(
            pose or [0.0, -0.5, 0.4, 0.0, 0.0, 0.0], dtype=float
        )
        self.wrench = np.array([1.0, 2.0, -5.0, 0.1, 0.2, 0.3])
        self.disconnected = False

    def getActualTCPPose(self):
        return self.pose.tolist()

    def getActualTCPForce(self):
        return self.wrench.tolist()

    def disconnect(self):
        self.disconnected = True


class FakeControl:
    def __init__(
        self,
        receive,
        clock,
        update_pose=True,
        fail_force=False,
        fail_zero=False,
        rtde_period_s=0.01,
    ):
        self.receive = receive
        self.clock = clock
        self.update_pose = update_pose
        self.fail_force = fail_force
        self.fail_zero = fail_zero
        self.rtde_period_s = rtde_period_s
        self.calls = []
        self.disconnected = False

    def zeroFtSensor(self):
        self.calls.append(("zeroFtSensor",))
        return not self.fail_zero

    def moveL(self, pose, speed, acceleration):
        self.calls.append(("moveL", list(pose), speed, acceleration))
        self.receive.pose = np.asarray(pose, dtype=float)
        return True

    def initPeriod(self):
        self.calls.append(("initPeriod",))
        return "period"

    def forceMode(self, task_frame, selection, wrench, force_type, limits):
        self.calls.append(
            (
                "forceMode",
                list(task_frame),
                list(selection),
                list(wrench),
                force_type,
                list(limits),
            )
        )
        return not self.fail_force

    def servoL(self, pose, velocity, acceleration, dt, lookahead, gain):
        self.calls.append(
            ("servoL", list(pose), velocity, acceleration, dt, lookahead, gain)
        )
        if self.update_pose:
            self.receive.pose = np.asarray(pose, dtype=float)
        return True

    def waitPeriod(self, period):
        self.calls.append(("waitPeriod", period))
        self.clock.sleep(self.rtde_period_s)

    def servoStop(self):
        self.calls.append(("servoStop",))

    def stopScript(self):
        self.calls.append(("stopScript",))

    def disconnect(self):
        self.calls.append(("disconnect",))
        self.disconnected = True


class FakeGripper:
    def __init__(self):
        self.commands = []

    def command(self, state):
        self.commands.append(state)


class CartesianExecutorTests(unittest.TestCase):
    def make_executor(
        self,
        update_pose=True,
        fail_force=False,
        fail_zero=False,
        lock_path=None,
        rtde_period_s=0.01,
    ):
        clock = FakeClock()
        receive = FakeReceive()
        controls = []

        def control_factory(_robot_ip):
            control = FakeControl(
                receive,
                clock,
                update_pose=update_pose,
                fail_force=(fail_force and not controls),
                fail_zero=(fail_zero and not controls),
                rtde_period_s=rtde_period_s,
            )
            controls.append(control)
            return control

        config = CartesianConfig(
            ownership_lock_path=str(lock_path or Path(tempfile.mktemp())),
        )
        events = []
        gripper = FakeGripper()
        executor = CartesianExecutor(
            config=config,
            control_factory=control_factory,
            receive_factory=lambda _robot_ip: receive,
            gripper_driver=gripper,
            event_sink=events.append,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        return executor, receive, controls, gripper, events, clock

    def test_move_to_calls_exact_blocking_movel_profile(self):
        executor, receive, controls, _, events, _ = self.make_executor()
        target = [0.2, -0.45, 0.35, -2.0, -2.1, 0.1]
        try:
            executor.move_to(*target)
            self.assertEqual(controls[0].calls[0], ("zeroFtSensor",))
            self.assertEqual(controls[0].calls[1], ("moveL", target, 0.1, 0.1))
            np.testing.assert_allclose(receive.pose, target)
            completed = next(event for event in events if event["event"] == "move_to_completed")
            self.assertAlmostEqual(completed["position_residual_norm_m"], 0.0)
        finally:
            executor.close()

    def test_ft_sensor_is_zeroed_once_at_startup_not_force_reconnect(self):
        executor, _, controls, _, events, _ = self.make_executor()
        try:
            executor.move_force(0.0, -0.5, 0.4, 0, 0, 0, 0, 0, 1, -5)
            self.assertEqual(
                [call[0] for call in controls[0].calls].count("zeroFtSensor"),
                1,
            )
            self.assertEqual(
                [call[0] for call in controls[1].calls].count("zeroFtSensor"),
                0,
            )
            self.assertEqual(
                [event["event"] for event in events].count("ft_sensor_zeroed"),
                1,
            )
        finally:
            executor.close()

    def test_ft_sensor_zero_failure_aborts_before_motion(self):
        executor, _, controls, _, _, _ = self.make_executor(fail_zero=True)
        try:
            with self.assertRaises(CartesianExecutionError):
                executor.move_to(0.0, -0.5, 0.4, 0, 0, 0)
            names = [call[0] for call in controls[0].calls]
            self.assertIn("zeroFtSensor", names)
            self.assertNotIn("moveL", names)
            self.assertTrue(controls[0].disconnected)
        finally:
            executor.close()

    def test_force_mode_precedes_servo_and_full_force_axis_error_is_sent(self):
        executor, _, controls, _, events, clock = self.make_executor()
        target = [0.2, -0.5, 0.4, 0.0, 0.0, 0.0]
        try:
            executor.move_force(*target, 1.0, 0.0, 0.0, -6.0)
            first = controls[0]
            names = [call[0] for call in first.calls]
            force_index = names.index("forceMode")
            servo_index = names.index("servoL")
            self.assertLess(force_index, servo_index)

            force_call = first.calls[force_index]
            task_rotation = R.from_rotvec(force_call[1][3:]).as_matrix()
            np.testing.assert_allclose(task_rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-9)
            self.assertEqual(force_call[2], [0, 0, 1, 0, 0, 0])
            self.assertEqual(force_call[3], [0.0, 0.0, -6.0, 0.0, 0.0, 0.0])

            # X is also the force axis. A projected implementation would send no
            # X step; this executor must pass the complete target error to RTDE.
            servo_call = first.calls[servo_index]
            self.assertGreater(servo_call[1][0], 0.0)
            self.assertAlmostEqual(servo_call[1][0], 0.008, places=9)
            expected_cycles = int(round(
                executor.config.force_duration_s * executor.config.control_hz
            ))
            self.assertEqual(names.count("forceMode"), expected_cycles)
            self.assertEqual(names.count("servoL"), expected_cycles)
            self.assertGreaterEqual(clock.now, executor.config.force_duration_s)

            self.assertTrue(first.disconnected)
            self.assertEqual(len(controls), 2)
            self.assertTrue(any(event["event"] == "force_control_reconnected" for event in events))
        finally:
            executor.close()

    def test_force_residual_is_logged_and_does_not_raise(self):
        executor, _, controls, _, events, _ = self.make_executor(update_pose=False)
        target = [0.1, -0.4, 0.45, 0.1, 0.0, 0.0]
        try:
            executor.move_force(*target, 0.0, 0.0, 1.0, 5.0)
            completed = next(event for event in events if event["event"] == "move_force_completed")
            self.assertFalse(completed["target_reached"])
            self.assertGreater(completed["position_residual_norm_m"], 0.0)
            self.assertEqual(completed["actual_wrench"], [1.0, 2.0, -5.0, 0.1, 0.2, 0.3])
            self.assertEqual(len(controls), 2)
        finally:
            executor.close()

    def test_force_loop_paces_native_500_hz_rtde_period_to_100_hz(self):
        executor, _, controls, _, events, _ = self.make_executor(rtde_period_s=0.002)
        try:
            executor.move_force(0.0, -0.5, 0.4, 0, 0, 0, 0, 0, 1, -5)
            first = controls[0]
            self.assertEqual(
                [call[0] for call in first.calls].count("forceMode"),
                int(round(
                    executor.config.force_duration_s * executor.config.control_hz
                )),
            )
            completed = next(
                event for event in events if event["event"] == "move_force_completed"
            )
            self.assertAlmostEqual(
                completed["elapsed_s"],
                executor.config.force_duration_s,
                places=6,
            )
        finally:
            executor.close()

    def test_positive_force_magnitude_preserves_sign(self):
        executor, _, controls, _, _, _ = self.make_executor()
        try:
            executor.move_force(0.0, -0.5, 0.4, 0, 0, 0, 0, 0, 1, 4.5)
            force_call = next(call for call in controls[0].calls if call[0] == "forceMode")
            self.assertEqual(force_call[3], [0.0, 0.0, 4.5, 0.0, 0.0, 0.0])
        finally:
            executor.close()

    def test_force_failure_still_reconnects(self):
        executor, _, controls, _, events, _ = self.make_executor(fail_force=True)
        try:
            with self.assertRaises(CartesianExecutionError):
                executor.move_force(0.1, -0.5, 0.4, 0, 0, 0, 0, 0, 1, -5)
            self.assertTrue(controls[0].disconnected)
            self.assertEqual(len(controls), 2)
            self.assertTrue(any(event["event"] == "move_force_failed" for event in events))
        finally:
            executor.close()

    def test_validation_rejects_bad_motion_before_force_call(self):
        executor, _, controls, gripper, _, _ = self.make_executor()
        try:
            with self.assertRaises(ValueError):
                executor.move_force(0, -0.5, 0.4, 0, 0, 0, 0, 0, 0, -5)
            with self.assertRaises(CartesianExecutionError):
                executor.move_force(0, -0.5, 0.4, 0, 0, 0, 0, 0, 1, 31)
            with self.assertRaises(CartesianExecutionError):
                executor.move_to(2.0, -0.5, 0.4, 0, 0, 0)
            with self.assertRaises(ValueError):
                executor.gripper("half")
            self.assertEqual(controls, [])
            self.assertEqual(gripper.commands, [])
        finally:
            executor.close()

    def test_gripper_accepts_only_blocking_open_close_commands(self):
        executor, _, _, gripper, events, _ = self.make_executor()
        try:
            executor.gripper("open")
            executor.gripper("close")
            self.assertEqual(gripper.commands, ["open", "close"])
            self.assertEqual(
                [event["event"] for event in events].count("gripper_completed"), 2
            )
        finally:
            executor.close()

    def test_process_lock_prevents_second_cartesian_owner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "rtde.lock"
            first, _, _, _, _, _ = self.make_executor(lock_path=lock_path)
            second, _, _, _, _, _ = self.make_executor(lock_path=lock_path)
            try:
                first.connect()
                with self.assertRaises(CartesianOwnershipError):
                    second.connect()
            finally:
                second.close()
                first.close()


if __name__ == "__main__":
    unittest.main()
