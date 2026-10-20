"""UR5e pose reader using a shared RTDE Receive connection."""

from rtde_receive import RTDEReceiveInterface
import numpy as np
import time

from config import Config
from scene.robot.base import BaseRobotState, RobotStateError


class RobotState(BaseRobotState):
    """Singleton UR5e RTDE connection using the configured robot address."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.robot_ip = Config.ROBOT_IP
            cls._instance.max_pose_stale_s = 0.25
            cls._instance._last_timestamp = None
            cls._instance._last_timestamp_wall_time = None
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self._last_timestamp = None
        self._last_timestamp_wall_time = None
        print(f"[Robot] RTDE initialized for {self.robot_ip}")

    def close(self):
        disconnect = getattr(self.rtde_r, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass

    def reconnect(self):
        self.close()
        time.sleep(0.25)
        self._connect()

    def _read_timestamp(self):
        get_timestamp = getattr(self.rtde_r, "getTimestamp", None)
        if not callable(get_timestamp):
            return None
        return float(get_timestamp())

    def _check_pose_freshness(self):
        timestamp = self._read_timestamp()
        if timestamp is None:
            return

        now = time.time()
        if self._last_timestamp is None or timestamp != self._last_timestamp:
            self._last_timestamp = timestamp
            self._last_timestamp_wall_time = now
            return

        if (
            self._last_timestamp_wall_time is not None
            and now - self._last_timestamp_wall_time > self.max_pose_stale_s
        ):
            raise RobotStateError(
                f"RTDE receive timestamp has not advanced for "
                f"{now - self._last_timestamp_wall_time:.3f}s"
            )

    def get_tcp_pose(self):
        """Get current TCP pose [x, y, z, rx, ry, rz]."""
        last_error = None

        for attempt in range(2):
            try:
                is_connected = getattr(self.rtde_r, "isConnected", None)
                if callable(is_connected) and not is_connected():
                    raise RobotStateError("RTDE receive interface is disconnected")

                pose = np.asarray(self.rtde_r.getActualTCPPose(), dtype=float).reshape(-1)
                if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                    raise RobotStateError(f"Invalid TCP pose from RTDE receive: {pose}")

                self._check_pose_freshness()
                return pose

            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    print(f"[Robot] RTDE receive error: {exc}. Reconnecting receive interface...")
                    self.reconnect()

        raise RobotStateError(f"Unable to read fresh TCP pose after reconnect: {last_error}")
