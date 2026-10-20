"""Continuously print the UR5e TCP force and torque readings."""

import math
import time

from rtde_receive import RTDEReceiveInterface


ROBOT_IP = "192.168.1.10"
PRINT_INTERVAL_SECONDS = 0.1


def main():
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    print("Connected to the UR5e. Press Ctrl+C to stop.")
    print("Values: Fx, Fy, Fz in newtons; Tx, Ty, Tz in newton-metres.")

    try:
        while True:
            fx, fy, fz, tx, ty, tz = rtde_r.getActualTCPForce()
            force_magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
            print(
                f"Fx={fx:8.3f} N  Fy={fy:8.3f} N  Fz={fz:8.3f} N  "
                f"|F|={force_magnitude:8.3f} N  "
                f"Tx={tx:8.3f} Nm  Ty={ty:8.3f} Nm  Tz={tz:8.3f} Nm",
                flush=True,
            )
            time.sleep(PRINT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped reading force values.")
    finally:
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
