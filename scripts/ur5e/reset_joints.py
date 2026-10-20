"""Move the UR5e back to the recorded reset joint configuration."""

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


ROBOT_IP = "192.168.1.10"

# Joint configuration captured from the robot on 2026-07-20 (radians).
RESET_JOINTS = [
    -1.2959921995746058,
    -1.5255071830800553,
    -1.6169557571411133,
    -1.5701475674034846,
    1.5704625844955444,
    0.29003363847732544,
]

JOINT_SPEED = 0.2
JOINT_ACCELERATION = 0.2


def main():
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    try:
        print("Current robot joint positions:", rtde_r.getActualQ())
        print("Moving to reset joint positions:", RESET_JOINTS)

        if not rtde_c.moveJ(RESET_JOINTS, JOINT_SPEED, JOINT_ACCELERATION):
            raise RuntimeError("Robot did not complete the joint reset move.")

        print("Final robot joint positions:", rtde_r.getActualQ())
    finally:
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
