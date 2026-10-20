from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


ROBOT_IP = "192.168.1.10"

# Joint configuration captured from the robot on 2026-07-18 (radians).
POURING_JOINTS = [
    -1.2487753073321741,
    -1.2348447305015107,
    -2.0893349647521973,
    -2.9590231380858363,
    -1.248725716267721,
    3.141561985015869,
]

JOINT_SPEED = 0.5
JOINT_ACCELERATION = 0.5


def main():
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    try:
        print("Current robot joint positions:", rtde_r.getActualQ())
        print("Moving to pouring joint positions:", POURING_JOINTS)

        if not rtde_c.moveJ(POURING_JOINTS, JOINT_SPEED, JOINT_ACCELERATION):
            raise RuntimeError("Robot did not complete the pouring reset move.")

        print("Final robot joint positions:", rtde_r.getActualQ())
    finally:
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
