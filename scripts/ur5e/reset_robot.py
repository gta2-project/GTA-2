import time
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R

robot_ip = "192.168.1.10"

rtde_c = RTDEControlInterface(robot_ip)
rtde_r = RTDEReceiveInterface(robot_ip)
print("RTDE Control and Receive interfaces initialized.")

print("Current robot pose:", rtde_r.getActualTCPPose())

reset_pose = [0, -0.491, 0.487, -2.221, -2.221, 0]
rtde_c.moveL(reset_pose, 0.1, 0.1)

current_joints = rtde_r.getActualQ()
current_joints[-1] = 1.00  # Set the last joint to 0.39

rtde_c.moveJ(current_joints, 0.5, 0.5)
