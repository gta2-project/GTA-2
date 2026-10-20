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
print("Current robot joint positions:", rtde_r.getActualQ())
