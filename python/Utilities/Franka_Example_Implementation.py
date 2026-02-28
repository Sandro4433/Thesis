
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rospy
import moveit_commander

from Execution_Module.robot import Robot



def main() -> None:
    rospy.init_node("franka_go_points")
    moveit_commander.roscpp_initialize([])

    robot = Robot("panda_arm", "panda_hand", moveit_commander)

    robot.MoveJ_J("Camera_Home")

    


if __name__ == "__main__":
    main()