# Robot_Main.py
from __future__ import annotations

import sys

import rospy
import moveit_commander

from Core.config import settings
from Execution_Module.robot import Robot
from Execution_Module.sequence_executor import execute_sequence


def main() -> int:
    rospy.init_node("franka_go_points")
    moveit_commander.roscpp_initialize([])

    robot = Robot(
        settings.arm_group,
        settings.hand_group,
        moveit_commander,
        finger_joint_1=settings.finger_joint_1,
        finger_joint_2=settings.finger_joint_2,
    )

    robot.MoveJ_J("Camera_Home")

    #Run a sequence from sequence.json (standard usage) ──────────
    completed = execute_sequence(robot)
    robot.MoveJ_J("Camera_Home")

    # Return step count so run_execute.py can apply only the completed portion
    return completed


if __name__ == "__main__":
    main()