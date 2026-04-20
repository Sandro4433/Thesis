# Robot_Main.py
from __future__ import annotations

import sys

import rospy
import moveit_commander

from Execution_Module.robot import Robot
from Execution_Module.sequence_executor import execute_sequence


# ── Robot hardware configuration ──────────────────────────────────────────────
# Change these if you are using a different robot.
# arm_group / hand_group must match the MoveIt planning group names.
# finger_joint_1/2 must match the URDF gripper finger joint names.
ROBOT_CONFIG = {
    "arm_group":      "panda_arm",
    "hand_group":     "panda_hand",
    "finger_joint_1": "panda_finger_joint1",
    "finger_joint_2": "panda_finger_joint2",
}


def main() -> None:
    rospy.init_node("franka_go_points")
    moveit_commander.roscpp_initialize([])

    robot = Robot(
        ROBOT_CONFIG["arm_group"],
        ROBOT_CONFIG["hand_group"],
        moveit_commander,
        finger_joint_1=ROBOT_CONFIG["finger_joint_1"],
        finger_joint_2=ROBOT_CONFIG["finger_joint_2"],
    )

    robot.MoveJ_J("Camera_Home")

    #Run a sequence from sequence.json (standard usage) ──────────
    completed = execute_sequence(robot)
    robot.MoveJ_J("Camera_Home")

    # Return step count so run_execute.py can apply only the completed portion
    return completed


if __name__ == "__main__":
    main()