# Robot_Main.py
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import rospy
import moveit_commander

from Execution_Module.robot import Robot
from Execution_Module.sequence_executor import execute_sequence


def main() -> None:
    rospy.init_node("franka_go_points")
    moveit_commander.roscpp_initialize([])

    robot = Robot("panda_arm", "panda_hand", moveit_commander)

    robot.MoveJ_J("Camera_Home")

    # ── Option A: Run a sequence from sequence.json (standard usage) ──────────
    completed = execute_sequence(robot)

    # ── Option B: Single manual pick-and-place (still works as before) ────────
    # robot.pick_and_place("Container_1_Pos_1", "Kit_0_Pos_1")

    robot.MoveJ_J("Camera_Home")

    # Return step count so run_execute.py can apply only the completed portion
    return completed


if __name__ == "__main__":
    main()