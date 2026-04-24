"""
move_camera_home.py — Move the robot to the Camera_Home position.

Called as a subprocess by the GUI so rospy.init_node() gets the main thread.
All init noise (rospy, moveit, Robot, ROS C++ loggers) is suppressed —
only clean status messages are printed for the GUI chat panel.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))




def main() -> None:
    # ── Suppress ALL init noise, including C-level ROS loggers ───────────────

    # Save original Python streams and OS file descriptors
    _py_stdout = sys.stdout
    _py_stderr = sys.stderr
    _fd_stdout = os.dup(1)
    _fd_stderr = os.dup(2)

    # Redirect both Python streams and OS file descriptors to /dev/null
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _devnull_py = open(os.devnull, "w")
    sys.stdout = _devnull_py
    sys.stderr = _devnull_py
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)

    try:
        import rospy
        import moveit_commander
        from Execution_Module.robot import Robot
        from Core.config import settings

        rospy.init_node("franka_camera_home")
        moveit_commander.roscpp_initialize([])
        robot = Robot(
            settings.arm_group,
            settings.hand_group,
            moveit_commander,
            finger_joint_1=settings.finger_joint_1,
            finger_joint_2=settings.finger_joint_2,
        )
    finally:
        # Restore everything
        os.dup2(_fd_stdout, 1)
        os.dup2(_fd_stderr, 2)
        os.close(_fd_stdout)
        os.close(_fd_stderr)
        os.close(_devnull_fd)
        _devnull_py.close()
        sys.stdout = _py_stdout
        sys.stderr = _py_stderr

    # ── Clean messages only ──────────────────────────────────────────────
    print("Moving to camera home position...", flush=True)
    robot.MoveJ_J("Camera_Home")
    print("Camera home position reached.", flush=True)


if __name__ == "__main__":
    main()