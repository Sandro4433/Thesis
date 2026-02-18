
# robot.py
import numpy as np
from geometry_msgs.msg import Pose
from moveit_commander.exception import MoveItCommanderException
import json

'''
# --- Set up FRANKA ---
FCI needs to be enabled, joints unlocked and enabeling Button not pressed (blue mode)

In your script initalize the Franka as follows
Example for FRANKA EMIKA

rospy.init_node("franka_go_points")
moveit_commander.roscpp_initialize([])

robot = Robot("panda_arm", "panda_hand", moveit_commander)


For trajectory planning, RViz needs to be running. Change the ip to your Franka controller.

source ~/ws_moveit/devel/setup.bash
roslaunch panda_moveit_config  franka_control.launch robot_ip:=192.168.1.100 load_gripper:=true use_rviz:=false


'''



class Robot:
    def __init__(self, arm_name, hand_name, moveit_commander):
        self.arm = moveit_commander.MoveGroupCommander(arm_name)
        self.gripper = moveit_commander.MoveGroupCommander(hand_name)

        self.arm.set_goal_orientation_tolerance(0.02)
        self.arm.set_goal_position_tolerance(0.02)
        self.set_mode_ptp()

        self._positions = self.load_points()
        print("Loaded positions:", list(self._positions.keys()))


    # --------------------------
    #       PLANNERS
    # --------------------------

    def set_mode_ptp(self):
        self.arm.set_planner_id("PTP")
        self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.arm.set_max_velocity_scaling_factor(0.2)
        self.arm.set_max_acceleration_scaling_factor(0.2)

    def set_mode_lin(self):
        self.arm.set_planner_id("LIN")
        self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.arm.set_max_velocity_scaling_factor(0.1)
        self.arm.set_max_acceleration_scaling_factor(0.1)


    # --------------------------
    #     MOTION FUNCTIONS
    # --------------------------

    # Linear movement
    def MoveL(self, name, positions=None, offset=None):
        if positions is None:
            positions = self._positions
        self.set_mode_lin()
        self.go_to_point_pose_only(name, positions, offset)

    # Non-linear movement
    def MoveJ(self, name, positions=None, offset=None):
        if positions is None:
            positions = self._positions
        self.set_mode_ptp()
        self.go_to_point_pose_only(name, positions, offset)

    # Joint movement
    def MoveJ_J(self, name, positions=None):
        if positions is None:
            positions = self._positions
        entry = positions.get(name)
        if entry is None:
            print(f"⚠ Position '{name}' not found.")
            return False

        joints = entry["joints"]
        if joints is None:
            raise MoveItCommanderException(f"❌ '{name}' has no joint data.")

        self.set_mode_ptp()
        self.arm.set_joint_value_target(joints)

        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

        if not success:
            raise MoveItCommanderException("❌ Joint PTP failed.")

        return success

    # --------------------------
    # For PTP/LIN movement
    # --------------------------
    def go_to_point_pose_only(self, name, positions, offset=None):
        entry = positions.get(name)
        if entry is None:
            print(f"⚠ Position '{name}' not found.")
            return False

        pose = entry["pose"]

        pose_goal = Pose()
        pose_goal.position.x = pose.position.x
        pose_goal.position.y = pose.position.y
        pose_goal.position.z = pose.position.z
        pose_goal.orientation = pose.orientation

        if offset is not None:
            pose_goal.position.x += offset[0]
            pose_goal.position.y += offset[1]
            pose_goal.position.z += offset[2]

        self.normalize_quaternion(pose_goal)
        self.arm.set_pose_target(pose_goal)

        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

        if not success:
            raise MoveItCommanderException("❌ Pose-PTP/LIN failed.")

        return success

    # --------------------------
    #      GRIPPER
    # --------------------------
    def gripper_open(self, width=None):
        if width is None:
            width = 0.04
        self.set_width(width)
        self.gripper.go(wait=True)

    def gripper_close(self):
        self.set_width(0.0005)
        self.gripper.go(wait=True)

    def set_width(self, width):
        target = width / 2.0
        self.gripper.set_joint_value_target("panda_finger_joint1", target)
        self.gripper.set_joint_value_target("panda_finger_joint2", target)
        self.gripper.go(wait=True)

    # --------------------------
    #   UTILITIES
    # --------------------------
    def normalize_quaternion(self, pose):
        q = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q


    def load_points(self):
        SAVE_FILE = "positions.jsonl"       # file in which the postions are saved
        positions = {}
        try:
            with open(SAVE_FILE, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line.strip())

                    pose = Pose()
                    pose.position.x = data["pos"][0]
                    pose.position.y = data["pos"][1]
                    pose.position.z = data["pos"][2]
                    pose.orientation.x = data["quat"][0]
                    pose.orientation.y = data["quat"][1]
                    pose.orientation.z = data["quat"][2]
                    pose.orientation.w = data["quat"][3]

                    joints = data.get("joints", None)

                    positions[data["name"]] = {
                        "pose": pose,
                        "joints": joints
                    }

        except FileNotFoundError:
            print(f"⚠ File not found: {SAVE_FILE}")

        return positions