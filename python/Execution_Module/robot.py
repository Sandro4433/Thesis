# robot.py
import json
import numpy as np
from geometry_msgs.msg import Pose
from moveit_commander.exception import MoveItCommanderException

from paths import CONFIGURATION_JSON, POSITIONS_FIXED_JSONL

class Robot:
    def __init__(self, arm_name, hand_name, moveit_commander):
        self.arm = moveit_commander.MoveGroupCommander(arm_name)
        self.gripper = moveit_commander.MoveGroupCommander(hand_name)

        self.arm.set_goal_orientation_tolerance(0.02)
        self.arm.set_goal_position_tolerance(0.02)
        self.set_mode_ptp()

        self._positions = {}
        self._positions.update(self.load_points_snapshot_json(str(CONFIGURATION_JSON.resolve())))
        self._positions.update(self.load_points_jsonl(str(POSITIONS_FIXED_JSONL.resolve())))

        print("Loaded positions:", list(self._positions.keys()))

    # --------------------------
    #       PLANNERS
    # --------------------------
    def set_mode_ptp(self):
        self.arm.set_planner_id("PTP")
        self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.arm.set_max_velocity_scaling_factor(0.5)
        self.arm.set_max_acceleration_scaling_factor(0.5)

    def set_mode_lin(self):
        self.arm.set_planner_id("LIN")
        self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.arm.set_max_velocity_scaling_factor(0.1)
        self.arm.set_max_acceleration_scaling_factor(0.1)

    def set_mode_ptp_fragile(self):
        self.arm.set_planner_id("PTP")
        self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.arm.set_max_velocity_scaling_factor(0.2)
        self.arm.set_max_acceleration_scaling_factor(0.2)

    def set_mode_lin_fragile(self):
        self.arm.set_planner_id("LIN")
        self.arm.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.arm.set_max_velocity_scaling_factor(0.05)
        self.arm.set_max_acceleration_scaling_factor(0.05)

    # --------------------------
    #     MOTION FUNCTIONS
    # --------------------------
    def MoveL(self, name, positions=None, offset=None, use_current_orientation: bool = False,
              fragile: bool = False):
        if positions is None:
            positions = self._positions
        self.set_mode_lin_fragile() if fragile else self.set_mode_lin()
        return self.go_to_point_pose_only(
            name, positions, offset=offset,
            use_current_orientation=use_current_orientation
        )

    def MoveJ(self, name, positions=None, offset=None, fragile: bool = False):
        if positions is None:
            positions = self._positions
        self.set_mode_ptp_fragile() if fragile else self.set_mode_ptp()
        self.go_to_point_pose_only(name, positions, offset)

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

    def MoveJointDelta(self, joint_index: int = 6, delta_deg: float = 0.0, target_name: str = ""):
        if joint_index < 0 or joint_index > 6:
            raise ValueError("joint_index must be between 0 and 6 (panda has 7 joints).")

        if target_name:
            entry = self._positions.get(target_name)
            if entry is None:
                raise MoveItCommanderException(f"❌ Target '{target_name}' not found.")
            raw_ori = entry.get("orientation_deg", None)
            if raw_ori is None:
                print(f"⚠ '{target_name}' has no orientation — skipping gripper rotation.")
                return True
            use_deg = float(-raw_ori)
        else:
            use_deg = float(delta_deg)

        self.set_mode_ptp()
        current_joints = self.arm.get_current_joint_values()
        delta_rad = float(np.deg2rad(use_deg))
        new_joints = list(current_joints)
        new_joints[joint_index] += delta_rad

        self.arm.set_joint_value_target(new_joints)
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

        if not success:
            raise MoveItCommanderException("❌ Joint delta move failed.")
        return success

    # --------------------------
    # For PTP/LIN movement
    # --------------------------
    def go_to_point_pose_only(self, name, positions, offset=None, use_current_orientation: bool = False):
        entry = positions.get(name)
        if entry is None:
            print(f"⚠ Position '{name}' not found.")
            return False

        pose = entry["pose"]
        pose_goal = Pose()
        pose_goal.position.x = pose.position.x
        pose_goal.position.y = pose.position.y
        pose_goal.position.z = pose.position.z

        if offset is not None:
            pose_goal.position.x += offset[0]
            pose_goal.position.y += offset[1]
            pose_goal.position.z += offset[2]

        if use_current_orientation:
            current_pose = self.arm.get_current_pose().pose
            pose_goal.orientation = current_pose.orientation
        else:
            pose_goal.orientation = pose.orientation

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
            width = 0.075
        self.set_width(width)
        self.gripper.go(wait=True)

    def gripper_close(self, width=None):
        if width is None:
            width = 0.06
        self.set_width(width)
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
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w
        ])
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm
            pose.orientation.x, pose.orientation.y, \
            pose.orientation.z, pose.orientation.w = q

    # --------------------------
    #   LOADERS
    # --------------------------
    def _entry_to_pose_record(self, name: str, data: dict) -> dict:
        pose = Pose()
        pose.position.x = data["pos"][0]
        pose.position.y = data["pos"][1]
        pose.position.z = data["pos"][2]
        pose.orientation.x = data["quat"][0]
        pose.orientation.y = data["quat"][1]
        pose.orientation.z = data["quat"][2]
        pose.orientation.w = data["quat"][3]
        return {
            "pose":            pose,
            "joints":          data.get("joints", None),
            "orientation_deg": data.get("orientation", None),
        }

    def load_points_snapshot_json(self, save_file: str) -> dict:
        """
        Load robot positions from the new PDDL-friendly configuration.json.

        Every entry in state["metric"] that has pos + quat becomes a named
        motion target. This covers:
          - slots (Kit_0_Pos_1, Container_3_Pos_2, …)   → place targets
          - embedded parts (Part_Blue_Nr_1, …)           → pick targets
          - standalone parts                              → pick targets
        """
        positions: dict = {}
        try:
            with open(save_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except FileNotFoundError:
            print(f"⚠ File not found: {save_file}")
            return positions
        except Exception as e:
            print(f"⚠ Failed to read {save_file}: {e}")
            return positions

        metric = state.get("metric", {})
        if not isinstance(metric, dict):
            print(f"⚠ 'metric' section missing or invalid in {save_file}")
            return positions

        for name, entry in metric.items():
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(entry, dict):
                continue
            if "pos" not in entry or "quat" not in entry:
                continue
            if entry["pos"] is None or entry["quat"] is None:
                continue
            positions[name] = self._entry_to_pose_record(name, entry)

        return positions

    def load_points_jsonl(self, save_file: str) -> dict:
        """Load fixed named positions from the legacy JSONL file (Home, Camera_Home, etc.)."""
        positions: dict = {}
        try:
            with open(save_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    name = data.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    positions[name] = self._entry_to_pose_record(name, data)
        except FileNotFoundError:
            print(f"⚠ File not found: {save_file}")
        return positions