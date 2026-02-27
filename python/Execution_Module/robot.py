# robot.py
import json
import numpy as np
from geometry_msgs.msg import Pose
from moveit_commander.exception import MoveItCommanderException

from paths import POSITIONS_JSON, POSITIONS_FIXED_JSONL

class Robot:
    def __init__(self, arm_name, hand_name, moveit_commander):
        self.arm = moveit_commander.MoveGroupCommander(arm_name)
        self.gripper = moveit_commander.MoveGroupCommander(hand_name)

        self.arm.set_goal_orientation_tolerance(0.02)
        self.arm.set_goal_position_tolerance(0.02)
        self.set_mode_ptp()

        self._positions = {}
        # Load positions from the shared exchange folder (stable paths)
        self._positions.update(self.load_points_snapshot_json(str(POSITIONS_JSON.resolve())))
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

    # --------------------------
    #     MOTION FUNCTIONS
    # --------------------------
    def MoveL(self, name, positions=None, offset=None, use_current_orientation: bool = False):
        if positions is None:
            positions = self._positions
        self.set_mode_lin()
        return self.go_to_point_pose_only(
            name,
            positions,
            offset=offset,
            use_current_orientation=use_current_orientation
        )

    def MoveJ(self, name, positions=None, offset=None):
        if positions is None:
            positions = self._positions
        self.set_mode_ptp()
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
                raise MoveItCommanderException(f"❌ Target '{target_name}' not found in loaded positions.")
            ori_deg = -entry.get("orientation_deg", None)
            if ori_deg is None:
                raise MoveItCommanderException(f"❌ Target '{target_name}' has no 'orientation' field in JSON.")
            use_deg = float(ori_deg)
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
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

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
            "pose": pose,
            "joints": data.get("joints", None),
            "orientation_deg": data.get("orientation", None),
        }

    def load_points_snapshot_json(self, save_file: str):
        """
        Loads the new JSON snapshot:
          {
            "slots": { name: {pos,quat,orientation,...} },
            "parts": { name: {...} }
          }

        Robot positions are built from:
          - each slot itself
          - if slot has child_part, also create a derived target "<slot_name>__child"
            (optional convenience)
        """
        positions = {}
        try:
            with open(save_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except FileNotFoundError:
            print(f"⚠ File not found: {save_file}")
            return positions
        except Exception as e:
            print(f"⚠ Failed to read {save_file}: {e}")
            return positions

        slots = state.get("slots", {}) if isinstance(state, dict) else {}
        if not isinstance(slots, dict):
            return positions

        for name, slot in slots.items():
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(slot, dict):
                continue

            # reconstruct a flat record with name
            record = dict(slot)
            record["name"] = name

            # must have pos/quat for motion
            if "pos" in record and "quat" in record:
                positions[name] = self._entry_to_pose_record(name, record)

            # Optional convenience: expose child pose as "<slot>__child"
            child = slot.get("child_part")
            if isinstance(child, dict) and "pos" in child and "quat" in child:
                child_name = f"{name}__child"
                child_record = dict(child)
                child_record["name"] = child_name
                positions[child_name] = self._entry_to_pose_record(child_name, child_record)

        return positions

    def load_points_jsonl(self, save_file: str):
        """
        Keeps your old JSONL loader for fixed points.
        """
        positions = {}
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