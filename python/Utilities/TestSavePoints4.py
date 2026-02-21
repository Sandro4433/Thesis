#!/usr/bin/env python3
import sys, json
import rospy
import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

SAVE_FILE = "positions.jsonl"

# ======================================================================
# Utils
# ======================================================================

def normalize_quaternion(pose: Pose):
    import numpy as np
    q = np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ], dtype=float)
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q.tolist()

def wait_for_valid_time():
    for _ in range(50):
        if rospy.Time.now().to_sec() > 0:
            return True
        rospy.sleep(0.1)
    print("❌ ROS time not valid.")
    return False

def wait_for_joint_state():
    try:
        rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
        return True
    except rospy.ROSException:
        print("❌ No /joint_states received in time.")
        return False

# ======================================================================
# Load / Save
# ======================================================================

def load_points():
    points = {}

    try:
        with open(SAVE_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not all(k in data for k in ("name", "pos", "quat")):
                    continue

                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = data["pos"]
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = data["quat"]

                normalize_quaternion(pose)

                joints = data.get("joints", None)

                points[data["name"]] = {
                    "pose": pose,
                    "joints": joints
                }

        return points

    except FileNotFoundError:
        return {}

def save_point(name: str, pose: Pose, joints: list):
    normalize_quaternion(pose)

    rec = {
        "name": name,
        "pos": [pose.position.x, pose.position.y, pose.position.z],
        "quat": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        "joints": joints
    }

    with open(SAVE_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")

def delete_point(name: str):
    pts = load_points()

    if name not in pts:
        return False

    del pts[name]

    with open(SAVE_FILE, "w") as f:
        for n, entry in pts.items():
            pose = entry["pose"]
            joints = entry["joints"]

            rec = {
                "name": n,
                "pos": [pose.position.x, pose.position.y, pose.position.z],
                "quat": [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                "joints": joints
            }

            f.write(json.dumps(rec) + "\n")

    return True

# ======================================================================
# Motion
# ======================================================================

def go_to_point(name: str, arm: MoveGroupCommander, points: dict):
    entry = points.get(name)
    if entry is None:
        print(f"❌ Point '{name}' not found.")
        return False

    pose = entry["pose"]
    joints = entry["joints"]

    # ✅ If joint values exist → use joint PTP
    if joints is not None:
        arm.set_joint_value_target(joints)
        ok = arm.go(wait=True)
        arm.stop()
        return ok

    # ❌ fallback (rare): use pose IK
    normalize_quaternion(pose)
    ps = PoseStamped()
    ps.header.frame_id = arm.get_planning_frame()
    ps.pose = pose

    arm.set_pose_target(ps)
    ok = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    return ok


def set_width(gripper: MoveGroupCommander, width: float):
    target = max(0.0, min(0.04, width / 2.0))
    gripper.set_joint_value_target({
        "panda_finger_joint1": target,
        "panda_finger_joint2": target
    })
    gripper.go(wait=True)

# ======================================================================
# Main
# ======================================================================

def main():
    rospy.init_node("teach_and_go_points", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)

    if not wait_for_valid_time() or not wait_for_joint_state():
        sys.exit(1)

    robot = RobotCommander()
    scene = PlanningSceneInterface()
    arm = MoveGroupCommander("panda_arm")
    gripper = MoveGroupCommander("panda_hand")

    arm.set_goal_position_tolerance(0.003)
    arm.set_goal_orientation_tolerance(0.01)

    print(f"Planning frame: {arm.get_planning_frame()}")
    print("Loading points...")

    points = load_points()
    print("Loaded:", list(points.keys()) if points else "none")

    print("\nCommands:")
    print("  s <name>       : save current pose + joints")
    print("  goto <name>    : move to saved pose")
    print("  list           : list all points")
    print("  del <name>     : delete point")
    print("  grip <meters>  : set gripper width")
    print("  q              : quit\n")

    while not rospy.is_shutdown():
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        if cmd == "q":
            break

        elif cmd == "list":
            print(sorted(points.keys()))

        elif cmd.startswith("s "):
            name = cmd.split(" ", 1)[1].strip()
            if not name:
                print("❌ name required")
                continue

            pose = arm.get_current_pose().pose
            joints = arm.get_current_joint_values()

            save_point(name, pose, joints)

            points[name] = {
                "pose": pose,
                "joints": joints
            }

            print(f"✅ Saved '{name}' with joints")

        elif cmd.startswith("goto "):
            name = cmd.split(" ", 1)[1].strip()
            go_to_point(name, arm, points)

        elif cmd.startswith("del "):
            name = cmd.split(" ", 1)[1].strip()
            if delete_point(name):
                points.pop(name, None)
                print(f"✅ Deleted '{name}'")
            else:
                print(f"❌ '{name}' not found")

        elif cmd.startswith("grip "):
            try:
                w = float(cmd.split(" ", 1)[1])
                set_width(gripper, w)
                print("✅ Gripper set")
            except:
                print("❌ usage: grip 0.03")

        else:
            print("Unknown command")

    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
