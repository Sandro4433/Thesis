from pick_and_place import pick_and_place
import rospy
import moveit_commander
from robot import Robot 
from geometry_msgs.msg import Pose

rospy.init_node("franka_go_points")
moveit_commander.roscpp_initialize([])

robot = Robot("panda_arm", "panda_hand", moveit_commander)

robot.MoveJ_J("Camera_Home")


"""
pick_and_place(robot, "April_Tag_1_Pos_1", "April_Tag_2_Pos_3")
pick_and_place(robot, "April_Tag_2_Pos_2", "April_Tag_1_Pos_1")

robot.MoveJ_J("Camera_Home")

"""