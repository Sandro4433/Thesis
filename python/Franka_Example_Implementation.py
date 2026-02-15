import rospy
import moveit_commander
from robot import Robot 
from geometry_msgs.msg import Pose

rospy.init_node("franka_go_points")
moveit_commander.roscpp_initialize([])

robot = Robot("panda_arm", "panda_hand", moveit_commander)


robot.MoveJ("Camera_Home")
    
