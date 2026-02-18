import rospy
import moveit_commander
from robot import Robot 
from geometry_msgs.msg import Pose

rospy.init_node("franka_go_points")
moveit_commander.roscpp_initialize([])

robot = Robot("panda_arm", "panda_hand", moveit_commander)


robot.MoveJ_J("Camera_Home")
robot.MoveJ("Camera_Home",offset=[0.0,0.0,-0.01])
robot.MoveJ("Camera_Home_test")

    
