def pick_and_place(
    robot,
    pick_name: str,
    place_name: str,
    home_name: str = "Home",
    approach_z_offset: float = -0.05,
    gripper_open_width: float = 0.075,
    gripper_close_width: float = 0.05,
    fragile: bool = False,
):
    # Home
    robot.MoveJ_J(home_name)

    # Approach pick location
    robot.MoveJ(pick_name, fragile=fragile)

    # Rotate gripper
    robot.MoveJointDelta(joint_index=6, target_name=pick_name)

    # Open gripper
    robot.gripper_open(gripper_open_width)

    # Go down to pick
    robot.MoveL(pick_name, offset=(0.0, 0.0, approach_z_offset),
               use_current_orientation=True, fragile=fragile)

    # Close gripper
    robot.gripper_close(gripper_close_width)

    # Retreat up
    robot.MoveL(pick_name, offset=(0.0, 0.0, 0.1),
               use_current_orientation=True, fragile=fragile)

    # Home
    robot.MoveJ_J(home_name)

    # Approach place location
    robot.MoveJ(place_name, fragile=fragile)

    # Rotate gripper
    robot.MoveJointDelta(joint_index=6, target_name=place_name)

    # Go down to place
    robot.MoveL(place_name, offset=(0.0, 0.0, approach_z_offset),
               use_current_orientation=True, fragile=fragile)

    # Open gripper
    robot.gripper_open(gripper_open_width)

    # Retreat up
    robot.MoveL(place_name, offset=(0.0, 0.0, 0.1),
               use_current_orientation=True, fragile=fragile)

    # Home
    robot.MoveJ_J(home_name)

    return True