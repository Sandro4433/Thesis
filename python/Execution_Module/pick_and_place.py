def pick_and_place(
    self,
    pick_name: str,
    place_name: str,
    home_name: str = "Home",
    approach_z_offset: float = -0.05,
    gripper_open_width: float = 0.075,
    gripper_close_width: float = 0.05,
):
   
    # Home
    self.MoveJ_J(home_name)

    # Approach pick location
    self.MoveJ(pick_name)
    
    # Rotate gripper
    self.MoveJointDelta(joint_index=6, target_name=pick_name)
    
    # Go down to pick
    self.MoveL(pick_name, offset=(0.0, 0.0, approach_z_offset), use_current_orientation=True)

    # Close gripper
    self.gripper_close(gripper_close_width)

    # Retreat up
    self.MoveL(pick_name, offset=(0.0, 0.0, 0.1), use_current_orientation=True)

    # Home
    self.MoveJ_J(home_name)

    # Approach place location
    self.MoveJ(place_name)

    # Rotate gripper
    self.MoveJointDelta(joint_index=6, target_name=place_name)
    
    # Go down to place
    self.MoveL(place_name, offset=(0.0, 0.0, approach_z_offset), use_current_orientation=True)

    # Open gripper
    self.gripper_open(gripper_open_width)

    # Retreat up
    self.MoveL(place_name, offset=(0.0, 0.0, 0.1), use_current_orientation=True)

    # Home
    self.MoveJ_J(home_name)

    return True
