def pick_and_place(
    self,
    pick_name: str,
    place_name: str,
    home_name: str = "Home",
    approach_z_offset: float = -0.05,
    gripper_open_width: float = 0.075,
    gripper_close_width: float = 0.06,
):
   

    self.MoveJ_J(home_name)

    self.MoveJ(pick_name)
    
    self.MoveJointDelta(joint_index=6, target_name=pick_name)
    
    self.MoveL(pick_name, offset=(0.0, 0.0, approach_z_offset), use_current_orientation=True)

    # close gripper
    self.gripper_close(gripper_close_width)

    #  retreat up
    self.MoveL(pick_name, offset=(0.0, 0.0, 0.1), use_current_orientation=True)

    #  go to place
    self.MoveJ_J(place_name)

  
    self.gripper_open(gripper_open_width)

    #  camera home
    self.MoveJ_J(home_name)

    return True
