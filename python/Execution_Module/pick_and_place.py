"""
pick_and_place.py — Single pick-and-place cycle for the Franka Panda arm.

Each call performs one complete pick-and-place in the following order:
  Home → approach pick (PTP) → rotate gripper → open → descend (LIN) →
  close → retreat (LIN) → Home → approach place (PTP) → rotate gripper →
  descend (LIN) → open → retreat (LIN) → Home
"""


def pick_and_place(
    robot,
    pick_name: str,
    place_name: str,
    home_name: str = "Home",
    approach_z_offset: float = -0.05,
    gripper_open_width: float = 0.075,
    gripper_close_width: float = 0.05,
    fragile: bool = False,
) -> bool:
    """Execute one pick-and-place cycle between two named positions.

    Parameters
    ----------
    robot:
        :class:`Execution_Module.robot.Robot` instance.
    pick_name:
        Named position of the part to pick.
    place_name:
        Named position of the destination slot.
    home_name:
        Named safe-transit position used between every major move.
    approach_z_offset:
        Vertical offset (metres) applied when descending to pick/place.
        Negative = downward.
    gripper_open_width:
        Finger spread (metres) used when opening the gripper.
    gripper_close_width:
        Finger spread (metres) used when gripping the part.
    fragile:
        If True, all moves use reduced velocity/acceleration limits.

    Returns
    -------
    bool
        Always ``True``; exceptions from the robot layer propagate unchanged.
    """
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