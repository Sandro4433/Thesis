import json
from pathlib import Path

from paths import LLM_RESPONSE_JSON
from Execution_Module.pick_and_place import pick_and_place

SEQUENCE_PATH = Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json"

# Gripper width defaults (fallback if not specified in sequence)
GRIPPER_OPEN_WIDTH  = 0.075   # never changes
GRIPPER_CLOSE_WIDTH_STANDARD = 0.05   # Size = null
GRIPPER_CLOSE_WIDTH_LARGE    = 0.06   # Size = "large"


def execute_sequence(robot):
    """
    Reads sequence.json and executes each pick-and-place step.

    Sequence format (3-element):
      [ ["<pick_name>", "<place_name>", <gripper_close_width>], ... ]

    Legacy 2-element format is still accepted and falls back to the
    standard gripper close width (0.05).
    """
    entries = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))

    for i, entry in enumerate(entries):
        if len(entry) == 3:
            pick_name, place_name, gripper_close_width = entry
            gripper_close_width = float(gripper_close_width)
        elif len(entry) == 2:
            pick_name, place_name = entry
            gripper_close_width = GRIPPER_CLOSE_WIDTH_STANDARD  # standard size default
        else:
            print(f"  [Step {i+1}] Skipping malformed entry: {entry!r}")
            continue

        print(
            f"  [Step {i+1}/{len(entries)}] pick='{pick_name}'  place='{place_name}'  "
            f"gripper_close={gripper_close_width}"
        )
        pick_and_place(
            robot,
            pick_name,
            place_name,
            gripper_open_width=GRIPPER_OPEN_WIDTH,
            gripper_close_width=gripper_close_width,
        )

    print(f"\n✅  Sequence complete ({len(entries)} step(s)).")