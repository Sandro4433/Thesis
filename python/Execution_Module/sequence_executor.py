import json
from pathlib import Path

from paths import LLM_RESPONSE_JSON, CONFIGURATION_JSON
from Execution_Module.pick_and_place import pick_and_place

SEQUENCE_PATH      = Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json"
CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())

# Gripper width defaults (fallback if not specified in sequence)
GRIPPER_OPEN_WIDTH           = 0.075   # never changes
GRIPPER_CLOSE_WIDTH_STANDARD = 0.05    # Size = null
GRIPPER_CLOSE_WIDTH_LARGE    = 0.06    # Size = "large"


def _load_fragility_map() -> dict:
    """Return {part_name: True} for every part marked fragile in configuration.json."""
    if not CONFIGURATION_PATH.exists():
        return {}
    try:
        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        return {
            e["part"]: True
            for e in state.get("predicates", {}).get("fragility", [])
            if e.get("fragility") == "fragile"
        }
    except Exception:
        return {}


def execute_sequence(robot):
    """
    Reads sequence.json and executes each pick-and-place step.

    Sequence format (3-element):
      [ ["<pick_name>", "<place_name>", <gripper_close_width>], ... ]

    Legacy 2-element format is still accepted and falls back to the
    standard gripper close width (0.05).

    Fragility is read from configuration.json at execution time — fragile
    parts use reduced speed throughout their pick-and-place cycle.
    """
    entries       = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    fragility_map = _load_fragility_map()

    for i, entry in enumerate(entries):
        if len(entry) == 3:
            pick_name, place_name, gripper_close_width = entry
            gripper_close_width = float(gripper_close_width)
        elif len(entry) == 2:
            pick_name, place_name = entry
            gripper_close_width = GRIPPER_CLOSE_WIDTH_STANDARD
        else:
            print(f"  [Step {i+1}] Skipping malformed entry: {entry!r}")
            continue

        fragile = fragility_map.get(pick_name, False)

        print(
            f"  [Step {i+1}/{len(entries)}] pick='{pick_name}'  place='{place_name}'  "
            f"gripper_close={gripper_close_width}"
            + ("  ⚠ FRAGILE" if fragile else "")
        )
        pick_and_place(
            robot,
            pick_name,
            place_name,
            gripper_open_width=GRIPPER_OPEN_WIDTH,
            gripper_close_width=gripper_close_width,
            fragile=fragile,
        )

    print(f"\n✅  Sequence complete ({len(entries)} step(s)).")