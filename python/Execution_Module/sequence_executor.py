import json
from pathlib import Path

from paths import SEQUENCE_PATH, CONFIGURATION_PATH, FILE_EXCHANGE_DIR
from Execution_Module.pick_and_place import pick_and_place

CANCEL_SENTINEL = FILE_EXCHANGE_DIR / ".cancel_execution"

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


def _is_cancelled() -> bool:
    """Check whether the GUI has requested cancellation."""
    return CANCEL_SENTINEL.exists()


def _clear_cancel() -> None:
    """Remove the cancel sentinel file."""
    try:
        CANCEL_SENTINEL.unlink(missing_ok=True)
    except Exception:
        pass


def execute_sequence(robot):
    """
    Reads sequence.json and executes each pick-and-place step.

    Sequence format (3-element):
      [ ["<pick_name>", "<place_name>", <gripper_close_width>], ... ]

    Legacy 2-element format is still accepted and falls back to the
    standard gripper close width (0.05).

    Fragility is read from configuration.json at execution time — fragile
    parts use reduced speed throughout their pick-and-place cycle.

    Returns the number of steps that were actually executed (may be less
    than total if the user cancelled mid-sequence).
    """
    entries       = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    fragility_map = _load_fragility_map()
    completed     = 0

    for i, entry in enumerate(entries):
        # ── check for cancellation before starting the next step ──────────
        if _is_cancelled():
            _clear_cancel()
            print(f"\n⚠  Cancellation received after step {completed}/{len(entries)} "
                  f"— skipping remaining steps.")
            return completed

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
        completed += 1

    print(f"\n✅  Sequence complete ({len(entries)} step(s)).")
    return completed