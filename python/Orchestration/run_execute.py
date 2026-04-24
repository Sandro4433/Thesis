"""
run_execute.py — Runs robot execution in a clean main thread.

Called as a subprocess by the GUI so that rospy.init_node() always gets the
process main thread, which ROS requires.

The sequence is executed step-by-step.  If the user cancels mid-run, only
the completed steps are applied back to configuration.json, leaving the
workspace state consistent with what actually happened physically.
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Core.paths import CONFIGURATION_PATH, SEQUENCE_PATH


def main() -> None:
    if not SEQUENCE_PATH.exists():
        print(f"\n[WARN] No sequence.json found at {SEQUENCE_PATH.resolve()}")
        print("   Plan a motion sequence first (option 2).\n")
        sys.exit(0)

    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    print(f"\n  Executing {len(sequence)} step(s) from: {SEQUENCE_PATH.resolve()}")

    # Robot_Main is imported here (not at the top) because importing it
    # triggers ROS/MoveIt initialisation, which must happen after the
    # module-level guard above confirms there is actually work to do.
    from Execution_Module.Robot_Main import main as robot_main  # type: ignore
    completed = robot_main()

    # Backwards-compat: older Robot_Main.main() returned None
    if completed is None:
        completed = len(sequence)

    if completed < len(sequence):
        print(f"\n-- Execution cancelled after {completed}/{len(sequence)} step(s). --\n")
    else:
        print("\n-- Execution complete. --\n")

    # Apply only the steps that were physically executed so that
    # configuration.json stays consistent with the real workspace state.
    executed_sequence = sequence[:completed]

    from Configuration_Module.apply_sequence_changes import apply_and_save
    if executed_sequence:
        apply_and_save(CONFIGURATION_PATH, executed_sequence, save_memory=True)
    else:
        print("  No steps were executed — configuration unchanged.")


if __name__ == "__main__":
    main()
