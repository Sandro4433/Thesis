"""
run_execute.py — Runs robot execution in a clean main thread.
Called as a subprocess by the GUI so rospy.init_node() gets the main thread.
"""
import sys
import json
from pathlib import Path


from Core.paths import CONFIGURATION_PATH, SEQUENCE_PATH

if not SEQUENCE_PATH.exists():
    print(f"\n[WARN] No sequence.json found at {SEQUENCE_PATH.resolve()}")
    print("   Plan a motion sequence first (option 2).\n")
    sys.exit(0)

sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
print(f"\n  Executing {len(sequence)} step(s) from: {SEQUENCE_PATH.resolve()}")

from Execution_Module.Robot_Main import main as robot_main  # type: ignore
completed = robot_main()

if completed is None:
    # Backwards-compat: if main() returned None treat as fully completed
    completed = len(sequence)

if completed < len(sequence):
    print(f"\n-- Execution cancelled after {completed}/{len(sequence)} step(s). --\n")
else:
    print("\n-- Execution complete. --\n")

# Only apply the steps that were actually executed
executed_sequence = sequence[:completed]

from Configuration_Module.apply_sequence_changes import apply_and_save
if executed_sequence:
    apply_and_save(CONFIGURATION_PATH, executed_sequence, save_memory=True)
else:
    print("  No steps were executed — configuration unchanged.")