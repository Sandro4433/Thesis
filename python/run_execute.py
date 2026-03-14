"""
run_execute.py — Runs robot execution in a clean main thread.
Called as a subprocess by the GUI so rospy.init_node() gets the main thread.
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import CONFIGURATION_JSON, LLM_RESPONSE_JSON
from pathlib import Path
import json

CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
SEQUENCE_PATH      = Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json"

if not SEQUENCE_PATH.exists():
    print(f"\n[WARN] No sequence.json found at {SEQUENCE_PATH.resolve()}")
    print("   Plan a motion sequence first (option 2).\n")
    sys.exit(0)

sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
print(f"\n  Executing {len(sequence)} step(s) from: {SEQUENCE_PATH.resolve()}")

from Execution_Module.Robot_Main import main as robot_main  # type: ignore
robot_main()

print("\n-- Execution complete. --\n")

from Configuration_Module.Apply_Sequence_Changes import apply_and_save  # type: ignore
apply_and_save(CONFIGURATION_PATH, sequence, save_memory=True)