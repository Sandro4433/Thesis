import json
from pathlib import Path

from paths import LLM_RESPONSE_JSON
from Execution_Module.pick_and_place import pick_and_place

SEQUENCE_PATH = Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json"


def execute_sequence(robot):
    pairs = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))

    for pick_name, place_name in pairs:
        print(f"pick_and_place('{pick_name}', '{place_name}')")
        pick_and_place(robot, pick_name, place_name)  # uses "Home" by default