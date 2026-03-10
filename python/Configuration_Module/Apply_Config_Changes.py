# apply_changes.py
# Loads llm_input.json + workspace_changes.json, applies the changes,
# and saves the result as config.json in the same directory.

from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Any, Dict

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON

LLM_INPUT_PATH   = Path(LLM_INPUT_JSON.resolve())
CHANGES_PATH     = Path(LLM_RESPONSE_JSON.resolve()).parent / "workspace_changes.json"
CONFIG_OUT_PATH  = Path(LLM_RESPONSE_JSON.resolve()).parent / "config.json"


# ── Apply logic ───────────────────────────────────────────────────────────────

def apply_changes(
    scene: Dict[str, Any],
    changes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Applies workspace_changes.json onto the llm_input scene dict.

    Lookup order per changed object name:
      1. Top-level slot  (scene["slots"][name])
      2. Top-level part  (scene["parts"][name])
      3. Embedded part   (scene["slots"][slot]["child_part"] where child_part name matches)

    Allowed attributes:
      Slots : Role
      Parts : Color, Size
    """
    import copy
    result = copy.deepcopy(scene)

    slots = result.get("slots", {})
    parts = result.get("parts", {})

    not_found: list[str] = []

    for obj_name, attrs in changes.items():

        # 1. Direct slot match
        if obj_name in slots:
            slot = slots[obj_name]
            for attr, val in attrs.items():
                if attr == "Role":
                    # Role belongs to the slot itself
                    slot[attr] = val
                elif attr in ("Color", "Size"):
                    # Part attributes — redirect to child_part if one exists
                    child = slot.get("child_part")
                    if isinstance(child, dict):
                        child[attr] = val
                    else:
                        print(f"  WARNING: '{obj_name}' has no child_part — cannot set '{attr}', skipped.")
                else:
                    print(f"  WARNING: Unknown attribute '{attr}' for '{obj_name}' — skipped.")
            continue

        # 2. Direct part match (standalone)
        if obj_name in parts:
            for attr, val in attrs.items():
                if attr not in ("Color", "Size"):
                    print(f"  WARNING: '{attr}' is not a valid part attribute — skipped for '{obj_name}'.")
                    continue
                parts[obj_name][attr] = val
            continue

        # 3. Embedded child_part inside a slot
        found_embedded = False
        for slot in slots.values():
            child = slot.get("child_part")
            if isinstance(child, dict) and child.get("name") == obj_name:
                for attr, val in attrs.items():
                    if attr not in ("Color", "Size"):
                        print(f"  WARNING: '{attr}' is not a valid part attribute — skipped for '{obj_name}'.")
                        continue
                    child[attr] = val
                found_embedded = True
                break

        if not found_embedded:
            not_found.append(obj_name)

    if not_found:
        print(f"  WARNING: The following names were not found in the scene and were skipped:")
        for name in not_found:
            print(f"    - {name}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load llm_input.json
    if not LLM_INPUT_PATH.exists():
        print(f"ERROR: LLM input file not found: {LLM_INPUT_PATH}")
        sys.exit(1)

    scene = json.loads(LLM_INPUT_PATH.read_text(encoding="utf-8"))
    print(f"Loaded scene:   {LLM_INPUT_PATH}")

    # Load workspace_changes.json
    if not CHANGES_PATH.exists():
        print(f"ERROR: Changes file not found: {CHANGES_PATH}")
        sys.exit(1)

    changes = json.loads(CHANGES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded changes: {CHANGES_PATH}")
    total_attrs = sum(len(v) for v in changes.values())
    print(f"  → {len(changes)} object(s), {total_attrs} attribute change(s)")

    # Apply
    updated = apply_changes(scene, changes)

    # Save config.json
    CONFIG_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(CONFIG_OUT_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(CONFIG_OUT_PATH)

    print(f"Saved config:   {CONFIG_OUT_PATH}")


if __name__ == "__main__":
    main()