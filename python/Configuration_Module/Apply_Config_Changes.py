# Apply_Config_Changes.py
# Can be run standalone OR imported by API_Main.py.
#
# Standalone: reads baseline and changes from paths.py paths, saves config.json to Memory/.
# Imported:   call apply_changes(scene_dict, changes_dict) directly.

from __future__ import annotations

from pathlib import Path
import sys
import json
import copy
from typing import Any, Dict

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON

LLM_INPUT_PATH  = Path(LLM_INPUT_JSON.resolve())
CHANGES_PATH    = Path(LLM_RESPONSE_JSON.resolve()).parent / "workspace_changes.json"
MEMORY_DIR      = PROJECT_DIR / "Memory"
CONFIG_OUT_PATH = MEMORY_DIR / "config.json"


# ── Core apply function (importable) ─────────────────────────────────────────

def apply_changes(
    scene: Dict[str, Any],
    changes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Applies a changes dict onto a scene dict and returns the updated scene.

    Lookup order per changed object name:
      1. Top-level slot  → scene["slots"][name]
         - Role applies to the slot itself.
         - Color/Size are redirected to the slot's child_part (if one exists).
      2. Top-level part  → scene["parts"][name]
         - Color/Size apply directly.
      3. Embedded child_part → searches all slot["child_part"] for matching name
         - Color/Size apply to the embedded part.

    Allowed attributes:
      Slots : Role, (Size/Color redirected to child_part)
      Parts : Color, Size
    """
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
                    slot[attr] = val
                elif attr in ("Color", "Size"):
                    # Part attributes — redirect to child_part
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
        print("  WARNING: The following names were not found in the scene and were skipped:")
        for name in not_found:
            print(f"    - {name}")

    return result


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    if not LLM_INPUT_PATH.exists():
        print(f"ERROR: LLM input file not found: {LLM_INPUT_PATH}")
        sys.exit(1)
    if not CHANGES_PATH.exists():
        print(f"ERROR: Changes file not found: {CHANGES_PATH}")
        sys.exit(1)

    scene   = json.loads(LLM_INPUT_PATH.read_text(encoding="utf-8"))
    changes = json.loads(CHANGES_PATH.read_text(encoding="utf-8"))

    print(f"Loaded scene:   {LLM_INPUT_PATH}")
    print(f"Loaded changes: {CHANGES_PATH}")
    total_attrs = sum(len(v) for v in changes.values())
    print(f"  → {len(changes)} object(s), {total_attrs} attribute change(s)")

    updated = apply_changes(scene, changes)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(CONFIG_OUT_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(CONFIG_OUT_PATH)

    print(f"Saved config:   {CONFIG_OUT_PATH}")


if __name__ == "__main__":
    main()