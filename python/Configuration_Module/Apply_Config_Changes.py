# Apply_Config_Changes.py
# Can be run standalone OR imported by API_Main.py.
#
# Applies workspace_changes.json to positions.json and saves it in-place.
#
# Changes format:
#   Role changes  → keyed by SLOT name   e.g. "Kit_0_Pos_1": {"Role": "output"}
#   Size/Color    → keyed by PART name   e.g. "Part_Blue_Nr_3": {"Size": "large"}
#
# The script searches positions.json for each name and applies changes in-place.
# Parent-child structure is never altered by this script — only attributes change.
# Structural changes (moving parts between slots) are handled by apply_sequence_to_scene
# in API_Main.py after robot execution.

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

from paths import POSITIONS_JSON, LLM_RESPONSE_JSON

POSITIONS_PATH = Path(POSITIONS_JSON.resolve())
CHANGES_PATH   = Path(LLM_RESPONSE_JSON.resolve()).parent / "workspace_changes.json"


# ── Core apply function (importable) ─────────────────────────────────────────

def apply_changes(
    scene: Dict[str, Any],
    changes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Applies a changes dict to a full positions.json-style scene dict.
    Returns an updated deep copy. Does NOT modify the input.

    Key lookup rules:
      - If the key matches a slot name  → apply Role to that slot.
      - If the key matches a part name  → find the part (in child_parts or
        standalone parts) and apply Size / Color to it.

    Allowed attributes:
      Slots  : Role  (null | "input" | "output")
      Parts  : Size  (null | "large")
               Color ("Blue" | "Red")
    """
    result = copy.deepcopy(scene)
    slots  = result.get("slots", {})
    parts  = result.get("parts", {})

    not_found: list[str] = []

    for obj_name, attrs in changes.items():

        # ── 1. Slot name → apply Role ─────────────────────────────────────────
        if obj_name in slots:
            for attr, val in attrs.items():
                if attr == "Role":
                    slots[obj_name]["Role"] = val
                elif attr in ("Color", "Size"):
                    # Convenience: if user accidentally keys by slot for part attrs,
                    # redirect to child_part if one exists.
                    child = slots[obj_name].get("child_part")
                    if isinstance(child, dict):
                        child[attr] = val
                        print(f"  Note: '{attr}' on slot '{obj_name}' redirected to its child_part.")
                    else:
                        print(f"  WARNING: '{obj_name}' has no child_part — cannot set '{attr}', skipped.")
                else:
                    print(f"  WARNING: Unknown attribute '{attr}' for slot '{obj_name}' — skipped.")
            continue

        # ── 2. Part name → find in child_parts or standalone parts ────────────
        found = False

        # Search child_parts in all slots
        for slot in slots.values():
            child = slot.get("child_part")
            if isinstance(child, dict) and child.get("name") == obj_name:
                for attr, val in attrs.items():
                    if attr in ("Color", "Size"):
                        child[attr] = val
                    elif attr == "Role":
                        print(f"  WARNING: Role is a slot attribute, not a part attribute — "
                              f"skipped for '{obj_name}'.")
                    else:
                        print(f"  WARNING: Unknown attribute '{attr}' for part '{obj_name}' — skipped.")
                found = True
                break

        if not found and obj_name in parts:
            for attr, val in attrs.items():
                if attr in ("Color", "Size"):
                    parts[obj_name][attr] = val
                elif attr == "Role":
                    print(f"  WARNING: Role is a slot attribute, not a part attribute — "
                          f"skipped for '{obj_name}'.")
                else:
                    print(f"  WARNING: Unknown attribute '{attr}' for part '{obj_name}' — skipped.")
            found = True

        if not found:
            not_found.append(obj_name)

    if not_found:
        print("  WARNING: The following names were not found in the scene and were skipped:")
        for name in not_found:
            print(f"    - {name}")

    return result


# ── Standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    if not POSITIONS_PATH.exists():
        print(f"ERROR: positions.json not found: {POSITIONS_PATH}")
        sys.exit(1)
    if not CHANGES_PATH.exists():
        print(f"ERROR: Changes file not found: {CHANGES_PATH}")
        sys.exit(1)

    scene   = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    changes = json.loads(CHANGES_PATH.read_text(encoding="utf-8"))

    print(f"Loaded positions: {POSITIONS_PATH}")
    print(f"Loaded changes:   {CHANGES_PATH}")
    total_attrs = sum(len(v) for v in changes.values())
    print(f"  → {len(changes)} object(s), {total_attrs} attribute change(s)")

    updated = apply_changes(scene, changes)

    tmp = str(POSITIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(POSITIONS_PATH)

    print(f"Saved:            {POSITIONS_PATH}")


if __name__ == "__main__":
    main()