# Apply_Config_Changes.py
# Applies a workspace_changes dict to the new PDDL-friendly configuration.json.
#
# Changes format (output by the LLM):
# {
#   "<receptacle_name>": {"role": "input"},        e.g. "Container_3"
#   "<part_name>":       {"size": "large"},        e.g. "Part_Blue_Nr_1"
#   "<part_name>":       {"color": "red"},
#   "workspace":         {"operation_mode": "kitting", "batch_size": 2},
#   "priority":          [{"color": "blue", "order": 1}, ...],
#   "kit_recipe":        [{"kit": "Kit_0", "color": "blue", "quantity": 2}, ...],
#   "part_compatibility":[{"part_color": "blue", "allowed_in": ["Kit_0"]}]
# }
#
# Backward-compatibility: slot-level role keys (e.g. "Container_3_Pos_1")
# are also accepted and translated to their parent receptacle.

from __future__ import annotations

from pathlib import Path
import sys
import json
import copy
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import CONFIGURATION_JSON, LLM_RESPONSE_JSON

CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
CHANGES_PATH   = Path(LLM_RESPONSE_JSON.resolve()).parent / "workspace_changes.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _parent_of(slot_name: str) -> str | None:
    idx = slot_name.rfind("_Pos_")
    return slot_name[:idx] if idx != -1 else None


def _upsert_list(lst: List[Dict], key_field: str, key_val: Any,
                 update: Dict[str, Any]) -> None:
    """Update first matching entry in a list-of-dicts; append if not found."""
    for item in lst:
        if item.get(key_field) == key_val:
            item.update(update)
            return
    lst.append({key_field: key_val, **update})


# ── core apply function ───────────────────────────────────────────────────────

def apply_changes(
    state: Dict[str, Any],
    changes: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply a changes dict to a PDDL-friendly state.
    Returns an updated deep copy; does NOT modify the input.

    Special top-level keys:
      "workspace"          → updates state["workspace"]
      "priority"           → replaces state["predicates"]["priority"]
      "kit_recipe"         → replaces state["predicates"]["kit_recipe"]
      "part_compatibility" → replaces state["predicates"]["part_compatibility"]

    Object keys:
      receptacle name (Kit_* / Container_*) → role update in predicates.role
      slot name (*_Pos_*)                   → role update (translated to parent)
      part name (Part_*)                    → size / color update in predicates
    """
    result = copy.deepcopy(state)
    preds  = result.setdefault("predicates", {})
    preds.setdefault("role",              [])
    preds.setdefault("size",              [])
    preds.setdefault("color",             [])
    preds.setdefault("priority",          [])
    preds.setdefault("kit_recipe",        [])
    preds.setdefault("part_compatibility",[])
    preds.setdefault("fragility",         [])
    result.setdefault("workspace", {"operation_mode": None, "batch_size": None})

    not_found: List[str] = []

    for key, value in changes.items():

        # ── workspace-level attributes ────────────────────────────────────────
        if key == "workspace" and isinstance(value, dict):
            result["workspace"].update(value)
            print(f"  workspace: {value}")
            continue

        # ── list-type predicate replacements ─────────────────────────────────
        if key == "priority" and isinstance(value, list):
            preds["priority"] = value
            print(f"  priority: {value}")
            continue

        if key == "kit_recipe" and isinstance(value, list):
            preds["kit_recipe"] = value
            print(f"  kit_recipe: {value}")
            continue

        if key == "part_compatibility" and isinstance(value, list):
            preds["part_compatibility"] = value
            print(f"  part_compatibility: {value}")
            continue

        if not isinstance(value, dict):
            print(f"  WARNING: unrecognised change entry '{key}': {value!r} — skipped.")
            continue

        # ── receptacle role (direct: "Container_3": {"role": "input"}) ────────
        all_receptacles = (
            result.get("objects", {}).get("kits", []) +
            result.get("objects", {}).get("containers", [])
        )

        if key in all_receptacles:
            role_val = value.get("role")
            _upsert_list(preds["role"], "object", key, {"role": role_val})
            print(f"  role: {key} → {role_val}")
            continue

        # ── slot-level role (backward-compat: "Container_3_Pos_1": {"Role":…}) ─
        parent = _parent_of(key)
        if parent is not None and parent in all_receptacles:
            role_val = value.get("role") or value.get("Role")
            if role_val is not None or "role" in value or "Role" in value:
                _upsert_list(preds["role"], "object", parent, {"role": role_val})
                print(f"  role (via slot): {parent} → {role_val}")
                continue

        # ── part attributes ───────────────────────────────────────────────────
        all_parts = result.get("objects", {}).get("parts", [])
        if key in all_parts:
            for attr, val in value.items():
                attr_lower = attr.lower()
                if attr_lower == "size":
                    _upsert_list(preds["size"], "part", key, {"size": (val or "standard").lower()})
                    print(f"  size: {key} → {val}")
                elif attr_lower == "color":
                    _upsert_list(preds["color"], "part", key, {"color": (val or "").lower()})
                    print(f"  color: {key} → {val}")
                elif attr_lower == "fragility":
                    frag_val = (val or "normal").lower()
                    if frag_val not in ("normal", "fragile"):
                        print(f"  WARNING: fragility for '{key}' must be 'normal' or 'fragile' — skipped.")
                    else:
                        if frag_val == "fragile":
                            _upsert_list(preds["fragility"], "part", key, {"fragility": "fragile"})
                        else:
                            # Remove entry (normal is the default — no entry needed)
                            preds["fragility"] = [e for e in preds["fragility"] if e["part"] != key]
                        print(f"  fragility: {key} → {frag_val}")
                else:
                    print(f"  WARNING: unknown part attribute '{attr}' for '{key}' — skipped.")
            continue

        not_found.append(key)

    if not_found:
        print("  WARNING: the following keys were not found in the state and were skipped:")
        for n in not_found:
            print(f"    - {n}")

    return result


# ── standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    if not CONFIGURATION_PATH.exists():
        print(f"ERROR: configuration.json not found: {CONFIGURATION_PATH}")
        sys.exit(1)
    if not CHANGES_PATH.exists():
        print(f"ERROR: Changes file not found: {CHANGES_PATH}")
        sys.exit(1)

    state   = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    changes = json.loads(CHANGES_PATH.read_text(encoding="utf-8"))

    print(f"Loaded positions: {CONFIGURATION_PATH}")
    print(f"Loaded changes:   {CHANGES_PATH}")

    updated = apply_changes(state, changes)

    tmp = str(CONFIGURATION_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(CONFIGURATION_PATH)
    print(f"Saved: {CONFIGURATION_PATH}")


if __name__ == "__main__":
    main()