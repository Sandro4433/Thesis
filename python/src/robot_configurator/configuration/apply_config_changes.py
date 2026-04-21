"""
apply_config_changes.py — Apply a workspace_changes dict to configuration.json.

Changes format (output by the LLM):

  {
    "<receptacle_name>": {"role": "input"},
    "<part_name>":       {"color": "red"},
    "<part_name>":       {"fragility": "fragile"},
    "workspace":         {"operation_mode": "kitting", "batch_size": 2},
    "priority":          [{"color": "blue", "order": 1}, ...],
    "kit_recipe":        [{"kit": "Kit_0", "color": "blue", "quantity": 2}, ...],
    "part_compatibility":[{"part_color": "blue", "allowed_in": ["Kit_0"]}]
  }

Backward-compatibility: slot-level role keys (e.g. ``Container_3_Pos_1``)
are also accepted and translated to their parent receptacle.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List

from robot_configurator.core.paths import (
    CHANGES_PATH,
    CONFIGURATION_PATH,
    parent_of_slot,
    save_atomic,
)

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _upsert_list(
    lst: List[Dict],
    key_field: str,
    key_val: Any,
    update: Dict[str, Any],
) -> None:
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
      ``workspace``          → updates ``state["workspace"]``
      ``priority``           → replaces ``state["predicates"]["priority"]``
      ``kit_recipe``         → replaces ``state["predicates"]["kit_recipe"]``
      ``part_compatibility`` → replaces ``state["predicates"]["part_compatibility"]``

    Object keys:
      receptacle name (Kit_* / Container_*) → role update in predicates.role
      slot name (*_Pos_*)                   → role update (translated to parent)
      part name (Part_*)                    → color / fragility update
    """
    result = copy.deepcopy(state)
    preds = result.setdefault("predicates", {})
    preds.setdefault("role", [])
    preds.setdefault("color", [])
    preds.setdefault("priority", [])
    preds.setdefault("kit_recipe", [])
    preds.setdefault("part_compatibility", [])
    preds.setdefault("fragility", [])
    result.setdefault("workspace", {"operation_mode": None, "batch_size": None})

    not_found: List[str] = []

    for key, value in changes.items():

        # ── workspace-level attributes ────────────────────────────────────────
        if key == "workspace" and isinstance(value, dict):
            result["workspace"].update(value)
            logger.debug("workspace: %s", value)
            continue

        # ── list-type predicate replacements ─────────────────────────────────
        if key == "priority":
            preds["priority"] = value if isinstance(value, list) else []
            logger.debug("priority: %s", preds["priority"])
            continue

        if key == "kit_recipe":
            preds["kit_recipe"] = value if isinstance(value, list) else []
            logger.debug("kit_recipe: %s", preds["kit_recipe"])
            continue

        if key == "part_compatibility":
            preds["part_compatibility"] = value if isinstance(value, list) else []
            logger.debug("part_compatibility: %s", preds["part_compatibility"])
            continue

        if not isinstance(value, dict):
            logger.warning("Unrecognised change entry '%s': %r — skipped.", key, value)
            continue

        all_receptacles = (
            result.get("objects", {}).get("kits", [])
            + result.get("objects", {}).get("containers", [])
        )

        # ── receptacle role (direct: "Container_3": {"role": "input"}) ───────
        if key in all_receptacles:
            role_val = value.get("role")
            _upsert_list(preds["role"], "object", key, {"role": role_val})
            logger.debug("role: %s → %s", key, role_val)
            continue

        # ── slot-level role (backward-compat) ─────────────────────────────────
        parent = parent_of_slot(key, result.get("slot_belongs_to", {}))
        if parent is not None and parent in all_receptacles:
            role_val = value.get("role") or value.get("Role")
            if role_val is not None or "role" in value or "Role" in value:
                _upsert_list(preds["role"], "object", parent, {"role": role_val})
                logger.debug("role (via slot): %s → %s", parent, role_val)
                continue

        # ── part attributes ───────────────────────────────────────────────────
        all_parts = result.get("objects", {}).get("parts", [])
        if key in all_parts:
            for attr, val in value.items():
                attr_lower = attr.lower()
                if attr_lower == "color":
                    _upsert_list(
                        preds["color"], "part", key,
                        {"color": (val or "").lower()},
                    )
                    logger.debug("color: %s → %s", key, val)
                elif attr_lower == "fragility":
                    frag_val = (val or "normal").lower()
                    if frag_val not in ("normal", "fragile"):
                        logger.warning(
                            "fragility for '%s' must be 'normal' or 'fragile' — skipped.", key
                        )
                    elif frag_val == "fragile":
                        _upsert_list(
                            preds["fragility"], "part", key,
                            {"fragility": "fragile"},
                        )
                        logger.debug("fragility: %s → fragile", key)
                    else:
                        preds["fragility"] = [
                            e for e in preds["fragility"] if e["part"] != key
                        ]
                        logger.debug("fragility: %s → normal (entry removed)", key)
                else:
                    logger.warning(
                        "Unknown part attribute '%s' for '%s' — skipped.", attr, key
                    )
            continue

        not_found.append(key)

    if not_found:
        logger.warning(
            "The following keys were not found in the state and were skipped: %s",
            not_found,
        )

    return result


# ── standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    import sys

    if not CONFIGURATION_PATH.exists():
        print(f"ERROR: configuration.json not found: {CONFIGURATION_PATH}")
        sys.exit(1)
    if not CHANGES_PATH.exists():
        print(f"ERROR: Changes file not found: {CHANGES_PATH}")
        sys.exit(1)

    state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    changes = json.loads(CHANGES_PATH.read_text(encoding="utf-8"))

    print(f"Loaded positions: {CONFIGURATION_PATH}")
    print(f"Loaded changes:   {CHANGES_PATH}")

    updated = apply_changes(state, changes)
    save_atomic(CONFIGURATION_PATH, updated)
    print(f"Saved: {CONFIGURATION_PATH}")


if __name__ == "__main__":
    main()
