# workspace_state.py
# Produces a PDDL-friendly workspace state from the flat list of vision entries.
#
# Structure:
#   workspace       – operation-level high-level attributes (mode, batch_size)
#   objects         – typed object lists for PDDL object declarations
#   slot_belongs_to – parent receptacle for every slot
#   predicates      – all semantic facts (at, slot_empty, role, color,
#                     priority, kit_recipe, part_compatibility)
#   metric          – robot-execution data (pos/quat/orientation) keyed by name
#
# The metric section contains full coordinates (pos/quat/orientation) for
# robot execution.  The LLM view replaces it with a lightweight positions_xy
# dict (name → [x, y]) so the LLM can resolve spatial references.

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

from Core.paths import parent_of_slot as _parent_of


def _is_slot(name: str) -> bool:
    return name.startswith("Kit_") or name.startswith("Container_")


def _is_part(name: str) -> bool:
    return name.startswith("Part_")


# ── core conversion ───────────────────────────────────────────────────────────

def entries_to_state(final_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert the flat list produced by assign_parts_to_slots into the
    PDDL-friendly workspace state dict.

    Low-level attributes (from vision):   pos, quat, orientation, Color
    High-level attributes (user-supplied): role, priority, kit_recipe,
                                           part_compatibility, operation_mode

    High-level attributes default to null/[] here; they are applied later via
    Apply_Config_Changes.
    """

    # ── categorise entries ────────────────────────────────────────────────────
    slot_entries: Dict[str, Dict[str, Any]] = {}
    part_entries: Dict[str, Dict[str, Any]] = {}   # standalone parts

    for obj in final_entries:
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue
        if _is_slot(name):
            slot_entries[name] = obj
        elif _is_part(name):
            part_entries[name] = obj

    # ── receptacle sets ───────────────────────────────────────────────────────
    kits: set = set()
    containers: set = set()

    for slot_name in slot_entries:
        parent = _parent_of(slot_name)
        if parent is None:
            continue
        if slot_name.startswith("Kit_"):
            kits.add(parent)
        else:
            containers.add(parent)

    # ── collect all part names (embedded + standalone) ────────────────────────
    embedded_parts: Dict[str, Dict[str, Any]] = {}   # name → child_part dict
    embedded_slot:  Dict[str, str] = {}              # part_name → slot_name

    for slot_name, slot in slot_entries.items():
        child = slot.get("child_part")
        if isinstance(child, dict) and isinstance(child.get("name"), str):
            pname = child["name"]
            embedded_parts[pname] = child
            embedded_slot[pname]  = slot_name

    all_parts: List[str] = sorted(
        list(embedded_parts.keys()) +
        [n for n in part_entries if n not in embedded_parts]
    )

    # ── slot_belongs_to ───────────────────────────────────────────────────────
    slot_belongs_to: Dict[str, str] = {}
    for slot_name in slot_entries:
        parent = _parent_of(slot_name)
        if parent:
            slot_belongs_to[slot_name] = parent

    # ── predicates ───────────────────────────────────────────────────────────
    at_list:         List[Dict] = []
    slot_empty_list: List[str]  = []
    color_list:      List[Dict] = []

    for slot_name, slot in sorted(slot_entries.items()):
        child = slot.get("child_part")
        if isinstance(child, dict) and isinstance(child.get("name"), str):
            pname = child["name"]
            at_list.append({"part": pname, "slot": slot_name})

            color = child.get("Color")
            if color:
                color_list.append({"part": pname, "color": color.lower()})
        else:
            slot_empty_list.append(slot_name)

    # standalone parts
    for pname, part in sorted(part_entries.items()):
        if pname in embedded_parts:
            continue
        color = part.get("Color")
        if color:
            color_list.append({"part": pname, "color": color.lower()})

    # role — one entry per receptacle, default null
    role_list: List[Dict] = [
        {"object": r, "role": None}
        for r in sorted(kits | containers)
    ]

    # ── metric ────────────────────────────────────────────────────────────────
    metric: Dict[str, Any] = {}

    for slot_name, slot in sorted(slot_entries.items()):
        metric[slot_name] = {
            "pos":         slot.get("pos"),
            "quat":        slot.get("quat"),
            "orientation": slot.get("orientation"),
        }

    # embedded parts get their own metric entry (actual detected position)
    for pname, child in embedded_parts.items():
        metric[pname] = {
            "pos":         child.get("pos"),
            "quat":        child.get("quat"),
            "orientation": child.get("orientation"),
        }

    # standalone parts
    for pname, part in part_entries.items():
        if pname not in embedded_parts:
            metric[pname] = {
                "pos":         part.get("pos"),
                "quat":        part.get("quat"),
                "orientation": part.get("orientation"),
            }

    return {
        "workspace": {
            "operation_mode": None,
            "batch_size":     None,
        },
        "objects": {
            "kits":       sorted(kits),
            "containers": sorted(containers),
            "parts":      all_parts,
            "slots":      sorted(slot_entries.keys()),
        },
        "slot_belongs_to": slot_belongs_to,
        "predicates": {
            "at":                at_list,
            "slot_empty":        slot_empty_list,
            "role":              role_list,
            "color":             color_list,
            "priority":          [],
            "kit_recipe":        [],
            "part_compatibility":[],
            "fragility":         [],
        },
        "metric": metric,
    }


# ── LLM slim view ────────────────────────────────────────────────────────────

def _extract_positions_xy(metric: Dict[str, Any]) -> Dict[str, List[float]]:
    """Build a lightweight {name: [x, y]} dict from the full metric section."""
    positions: Dict[str, List[float]] = {}
    for name, entry in sorted(metric.items()):
        pos = entry.get("pos") if isinstance(entry, dict) else None
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            positions[name] = [round(pos[0], 4), round(pos[1], 4)]
    return positions


def save_json_snapshot(path: str, state: Dict[str, Any], pretty: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(state, f, indent=2, ensure_ascii=False)
        else:
            json.dump(state, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


