"""
scene_helpers.py — Build the LLM-facing view of the workspace state.

Converts the full PDDL-friendly configuration.json into a compact representation
that the LLM can reason about without drowning in structural detail.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _extract_xy(metric: Dict[str, Any], name: str) -> Optional[List[float]]:
    """Return [x, y] from metric[name]["pos"], or None if unavailable."""
    entry = metric.get(name)
    if not entry:
        return None
    pos = entry.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return [round(pos[0], 4), round(pos[1], 4)]
    return None


def slim_scene(state: dict) -> dict:
    """
    Produce the LLM-facing view from the PDDL-friendly configuration.json.

    Returns:
    {
        "workspace": {"operation_mode": ..., "batch_size": ...},
        "receptacle_xy": {"Container_3": [x, y], "Kit_0": [x, y]},
        "slots": {
            "Kit_0_Pos_1": {
                "xy": [x, y],
                "role": "output",
                "child_part": null
            },
            "Container_3_Pos_1": {
                "xy": [x, y],
                "role": "input",
                "child_part": {"name": "Part_1", "color": "blue"}
            }
        },
        "parts": {
            "Part_5": {"xy": [x, y], "color": "red", "fragility": "normal"}
        }
    }
    """
    preds = state.get("predicates", {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs = state.get("objects", {})
    metric = state.get("metric", {})

    # Role per receptacle
    role_map: Dict[str, Optional[str]] = {
        e["object"]: e.get("role") for e in preds.get("role", [])
    }

    # Part attributes
    color_map = {e["part"]: e.get("color") for e in preds.get("color", [])}
    frag_map: Dict[str, str] = {
        e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])
    }

    # Part → slot mapping
    part_in_slot: Dict[str, str] = {
        e["part"]: e["slot"] for e in preds.get("at", [])
    }

    # Build slot view
    slots_view: Dict[str, Any] = {}
    for slot_name in objs.get("slots", []):
        parent = slot_belongs.get(slot_name)
        role = role_map.get(parent) if parent else None
        entry: Dict[str, Any] = {"role": role, "child_part": None}
        xy = _extract_xy(metric, slot_name)
        if xy is not None:
            entry["xy"] = xy
        slots_view[slot_name] = entry

    # Embed parts into their slots
    for part_name, slot_name in part_in_slot.items():
        if slot_name not in slots_view:
            continue
        slots_view[slot_name]["child_part"] = {
            "name": part_name,
            "color": color_map.get(part_name),
            "fragility": frag_map.get(part_name, "normal"),
        }

    # Standalone parts (not in any slot)
    in_slot_set = set(part_in_slot.keys())
    parts_view: Dict[str, Any] = {}
    for p in objs.get("parts", []):
        if p in in_slot_set:
            continue
        entry = {
            "color": color_map.get(p),
            "fragility": frag_map.get(p, "normal"),
        }
        xy = _extract_xy(metric, p)
        if xy is not None:
            entry["xy"] = xy
        parts_view[p] = entry

    # Receptacle centroids (average of child slot positions)
    receptacle_xy: Dict[str, List[float]] = {}
    recept_accum: Dict[str, List[List[float]]] = {}
    for slot_name in objs.get("slots", []):
        parent = slot_belongs.get(slot_name)
        if parent is None:
            continue
        xy = _extract_xy(metric, slot_name)
        if xy is not None:
            recept_accum.setdefault(parent, []).append(xy)
    for name, xys in sorted(recept_accum.items()):
        cx = round(sum(p[0] for p in xys) / len(xys), 4)
        cy = round(sum(p[1] for p in xys) / len(xys), 4)
        receptacle_xy[name] = [cx, cy]

    return {
        "workspace": state.get("workspace", {"operation_mode": None, "batch_size": None}),
        "receptacle_xy": receptacle_xy,
        "slots": slots_view,
        "parts": parts_view,
    }
