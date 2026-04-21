"""
scene_helpers.py — Build the LLM-facing view of the workspace state.

Converts the full PDDL-friendly configuration.json into a compact
representation that the LLM can reason about without drowning in detail.
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

    Returns a compact dict with keys:
      workspace, receptacle_xy, capacity, slots, parts,
      priority, kit_recipe, part_compatibility.
    """
    preds = state.get("predicates", {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs = state.get("objects", {})
    metric = state.get("metric", {})

    role_map: Dict[str, Optional[str]] = {
        e["object"]: e.get("role") for e in preds.get("role", [])
    }

    color_map = {e["part"]: e.get("color") for e in preds.get("color", [])}
    frag_map: Dict[str, str] = {
        e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])
    }

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

    for part_name, slot_name in part_in_slot.items():
        if slot_name not in slots_view:
            continue
        slots_view[slot_name]["child_part"] = {
            "name": part_name,
            "color": color_map.get(part_name),
            "fragility": frag_map.get(part_name, "normal"),
        }

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

    # Capacity summary
    capacity: Dict[str, Any] = {}
    for rec_name in sorted(set(objs.get("kits", []) + objs.get("containers", []))):
        rec_slots = [s for s, p in slot_belongs.items() if p == rec_name]
        empty_count = 0
        color_counts: Dict[str, int] = {}
        for s in rec_slots:
            sv = slots_view.get(s, {})
            cp = sv.get("child_part")
            if cp is not None:
                c = (cp.get("color") or "unknown").lower()
                color_counts[c] = color_counts.get(c, 0) + 1
            else:
                empty_count += 1
        capacity[rec_name] = {
            "total_slots": len(rec_slots),
            "occupied": len(rec_slots) - empty_count,
            "empty": empty_count,
            "parts_by_color": color_counts if color_counts else {},
            "role": role_map.get(rec_name),
        }

    return {
        "workspace": state.get("workspace", {"operation_mode": None, "batch_size": None}),
        "receptacle_xy": receptacle_xy,
        "capacity": capacity,
        "slots": slots_view,
        "parts": parts_view,
        "priority": preds.get("priority", []),
        "kit_recipe": preds.get("kit_recipe", []),
        "part_compatibility": preds.get("part_compatibility", []),
    }
