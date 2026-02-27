# assign_parts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class Match:
    part_i: int
    slot_i: int
    dist_xy: float


def _xy(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = obj.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        return None
    try:
        return float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None


def _dist_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_slot(name: str) -> bool:
    return name.startswith("Kit_") or name.startswith("Container_")


def _is_part(name: str) -> bool:
    return name.startswith("Part_")


def assign_parts_to_slots(
    objects: List[Dict[str, Any]],
    xy_threshold_m: float,
) -> List[Dict[str, Any]]:
    """
    If Part is within xy_threshold_m of a slot in XY:
      - remove Part from top-level
      - set slot["child_part"] = part (embedded)
      - set child_part["orientation"] = slot["orientation"]
    Each slot gets at most 1 part (closest wins).
    """
    out = [dict(o) for o in objects]  # shallow copy

    slot_idxs: List[int] = []
    part_idxs: List[int] = []

    for i, o in enumerate(out):
        n = o.get("name")
        if not isinstance(n, str) or not n:
            continue
        if _is_slot(n):
            o.setdefault("child_part", None)
            slot_idxs.append(i)
        elif _is_part(n):
            part_idxs.append(i)

    slot_xy = {i: _xy(out[i]) for i in slot_idxs}
    slot_xy = {i: xy for i, xy in slot_xy.items() if xy is not None}

    part_xy = {i: _xy(out[i]) for i in part_idxs}
    part_xy = {i: xy for i, xy in part_xy.items() if xy is not None}

    candidates: List[Match] = []
    for pi, pxy in part_xy.items():
        for si, sxy in slot_xy.items():
            d = _dist_xy(pxy, sxy)
            if d <= xy_threshold_m:
                candidates.append(Match(pi, si, d))

    candidates.sort(key=lambda m: m.dist_xy)

    used_slots = set()
    used_parts = set()
    chosen: List[Match] = []
    for m in candidates:
        if m.slot_i in used_slots or m.part_i in used_parts:
            continue
        used_slots.add(m.slot_i)
        used_parts.add(m.part_i)
        chosen.append(m)

    parts_to_remove = set()
    for m in chosen:
        slot = out[m.slot_i]
        part = out[m.part_i]

        child = dict(part)
        child["orientation"] = slot.get("orientation", None)

        slot["child_part"] = child
        parts_to_remove.add(m.part_i)

    final: List[Dict[str, Any]] = []
    for i, obj in enumerate(out):
        if i in parts_to_remove:
            continue
        final.append(obj)

    return final