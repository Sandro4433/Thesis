# Update_Scene.py
# Scene Update Module — merges a fresh camera scan with the existing memory state.
#
# Workflow
# ────────
#   1. Load memory state (positions.json) — the ground truth with stable IDs.
#   2. Trigger the vision pipeline → fresh positions.json (overwrites memory
#      on disk, but we already hold memory in RAM).
#   3. Match slots by exact name (AprilTag-derived, so naturally stable).
#      Match parts by XY proximity (detection order varies, so proximity is
#      the only reliable link between scans).
#   4. Re-number unmatched fresh parts so they continue from
#      max(existing Part_N) + 1, preserving all existing IDs.
#   5. Compute a structured diff: new / removed / moved / unchanged.
#   6. LLM dialogue: user picks which changes to apply.
#   7. Merge only the confirmed changes into memory state, save positions.json,
#      and archive a timestamped copy to Memory/.

from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import POSITIONS_JSON  # type: ignore

POSITIONS_PATH = Path(POSITIONS_JSON.resolve())

# ── Tuning constants ──────────────────────────────────────────────────────────
# Parts within this XY distance are considered the SAME physical part.
XY_MATCH_THRESHOLD_M = 0.05    # 5 cm

# If a matched part/slot moved MORE than this it is flagged as "moved".
MOVED_THRESHOLD_M    = 0.008   # 8 mm

# LLM model (keep in sync with API_Main.MODEL)
_LLM_MODEL = "gpt-4.1"


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xy_dist(p1: Optional[List], p2: Optional[List]) -> float:
    if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
        return float("inf")
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def _is_slot(name: str) -> bool:
    return name.startswith("Kit_") or name.startswith("Container_")


def _is_part(name: str) -> bool:
    return name.startswith("Part_")


def _is_receptacle(name: str) -> bool:
    """True for e.g. "Kit_0" or "Container_3" (no _Pos_ suffix)."""
    return (_is_slot(name) or name.startswith("Kit_") or
            name.startswith("Container_")) and "_Pos_" not in name


def _part_number(name: str) -> int:
    """'Part_3' → 3, -1 on failure."""
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _slot_parent(slot_name: str) -> Optional[str]:
    idx = slot_name.rfind("_Pos_")
    return slot_name[:idx] if idx != -1 else None


def _metric_pos(state: Dict, name: str) -> Optional[List]:
    return state.get("metric", {}).get(name, {}).get("pos")


def _part_color(state: Dict, name: str) -> Optional[str]:
    for e in state.get("predicates", {}).get("color", []):
        if e["part"] == name:
            return e.get("color")
    return None


def _part_size(state: Dict, name: str) -> Optional[str]:
    for e in state.get("predicates", {}).get("size", []):
        if e["part"] == name:
            return e.get("size")
    return None


def _part_slot(state: Dict, name: str) -> Optional[str]:
    for e in state.get("predicates", {}).get("at", []):
        if e["part"] == name:
            return e.get("slot")
    return None


def _empty_state() -> Dict[str, Any]:
    return {
        "workspace": {"operation_mode": None, "batch_size": None},
        "objects":   {"kits": [], "containers": [], "parts": [], "slots": []},
        "slot_belongs_to": {},
        "predicates": {
            "at": [], "slot_empty": [], "role": [],
            "color": [], "size": [],
            "priority": [], "kit_recipe": [], "part_compatibility": [],
        },
        "metric": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Run vision, return (memory_state, fresh_state)
# ─────────────────────────────────────────────────────────────────────────────

def run_vision_and_load() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load memory BEFORE triggering Vision_Main (which overwrites positions.json).
    Returns (memory_state, fresh_state).
    """
    memory_state: Dict[str, Any] = (
        json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
        if POSITIONS_PATH.exists()
        else _empty_state()
    )

    print("\nStarting vision module …")
    from Vision_Module.Vision_Main import main as vision_main  # type: ignore
    vision_main()

    if not POSITIONS_PATH.exists():
        raise RuntimeError("Vision module did not produce positions.json.")

    fresh_state = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    print("Vision complete.\n")
    return memory_state, fresh_state


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Compute diff
# ─────────────────────────────────────────────────────────────────────────────

def compute_scene_diff(
    memory_state: Dict[str, Any],
    fresh_state:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a diff dict with keys:
      matched_parts   {fresh_name → memory_name}   position-matched pairs
      new_slots       [names]   in fresh, absent from memory
      removed_slots   [names]   in memory, absent from fresh
      moved_slots     {name: {memory_pos, fresh_pos, dist_m}}
      unchanged_slots [names]
      new_parts       [fresh names]   unmatched fresh parts
      removed_parts   [memory names] unmatched memory parts
      moved_parts     {memory_name: {fresh_name, memory_pos, fresh_pos, dist_m}}
      unchanged_parts [memory names]
    """
    m_slots = set(memory_state.get("objects", {}).get("slots", []))
    f_slots = set(fresh_state.get("objects",  {}).get("slots", []))
    m_parts = set(memory_state.get("objects", {}).get("parts", []))
    f_parts = set(fresh_state.get("objects",  {}).get("parts", []))

    # ── Slots: exact name match ───────────────────────────────────────────────
    new_slots:       List[str]      = sorted(f_slots - m_slots)
    removed_slots:   List[str]      = sorted(m_slots - f_slots)
    moved_slots:     Dict[str, Any] = {}
    unchanged_slots: List[str]      = []

    for sname in sorted(m_slots & f_slots):
        d = _xy_dist(_metric_pos(memory_state, sname),
                     _metric_pos(fresh_state,  sname))
        if d > MOVED_THRESHOLD_M:
            moved_slots[sname] = {
                "memory_pos": _metric_pos(memory_state, sname),
                "fresh_pos":  _metric_pos(fresh_state,  sname),
                "dist_m":     round(d, 4),
            }
        else:
            unchanged_slots.append(sname)

    # ── Parts: XY proximity match ─────────────────────────────────────────────
    m_pos = {p: _metric_pos(memory_state, p) for p in m_parts}
    f_pos = {p: _metric_pos(fresh_state,  p) for p in f_parts}
    m_pos = {p: xy for p, xy in m_pos.items() if xy}
    f_pos = {p: xy for p, xy in f_pos.items() if xy}

    # Build all candidates within threshold, sorted by distance (greedy match)
    candidates: List[Tuple[float, str, str]] = sorted(
        (d, fp, mp)
        for fp, fxy in f_pos.items()
        for mp, mxy in m_pos.items()
        if (d := _xy_dist(fxy, mxy)) <= XY_MATCH_THRESHOLD_M
    )

    matched_parts: Dict[str, str] = {}   # fresh → memory
    used_f: Set[str] = set()
    used_m: Set[str] = set()
    for d, fp, mp in candidates:
        if fp in used_f or mp in used_m:
            continue
        matched_parts[fp] = mp
        used_f.add(fp)
        used_m.add(mp)

    new_parts:     List[str] = sorted(f_parts - used_f,  key=_part_number)
    removed_parts: List[str] = sorted(m_parts - used_m,  key=_part_number)
    moved_parts:   Dict[str, Any] = {}
    unchanged_parts: List[str] = []

    for fp, mp in matched_parts.items():
        d = _xy_dist(f_pos.get(fp), m_pos.get(mp))
        if d > MOVED_THRESHOLD_M:
            moved_parts[mp] = {
                "fresh_name": fp,
                "memory_pos": _metric_pos(memory_state, mp),
                "fresh_pos":  _metric_pos(fresh_state,  fp),
                "dist_m":     round(d, 4),
            }
        else:
            unchanged_parts.append(mp)

    return {
        "matched_parts":   matched_parts,
        "new_slots":       new_slots,
        "removed_slots":   removed_slots,
        "moved_slots":     moved_slots,
        "unchanged_slots": unchanged_slots,
        "new_parts":       new_parts,
        "removed_parts":   removed_parts,
        "moved_parts":     moved_parts,
        "unchanged_parts": unchanged_parts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Re-number fresh parts to use stable IDs
# ─────────────────────────────────────────────────────────────────────────────

def renumber_fresh_parts(
    fresh_state:  Dict[str, Any],
    diff:         Dict[str, Any],
    memory_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Build rename_map {fresh_name → stable_name}:
      - Matched parts  → keep their memory name (e.g. fresh Part_2 → memory Part_5)
      - New parts      → assign next available Part_N after max(existing)

    Returns (renamed_fresh_state, rename_map).
    """
    rename_map: Dict[str, str] = dict(diff["matched_parts"])   # fresh → memory

    existing_nrs: Set[int] = {
        _part_number(n)
        for n in memory_state.get("objects", {}).get("parts", [])
        if _part_number(n) > 0
    }
    next_nr = max(existing_nrs, default=0) + 1

    for fp in sorted(diff["new_parts"], key=_part_number):
        while next_nr in existing_nrs:
            next_nr += 1
        rename_map[fp] = f"Part_{next_nr}"
        existing_nrs.add(next_nr)
        next_nr += 1

    # Also update moved_parts entries: replace fresh_name with stable name
    for mp_info in diff["moved_parts"].values():
        fp = mp_info["fresh_name"]
        mp_info["fresh_name"] = rename_map.get(fp, fp)

    renamed = _apply_part_rename(fresh_state, rename_map)
    return renamed, rename_map


def _apply_part_rename(
    state: Dict[str, Any], rename: Dict[str, str]
) -> Dict[str, Any]:
    if not rename:
        return copy.deepcopy(state)

    s = copy.deepcopy(state)

    def rn(n: str) -> str:
        return rename.get(n, n)

    s["objects"]["parts"] = [rn(p) for p in s["objects"].get("parts", [])]

    preds = s.get("predicates", {})
    preds["at"]    = [{"part": rn(e["part"]), "slot": e["slot"]} for e in preds.get("at",    [])]
    preds["color"] = [{"part": rn(e["part"]), "color": e["color"]} for e in preds.get("color", [])]
    preds["size"]  = [{"part": rn(e["part"]), "size":  e["size"]}  for e in preds.get("size",  [])]

    new_metric: Dict[str, Any] = {}
    for k, v in s.get("metric", {}).items():
        new_metric[rn(k) if _is_part(k) else k] = v
    s["metric"] = new_metric

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Format diff for display and LLM
# ─────────────────────────────────────────────────────────────────────────────

def build_diff_summary(
    diff:          Dict[str, Any],
    memory_state:  Dict[str, Any],
    fresh_renamed: Dict[str, Any],
    rename_map:    Dict[str, str],   # fresh_orig → stable
) -> str:
    """Human-readable terminal summary of detected changes."""
    lines: List[str] = []

    # ── New receptacles ───────────────────────────────────────────────────────
    parents: Dict[str, List[str]] = {}
    for s in diff["new_slots"]:
        parents.setdefault(_slot_parent(s) or s, []).append(s)
    for parent, slots in sorted(parents.items()):
        kind = "kit" if parent.startswith("Kit_") else "container"
        lines.append(f"  + {parent}  [{kind}, {len(slots)} slot(s) — NEW]")

    # ── Removed receptacles ───────────────────────────────────────────────────
    parents = {}
    for s in diff["removed_slots"]:
        parents.setdefault(_slot_parent(s) or s, []).append(s)
    for parent, slots in sorted(parents.items()):
        lines.append(f"  - {parent}  [{len(slots)} slot(s) no longer detected — REMOVED]")

    # ── Moved receptacles ─────────────────────────────────────────────────────
    parents = {}
    for s in diff["moved_slots"]:
        parents.setdefault(_slot_parent(s) or s, []).append(s)
    for parent, slots in sorted(parents.items()):
        max_d = max(diff["moved_slots"][s]["dist_m"] for s in slots)
        lines.append(f"  ~ {parent}  [repositioned ≤{max_d*100:.1f} cm — MOVED]")

    # ── New parts ─────────────────────────────────────────────────────────────
    for fp_orig in diff["new_parts"]:
        stable = rename_map.get(fp_orig, fp_orig)
        color  = _part_color(fresh_renamed, stable) or "?"
        size   = _part_size(fresh_renamed,  stable) or "standard"
        slot   = _part_slot(fresh_renamed,  stable)
        loc    = f"in {slot}" if slot else "standalone"
        lines.append(f"  + {stable}  [{color}, {size}, {loc} — NEW]")

    # ── Removed parts ─────────────────────────────────────────────────────────
    for mp in diff["removed_parts"]:
        color = _part_color(memory_state, mp) or "?"
        size  = _part_size(memory_state,  mp) or "standard"
        slot  = _part_slot(memory_state,  mp)
        loc   = f"in {slot}" if slot else "standalone"
        lines.append(f"  - {mp}  [{color}, {size}, {loc} — REMOVED]")

    # ── Moved parts ───────────────────────────────────────────────────────────
    for mp, info in sorted(diff["moved_parts"].items()):
        stable  = info["fresh_name"]   # already renamed
        color   = _part_color(memory_state, mp) or "?"
        m_slot  = _part_slot(memory_state,  mp)
        f_slot  = _part_slot(fresh_renamed, stable)
        d_cm    = info["dist_m"] * 100
        if m_slot and f_slot and m_slot != f_slot:
            move_desc = f"{m_slot} → {f_slot}"
        elif m_slot and not f_slot:
            move_desc = f"{m_slot} → standalone"
        elif not m_slot and f_slot:
            move_desc = f"standalone → {f_slot}"
        else:
            move_desc = f"moved {d_cm:.1f} cm"
        lines.append(f"  ~ {mp}  [{color}, {move_desc}, {d_cm:.1f} cm — MOVED]")

    # ── Unchanged summary ─────────────────────────────────────────────────────
    u_slots = len(diff["unchanged_slots"])
    u_parts = len(diff["unchanged_parts"])
    if u_slots or u_parts:
        lines.append(f"  ✓ {u_slots} slot(s) and {u_parts} part(s) unchanged")

    return "\n".join(lines) if lines else "  (no changes detected)"


def format_diff_for_llm(
    diff:          Dict[str, Any],
    memory_state:  Dict[str, Any],
    fresh_renamed: Dict[str, Any],
    rename_map:    Dict[str, str],
) -> Dict[str, Any]:
    """Compact structured diff dict for the LLM prompt."""

    def _pinfo(state: Dict, name: str) -> Dict[str, Any]:
        return {
            "name":  name,
            "color": _part_color(state, name),
            "size":  _part_size(state,  name),
            "slot":  _part_slot(state,  name),
        }

    # Group slot changes by receptacle
    def _group_slots(slots: List[str]) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {}
        for s in slots:
            g.setdefault(_slot_parent(s) or s, []).append(s)
        return dict(sorted(g.items()))

    new_rec = _group_slots(diff["new_slots"])
    rem_rec = _group_slots(diff["removed_slots"])

    moved_rec: Dict[str, Any] = {}
    for s, info in diff["moved_slots"].items():
        p = _slot_parent(s) or s
        rec = moved_rec.setdefault(p, {"max_dist_cm": 0.0, "slots": []})
        rec["slots"].append(s)
        rec["max_dist_cm"] = max(rec["max_dist_cm"], round(info["dist_m"] * 100, 1))

    new_parts_llm = [
        _pinfo(fresh_renamed, rename_map.get(fp, fp))
        for fp in diff["new_parts"]
    ]
    rem_parts_llm = [_pinfo(memory_state, mp) for mp in diff["removed_parts"]]

    moved_parts_llm: Dict[str, Any] = {
        mp: {
            "color":     _part_color(memory_state, mp),
            "dist_cm":   round(info["dist_m"] * 100, 1),
            "from_slot": _part_slot(memory_state, mp),
            "to_slot":   _part_slot(fresh_renamed, info["fresh_name"]),
        }
        for mp, info in diff["moved_parts"].items()
    }

    return {
        "new_receptacles":     new_rec,
        "removed_receptacles": rem_rec,
        "moved_receptacles":   moved_rec,
        "new_parts":           new_parts_llm,
        "removed_parts":       rem_parts_llm,
        "moved_parts":         moved_parts_llm,
        "unchanged_summary":   {
            "slots": len(diff["unchanged_slots"]),
            "parts": len(diff["unchanged_parts"]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Merge confirmed changes into memory state
# ─────────────────────────────────────────────────────────────────────────────

def _expand_to_slots(
    names:         List[str],
    memory_state:  Dict[str, Any],
    fresh_renamed: Dict[str, Any],
) -> Set[str]:
    """
    Expand receptacle names (e.g. "Kit_0") to all their slot names.
    Individual slot names are passed through unchanged.
    """
    result: Set[str] = set()
    all_slots = (
        set(memory_state.get("objects", {}).get("slots", []))
        | set(fresh_renamed.get("objects", {}).get("slots", []))
    )
    for name in names:
        if "_Pos_" in name:
            result.add(name)          # already a slot
        elif _is_slot(name) or name.startswith("Kit_") or name.startswith("Container_"):
            # Receptacle — expand to all its slots from either state
            for s in all_slots:
                if _slot_parent(s) == name:
                    result.add(s)
    return result


def merge_confirmed(
    memory_state:  Dict[str, Any],
    fresh_renamed: Dict[str, Any],
    diff:          Dict[str, Any],
    add_names:     List[str],    # new objects to add  (stable names or receptacles)
    remove_names:  List[str],    # objects to remove   (stable names or receptacles)
    move_names:    List[str],    # objects to re-locate (stable names or receptacles)
) -> Dict[str, Any]:
    """
    Apply confirmed changes.  Returns a new merged deep-copy of memory_state.
    Processing order: REMOVE → ADD (slots first, then parts) → MOVE.
    """
    result = copy.deepcopy(memory_state)

    f_preds     = fresh_renamed.get("predicates", {})
    f_metric    = fresh_renamed.get("metric", {})
    f_at_by_slot = {e["slot"]: e["part"] for e in f_preds.get("at", [])}
    f_at_by_part = {e["part"]: e["slot"] for e in f_preds.get("at", [])}
    f_empty      = set(f_preds.get("slot_empty", []))
    f_color_map  = {e["part"]: e["color"] for e in f_preds.get("color", [])}
    f_size_map   = {e["part"]: e["size"]  for e in f_preds.get("size",  [])}

    # ── REMOVE ────────────────────────────────────────────────────────────────
    rem_slots = _expand_to_slots(
        [n for n in remove_names if not _is_part(n)],
        memory_state, fresh_renamed,
    )
    for s in sorted(rem_slots):
        _remove_inplace(result, s)
    for p in [n for n in remove_names if _is_part(n)]:
        _remove_inplace(result, p)

    # ── ADD — slots first ────────────────────────────────────────────────────
    add_slots = _expand_to_slots(
        [n for n in add_names if not _is_part(n)],
        memory_state, fresh_renamed,
    )
    add_parts = {n for n in add_names if _is_part(n)}

    for slot_name in sorted(add_slots):
        if slot_name in result["objects"]["slots"]:
            continue
        result["objects"]["slots"].append(slot_name)

        parent = _slot_parent(slot_name)
        if parent:
            result["slot_belongs_to"][slot_name] = parent
            if slot_name.startswith("Kit_") and parent not in result["objects"]["kits"]:
                result["objects"]["kits"].append(parent)
                result["objects"]["kits"].sort()
            elif slot_name.startswith("Container_") and parent not in result["objects"]["containers"]:
                result["objects"]["containers"].append(parent)
                result["objects"]["containers"].sort()
            existing_role_objects = {e["object"] for e in result["predicates"]["role"]}
            if parent not in existing_role_objects:
                result["predicates"]["role"].append({"object": parent, "role": None})

        if slot_name in f_metric:
            result["metric"][slot_name] = copy.deepcopy(f_metric[slot_name])

        # Mark empty (at predicates are managed by the ADD-parts step)
        if slot_name not in [e["slot"] for e in f_preds.get("at", [])]:
            if slot_name not in result["predicates"]["slot_empty"]:
                result["predicates"]["slot_empty"].append(slot_name)

    result["objects"]["slots"].sort()

    # ── ADD — parts ───────────────────────────────────────────────────────────
    for part_name in sorted(add_parts, key=_part_number):
        if part_name in result["objects"]["parts"]:
            continue
        result["objects"]["parts"].append(part_name)

        if part_name in f_color_map:
            result["predicates"]["color"].append(
                {"part": part_name, "color": f_color_map[part_name]}
            )
        if part_name in f_size_map:
            result["predicates"]["size"].append(
                {"part": part_name, "size": f_size_map[part_name]}
            )
        if part_name in f_metric:
            result["metric"][part_name] = copy.deepcopy(f_metric[part_name])

        # Link to slot only if that slot is already in the result
        if part_name in f_at_by_part:
            dest_slot = f_at_by_part[part_name]
            if dest_slot in result["objects"]["slots"]:
                result["predicates"]["at"].append(
                    {"part": part_name, "slot": dest_slot}
                )
                if dest_slot in result["predicates"]["slot_empty"]:
                    result["predicates"]["slot_empty"].remove(dest_slot)

    result["objects"]["parts"].sort(key=_part_number)

    # ── MOVE — update metric + slot assignment ────────────────────────────────
    move_slots = _expand_to_slots(
        [n for n in move_names if not _is_part(n)],
        memory_state, fresh_renamed,
    )
    move_parts = {n for n in move_names if _is_part(n)}

    for slot_name in move_slots:
        if slot_name in f_metric and slot_name in result.get("metric", {}):
            for k in ("pos", "quat", "orientation"):
                if k in f_metric[slot_name]:
                    result["metric"][slot_name][k] = f_metric[slot_name][k]

    for part_name in move_parts:
        if part_name not in result.get("metric", {}):
            continue
        # Update position
        if part_name in f_metric:
            for k in ("pos", "quat", "orientation"):
                if k in f_metric[part_name]:
                    result["metric"][part_name][k] = f_metric[part_name][k]
        # Update slot assignment if it changed
        if part_name in f_at_by_part:
            new_slot  = f_at_by_part[part_name]
            old_slot  = _part_slot(result, part_name)   # current slot in result
            if new_slot != old_slot:
                # Remove old at entry
                result["predicates"]["at"] = [
                    e for e in result["predicates"]["at"] if e["part"] != part_name
                ]
                # Free old slot
                if old_slot and old_slot not in result["predicates"]["slot_empty"]:
                    result["predicates"]["slot_empty"].append(old_slot)
                # Add new at entry if the destination slot exists
                if new_slot in result["objects"]["slots"]:
                    result["predicates"]["at"].append(
                        {"part": part_name, "slot": new_slot}
                    )
                    if new_slot in result["predicates"]["slot_empty"]:
                        result["predicates"]["slot_empty"].remove(new_slot)

    return result


def _remove_inplace(state: Dict[str, Any], name: str) -> None:
    """Remove a single slot or part from all sections of state (in-place)."""
    preds = state.get("predicates", {})

    if _is_part(name):
        if name in state["objects"].get("parts", []):
            state["objects"]["parts"].remove(name)
        old_slot = None
        new_at   = []
        for e in preds.get("at", []):
            if e["part"] == name:
                old_slot = e["slot"]
            else:
                new_at.append(e)
        preds["at"] = new_at
        if old_slot and old_slot not in preds.get("slot_empty", []):
            preds["slot_empty"].append(old_slot)
        preds["color"] = [e for e in preds.get("color", []) if e["part"] != name]
        preds["size"]  = [e for e in preds.get("size",  []) if e["part"] != name]
        state.get("metric", {}).pop(name, None)

    else:   # slot
        if name in state["objects"].get("slots", []):
            state["objects"]["slots"].remove(name)
        state.get("slot_belongs_to", {}).pop(name, None)
        preds["slot_empty"] = [s for s in preds.get("slot_empty", []) if s != name]
        preds["at"]         = [e for e in preds.get("at",         []) if e["slot"] != name]
        state.get("metric", {}).pop(name, None)

        parent = _slot_parent(name)
        if parent:
            remaining = [s for s in state["objects"].get("slots", [])
                         if _slot_parent(s) == parent]
            if not remaining:
                for lst in ("kits", "containers"):
                    objs = state["objects"].get(lst, [])
                    if parent in objs:
                        objs.remove(parent)
                preds["role"] = [e for e in preds.get("role", []) if e["object"] != parent]


# ─────────────────────────────────────────────────────────────────────────────
# LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_update_system_prompt() -> str:
    return """\
You are a robot workspace update assistant.

The user has taken a new camera scan of the workspace. The scan has been
compared to the stored configuration (memory). Your job is to:
  1. Summarise the detected changes clearly.
  2. Ask the user which changes to apply to the memory config.
  3. Once the user confirms, output a scene_update block.

══════════════════════════════════════════════════════════════
INPUT STRUCTURE
══════════════════════════════════════════════════════════════
  "memory_scene"       – current stored config (receptacles with roles, parts).
  "detected_changes"   – structured diff from camera vs memory:
    new_receptacles      newly detected kits/containers  {name: {slots: [...]}}
    removed_receptacles  receptacles no longer visible   {name: {slots: [...]}}
    moved_receptacles    boards repositioned             {name: {max_dist_cm, slots}}
    new_parts            new parts found                 [{name, color, size, slot}]
    removed_parts        memory parts no longer seen     [{name, color, size, slot}]
    moved_parts          parts that moved                {name: {color, dist_cm, from_slot, to_slot}}
    unchanged_summary    counts of stable objects        {slots: N, parts: N}

══════════════════════════════════════════════════════════════
WORKFLOW
══════════════════════════════════════════════════════════════
1. Give a clear, concise summary of the detected changes.
2. Ask which changes the user wants to apply.
   Handle each category (new / removed / moved) separately if needed.
   If the user says "all" → include everything.
   If the user says "only [X]" → include only those items.
3. If ambiguous, ask ONE focused clarification question.
4. Once the scope is clear, output the scene_update block and ask "Confirm?"

══════════════════════════════════════════════════════════════
OUTPUT BLOCK
══════════════════════════════════════════════════════════════
```scene_update
{
  "add":    ["<name>", ...],
  "remove": ["<name>", ...],
  "move":   ["<name>", ...]
}
```

RULES:
  "add"    → new objects from the camera scan to add to memory.
             Use stable part names (e.g. "Part_4") from detected_changes.new_parts.
             For receptacles, use the receptacle name (e.g. "Kit_2") — all its
             slots are included automatically. Individual slot names also work.
  "remove" → objects to remove from memory.
             Use names from memory (detected_changes.removed_* or memory_scene).
             Receptacle name removes all its slots.
  "move"   → objects whose positions to update from the new scan.
             Receptacle name updates all its slots.
  All three arrays may be empty ([]) but the block must always contain all three keys.
  Never invent names. Use only names present in the input JSON.

══════════════════════════════════════════════════════════════
CONFIRMATION
══════════════════════════════════════════════════════════════
  After proposing the block, always ask: "Confirm?"
  If confirmed → config is updated. Ask if there is more to do.
  If rejected  → discard and ask what to change.

══════════════════════════════════════════════════════════════
AMBIGUITY RULES
══════════════════════════════════════════════════════════════
  Never guess. Never infer. Ask.
  When asking clarification, do NOT re-print the full JSON.
  Only mention the relevant candidates with name and one distinguishing attribute.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Parse LLM output block
# ─────────────────────────────────────────────────────────────────────────────

_UPDATE_BLOCK_RE = re.compile(
    r"```scene_update\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)


def extract_scene_update_block(text: str) -> Dict[str, Any]:
    m = _UPDATE_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```scene_update``` block found.")
    data = json.loads(m.group(1).strip())
    if not isinstance(data, dict):
        raise ValueError("scene_update block must be a JSON object.")
    for key in ("add", "remove", "move"):
        if key in data and not isinstance(data[key], list):
            raise ValueError(f"'{key}' must be a JSON array.")
        data.setdefault(key, [])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_atomic(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _slim_memory(memory_state: Dict[str, Any]) -> Dict[str, Any]:
    """Compact LLM-facing view of current stored config (no coordinates)."""
    preds     = memory_state.get("predicates", {})
    objs      = memory_state.get("objects",    {})
    role_map  = {e["object"]: e.get("role") for e in preds.get("role",  [])}
    at_map    = {e["part"]:   e["slot"]     for e in preds.get("at",    [])}
    color_map = {e["part"]:   e.get("color") for e in preds.get("color", [])}
    size_map  = {e["part"]:   e.get("size")  for e in preds.get("size",  [])}

    receptacles: Dict[str, Any] = {}
    for r in sorted(set(objs.get("kits", []) + objs.get("containers", []))):
        slots = sorted(s for s in objs.get("slots", []) if _slot_parent(s) == r)
        receptacles[r] = {"role": role_map.get(r), "slots": slots}

    parts: Dict[str, Any] = {}
    for p in objs.get("parts", []):
        parts[p] = {
            "color": color_map.get(p),
            "size":  size_map.get(p),
            "slot":  at_map.get(p),
        }

    return {
        "workspace":   memory_state.get("workspace", {}),
        "receptacles": receptacles,
        "parts":       parts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simple dialogue helpers (copied here to avoid circular import with API_Main)
# ─────────────────────────────────────────────────────────────────────────────

def _contains_word(text: str, words) -> bool:
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", text):
            return True
    return False


def _is_finish(text: str) -> bool:
    t = text.strip().lower()
    return _contains_word(t, [
        "finish", "finalize", "done", "end", "export",
        "save", "write", "quit", "exit",
    ]) or any(x in t for x in ["last step", "that's all", "thats all"])


def _is_yes(text: str) -> bool:
    t = text.strip().lower()
    return _contains_word(t, [
        "yes", "ok", "okay", "confirm", "confirmed", "sure", "correct",
    ]) or any(x in t for x in ["go ahead", "do it", "looks good"]) or t == "y"


def _is_no(text: str) -> bool:
    t = text.strip().lower()
    return _contains_word(t, [
        "no", "nope", "cancel", "reject", "wrong", "redo",
    ]) or t == "n"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called from API_Main
# ─────────────────────────────────────────────────────────────────────────────

def run_update_session(client: Any) -> None:
    """
    Full scene-update dialogue:
      1. Run vision → compute diff → re-number parts.
      2. Show terminal summary.
      3. LLM dialogue to select changes.
      4. Apply confirmed changes, save positions.json, archive to Memory/.
    """
    # ── Acquire both states ───────────────────────────────────────────────────
    try:
        memory_state, fresh_state = run_vision_and_load()
    except Exception as exc:
        print(f"\n❌  Vision failed: {exc}\n")
        return

    # ── Compute diff & renumber ───────────────────────────────────────────────
    diff = compute_scene_diff(memory_state, fresh_state)
    fresh_renamed, rename_map = renumber_fresh_parts(fresh_state, diff, memory_state)

    has_changes = any([
        diff["new_slots"], diff["removed_slots"], diff["moved_slots"],
        diff["new_parts"], diff["removed_parts"], diff["moved_parts"],
    ])

    # ── Terminal summary ──────────────────────────────────────────────────────
    print("\n── Scene Update — Detected Changes ──")
    print(build_diff_summary(diff, memory_state, fresh_renamed, rename_map))

    if not has_changes:
        print("\n✅  Camera scan matches memory exactly — nothing to update.\n")
        # Restore fresh state was already saved by Vision_Main; write back memory.
        _save_atomic(POSITIONS_PATH, memory_state)
        return

    # ── Build LLM context ─────────────────────────────────────────────────────
    llm_diff = format_diff_for_llm(diff, memory_state, fresh_renamed, rename_map)
    slim_mem = _slim_memory(memory_state)

    messages = [
        {"role": "system", "content": _build_update_system_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {"memory_scene": slim_mem, "detected_changes": llm_diff},
                indent=2, ensure_ascii=False,
            ),
        },
        {
            "role": "user",
            "content": (
                "Please give a brief summary of the detected changes "
                "and ask which ones I want to apply."
            ),
        },
    ]

    def _chat(msgs: List[Dict]) -> str:
        resp = client.chat.completions.create(
            model=_LLM_MODEL, messages=msgs, temperature=0.2
        )
        return (resp.choices[0].message.content or "").strip()

    pending_update: Optional[Dict[str, Any]] = None
    print(f"\n── Mode: Scene Update ──\n")

    # ── Dialogue loop ─────────────────────────────────────────────────────────
    while True:
        assistant_text = _chat(messages)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        # Try to parse a scene_update block
        try:
            pending_update = extract_scene_update_block(assistant_text)
            n_ops = sum(len(v) for v in pending_update.values() if isinstance(v, list))
            print(f"  [Scene-update proposal: {n_ops} operation(s)]\n")
        except ValueError:
            if "```scene_update" in (assistant_text or ""):
                # Malformed block — feed parse error back to LLM
                try:
                    extract_scene_update_block(assistant_text)
                except ValueError as e:
                    print(f"  [WARNING: scene_update block parse error — {e}]")
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your scene_update block failed to parse: {e}\n"
                            'Expected format: {"add": [...], "remove": [...], "move": [...]}\n'
                            "Please rewrite the block."
                        ),
                    })
                    continue

        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        # ── finish / done ────────────────────────────────────────────────────
        if _is_finish(user_input):
            if pending_update:
                _apply_and_save(
                    memory_state, fresh_renamed, diff, pending_update
                )
            else:
                print("  [No confirmed changes — positions.json left as-is.]\n")
                # Restore memory (vision_main overwrote it)
                _save_atomic(POSITIONS_PATH, memory_state)
            print("\n── Scene update session complete. ──\n")
            return

        # ── confirm ──────────────────────────────────────────────────────────
        if pending_update is not None and _is_yes(user_input):
            _apply_and_save(memory_state, fresh_renamed, diff, pending_update)
            pending_update = None
            messages.append({"role": "user",      "content": "Confirmed the update."})
            messages.append({"role": "assistant",  "content": "Update applied."})
            user_input2 = input("Anything else? (or type 'done')\nYOU: ").strip()
            if _is_finish(user_input2):
                print("\n── Scene update session complete. ──\n")
                return
            if user_input2:
                messages.append({"role": "user", "content": user_input2})
            continue

        # ── reject ───────────────────────────────────────────────────────────
        if pending_update is not None and _is_no(user_input):
            print("  [Proposal rejected.]\n")
            pending_update = None
            messages.append({
                "role": "user",
                "content": "Rejected. Discard that proposal and ask what changes I want.",
            })
            continue

        messages.append({"role": "user", "content": user_input})


def _apply_and_save(
    memory_state:  Dict[str, Any],
    fresh_renamed: Dict[str, Any],
    diff:          Dict[str, Any],
    update_block:  Dict[str, Any],
) -> None:
    """Merge, save positions.json, archive to Memory/."""
    merged = merge_confirmed(
        memory_state  = memory_state,
        fresh_renamed = fresh_renamed,
        diff          = diff,
        add_names     = update_block.get("add",    []),
        remove_names  = update_block.get("remove", []),
        move_names    = update_block.get("move",   []),
    )

    _save_atomic(POSITIONS_PATH, merged)
    print(f"✅  positions.json updated → {POSITIONS_PATH.resolve()}")

    try:
        from Configuration_Module.Apply_Sequence_Changes import save_to_memory  # type: ignore
        mem_path = save_to_memory(merged, label="scene_update")
        print(f"✅  State archived → {mem_path.resolve()}\n")
    except ImportError:
        print("  (Memory archive skipped — Apply_Sequence_Changes not found)\n")