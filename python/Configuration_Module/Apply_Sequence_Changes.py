# Apply_Sequence_Changes.py
#
# Configuration Module — post-execution state update.
#
# Receives the motion sequence produced by the PDDL planner (sequence.json)
# and applies each pick-and-place step to the current positions.json, yielding
# an accurate post-execution workspace state.
#
# What changes per step [pick_name, place_name, gripper_width]:
#   predicates.at       — part moves from source_slot → place_name
#                         (or is removed from standalone if it was not in a slot)
#   predicates.slot_empty — source_slot becomes empty; place_name is no longer empty
#   objects.parts       — standalone list unchanged (part identity is preserved)
#   metric[pick_name]   — pos/quat/orientation updated to match metric[place_name]
#
# Called from API_Main after Robot_Main returns (i.e. robot has fully stopped).
# Can also be run standalone for debugging:
#   python Apply_Sequence_Changes.py
#
# Output:
#   - Overwrites positions.json with the post-execution state (atomic write).
#   - Saves a timestamped copy to Memory/ so previous states are never lost.

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import POSITIONS_JSON, LLM_RESPONSE_JSON

POSITIONS_PATH = Path(POSITIONS_JSON.resolve())
SEQUENCE_PATH  = Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json"
MEMORY_DIR     = PROJECT_DIR / "Memory"


# ── core apply function ───────────────────────────────────────────────────────

def apply_sequence(
    state: Dict[str, Any],
    sequence: List[List],
) -> Dict[str, Any]:
    """
    Apply a completed motion sequence to the workspace state.
    Returns an updated deep copy — the input is never modified.

    Handles both cases:
      - Part was in a slot  (listed in predicates.at)
      - Part was standalone (in objects.parts but not in predicates.at)

    After each step the part's metric entry is updated to reflect its new
    physical position (the destination slot's pos/quat/orientation).
    """
    state  = copy.deepcopy(state)
    preds  = state.setdefault("predicates", {})
    metric = state.setdefault("metric", {})

    at_list:    List[Dict[str, str]] = preds.setdefault("at", [])
    empty_list: List[str]            = preds.setdefault("slot_empty", [])

    # build fast lookup: part_name → source_slot (or None if standalone)
    at_index: Dict[str, str] = {item["part"]: item["slot"] for item in at_list}

    for entry in sequence:
        if len(entry) < 2:
            print(f"  ⚠  Malformed sequence entry (needs at least 2 elements): {entry!r} — skipped.")
            continue

        pick_name:  str = entry[0]
        place_name: str = entry[1]

        source_slot: Optional[str] = at_index.get(pick_name)   # None → standalone

        # ── validate destination ───────────────────────────────────────────────
        known_slots = {e["slot"] for e in at_list} | set(empty_list)
        if place_name not in known_slots:
            print(f"  ⚠  Destination slot '{place_name}' not found in state — skipped.")
            continue

        # ── update predicates.at ───────────────────────────────────────────────
        # Remove part from wherever it was (slot or standalone — no at entry to remove
        # for standalone, but we still need to add it to the destination)
        at_list[:] = [i for i in at_list if i["part"] != pick_name]
        at_list.append({"part": pick_name, "slot": place_name})
        at_index[pick_name] = place_name    # keep lookup in sync

        # ── update slot_empty ──────────────────────────────────────────────────
        if source_slot is not None and source_slot not in empty_list:
            empty_list.append(source_slot)
        if place_name in empty_list:
            empty_list.remove(place_name)

        # ── update metric (position follows the destination slot) ──────────────
        dest_metric = metric.get(place_name)
        if dest_metric:
            if pick_name not in metric:
                metric[pick_name] = {}
            for key in ("pos", "quat", "orientation"):
                if key in dest_metric:
                    metric[pick_name][key] = dest_metric[key]
        else:
            print(f"  ⚠  No metric entry found for slot '{place_name}' — position not updated.")

        src_label = source_slot if source_slot else "(standalone)"
        print(f"  ✓  {pick_name}  {src_label} → {place_name}")

    return state


# ── persistence helpers ───────────────────────────────────────────────────────

def _save_atomic(path: Path, state: Dict[str, Any]) -> None:
    """Write JSON atomically via a .tmp file → os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def save_to_memory(state: Dict[str, Any], label: str = "post_exec") -> Path:
    """
    Save a timestamped copy to Memory/ so that no prior state is ever lost.
    Returns the path it was saved to.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"positions_{label}_{ts}.json"
    dest = MEMORY_DIR / name
    _save_atomic(dest, state)
    return dest


# ── top-level entry point (called from API_Main) ──────────────────────────────

def apply_and_save(
    positions_path: Path,
    sequence: List[List],
    save_memory: bool = True,
) -> Dict[str, Any]:
    """
    Load positions.json, apply the sequence, overwrite positions.json, and
    optionally archive a copy to Memory/.

    Called by API_Main immediately after Robot_Main returns (robot fully stopped).

    Parameters
    ----------
    positions_path : path to positions.json
    sequence       : list of [pick_name, place_name, gripper_width] entries
    save_memory    : if True, save a timestamped archive to Memory/

    Returns the updated state dict.
    """
    if not positions_path.exists():
        print(f"⚠  positions.json not found at {positions_path} — post-execution update skipped.")
        return {}

    state = json.loads(positions_path.read_text(encoding="utf-8"))

    print("\n── Applying sequence to workspace state ──")
    updated = apply_sequence(state, sequence)

    # Overwrite positions.json
    _save_atomic(positions_path, updated)
    print(f"✅  positions.json updated → {positions_path.resolve()}")

    # Archive to Memory/
    if save_memory:
        mem_path = save_to_memory(updated, label="post_exec")
        print(f"✅  State archived → {mem_path.resolve()}\n")

    return updated


# ── standalone entry point (for debugging) ───────────────────────────────────

def main() -> None:
    if not POSITIONS_PATH.exists():
        print(f"ERROR: positions.json not found: {POSITIONS_PATH}")
        sys.exit(1)
    if not SEQUENCE_PATH.exists():
        print(f"ERROR: sequence.json not found: {SEQUENCE_PATH}")
        sys.exit(1)

    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded sequence: {SEQUENCE_PATH}  ({len(sequence)} step(s))")

    apply_and_save(POSITIONS_PATH, sequence, save_memory=True)


if __name__ == "__main__":
    main()