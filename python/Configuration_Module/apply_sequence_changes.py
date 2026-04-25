"""
apply_sequence_changes.py — Post-execution state update.

Receives the motion sequence produced by the PDDL planner (sequence.json)
and applies each pick-and-place step to the current configuration.json,
yielding an accurate post-execution workspace state.

What changes per step [pick_name, place_name]:
  predicates.at       — part moves from source_slot → place_name
  predicates.slot_empty — source_slot becomes empty; place_name is no longer empty
  metric[pick_name]   — pos/quat/orientation updated to match metric[place_name]

Can also be run standalone for debugging::

    python -m robot_configurator.configuration.apply_sequence_changes
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from Core.paths import (
    CONFIGURATION_PATH,
    SEQUENCE_PATH,
    save_atomic,
    save_to_memory,
)

logger = logging.getLogger(__name__)


# ── core apply function ───────────────────────────────────────────────────────

def apply_sequence(
    state: Dict[str, Any],
    sequence: List[List],
) -> Dict[str, Any]:
    """
    Apply a completed motion sequence to the workspace state.

    Returns an updated deep copy — the input is never modified.

    Handles both cases:

    * Part was in a slot (listed in ``predicates.at``)
    * Part was standalone (in ``objects.parts`` but not in ``predicates.at``)

    After each step the part's metric entry is updated to reflect its new
    physical position (the destination slot's pos/quat/orientation).
    """
    state = copy.deepcopy(state)
    preds = state.setdefault("predicates", {})
    metric = state.setdefault("metric", {})

    at_list: List[Dict[str, str]] = preds.setdefault("at", [])
    empty_list: List[str] = preds.setdefault("slot_empty", [])

    at_index: Dict[str, str] = {item["part"]: item["slot"] for item in at_list}

    for entry in sequence:
        if len(entry) < 2:
            logger.warning("Malformed sequence entry (needs ≥2 elements): %r — skipped.", entry)
            continue

        pick_name: str = entry[0]
        place_name: str = entry[1]
        source_slot: Optional[str] = at_index.get(pick_name)

        known_slots = {e["slot"] for e in at_list} | set(empty_list)
        if place_name not in known_slots:
            logger.warning("Destination slot '%s' not found in state — skipped.", place_name)
            continue

        at_list[:] = [i for i in at_list if i["part"] != pick_name]
        at_list.append({"part": pick_name, "slot": place_name})
        at_index[pick_name] = place_name

        if source_slot is not None and source_slot not in empty_list:
            empty_list.append(source_slot)
        if place_name in empty_list:
            empty_list.remove(place_name)

        dest_metric = metric.get(place_name)
        if dest_metric:
            metric.setdefault(pick_name, {})
            for key in ("pos", "quat", "orientation"):
                if key in dest_metric:
                    metric[pick_name][key] = dest_metric[key]
        else:
            logger.warning(
                "No metric entry found for slot '%s' — position not updated.", place_name
            )

        src_label = source_slot if source_slot else "(standalone)"
        print(f"  ✓  {pick_name}  {src_label} → {place_name}")
        print(f"[DEBUG] metric[{pick_name}] after update: {metric.get(pick_name)}")
    
    return state


# ── top-level entry point (called from api_main) ──────────────────────────────

def apply_and_save(
    positions_path: Path,
    sequence: List[List],
    save_memory: bool = True,
) -> Dict[str, Any]:
    """
    Load configuration.json, apply the sequence, overwrite configuration.json,
    and optionally archive a timestamped copy to Memory/.

    Parameters
    ----------
    positions_path:
        Path to ``configuration.json``.
    sequence:
        List of ``[pick_name, place_name]`` entries.
    save_memory:
        If ``True``, save a timestamped archive to Memory/.

    Returns the updated state dict.
    """
    if not positions_path.exists():
        logger.warning(
            "configuration.json not found at %s — post-execution update skipped.",
            positions_path,
        )
        return {}

    state = json.loads(positions_path.read_text(encoding="utf-8"))

    print("\n── Applying sequence to workspace state ──")
    updated = apply_sequence(state, sequence)

    save_atomic(positions_path, updated)
    print(f"✅  configuration.json updated → {positions_path.resolve()}")

    if save_memory:
        mem_path = save_to_memory(updated, label="post_exec")
        print(f"✅  State archived → {mem_path.resolve()}\n")

    return updated


# ── standalone entry point ────────────────────────────────────────────────────

def main() -> None:
    import sys

    if not CONFIGURATION_PATH.exists():
        print(f"ERROR: configuration.json not found: {CONFIGURATION_PATH}")
        sys.exit(1)
    if not SEQUENCE_PATH.exists():
        print(f"ERROR: sequence.json not found: {SEQUENCE_PATH}")
        sys.exit(1)

    sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded sequence: {SEQUENCE_PATH}  ({len(sequence)} step(s))")
    apply_and_save(CONFIGURATION_PATH, sequence, save_memory=True)


if __name__ == "__main__":
    main()
