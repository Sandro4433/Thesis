# planner_interface.py
#
# Single entry point for sequence generation.
# Reads RC_USE_PDDL_PLANNER from .env (via settings) to decide which backend to call.
#
# Both planners receive the same input (positions.json) and produce the
# same output (sequence.json), so the execution module is unaffected.
#
# Workspace RECONFIGURATION is always handled by the LLM (API_Main.py).
# This module is only called when a sequence needs to be generated.

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def generate_sequence(
    state: Dict[str, Any],
    sequence_path: str,
) -> Optional[List[List]]:
    """
    Generate a pick-and-place sequence from the current workspace state
    and save it to sequence_path.

    The backend is selected by RC_USE_PDDL_PLANNER in .env (settings.use_pddl_planner):
      False (default) → the LLM dialogue in API_Main already produced the
                        sequence; this function is a no-op (returns None to
                        signal that the caller should not overwrite the file).
      True            → run the PDDL planner and save the result.

    Returns the sequence list, or None if skipped / failed.
    """
    # Import here to avoid circular imports at module load time
    from Core.config import settings

    if not settings.use_pddl_planner:
        # LLM dialogue flow — sequence already written by API_Main.py
        return None

    # ── PDDL path ─────────────────────────────────────────────────────────────
    from Planning_Module.pddl_planner import plan_sequence  # type: ignore

    print("\n── PDDL Planner ──")
    sequence = plan_sequence(
        state,
        output_path=sequence_path,
        keep_pddl=True,
    )

    if sequence is None:
        print("❌ PDDL planning failed — no sequence generated.")
    return sequence


def load_state_and_plan(positions_path: str, sequence_path: str) -> Optional[List[List]]:
    """
    Convenience wrapper: load positions.json → generate sequence.

    Typical usage (called from API_Main.py after workspace reconfiguration
    is confirmed and the user requests execution):

        from Planning_Module.planner_interface import load_state_and_plan
        load_state_and_plan(str(POSITIONS_PATH), str(SEQUENCE_PATH))
    """
    try:
        with open(positions_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        print(f"❌ positions.json not found: {positions_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse positions.json: {e}")
        return None

    return generate_sequence(state, sequence_path)