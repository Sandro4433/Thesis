"""Shared, stable filesystem paths and utilities for the project.

All paths are defined relative to the project root directory (the `python/` folder).
Use these constants instead of `os.path.abspath(...)` so code works regardless of
the current working directory.

This module also provides small shared helpers (atomic save, slot-name parsing)
that multiple modules need, avoiding duplication.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# ── Project root setup ────────────────────────────────────────────────────────
# Project root = the folder containing this file (python/)
PROJECT_DIR = Path(__file__).resolve().parent

# Ensure project root is on sys.path (idempotent).
# Every module that previously did its own sys.path.insert can now just
# ``import paths`` and the path is guaranteed to be set.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ── Shared exchange folder ────────────────────────────────────────────────────
FILE_EXCHANGE_DIR = PROJECT_DIR / "File_Exchange"


# ── Common file paths ────────────────────────────────────────────────────────
CONFIGURATION_JSON    = FILE_EXCHANGE_DIR / "configuration.json"
LLM_INPUT_JSON        = FILE_EXCHANGE_DIR / "llm_input.json"
LLM_RESPONSE_JSON     = FILE_EXCHANGE_DIR / "llm_response.json"
POSITIONS_FIXED_JSONL = FILE_EXCHANGE_DIR / "positions_fixed.jsonl"

# Derived paths used by multiple modules
CONFIGURATION_PATH = CONFIGURATION_JSON   # Path object — ready to use
SEQUENCE_PATH      = FILE_EXCHANGE_DIR / "sequence.json"
CHANGES_PATH       = FILE_EXCHANGE_DIR / "workspace_changes.json"
MEMORY_DIR         = PROJECT_DIR / "Memory"


# ── Shared utilities ─────────────────────────────────────────────────────────

def save_atomic(path: Path, state: Dict[str, Any]) -> None:
    """Write JSON atomically via a .tmp file → Path.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def parent_of_slot(slot_name: str) -> Optional[str]:
    """Extract the parent receptacle name from a slot name.

    Examples:
        parent_of_slot("Kit_1_Pos_2")       → "Kit_1"
        parent_of_slot("Container_3_Pos_1") → "Container_3"
        parent_of_slot("Part_5")            → None
    """
    idx = slot_name.rfind("_Pos_")
    if idx == -1:
        return None
    return slot_name[:idx]


def save_to_memory(state: Dict[str, Any], label: str = "session") -> Path:
    """Save a timestamped configuration to Memory/.

    Uses a single consistent naming convention:
        configuration_{label}_YYYYMMDD_HHMMSS.json

    Returns the path it was saved to.
    """
    from datetime import datetime as _dt

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    name = f"configuration_{label}_{ts}.json"
    dest = MEMORY_DIR / name
    save_atomic(dest, state)
    return dest


def empty_state() -> Dict[str, Any]:
    """Return a minimal empty workspace state skeleton."""
    return {
        "workspace": {"operation_mode": None, "batch_size": None},
        "objects":   {"kits": [], "containers": [], "parts": [], "slots": []},
        "slot_belongs_to": {},
        "predicates": {
            "at": [], "slot_empty": [], "role": [],
            "color": [],
            "priority": [], "kit_recipe": [], "part_compatibility": [],
            "fragility": [],
        },
        "metric": {},
    }
