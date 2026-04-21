"""
core/paths.py — Path constants and atomic file I/O helpers.

All filesystem paths are derived from Settings so they can be overridden
via environment variables without touching source code.

Public API
----------
    from Core.paths import (
        PROJECT_DIR, CONFIGURATION_PATH, SEQUENCE_PATH,
        CHANGES_PATH, MEMORY_DIR, WORKSPACE_DIR,
        save_atomic, save_to_memory, empty_state, parent_of_slot,
    )
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from Core.config import settings

# ── Re-exported path constants ───────────────────────────────────────────────

PROJECT_DIR: Path = settings.project_root
CONFIGURATION_PATH: Path = settings.configuration_path
SEQUENCE_PATH: Path = settings.sequence_path
CHANGES_PATH: Path = settings.changes_path
MEMORY_DIR: Path = settings.memory_dir
WORKSPACE_DIR: Path = settings.workspace_dir


# ── Atomic write helper ──────────────────────────────────────────────────────

def save_atomic(path: Path, data: Any, *, indent: int = 2) -> None:
    """
    Write *data* as JSON to *path* atomically.

    Writes to a sibling temp file first, then renames — ensuring that a
    crash mid-write can never leave a half-written file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ── Memory / versioned snapshot helper ──────────────────────────────────────

def save_to_memory(state: Dict[str, Any], label: str = "config") -> Path:
    """
    Save *state* to Memory/ with a timestamp-based filename.

    Returns the path of the saved file.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = MEMORY_DIR / f"{label}_{timestamp}.json"
    save_atomic(dest, state)
    return dest


# ── Empty state factory ──────────────────────────────────────────────────────

def empty_state() -> Dict[str, Any]:
    """Return a blank PDDL-friendly configuration state."""
    return {
        "workspace": {
            "operation_mode": None,
            "batch_size": None,
        },
        "objects": {
            "kits": [],
            "containers": [],
            "slots": [],
            "parts": [],
        },
        "slot_belongs_to": {},
        "predicates": {
            "at": [],
            "slot_empty": [],
            "color": [],
            "role": [],
            "priority": [],
            "kit_recipe": [],
            "part_compatibility": [],
            "fragility": [],
        },
        "metric": {},
    }


# ── PDDL helpers ─────────────────────────────────────────────────────────────

def parent_of_slot(slot_name: str) -> Optional[str]:
    """Return the receptacle name that owns *slot_name*, or None."""
    idx = slot_name.rfind("_Pos_")
    if idx == -1:
        return None
    return slot_name[:idx]
