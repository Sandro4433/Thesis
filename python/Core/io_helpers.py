"""
Core/io_helpers.py — Thin I/O helpers shared by api_main and session_handler.

Extracted here to break the circular import that would arise if api_main
imported from session_handler (session_handler → api_main → session_handler).

These functions have no LLM or pipeline logic — they are pure file I/O.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from Core.paths import (
    SEQUENCE_PATH,
    CHANGES_PATH,
    CONFIGURATION_PATH,
    save_atomic,
    save_to_memory,
)


def save_sequence(sequence: List[List]) -> Path:
    """Write *sequence* to sequence.json atomically. Returns the path."""
    SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENCE_PATH.write_text(
        json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return SEQUENCE_PATH


def save_changes(changes: Dict[str, Any]) -> Path:
    """Write *changes* to changes.json atomically. Returns the path."""
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(
        json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return CHANGES_PATH


def save_config_to_memory(state: Dict[str, Any]) -> Path:
    """Archive *state* as a timestamped snapshot in Memory/. Returns the path."""
    dest = save_to_memory(state, label="configuration")
    print(f"✅  State archived → {dest.name}")
    return dest
