"""
block_parsing.py — Extract and validate fenced JSON blocks from LLM responses.

Handles ```sequence```, ```changes```, and ```mapping``` blocks.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# ── Fenced-block regexes ─────────────────────────────────────────────────────

SEQUENCE_BLOCK_RE = re.compile(
    r"```sequence\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)
CHANGES_BLOCK_RE = re.compile(
    r"```changes\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)
CHANGES_BLOCK_FALLBACK_RE = re.compile(
    r"```changes\s*(.*?)\s*(?=Confirm|```|\Z)", re.DOTALL | re.IGNORECASE
)
MAPPING_BLOCK_RE = re.compile(
    r"```mapping\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)
# Fallback 1: "mapping" on its own line, then a separate ``` block with JSON
MAPPING_BLOCK_FALLBACK_RE = re.compile(
    r"(?:^|\n)\s*mapping\s*\n\s*```[^\n]*\n\s*(\{.*?\})\s*\n\s*```",
    re.DOTALL | re.IGNORECASE,
)
# Fallback 2: "mapping" on its own line, then bare JSON (no fences)
MAPPING_BLOCK_FALLBACK2_RE = re.compile(
    r"(?:^|\n)\s*mapping\s*\n\s*(\{.*?\})", re.DOTALL | re.IGNORECASE
)

# ── Valid attribute values ────────────────────────────────────────────────────

VALID_ROLE = {"input", "output", None}
VALID_COLOR = {"Blue", "Red", "Green", "blue", "red", "green"}
VALID_FRAGILITY = {"normal", "fragile", None}

SPECIAL_KEYS = ("workspace", "priority", "kit_recipe", "part_compatibility")
ALLOWED_PART_ATTRS = {"role", "color", "fragility"}


def extract_sequence_block(text: str) -> List[List]:
    """Parse a ```sequence``` block into a list of [pick, place] entries."""
    m = SEQUENCE_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```sequence``` block found.")

    data = json.loads(m.group(1).strip())
    if not isinstance(data, list) or not data:
        raise ValueError("Sequence block must be a non-empty JSON array.")

    for i, entry in enumerate(data):
        if not isinstance(entry, list) or len(entry) not in (2, 3):
            raise ValueError(
                f"Entry {i} must be [pick_name, place_name] or "
                f"[pick_name, place_name, 0.05], got: {entry!r}"
            )
        if not isinstance(entry[0], str) or not isinstance(entry[1], str):
            raise ValueError(f"Entry {i}: pick_name and place_name must be strings.")
        if not entry[0].strip() or not entry[1].strip():
            raise ValueError(f"Entry {i}: names must not be empty.")
        if len(entry) == 3:
            if not isinstance(entry[2], (int, float)) or entry[2] <= 0:
                raise ValueError(
                    f"Entry {i}: gripper_close_width must be a positive number, got: {entry[2]!r}"
                )
    return data


def extract_changes_block(text: str) -> Dict[str, Any]:
    """Parse a ```changes``` block into a dict of workspace changes."""
    m = CHANGES_BLOCK_RE.search(text or "")
    if not m:
        m = CHANGES_BLOCK_FALLBACK_RE.search(text or "")
    if not m:
        raise ValueError("No ```changes``` block found.")

    data = json.loads(m.group(1).strip())
    if not isinstance(data, dict) or not data:
        raise ValueError("Changes block must be a non-empty JSON object.")

    for obj_name, attrs in data.items():
        if not isinstance(obj_name, str) or not obj_name.strip():
            raise ValueError(f"Object name must be a non-empty string, got: {obj_name!r}")

        if obj_name in SPECIAL_KEYS:
            continue

        if not isinstance(attrs, dict) or not attrs:
            raise ValueError(f"Attributes for '{obj_name}' must be a non-empty object.")

        for attr, val in attrs.items():
            key = attr.lower()
            if key == "role" and val not in VALID_ROLE:
                raise ValueError(f"'{obj_name}'.role must be 'input', 'output', or null.")
            if key == "color" and val not in VALID_COLOR:
                raise ValueError(f"'{obj_name}'.color must be 'Blue', 'Red', or 'Green'.")
            if key == "fragility" and val not in VALID_FRAGILITY:
                raise ValueError(f"'{obj_name}'.fragility must be 'normal' or 'fragile'.")
            if key not in ALLOWED_PART_ATTRS:
                raise ValueError(
                    f"'{obj_name}': unknown attribute '{attr}'. Allowed: role, color, fragility."
                )
    return data


def extract_mapping_block(text: str) -> Dict[str, str]:
    """Parse a ```mapping``` block → {image_name: old_name | "new"}.

    Accepts several formats the LLM might produce:
      1.  ```mapping\\n{...}\\n```        (canonical)
      2.  mapping\\n```\\n{...}\\n```     (label outside fences)
      3.  mapping\\n{...}                 (no fences at all)
    """
    m = MAPPING_BLOCK_RE.search(text or "")
    if not m:
        m = MAPPING_BLOCK_FALLBACK_RE.search(text or "")
    if not m:
        m = MAPPING_BLOCK_FALLBACK2_RE.search(text or "")
    if not m:
        raise ValueError("No ```mapping``` block found.")

    data = json.loads(m.group(1).strip())
    if not isinstance(data, dict):
        raise ValueError("Mapping block must be a JSON object.")

    # Empty mapping is valid — means all parts are auto-matched
    if not data:
        return data

    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"Mapping entries must be string→string, got {k!r}→{v!r}")
        if not k.startswith("Part_"):
            raise ValueError(f"Keys must be Part_* names from the new scan, got {k!r}")
        if v != "new" and not v.startswith("Part_"):
            raise ValueError(f"Values must be old Part_* names or 'new', got {v!r}")
    return data