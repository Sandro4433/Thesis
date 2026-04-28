"""
block_parsing.py — Extract and validate fenced JSON blocks from LLM responses.

Handles ``changes`` and ``mapping`` blocks.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# ── Fenced-block regexes ─────────────────────────────────────────────────────

CHANGES_BLOCK_RE = re.compile(
    r"```changes\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)
CHANGES_BLOCK_FALLBACK_RE = re.compile(
    # Fallback for LLM responses where the closing ``` is absent.
    # The lookahead stops at "Confirm", a new fence, or end-of-string.
    # NOTE: "Confirm" is hardcoded because GPT models consistently follow
    # their changes block with a confirmation prompt starting with that word.
    # If the model changes phrasing this fallback may silently fail — the
    # primary CHANGES_BLOCK_RE (which requires a closing ```) is preferred.
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


def _merge_duplicate_workspace_keys(raw: str) -> str:
    """
    Guard against LLM-generated changes blocks with duplicate 'workspace' keys.
    json.loads silently keeps only the last value, discarding e.g. operation_mode.
    This pre-pass merges all workspace objects into one before parsing.
    """
    try:
        # Use a custom decoder to collect all values for duplicate keys
        from collections import defaultdict
        pairs: list = []

        def pairs_hook(p):
            pairs.extend(p)
            return dict(p)

        import json as _json
        _json.loads(raw, object_pairs_hook=pairs_hook)

        workspace_values = [v for k, v in pairs if k == "workspace" and isinstance(v, dict)]
        if len(workspace_values) <= 1:
            return raw  # no duplicates, nothing to do

        # Merge all workspace dicts (left to right, last wins per key)
        merged_ws: dict = {}
        for ws in workspace_values:
            merged_ws.update(ws)

        # Re-encode the full object with a single merged workspace key
        all_pairs_merged: dict = {}
        seen_workspace = False
        for k, v in pairs:
            if k == "workspace":
                if not seen_workspace:
                    all_pairs_merged[k] = merged_ws
                    seen_workspace = True
                # skip subsequent workspace entries
            else:
                all_pairs_merged[k] = v

        return _json.dumps(all_pairs_merged)
    except Exception:
        return raw  # if anything fails, let json.loads handle it and surface the real error


def extract_changes_block(text: str) -> Dict[str, Any]:
    """Parse a ``changes`` block into a dict of workspace changes."""
    m = CHANGES_BLOCK_RE.search(text or "")
    if not m:
        m = CHANGES_BLOCK_FALLBACK_RE.search(text or "")
    if not m:
        raise ValueError("No ```changes``` block found.")

    raw_json = _merge_duplicate_workspace_keys(m.group(1).strip())
    data = json.loads(raw_json)
    if not isinstance(data, dict) or not data:
        raise ValueError("Changes block must be a non-empty JSON object.")

    for obj_name, attrs in data.items():
        if not isinstance(obj_name, str) or not obj_name.strip():
            raise ValueError(f"Object name must be a non-empty string, got: {obj_name!r}")

        if obj_name in SPECIAL_KEYS:
            if attrs is None or (isinstance(attrs, list) and len(attrs) == 0):
                continue
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
                    f"'{obj_name}': unknown attribute '{attr}'. "
                    f"Allowed: role, color, fragility."
                )
    return data


def extract_mapping_block(text: str) -> Dict[str, str]:
    """Parse a ``mapping`` block → {image_name: old_name | "new"}.

    Accepts several formats the LLM might produce:

    1.  ``mapping\\n{...}\\n```        (canonical)
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