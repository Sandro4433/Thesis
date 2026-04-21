"""
change_management.py — Merge, conflict-detect, and resolve workspace changes.

Handles the accumulation of changes across multiple LLM confirmations within
a single session, including conflict detection and resolution.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Merging ──────────────────────────────────────────────────────────────────

def merge_changes(
    accumulated: Dict[str, Any],
    new_block: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge new changes into accumulated changes.

    - part_compatibility: extend (accumulate rules across confirmations)
    - priority, kit_recipe: replace (new supersedes previous)
    - workspace: merge dict keys
    - everything else: merge/update
    """
    merged = {
        k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
        for k, v in accumulated.items()
    }

    for key, value in new_block.items():
        if key in ("priority", "kit_recipe", "part_compatibility") and value is None:
            value = []

        if key == "part_compatibility" and isinstance(value, list):
            if len(value) == 0:
                merged[key] = []
            else:
                existing = merged.get(key, [])
                if not isinstance(existing, list):
                    existing = []
                merged[key] = existing + value

        elif key in ("priority", "kit_recipe") and isinstance(value, list):
            merged[key] = list(value)

        elif key == "workspace" and isinstance(value, dict):
            merged.setdefault("workspace", {})
            merged["workspace"].update(value)

        elif isinstance(value, dict):
            if key not in merged or not isinstance(merged[key], dict):
                merged[key] = {}
            merged[key].update(value)

        else:
            merged[key] = value

    return merged


# ── Conflict detection ───────────────────────────────────────────────────────

def detect_conflicts(
    accumulated: Dict[str, Any],
    new_block: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Detect conflicts between accumulated and new changes.

    Returns a list of conflict dicts with keys:
        type, key, old_value, new_value, description
    """
    conflicts: List[Dict[str, Any]] = []

    acc_ws = accumulated.get("workspace", {})
    new_ws = new_block.get("workspace", {})
    for attr in ("operation_mode", "batch_size"):
        if attr in acc_ws and attr in new_ws:
            old, new = acc_ws[attr], new_ws[attr]
            if old != new and old is not None and new is not None:
                conflicts.append({
                    "type": "workspace",
                    "key": attr,
                    "old_value": old,
                    "new_value": new,
                    "description": (
                        f"Workspace '{attr}' was set to '{old}', now changing to '{new}'"
                    ),
                })

    for key in new_block:
        if not key.startswith(("Kit_", "Container_")) or not isinstance(new_block[key], dict):
            continue
        if key not in accumulated or not isinstance(accumulated[key], dict):
            continue
        for attr in ("role",):
            if attr in accumulated[key] and attr in new_block[key]:
                old, new = accumulated[key][attr], new_block[key][attr]
                if old != new:
                    conflicts.append({
                        "type": "receptacle",
                        "key": f"{key}.{attr}",
                        "old_value": old,
                        "new_value": new,
                        "description": (
                            f"{key} '{attr}' was set to '{old}', now changing to '{new}'"
                        ),
                    })

    for key in new_block:
        if not key.startswith("Part_") or not isinstance(new_block[key], dict):
            continue
        if key not in accumulated or not isinstance(accumulated[key], dict):
            continue
        for attr in ("fragility", "color"):
            if attr in accumulated[key] and attr in new_block[key]:
                old, new = accumulated[key][attr], new_block[key][attr]
                if old != new:
                    conflicts.append({
                        "type": "part",
                        "key": f"{key}.{attr}",
                        "old_value": old,
                        "new_value": new,
                        "description": (
                            f"{key} '{attr}' was set to '{old}', now changing to '{new}'"
                        ),
                    })

    acc_compat = accumulated.get("part_compatibility", [])
    new_compat = new_block.get("part_compatibility", [])
    if acc_compat and new_compat:
        for new_rule in new_compat:
            new_allowed = set(new_rule.get("allowed_in", []))
            new_excluded = set(new_rule.get("not_allowed_in", []))
            for old_rule in acc_compat:
                if not _rules_overlap(new_rule, old_rule):
                    continue
                old_excluded = set(old_rule.get("not_allowed_in", []))
                old_allowed = set(old_rule.get("allowed_in", []))

                overlap = new_allowed & old_excluded
                if overlap:
                    conflicts.append({
                        "type": "list_rule",
                        "key": "part_compatibility",
                        "old_value": old_rule,
                        "new_value": new_rule,
                        "description": (
                            f"New rule allows {overlap} but previous rule excludes it"
                        ),
                    })
                overlap = new_excluded & old_allowed
                if overlap:
                    conflicts.append({
                        "type": "list_rule",
                        "key": "part_compatibility",
                        "old_value": old_rule,
                        "new_value": new_rule,
                        "description": (
                            f"New rule excludes {overlap} but previous rule allows it"
                        ),
                    })

    return conflicts


def _rules_overlap(rule1: Dict[str, Any], rule2: Dict[str, Any]) -> bool:
    """Check if two part_compatibility rules might apply to overlapping parts."""
    if "part_name" in rule1 and "part_name" in rule2:
        return rule1["part_name"] == rule2["part_name"]
    if "part_name" in rule1 or "part_name" in rule2:
        return True
    if "part_color" in rule1 and "part_color" in rule2:
        if rule1["part_color"].lower() != rule2["part_color"].lower():
            return False
    if "part_fragility" in rule1 and "part_fragility" in rule2:
        if rule1["part_fragility"].lower() != rule2["part_fragility"].lower():
            return False
    return True


# ── Conflict formatting & resolution ────────────────────────────────────────

def format_conflicts_for_user(conflicts: List[Dict[str, Any]]) -> str:
    """Format conflicts into a user-friendly message."""
    if not conflicts:
        return ""
    lines = ["⚠️  Detected conflicting changes in this session:\n"]
    for i, c in enumerate(conflicts, 1):
        lines.append(f"  {i}. {c['description']}")
        lines.append(f"     - Previous: {c['old_value']}")
        lines.append(f"     - New: {c['new_value']}")
    lines.append("\nDescribe what you would like to keep:")
    return "\n".join(lines)


def resolve_conflicts(
    accumulated: Dict[str, Any],
    new_block: Dict[str, Any],
    conflicts: List[Dict[str, Any]],
    keep_new: bool,
) -> Dict[str, Any]:
    """Resolve conflicts by choosing old or new values, then merge."""
    if keep_new:
        return merge_changes(accumulated, new_block)

    filtered = json.loads(json.dumps(new_block))

    for conflict in conflicts:
        if conflict["type"] == "workspace":
            ws = filtered.get("workspace", {})
            ws.pop(conflict["key"], None)

        elif conflict["type"] in ("receptacle", "part"):
            parts = conflict["key"].split(".")
            if len(parts) == 2:
                obj_key, attr = parts
                obj = filtered.get(obj_key, {})
                if isinstance(obj, dict):
                    obj.pop(attr, None)

        elif conflict["type"] == "list_rule":
            compat = filtered.get("part_compatibility", [])
            filtered["part_compatibility"] = [
                r for r in compat if r != conflict["new_value"]
            ]

    return merge_changes(accumulated, filtered)


def interpret_conflict_resolution(
    client: OpenAI,
    model: str,
    user_input: str,
    conflicts: List[Dict[str, Any]],
) -> bool:
    """
    Use LLM to interpret user's conflict resolution response.
    Returns True to keep new values, False to keep old.
    """
    descriptions = [
        f"Conflict {i}: {c['description']}\n"
        f"  - Previous value: {c['old_value']}\n"
        f"  - New value: {c['new_value']}"
        for i, c in enumerate(conflicts, 1)
    ]

    prompt = (
        f"The user was asked to resolve conflicting configuration changes.\n\n"
        f"CONFLICTS:\n{chr(10).join(descriptions)}\n\n"
        f'USER\'S RESPONSE: "{user_input}"\n\n'
        "Based on the user's response, which values do they want to keep?\n"
        "Reply with ONLY one word: \"NEW\" or \"OLD\"\n\n"
        "- \"NEW\" means use the new/recent/second values\n"
        "- \"OLD\" means keep the previous/first/original values\n\n"
        "Your answer (NEW or OLD):"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        return "NEW" in answer
    except Exception as exc:
        logger.warning("Could not interpret conflict resolution response: %s", exc)
        return True


# ── Priority ambiguity detection ─────────────────────────────────────────────

def detect_priority_ambiguity(
    priority_list: List[Dict[str, Any]],
    color_map: Dict[str, str],
) -> List[str]:
    """
    Detect duplicate kit/container fill-order ranks.

    Returns a list of human-readable ambiguity descriptions.
    """
    ambiguities: List[str] = []

    order_to_recs: Dict[int, List[str]] = {}
    for entry in priority_list:
        rec = (
            entry.get("kit")
            or entry.get("container")
            or entry.get("destination")
            or (entry.get("receptacle") if "source" not in entry else None)
        )
        if rec is not None:
            order = int(entry["order"])
            order_to_recs.setdefault(order, []).append(rec)

    for order, recs in order_to_recs.items():
        if len(recs) > 1:
            ambiguities.append(
                f"Fill-order conflict: {' and '.join(recs)} are both "
                f"assigned the same fill position. Which should be filled first?"
            )

    return ambiguities


def format_priority_ambiguities(ambiguities: List[str]) -> str:
    """Format priority ambiguities into a user-facing message."""
    if not ambiguities:
        return ""
    lines = ["⚠️  Priority conflict detected — please clarify:\n"]
    for i, a in enumerate(ambiguities, 1):
        lines.append(f"  {i}. {a}")
    return "\n".join(lines)
