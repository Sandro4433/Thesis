"""
capacity_tools.py — Capacity-checking tool the LLM can call via function calling.

The LLM is good at understanding user intent and extracting the right
parameters, but unreliable at arithmetic.  This module exposes a
`check_capacity` function as an OpenAI-compatible tool so the LLM can
offload the counting and multiplication to deterministic code.

Usage in the conversation loop:
  1. Pass CAPACITY_TOOL_SCHEMA in the `tools` list when calling the LLM.
  2. If the LLM emits a tool_call for "check_capacity", call
     execute_capacity_check(args, scene) and feed the result back.
  3. The LLM sees the exact numbers and can respond correctly.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


# ── Shared spatial helper ────────────────────────────────────────────────────

def compute_position_labels(scene: Dict[str, Any]) -> Dict[str, str]:
    """
    Compute human-readable relative position labels for all receptacles.

    Uses the inverted-axis convention:
      LARGER X = LEFT,  SMALLER X = RIGHT
      LARGER Y = LOWER, SMALLER Y = UPPER

    Returns e.g. {"Container_1": "bottom-left", "Kit_2": "top"}.
    """
    receptacle_xy = scene.get("receptacle_xy", {})
    capacity = scene.get("capacity", {})
    names = sorted(capacity.keys())

    if not names:
        return {}

    all_xs = [receptacle_xy.get(n, [0, 0])[0] for n in names]
    all_ys = [receptacle_xy.get(n, [0, 0])[1] for n in names]
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    x_range = x_max - x_min
    y_range = y_max - y_min

    labels: Dict[str, str] = {}
    for name in names:
        xy = receptacle_xy.get(name, [0, 0])
        x, y = xy[0], xy[1]

        # Vertical: smaller Y = upper
        if y_range < 0.02:
            v = "center"
        elif y - y_min < y_range * 0.33:
            v = "top"
        elif y - y_min > y_range * 0.66:
            v = "bottom"
        else:
            v = "middle"

        # Horizontal: larger X = left
        if x_range < 0.02:
            h = "center"
        elif x - x_min > x_range * 0.66:
            h = "left"
        elif x - x_min < x_range * 0.33:
            h = "right"
        else:
            h = "center"

        if v and h and h != "center":
            labels[name] = f"{v}-{h}"
        elif v:
            labels[name] = v
        else:
            labels[name] = h or "center"

    return labels


# ── OpenAI function-calling tool schema ──────────────────────────────────────

CAPACITY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_capacity",
        "description": (
            "Check whether the workspace has enough parts and slots for a "
            "proposed kitting or sorting operation. Call this BEFORE proposing "
            "a changes block whenever the user requests kitting or sorting. "
            "The tool counts parts by color in the specified input containers, "
            "computes how many are needed for the recipe/operation, and checks "
            "that output receptacles have enough empty slots. Returns exact "
            "numbers you can quote directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["kitting", "sorting"],
                    "description": "The operation mode the user wants.",
                },
                "input_containers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Receptacle names to use as input (pick parts from). "
                        "Use the exact names from the scene JSON, e.g. "
                        '["Container_1", "Container_2"].'
                    ),
                },
                "output_receptacles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Receptacle names to use as output (place parts into). "
                        "e.g. [\"Kit_1\", \"Kit_2\"] for kitting or "
                        "[\"Container_3\"] for sorting."
                    ),
                },
                "kit_recipe": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                        "required": ["color", "quantity"],
                    },
                    "description": (
                        "For kitting: the per-kit recipe. "
                        'e.g. [{"color": "red", "quantity": 1}, '
                        '{"color": "blue", "quantity": 2}]. '
                        "Omit for sorting."
                    ),
                },
                "sorting_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For sorting: list of colors being sorted. "
                        'e.g. ["red", "blue"]. Omit for kitting.'
                    ),
                },
            },
            "required": ["operation", "input_containers", "output_receptacles"],
        },
    },
}


# ── Tool execution ───────────────────────────────────────────────────────────

def execute_capacity_check(
    args: Dict[str, Any],
    scene: Dict[str, Any],
) -> str:
    """
    Execute the check_capacity tool call.

    Takes the LLM-provided arguments and the current scene, runs
    deterministic arithmetic, and returns a plain-text result the LLM
    can include in its reasoning.
    """
    operation = args.get("operation", "kitting")
    input_containers = set(args.get("input_containers", []))
    output_receptacles = set(args.get("output_receptacles", []))
    kit_recipe = args.get("kit_recipe", [])
    sorting_colors = [c.lower() for c in args.get("sorting_colors", [])]

    capacity = scene.get("capacity", {})
    lines: List[str] = []

    # ── Validate names exist ─────────────────────────────────────────
    all_known = set(capacity.keys())
    unknown_inputs = input_containers - all_known
    unknown_outputs = output_receptacles - all_known
    if unknown_inputs:
        lines.append(f"WARNING: Unknown input receptacles: {', '.join(sorted(unknown_inputs))}")
    if unknown_outputs:
        lines.append(f"WARNING: Unknown output receptacles: {', '.join(sorted(unknown_outputs))}")

    # ── Count available parts by color in input containers ───────────
    available: Dict[str, int] = {}
    for rec_name in sorted(input_containers & all_known):
        rec_cap = capacity.get(rec_name, {})
        for color, count in rec_cap.get("parts_by_color", {}).items():
            available[color] = available.get(color, 0) + count

    lines.append(f"Available parts in {', '.join(sorted(input_containers & all_known))}:")
    if available:
        for color in sorted(available):
            lines.append(f"  {color}: {available[color]}")
    else:
        lines.append("  (none)")

    # ── Count empty slots in output receptacles ──────────────────────
    lines.append("")
    lines.append("Output receptacle capacity:")
    total_output_empty = 0
    for rec_name in sorted(output_receptacles & all_known):
        rec_cap = capacity.get(rec_name, {})
        empty = rec_cap.get("empty", 0)
        total = rec_cap.get("total_slots", 0)
        occupied = rec_cap.get("occupied", 0)
        total_output_empty += empty
        lines.append(f"  {rec_name}: {empty} empty / {total} total (occupied: {occupied})")

    # ── Operation-specific checks ────────────────────────────────────
    lines.append("")
    problems: List[str] = []

    if operation == "kitting" and kit_recipe:
        num_kits = len(output_receptacles & all_known)
        recipe_per_kit = sum(e.get("quantity", 0) for e in kit_recipe)

        lines.append(f"Kitting check ({num_kits} output kits, {recipe_per_kit} parts per kit):")

        for entry in kit_recipe:
            color = entry.get("color", "").lower()
            qty_per_kit = entry.get("quantity", 0)
            total_needed = qty_per_kit * num_kits
            total_avail = available.get(color, 0)
            ok = total_avail >= total_needed

            lines.append(
                f"  {color}: {qty_per_kit} per kit × {num_kits} kits = "
                f"{total_needed} needed, {total_avail} available → "
                f"{'OK' if ok else 'INSUFFICIENT'}"
            )
            if not ok:
                problems.append(
                    f"Not enough {color} parts: need {total_needed}, have {total_avail}"
                )

        # Check slot capacity per output kit
        for rec_name in sorted(output_receptacles & all_known):
            rec_cap = capacity.get(rec_name, {})
            empty = rec_cap.get("empty", 0)
            if recipe_per_kit > empty:
                problems.append(
                    f"{rec_name}: recipe needs {recipe_per_kit} slots but only "
                    f"{empty} are empty"
                )
                lines.append(
                    f"  {rec_name} slot check: need {recipe_per_kit}, "
                    f"empty {empty} → INSUFFICIENT"
                )
            else:
                lines.append(
                    f"  {rec_name} slot check: need {recipe_per_kit}, "
                    f"empty {empty} → OK"
                )

    elif operation == "sorting":
        colors_to_sort = sorting_colors if sorting_colors else list(available.keys())
        total_to_sort = sum(available.get(c, 0) for c in colors_to_sort)

        lines.append(f"Sorting check ({total_to_sort} parts to sort):")
        for color in sorted(colors_to_sort):
            count = available.get(color, 0)
            lines.append(f"  {color}: {count} parts to sort")

        if total_to_sort > total_output_empty:
            problems.append(
                f"Total parts to sort ({total_to_sort}) exceeds total empty "
                f"output slots ({total_output_empty})"
            )

    # ── Summary ──────────────────────────────────────────────────────
    lines.append("")
    if problems:
        lines.append("RESULT: PROBLEMS FOUND")
        for p in problems:
            lines.append(f"  ✗ {p}")
        lines.append("Ask the user how to resolve before proposing a changes block.")
    else:
        lines.append("RESULT: ALL CHECKS PASSED")
        lines.append("You may proceed with the changes block.")

    return "\n".join(lines)


# ── Describe-scene tool ──────────────────────────────────────────────────────

DESCRIBE_SCENE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "describe_scene",
        "description": (
            "Generate a human-readable description of the workspace layout. "
            "Call this ONLY when the user explicitly asks you to describe, "
            "explain, or summarise what is in the scene (e.g. 'describe the "
            "scene', 'what does the workspace look like', 'where are the "
            "containers', 'which container is on the left'). "
            "NEVER call this on your own initiative."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "include_parts": {
                    "type": "boolean",
                    "description": (
                        "Whether to include per-slot part details. Set true "
                        "when the user asks about parts/colors, false when "
                        "they only ask about layout/positions."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "Optional: narrow the description to a specific area. "
                        "Use a receptacle name (e.g. 'Container_1') or a "
                        "category ('kits', 'containers'). Leave empty for "
                        "a full overview."
                    ),
                },
            },
            "required": [],
        },
    },
}


def execute_describe_scene(
    args: Dict[str, Any],
    scene: Dict[str, Any],
) -> str:
    """
    Build a plain-language workspace description from the scene JSON.

    The LLM receives this text and can relay it to the user in natural
    language.
    """
    include_parts = args.get("include_parts", True)
    focus = (args.get("focus") or "").strip()

    capacity = scene.get("capacity", {})
    receptacle_xy = scene.get("receptacle_xy", {})
    pos_labels = compute_position_labels(scene)

    # Determine which receptacles to describe
    if focus:
        focus_lower = focus.lower()
        if focus_lower == "kits":
            names = sorted(n for n in capacity if n.startswith("Kit_"))
        elif focus_lower == "containers":
            names = sorted(n for n in capacity if n.startswith("Container_"))
        else:
            names = [n for n in capacity if n.lower() == focus_lower or n == focus]
            if not names:
                return f"No receptacle named '{focus}' found in the scene."
    else:
        names = sorted(capacity.keys())

    if not names:
        return "No receptacles found."

    # Sort: upper-left first
    def _sort_key(name: str):
        xy = receptacle_xy.get(name, [0, 0])
        return (xy[1], -xy[0])

    names.sort(key=_sort_key)

    lines: List[str] = []
    for name in names:
        cap = capacity.get(name, {})
        pos = pos_labels.get(name, "")
        role = cap.get("role")
        role_str = f", role={role}" if role else ""
        total = cap.get("total_slots", 0)
        occupied = cap.get("occupied", 0)
        empty = cap.get("empty", 0)

        header = f"{name} ({pos}{role_str}): {occupied}/{total} slots occupied, {empty} empty"
        lines.append(header)

        if include_parts:
            pbc = cap.get("parts_by_color", {})
            if pbc:
                color_str = ", ".join(f"{cnt} {col}" for col, cnt in sorted(pbc.items()))
                lines.append(f"  Contains: {color_str}")
            else:
                lines.append("  Empty")

    return "\n".join(lines)