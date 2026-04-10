"""
API_Main.py — LLM conversation engine.

This module handles all direct interaction with the language model:
  - The main reconfiguration / motion-planning conversation loop
  - The update-scene dialogue (part identity matching with the user)
  - Prompt building, response parsing, conflict resolution
  - Tool-use handling (capacity checks, scene description)

Pipeline orchestration (vision subprocess, scene loading, config saving,
robot execution) lives in session_handler.py, which calls into this module
only when it needs the LLM.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Project root setup ───────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Config ───────────────────────────────────────────────────────────────────

MODEL = "gpt-4.1"

from paths import CONFIGURATION_JSON

CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())

# ── Sibling module imports ───────────────────────────────────────────────────

from Communication_Module.block_parsing import (
    extract_sequence_block,
    extract_changes_block,
    extract_mapping_block,
)
from Communication_Module.change_management import (
    merge_changes,
    detect_conflicts,
    format_conflicts_for_user,
    resolve_conflicts,
    interpret_conflict_resolution,
    detect_priority_ambiguity,
    format_priority_ambiguities,
)
from Communication_Module.scene_helpers import slim_scene
from Communication_Module.prompts import build_system_prompt
from Communication_Module.ambiguity_detection import (
    detect_ambiguity,
    format_ambiguity_hint,
)
from Communication_Module.capacity_tools import (
    CAPACITY_TOOL_SCHEMA,
    execute_capacity_check,
    DESCRIBE_SCENE_TOOL_SCHEMA,
    execute_describe_scene,
)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM call helpers
# ═══════════════════════════════════════════════════════════════════════════════

def chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    scene: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call the LLM, handling optional tool use.

    If `scene` is provided, the capacity-check and describe-scene tools are
    offered. When the LLM calls one, the tool is executed locally and the
    result is fed back so the LLM can continue.  This loop runs at most
    MAX_TOOL_ROUNDS times to avoid runaway calls.
    """
    MAX_TOOL_ROUNDS = 3
    tools = (
        [CAPACITY_TOOL_SCHEMA, DESCRIBE_SCENE_TOOL_SCHEMA]
        if scene is not None else None
    )

    # Copy messages so tool-call bookkeeping doesn't leak into the caller's list
    working_messages = list(messages)

    for _ in range(MAX_TOOL_ROUNDS + 1):
        kwargs: Dict[str, Any] = dict(
            model=MODEL, messages=working_messages, temperature=temperature
        )
        if tools:
            kwargs["tools"] = tools

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        # If the LLM produced a normal text response, we're done
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return (choice.message.content or "").strip()

        # ── Handle tool calls ────────────────────────────────────────
        # Append the assistant message with tool calls to the working history
        working_messages.append(choice.message)

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "check_capacity" and scene is not None:
                result = execute_capacity_check(args, scene)
                print(f"  [Tool: check_capacity → result injected]")
            elif tc.function.name == "describe_scene" and scene is not None:
                result = execute_describe_scene(args, scene)
                print(f"  [Tool: describe_scene → result injected]")
            else:
                result = f"Unknown tool: {tc.function.name}"

            working_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Fallback: if we exhausted tool rounds, return whatever text we have
    return (choice.message.content or "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# User intent helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _contains_word(text: str, words) -> bool:
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", text):
            return True
    return False


def is_finish(text: str) -> bool:
    t = text.strip().lower()

    if t in ("done", "end", "quit", "exit", "save", "finalize"):
        return True

    if _contains_word(t, ["finished"]):
        return True

    finish_phrases = [
        "last step", "that's all", "thats all", "i'm done", "im done",
        "we're done", "were done", "finish up", "wrap up",
    ]
    if any(p in t for p in finish_phrases):
        return True

    # "finish" alone = session end. "finish [something]" = task instruction.
    if _contains_word(t, ["finish", "finalize"]):
        m = re.search(r"\b(finish|finalize)\b", t)
        if m:
            rest = t[m.end():].strip().lstrip(".,!?").strip()
            if rest:
                return False
        return True

    return False


def _is_pure_short(text: str, max_words: int = 5) -> bool:
    """True if the text is short enough to be a pure confirmation/rejection."""
    return len(text.split()) <= max_words


def is_yes(text: str) -> bool:
    t = text.strip().lower()
    if not _is_pure_short(t):
        return False
    return (
        _contains_word(t, ["yes", "ok", "okay", "confirm", "confirmed", "sure", "correct"])
        or any(p in t for p in ["go ahead", "do it", "looks good"])
        or t == "y"
    )


def is_no(text: str) -> bool:
    t = text.strip().lower()
    if not _is_pure_short(t):
        return False
    return (
        _contains_word(t, ["no", "nope", "cancel", "reject", "wrong", "redo"])
        or t == "n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Capacity validation (used mid-conversation)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_capacity(
    changes: Dict[str, Any],
) -> Optional[str]:
    """
    Check whether a proposed changes block would route more parts to an output
    receptacle than it has empty slots.

    Returns a human-readable warning string if a problem is found, or None if
    everything fits.  Works for both sorting and kitting.
    """
    if not CONFIGURATION_PATH.exists():
        return None

    try:
        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    preds  = state.get("predicates", {})
    objs   = state.get("objects", {})
    s2p    = state.get("slot_belongs_to", {})

    # Build role map: start from current roles, overlay proposed changes
    role_map: Dict[str, Optional[str]] = {
        e["object"]: e.get("role") for e in preds.get("role", [])
    }
    for key, val in changes.items():
        if isinstance(val, dict) and "role" in val:
            role_map[key] = val["role"]

    # Identify output and input receptacles
    all_recs = set(objs.get("kits", []) + objs.get("containers", []))
    output_recs = {r for r in all_recs if role_map.get(r) == "output"}
    input_recs  = {r for r in all_recs if role_map.get(r) == "input"}

    if not output_recs:
        return None

    # Count empty slots per output receptacle
    empty_slots: Dict[str, int] = {}
    for rec in output_recs:
        rec_slots = [s for s, p in s2p.items() if p == rec]
        occupied = set()
        for e in preds.get("at", []):
            if e["slot"] in rec_slots:
                occupied.add(e["slot"])
        empty_slots[rec] = len(rec_slots) - len(occupied)

    # Build color map for parts
    color_of: Dict[str, str] = {
        e["part"]: (e.get("color") or "unknown").lower()
        for e in preds.get("color", [])
    }

    # Determine which parts are in input receptacles
    slot_to_rec = s2p
    part_at: Dict[str, str] = {e["part"]: e["slot"] for e in preds.get("at", [])}

    input_parts: list = []
    for part, slot in part_at.items():
        rec = slot_to_rec.get(slot)
        if rec in input_recs:
            input_parts.append(part)

    # Also count standalone parts (not in any slot) — they're implicitly available
    in_slot_set = set(part_at.keys())
    for p in objs.get("parts", []):
        if p not in in_slot_set:
            input_parts.append(p)

    op_mode = (changes.get("workspace") or {}).get("operation_mode") or \
              (state.get("workspace") or {}).get("operation_mode")

    problems: list = []

    if op_mode == "sorting":
        # Determine routing from part_compatibility rules
        compat = changes.get("part_compatibility") or preds.get("part_compatibility", [])
        # Count how many parts would go to each output
        routed: Dict[str, int] = {r: 0 for r in output_recs}

        for part in input_parts:
            pc = color_of.get(part, "unknown")
            # Find which output this part would go to
            destinations: list = []
            for rule in compat:
                match = True
                if "part_color" in rule and rule["part_color"].lower() != pc:
                    match = False
                if "part_name" in rule and rule["part_name"] != part:
                    match = False
                if match and "allowed_in" in rule:
                    destinations.extend(
                        r for r in rule["allowed_in"] if r in output_recs
                    )
            # If no specific rule but there's a catch-all (no part_color filter)
            if not destinations:
                for rule in compat:
                    if "part_color" not in rule and "part_name" not in rule:
                        if "allowed_in" in rule:
                            destinations.extend(
                                r for r in rule["allowed_in"] if r in output_recs
                            )
            # If only one output, everything goes there
            if not destinations and len(output_recs) == 1:
                destinations = list(output_recs)

            for dest in destinations:
                routed[dest] = routed.get(dest, 0) + 1

        for rec in output_recs:
            needed = routed.get(rec, 0)
            avail = empty_slots.get(rec, 0)
            if needed > avail:
                problems.append(
                    f"{rec} would receive {needed} parts but only has "
                    f"{avail} empty slot{'s' if avail != 1 else ''}."
                )

    elif op_mode == "kitting":
        recipe = changes.get("kit_recipe") or preds.get("kit_recipe", [])
        if recipe:
            total_per_kit = sum(e.get("quantity", 0) for e in recipe)
            for rec in output_recs:
                avail = empty_slots.get(rec, 0)
                if total_per_kit > avail:
                    problems.append(
                        f"{rec} needs {total_per_kit} slots for the recipe "
                        f"but only has {avail} empty."
                    )

    if problems:
        return "Capacity problem:\n" + "\n".join(f"  - {p}" for p in problems)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Conflict resolution helper
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_conflicts(
    client: OpenAI,
    accumulated: Dict[str, Any],
    pending: Dict[str, Any],
) -> Dict[str, Any]:
    """Check for conflicts, prompt user if found, merge, and return updated accumulated."""
    conflicts = detect_conflicts(accumulated, pending)
    if not conflicts:
        return merge_changes(accumulated, pending)

    print(format_conflicts_for_user(conflicts))
    resolve_input = input("YOU: ").strip()
    keep_new = interpret_conflict_resolution(client, MODEL, resolve_input, conflicts)
    result = resolve_conflicts(accumulated, pending, conflicts, keep_new)

    if keep_new:
        print("✅  Using new values.\n")
    else:
        print("✅  Keeping previous values.\n")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LLM response extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _try_extract_sequence(
    text: str, messages: List[Dict[str, str]]
) -> Any:
    """Try to extract a sequence block. Returns the sequence, None, or False (parse error)."""
    try:
        return extract_sequence_block(text)
    except Exception as e:
        if "```sequence" in (text or ""):
            print(f"  [WARNING: sequence block parse error — {e}]")
            messages.append({
                "role": "user",
                "content": (
                    f"Your sequence block failed to parse: {e}\n"
                    'Each entry must be ["pick", "place"].\n'
                    "Please rewrite the sequence block."
                ),
            })
            return False  # Signal: parse error, retry
        return None


def _try_extract_changes(
    text: str, messages: List[Dict[str, str]]
) -> Any:
    """Try to extract a changes block. Returns the changes, None, or False (parse error)."""
    try:
        return extract_changes_block(text)
    except Exception as e:
        if "```changes" in (text or ""):
            print(f"  [WARNING: changes block parse error — {e}]")
            messages.append({
                "role": "user",
                "content": (
                    f"Your changes block failed to parse: {e}\n"
                    "Use receptacle names (e.g. 'Container_3') for role changes.\n"
                    "Use part names (e.g. 'Part_1') for color changes.\n"
                    "Please rewrite the changes block."
                ),
            })
            return False
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Session lifecycle helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _confirm_pending(
    client: OpenAI,
    messages: List[Dict[str, str]],
    accumulated: Dict[str, Any],
    pending_sequence: Optional[List[List]],
    pending_changes: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle confirmation of pending sequence and/or changes. Returns updated accumulated."""
    import session_handler as sh

    if pending_sequence is not None:
        sh.save_sequence(pending_sequence)
        print("✅  Sequence confirmed.\n")
        messages.append({"role": "user", "content": "Confirmed the sequence."})
        messages.append({"role": "assistant", "content": "Sequence saved."})

    if pending_changes is not None:
        accumulated = _handle_conflicts(client, accumulated, pending_changes)
        print("✅  Changes noted.\n")
        messages.append({"role": "user", "content": "Confirmed the changes."})
        messages.append({"role": "assistant", "content": "Changes noted."})

    return accumulated


def _finalize_session(
    client: OpenAI,
    accumulated: Dict[str, Any],
    pending_sequence: Optional[List[List]],
    pending_changes: Optional[Dict[str, Any]],
) -> None:
    """Save any pending work and end the session."""
    import session_handler as sh

    if pending_sequence is not None:
        sh.save_sequence(pending_sequence)
        print("✅  Sequence saved.")

    if pending_changes is not None:
        accumulated = _handle_conflicts(client, accumulated, pending_changes)

    if accumulated:
        sh.save_changes(accumulated)
        sh.apply_and_save_config(accumulated)
        print("✅  Changes saved.")

    print("\n── Configuration complete. ──\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Update Scene dialogue (LLM part only)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_update_prompt(
    old_state: Dict, fresh_state: Dict,
    image_rename_map: Optional[Dict[str, str]] = None,
) -> str:
    from Configuration_Module.Update_Scene import build_update_context

    context = build_update_context(old_state, fresh_state, image_rename_map)
    if not isinstance(context, str):
        raise ValueError(f"build_update_context returned unexpected type: {type(context)}")

    return "\n".join([
        "Fresh vision scan compared to old scene:",
        "",
        context,
        "",
        "Call submit_mapping with only overrides",
        "(auto-matches are applied automatically).",
        "Use the part names as they appear in the image.",
        "Mapping entries:",
        "- To reassign: {\"<name_in_image>\": \"<desired_old_name>\"}",
        "- For new parts: {\"<name_in_image>\": \"new\"}",
        "- No overrides needed: {}",
        "",
        "No duplicate IDs allowed — if a reassignment conflicts with an auto-match,",
        "flag it and ask which part should keep the identity.",
        "",
        "Start by asking the user if the auto-matched IDs are correct.",
    ])


# ── Tool definition for structured mapping output ────────────────────────────

_SUBMIT_MAPPING_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_mapping",
        "description": (
            "Submit the part identity mapping overrides. Keys are the part "
            "names visible in the image (e.g. 'Part_6'), values are either "
            "an old part name (e.g. 'Part_1') or the literal string 'new'. "
            "Submit an empty object {} when no overrides are needed yet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Your conversational reply to the user (questions, "
                        "confirmations, clarifications). Always provide this."
                    ),
                },
                "mapping": {
                    "type": "object",
                    "description": (
                        "The mapping overrides. Each key is a Part_* name "
                        "from the image, each value is either an old Part_* "
                        "name or 'new'."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["message", "mapping"],
        },
    },
}


def _chat_update(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> tuple:
    """
    Call the LLM with the submit_mapping tool.

    Returns (message_text, mapping_dict).
    The LLM is forced to call submit_mapping via tool_choice, so we always
    get structured output — no regex parsing needed.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        tools=[_SUBMIT_MAPPING_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_mapping"}},
    )
    choice = resp.choices[0]

    # Extract the tool call arguments
    if choice.message.tool_calls:
        tc = choice.message.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        msg_text = args.get("message", "")
        mapping  = args.get("mapping", {})

        # Validate mapping values
        clean: Dict[str, str] = {}
        for k, v in mapping.items():
            if isinstance(k, str) and isinstance(v, str):
                clean[k] = v
        return msg_text, clean

    # Fallback: if the model returned plain text instead of a tool call
    text = (choice.message.content or "").strip()
    return text, {}


def run_update_dialogue(
    client: OpenAI,
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    image_rename_map: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """
    Run the LLM conversation for the update-scene flow.

    Called by session_handler.run_update_pipeline() AFTER vision and
    auto-matching are already done.

    Parameters
    ----------
    old_state        : previous configuration state
    fresh_state      : fresh vision scan state
    image_rename_map : {fresh_vision_name: image_label} from auto-matching

    Returns
    -------
    mapping : dict of user-confirmed overrides, or None if cancelled.
    """
    prompt_text = _build_update_prompt(old_state, fresh_state, image_rename_map)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You help match parts between vision scans.\n\n"
                "STYLE: Be brief. Do NOT describe or narrate the scene, do NOT "
                "explain what the auto-matcher did, do NOT list what parts are "
                "where. The user has a GUI and can see the scene. Just ask "
                "your questions and provide the mapping. One or two sentences "
                "max in the message field.\n\n"
                "FIRST MESSAGE: Your very first message must simply ask the "
                "user whether the auto-matched part identities look correct, "
                "and if not, to explain what changed. Do NOT guess what might "
                "be wrong, do NOT ask about specific parts or possible moves. "
                "Example: 'Do the auto-matched part IDs look correct? If not, "
                "tell me what changed.' Submit an empty mapping {} while "
                "waiting for the user's answer.\n\n"
                "ALWAYS call the submit_mapping tool. Put your conversational "
                "text in the 'message' field and the mapping overrides in the "
                "'mapping' field.\n\n"
                "CRITICAL THINKING — SANITY-CHECK USER CLAIMS:\n"
                "Before accepting any user statement, cross-check it against "
                "the INVENTORY SUMMARY and the SLOT-BY-SLOT COMPARISON. The "
                "camera sees what is physically there — it is ground truth.\n"
                "When the user's description implies something that doesn't "
                "add up with the physical evidence, point out the discrepancy "
                "and ask for clarification. Examples of discrepancies:\n"
                "- User says N parts of a colour were removed, but the count "
                "delta doesn't support that (would require new parts the user "
                "didn't mention).\n"
                "- User says a part was moved somewhere, but that slot is "
                "empty in the current scan (or occupied by a different colour).\n"
                "- User's total adds/removes don't match the actual delta in "
                "part count.\n"
                "- User says nothing was added, but the current scan has more "
                "parts than before.\n"
                "When you spot a discrepancy, ask a short clarifying question "
                "that states the concrete numbers. For example: 'We had 3 red "
                "parts and now see 2. If you removed 2 red parts, that would "
                "mean a new red part was also added — is that right, or did "
                "you mean you removed just 1?' Do NOT silently accept claims "
                "that conflict with the physical evidence. Do NOT block the "
                "user — if they confirm after your question, proceed with "
                "their answer.\n\n"
                "CONSTRAINT: Every part must have a unique ID. If a user "
                "override would create a duplicate, point out the conflict in "
                "one sentence and ask how to resolve it.\n\n"
                "IMAGE NAMES: The image shows auto-matched parts with their "
                "old-config names and new detections with fresh vision names. "
                "The mapping uses these image-visible names as keys. "
                "Use the names exactly as they appear in the analysis.\n\n"
                "UNDERSTANDING SWAPS AND MOVES:\n"
                "- 'Part_X was swapped with Part_Y' or 'X and Y switched "
                "places' means BOTH parts moved to each other's old location. "
                "The auto-matcher matches by slot, so after a swap it will "
                "have assigned the WRONG identity to both. You must CROSS-"
                "REASSIGN: map the image-name in Part_X's old slot to Part_Y, "
                "and the image-name in Part_Y's old slot to Part_X.\n"
                "- 'Part_X was moved to [slot]' means that part kept its "
                "identity but is in a new location. If the auto-matcher gave "
                "a new ID to the part now in that slot, map that new ID back "
                "to Part_X.\n"
                "- When parts are described as swapped/switched/moved, they "
                "were NOT removed and NOT newly added — never mark moved "
                "parts as \"new\".\n\n"
                "NEW DETECTIONS: Parts listed under NEW DETECTIONS are "
                "genuinely new. Include them as {\"<n>\": \"new\"} in your "
                "mapping.\n\n"
                "HANDLING SPATIAL REFERENCES: If the user refers to positions "
                "(left, right, top, bottom), consult the SLOT-BY-SLOT "
                "COMPARISON table to identify which part they mean. Remember "
                "the axis convention: LARGER X = LEFT, LARGER Y = LOWER."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]

    print("\n── Update Scene Dialogue ──\n")

    mapping: Dict[str, str] = {}

    while True:
        msg_text, mapping = _chat_update(client, messages)
        print(f"ASSISTANT:\n{msg_text}\n")
        if mapping:
            print(f"  [Mapping: {json.dumps(mapping)}]")

        # Record the assistant turn in conversation history.
        messages.append({"role": "assistant", "content": msg_text})

        user = input("YOU: ").strip()
        if not user or is_yes(user) or user.lower() in ("done", "skip"):
            if mapping:
                print(f"\n  Applying mapping: {json.dumps(mapping)}")
            else:
                print("\n  No overrides — accepting auto-matches only.")
            return mapping
        if is_no(user):
            messages.append({"role": "user", "content": "Rejected. Ask me again."})
            continue
        messages.append({"role": "user", "content": user})

    # Should not be reached, but for safety
    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# Main conversation loop  (reconfiguration / motion planning)
# ═══════════════════════════════════════════════════════════════════════════════

def run_conversation(
    client: OpenAI,
    mode: str,
    scene: Dict[str, Any],
) -> None:
    """
    The LLM conversation loop for reconfiguration or motion-planning mode.

    Called by session_handler.run_session() after the scene has been loaded.
    All vision, execution, and config-loading logic is handled by the caller.
    """
    import session_handler as sh

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(mode)},
        {
            "role": "user",
            "content": (
                "SCENE JSON:\n"
                + json.dumps(scene, indent=2, ensure_ascii=False)
                + "\n\nAsk what I want to do."
            ),
        },
    ]

    pending_sequence: Optional[List[List]] = None
    pending_changes: Optional[Dict[str, Any]] = None
    accumulated_changes: Dict[str, Any] = {}

    mode_label = "Reconfiguration" if mode == "reconfig" else "Motion Sequence"
    print(f"\n── Mode: {mode_label} ──\n")

    while True:
        # ── Get LLM response ─────────────────────────────────────────────
        assistant_text = chat(client, messages, scene=scene)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        if "SWITCH_TO_RECONFIG" in assistant_text:
            print("  [Role conflict — returning to mode selection.]\n")
            return

        # ── Try extracting blocks from response ──────────────────────────
        pending_sequence = _try_extract_sequence(assistant_text, messages)
        pending_changes = _try_extract_changes(assistant_text, messages)

        # If a parse error triggered a retry message, skip to next LLM turn
        if pending_sequence is False or pending_changes is False:
            pending_sequence = None if pending_sequence is False else pending_sequence
            pending_changes = None if pending_changes is False else pending_changes
            continue

        # ── Priority ambiguity guard ─────────────────────────────────────
        if pending_changes and "priority" in pending_changes:
            state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
            color_map = {
                e["part"]: (e.get("color") or "").lower()
                for e in state.get("predicates", {}).get("color", [])
            }
            ambiguities = detect_priority_ambiguity(pending_changes["priority"], color_map)
            if ambiguities:
                msg = format_priority_ambiguities(ambiguities)
                print(f"  [Priority ambiguity — asking LLM to clarify]\n{msg}")
                messages.append({
                    "role": "user",
                    "content": (
                        "Before I confirm, there's a conflict in your priority proposal:\n\n"
                        + msg
                        + "\n\nDo NOT output a changes block yet. "
                        "Ask me the clarification question above first."
                    ),
                })
                pending_changes = None
                continue

        # ── Capacity validation guard ────────────────────────────────────
        if pending_changes:
            cap_warning = _validate_capacity(pending_changes)
            if cap_warning:
                print(f"  [Capacity check failed]\n  {cap_warning}")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Before I confirm — there is a capacity problem with "
                        f"your proposal:\n\n{cap_warning}\n\n"
                        "Do NOT output a changes block yet. "
                        "Ask the user how to resolve this (e.g. leave some "
                        "parts unsorted, use a different container, etc.)."
                    ),
                })
                pending_changes = None
                continue

        # ── Get user input ───────────────────────────────────────────────
        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        # ── Check for session end ────────────────────────────────────────
        if is_finish(user_input):
            _finalize_session(client, accumulated_changes, pending_sequence, pending_changes)
            return

        has_pending = pending_sequence is not None or pending_changes is not None

        # ── Handle confirmation ──────────────────────────────────────────
        if has_pending and is_yes(user_input):
            accumulated_changes = _confirm_pending(
                client, messages, accumulated_changes,
                pending_sequence, pending_changes,
            )
            pending_sequence = None
            pending_changes = None

            user_input = input("Anything else? If not, type or press 'done'.\nYOU: ").strip()
            if is_finish(user_input):
                if accumulated_changes:
                    sh.save_changes(accumulated_changes)
                    sh.apply_and_save_config(accumulated_changes)
                    print("✅  Changes saved.")
                print("\n── Configuration complete. ──\n")
                return
            if user_input:
                # Run ambiguity detection on the follow-up instruction too
                ambiguity_result = detect_ambiguity(
                    client, MODEL, user_input, scene, mode,
                    conversation_history=messages,
                )
                if ambiguity_result:
                    hint = format_ambiguity_hint(ambiguity_result)
                    if hint:
                        messages.append({"role": "system", "content": hint})
                        print("  [Ambiguity detected — guiding LLM to ask targeted question]")
                messages.append({"role": "user", "content": user_input})
            continue

        # ── Handle rejection ─────────────────────────────────────────────
        if has_pending and is_no(user_input):
            pending_sequence = None
            pending_changes = None
            messages.append({
                "role": "user",
                "content": "Rejected. Discard that proposal and ask what I want to change.",
            })
            continue

        # ── Regular user message ─────────────────────────────────────────
        # Run background ambiguity detection against the scene
        ambiguity_result = detect_ambiguity(
            client, MODEL, user_input, scene, mode,
            conversation_history=messages,
        )
        if ambiguity_result:
            hint = format_ambiguity_hint(ambiguity_result)
            if hint:
                messages.append({"role": "system", "content": hint})
                print("  [Ambiguity detected — guiding LLM to ask targeted question]")

        messages.append({"role": "user", "content": user_input})