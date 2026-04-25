"""
api_main.py — LLM conversation engine.

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
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from Core.config import settings
from Core.paths import CONFIGURATION_PATH
from Core.io_helpers import save_sequence, save_changes, save_config_to_memory
from Communication_Module.block_parsing import (
    extract_changes_block,
    extract_mapping_block,
    extract_sequence_block,
)
from Communication_Module.change_management import (
    detect_conflicts,
    detect_priority_ambiguity,
    format_conflicts_for_user,
    format_priority_ambiguities,
    interpret_conflict_resolution,
    merge_changes,
    resolve_conflicts,
)
from Communication_Module.prompts import build_system_prompt
from Communication_Module.user_intent import (
    classify_pending_reply,
    extract_yes_with_extra,
    is_finish,
    is_no,
    is_yes,
    resolve_pending_reply,
)
from Communication_Module.ambiguity_detection import (
    detect_ambiguity,
    format_ambiguity_hint,
)
from Communication_Module.capacity_tools import (
    CAPACITY_TOOL_SCHEMA,
    DESCRIBE_SCENE_TOOL_SCHEMA,
    execute_capacity_check,
    execute_describe_scene,
)

logger = logging.getLogger(__name__)

# ── Cancel event (set by the GUI when the user presses Done during LLM call) ─
_cancel_event: Optional[threading.Event] = None


class LLMCancelled(Exception):
    """Raised when an in-flight LLM call is aborted by the user."""


def set_cancel_event(event: threading.Event) -> None:
    """Register the cancel event from the GUI layer."""
    global _cancel_event
    _cancel_event = event


def _check_cancelled() -> None:
    """Raise LLMCancelled if the cancel event has been set."""
    if _cancel_event is not None and _cancel_event.is_set():
        raise LLMCancelled("LLM call cancelled by user")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM call helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cancellable_api_call(client: OpenAI, **kwargs) -> Any:
    """Run a blocking OpenAI API call in a sub-thread so the cancel event
    can interrupt the wait.  Returns the response or raises LLMCancelled."""
    result: list = [None]
    error: list = [None]
    done = threading.Event()

    def _call() -> None:
        try:
            result[0] = client.chat.completions.create(**kwargs)
        except Exception as exc:
            error[0] = exc
        finally:
            done.set()

    t = threading.Thread(target=_call, daemon=True)
    t.start()

    while not done.wait(timeout=0.25):
        _check_cancelled()

    _check_cancelled()

    if error[0] is not None:
        raise error[0]
    return result[0]


def chat(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    scene: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call the LLM, handling optional tool use.

    If ``scene`` is provided, the capacity-check and describe-scene tools are
    offered.  When the LLM calls one, the tool is executed locally and the
    result is fed back so the LLM can continue.  This loop runs at most
    ``settings.max_tool_rounds`` times to avoid runaway calls.

    Raises LLMCancelled if the user presses Done during the call.
    """
    tools = (
        [CAPACITY_TOOL_SCHEMA, DESCRIBE_SCENE_TOOL_SCHEMA]
        if scene is not None
        else None
    )
    working_messages = list(messages)

    for _ in range(settings.max_tool_rounds + 1):
        _check_cancelled()

        kwargs: Dict[str, Any] = dict(
            model=settings.model,
            messages=working_messages,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools

        resp = _cancellable_api_call(client, **kwargs)
        choice = resp.choices[0]

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return (choice.message.content or "").strip()

        working_messages.append(choice.message)

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "check_capacity" and scene is not None:
                tool_result = execute_capacity_check(args, scene)
                logger.debug("Tool: check_capacity → result injected")
            elif tc.function.name == "describe_scene" and scene is not None:
                tool_result = execute_describe_scene(args, scene)
                logger.debug("Tool: describe_scene → result injected")
            else:
                tool_result = f"Unknown tool: {tc.function.name}"

            working_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    return (choice.message.content or "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Capacity validation (used mid-conversation)
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_capacity(changes: Dict[str, Any]) -> Optional[str]:
    """
    Check whether a proposed changes block would route more parts to an output
    receptacle than it has empty slots.

    Returns a human-readable warning string, or None if everything fits.
    """
    if not CONFIGURATION_PATH.exists():
        return None

    try:
        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    preds = state.get("predicates", {})
    objs = state.get("objects", {})
    s2p = state.get("slot_belongs_to", {})

    role_map: Dict[str, Optional[str]] = {
        e["object"]: e.get("role") for e in preds.get("role", [])
    }
    for key, val in changes.items():
        if isinstance(val, dict) and "role" in val:
            role_map[key] = val["role"]

    all_recs = set(objs.get("kits", []) + objs.get("containers", []))
    output_recs = {r for r in all_recs if role_map.get(r) == "output"}
    input_recs = {r for r in all_recs if role_map.get(r) == "input"}

    if not output_recs:
        return None

    empty_slots: Dict[str, int] = {}
    for rec in output_recs:
        rec_slots = [s for s, p in s2p.items() if p == rec]
        occupied = {e["slot"] for e in preds.get("at", []) if e["slot"] in rec_slots}
        empty_slots[rec] = len(rec_slots) - len(occupied)

    color_of: Dict[str, str] = {
        e["part"]: (e.get("color") or "unknown").lower()
        for e in preds.get("color", [])
    }

    part_at: Dict[str, str] = {e["part"]: e["slot"] for e in preds.get("at", [])}
    input_parts: List[str] = [
        part for part, slot in part_at.items() if s2p.get(slot) in input_recs
    ]
    in_slot_set = set(part_at.keys())
    for p in objs.get("parts", []):
        if p not in in_slot_set:
            input_parts.append(p)

    op_mode = (
        (changes.get("workspace") or {}).get("operation_mode")
        or (state.get("workspace") or {}).get("operation_mode")
    )

    problems: List[str] = []

    if op_mode == "sorting":
        compat = changes.get("part_compatibility") or preds.get("part_compatibility", [])
        routed: Dict[str, int] = {r: 0 for r in output_recs}
        for part in input_parts:
            pc = color_of.get(part, "unknown")
            destinations: List[str] = []
            for rule in compat:
                match = True
                if "part_color" in rule and rule["part_color"].lower() != pc:
                    match = False
                if "part_name" in rule and rule["part_name"] != part:
                    match = False
                if match and "allowed_in" in rule:
                    destinations.extend(r for r in rule["allowed_in"] if r in output_recs)
            if not destinations:
                for rule in compat:
                    if "part_color" not in rule and "part_name" not in rule:
                        if "allowed_in" in rule:
                            destinations.extend(
                                r for r in rule["allowed_in"] if r in output_recs
                            )
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

    proposed_ws = changes.get("workspace") or {}
    current_ws = state.get("workspace") or {}
    batch_size = proposed_ws.get("batch_size") or current_ws.get("batch_size")
    num_kits = len(objs.get("kits", []))
    if batch_size is not None and num_kits > 0:
        if not isinstance(batch_size, int) or batch_size <= 0:
            problems.append(f"Batch size must be a positive integer (got {batch_size!r}).")
        elif batch_size > num_kits:
            problems.append(
                f"Batch size {batch_size} exceeds the number of available kits "
                f"({num_kits}). Please set a batch size of {num_kits} or fewer."
            )

    if op_mode == "kitting":
        recipe = changes.get("kit_recipe") or preds.get("kit_recipe", [])
        if recipe:
            for entry in recipe:
                qty = entry.get("quantity", 0)
                color = entry.get("color", "unknown")
                if qty <= 0:
                    problems.append(
                        f"Kit recipe quantity for '{color}' is {qty}. "
                        f"Each part in the recipe must have a quantity of at least 1."
                    )
            total_per_kit = sum(e.get("quantity", 0) for e in recipe)
            if total_per_kit > 3:
                problems.append(
                    f"Kit recipe total is {total_per_kit} parts, "
                    f"but the maximum allowed is 3."
                )
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
    keep_new = interpret_conflict_resolution(client, settings.model, resolve_input, conflicts)
    result = resolve_conflicts(accumulated, pending, conflicts, keep_new)
    print("✅  Using new values.\n" if keep_new else "✅  Keeping previous values.\n")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LLM response extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _try_extract_sequence(text: str, messages: List[Dict[str, str]]) -> Any:
    """Try to extract a sequence block. Returns the sequence, None, or False (parse error)."""
    try:
        return extract_sequence_block(text)
    except Exception as exc:
        if "```sequence" in (text or ""):
            logger.warning("Sequence block parse error: %s", exc)
            messages.append({
                "role": "user",
                "content": (
                    f"Your sequence block failed to parse: {exc}\n"
                    'Each entry must be ["pick", "place"].\n'
                    "Please rewrite the sequence block."
                ),
            })
            return False
        return None


def _try_extract_changes(text: str, messages: List[Dict[str, str]]) -> Any:
    """Try to extract a changes block. Returns the changes, None, or False (parse error)."""
    try:
        return extract_changes_block(text)
    except Exception as exc:
        if "```changes" in (text or ""):
            logger.warning("Changes block parse error: %s", exc)
            messages.append({
                "role": "user",
                "content": (
                    f"Your changes block failed to parse: {exc}\n"
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

def _describe_pending(
    pending_sequence: Optional[List[List]],
    pending_changes: Optional[Dict[str, Any]],
) -> str:
    """Short one-liner describing what's pending, for the reply classifier."""
    parts = []
    if pending_changes is not None:
        keys = ", ".join(sorted(pending_changes.keys())) or "(empty)"
        parts.append(f"a changes block with keys: {keys}")
    if pending_sequence is not None:
        parts.append(f"a sequence block with {len(pending_sequence)} pick-and-place steps")
    return " and ".join(parts) if parts else "a proposal awaiting confirmation"


def _confirm_pending(
    client: OpenAI,
    messages: List[Dict[str, str]],
    accumulated: Dict[str, Any],
    pending_sequence: Optional[List[List]],
    pending_changes: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle confirmation of pending sequence and/or changes. Returns updated accumulated."""
    if pending_sequence is not None:
        save_sequence(pending_sequence)
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
    # apply_and_save_config lives in session_handler (it orchestrates vision +
    # config modules) — imported late to avoid the circular import chain.
    from Orchestration.session_handler import apply_and_save_config

    if pending_sequence is not None:
        save_sequence(pending_sequence)
        print("✅  Sequence saved.")

    if pending_changes is not None:
        accumulated = _handle_conflicts(client, accumulated, pending_changes)

    if accumulated:
        save_changes(accumulated)
        apply_and_save_config(accumulated)
    else:
        if CONFIGURATION_PATH.exists():
            state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
            save_config_to_memory(state)

    print("\n── Configuration complete. ──\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Update Scene dialogue (LLM part only)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_update_prompt(
    old_state: Dict,
    fresh_state: Dict,
    image_rename_map: Optional[Dict[str, str]] = None,
) -> str:
    from Configuration_Module.update_scene import build_update_context

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
        '- To reassign to existing identity: {"<name_in_image>": "<old_Part_N>"}',
        '- For new part with auto-assigned ID: {"<name_in_image>": "new"}',
        '- For new part with specific ID:      {"<name_in_image>": "Part_8"}',
        "- No overrides to propose right now: {}",
        "",
        "When proposing a mapping, include the overrides in the tool call.",
        "Confirmed mappings are accumulated — the user can add more changes",
        "before saying 'done' to apply everything.",
        "",
        "No duplicate IDs allowed — if a reassignment conflicts with an auto-match,",
        "flag it and ask which part should keep the identity.",
        "",
        "Start by asking the user if the auto-matched IDs are correct.",
    ])


_SUBMIT_MAPPING_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_mapping",
        "description": (
            "Submit the part identity mapping overrides. Keys are the part "
            "names visible in the image (e.g. 'Part_6'), values are either "
            "an existing old part name (e.g. 'Part_1') for reassignment, "
            "the literal string 'new' for auto-assigned ID, or a specific "
            "Part_N (e.g. 'Part_8') for a custom new ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Your conversational reply to the user.",
                },
                "mapping": {
                    "type": "object",
                    "description": "The mapping overrides.",
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
) -> Tuple[str, Dict[str, str]]:
    """Call the LLM with the submit_mapping tool. Returns (message_text, mapping_dict)."""
    _check_cancelled()
    resp = _cancellable_api_call(
        client,
        model=settings.model,
        messages=messages,
        temperature=temperature,
        tools=[_SUBMIT_MAPPING_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_mapping"}},
    )
    choice = resp.choices[0]

    if choice.message.tool_calls:
        tc = choice.message.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        msg_text = args.get("message", "")
        mapping = args.get("mapping", {})

        clean: Dict[str, str] = {
            k: v for k, v in mapping.items()
            if isinstance(k, str) and isinstance(v, str)
        }

        if not clean and msg_text:
            fallback = _extract_mapping_from_text(msg_text)
            if fallback:
                logger.debug("Mapping extracted from message text (tool field was empty)")
                clean = fallback

        return msg_text, clean

    text = (choice.message.content or "").strip()
    return text, {}


def _extract_mapping_from_text(text: str) -> Dict[str, str]:
    """Try to extract a Part_* mapping from the LLM's message text."""
    pattern = r'\{[^{}]*"Part_\d+"[^{}]*\}'
    for match in re.findall(pattern, text):
        try:
            candidate = json.loads(match)
            if isinstance(candidate, dict):
                clean = {
                    k: v for k, v in candidate.items()
                    if isinstance(k, str) and isinstance(v, str) and k.startswith("Part_")
                }
                if clean:
                    return clean
        except json.JSONDecodeError:
            continue
    return {}


def _validate_mapping_proposal(
    proposal: Dict[str, str],
    accumulated: Dict[str, str],
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    image_rename_map: Dict[str, str],
) -> List[str]:
    """Validate a proposed mapping batch. Returns a list of issue strings."""
    from Configuration_Module.update_scene import _match_parts_by_position

    issues: List[str] = []
    old_parts = set(old_state.get("objects", {}).get("parts", []))

    auto_matched, _new, _missing = _match_parts_by_position(old_state, fresh_state)
    current_assignments: Dict[str, str] = {}
    for old_name, fresh_name in auto_matched:
        img_label = image_rename_map.get(fresh_name, old_name)
        current_assignments[img_label] = old_name
    for img_key, target in accumulated.items():
        current_assignments[img_key] = target

    identity_holders: Dict[str, str] = {
        identity: img_label
        for img_label, identity in current_assignments.items()
        if identity != "new"
    }

    proposal_targets: Dict[str, List[str]] = {}
    for img_key, target in proposal.items():
        if target != "new" and not re.match(r"^Part_\d+$", target):
            issues.append(
                f"'{img_key}' → '{target}': invalid format (must be Part_N or 'new')."
            )
            continue

        if target == "new":
            continue

        proposal_targets.setdefault(target, []).append(img_key)

        if target in identity_holders:
            current_holder = identity_holders[target]
            if current_holder != img_key and current_holder not in proposal:
                issues.append(
                    f"'{img_key}' → '{target}': ID '{target}' is "
                    f"currently assigned to '{current_holder}'. "
                    f"What should happen to '{current_holder}'?"
                )

        if target not in old_parts:
            acc_targets = set(accumulated.values()) - {"new"}
            if target in acc_targets:
                claimer = next((k for k, v in accumulated.items() if v == target), "?")
                issues.append(
                    f"'{img_key}' → '{target}': ID '{target}' was "
                    f"already assigned to '{claimer}' in a previous mapping."
                )

    for target, keys in proposal_targets.items():
        if len(keys) > 1:
            issues.append(
                f"Duplicate target: {', '.join(keys)} all map to '{target}'. "
                f"Each part needs a unique ID."
            )

    return issues


def run_update_dialogue(
    client: OpenAI,
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    image_rename_map: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """
    Run the LLM conversation for the update-scene flow.

    Called by session_handler.run_update_pipeline() after vision and
    auto-matching are already done.

    Returns a dict of user-confirmed overrides, or None if cancelled.
    """
    prompt_text = _build_update_prompt(old_state, fresh_state, image_rename_map)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You help match parts between vision scans.\n\n"
                "STYLE: Be brief. Do NOT narrate the scene. Just ask questions "
                "and provide the mapping.\n\n"
                "FLOW: Ask if auto-matched IDs are correct → propose mappings → "
                "accumulate confirmations → user says 'done' to finish.\n\n"
                "ALWAYS call the submit_mapping tool. Put conversational text in "
                "the 'message' field and mapping overrides in the 'mapping' field.\n\n"
                "CRITICAL: If you describe a mapping in your message and ask "
                "'Confirm?', you MUST also put it in the mapping field — otherwise "
                "nothing gets saved.\n\n"
                "CRITICAL: Once a mapping has been confirmed and saved, do NOT "
                "include it in the mapping field again. Only put NEW, unconfirmed "
                "overrides in the mapping field. Mentioning saved mappings in your "
                "message text is fine, but the mapping field must only contain "
                "fresh proposals awaiting confirmation."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]

    print("\n── Update Scene Dialogue ──\n")
    accumulated_mapping: Dict[str, str] = {}

    try:
        while True:
            msg_text, turn_mapping = _chat_update(client, messages)
            print(f"ASSISTANT:\n{msg_text}\n")

            # Strip entries that are already saved — the LLM sometimes echoes
            # previously confirmed mappings in the tool field when summarising.
            # Those must not be treated as new pending proposals.
            turn_mapping = {
                k: v for k, v in turn_mapping.items()
                if not (k in accumulated_mapping and accumulated_mapping[k] == v)
            }

            if turn_mapping:
                logger.debug("Proposed mapping: %s", json.dumps(turn_mapping))

            messages.append({"role": "assistant", "content": msg_text})

            # Show the proposed mapping to the user before asking for input
            if turn_mapping:
                print("  Proposed mapping:")
                for img_label, target in turn_mapping.items():
                    print(f"    {img_label}  →  {target}")
                print()

            user = input("YOU: ").strip()

            if not user or user.lower() in ("done", "skip"):
                if accumulated_mapping:
                    print(f"\n  Applying mapping: {json.dumps(accumulated_mapping)}")
                else:
                    print("\n  No overrides — accepting auto-matches only.")
                return accumulated_mapping

            # Only treat the reply as yes/no when the assistant actually
            # proposed a concrete mapping.  If it just asked an open question
            # (turn_mapping is empty), pass the user's message straight through
            # so instructions like "part 15 is actually part 33" are never
            # swallowed by the yes/no classifier.
            if turn_mapping:
                pending_summary = (
                    f"a mapping proposal with {len(turn_mapping)} override(s)"
                )
                decision, user = resolve_pending_reply(
                    client, settings.model, user, pending_summary,
                )
            else:
                decision = "other"

            if decision == "yes":
                validation_issues = _validate_mapping_proposal(
                    turn_mapping, accumulated_mapping,
                    old_state, fresh_state, image_rename_map,
                )
                if validation_issues:
                    messages.append({
                        "role": "user",
                        "content": (
                            "The system detected issues with the proposed mapping:\n"
                            + "\n".join(f"- {i}" for i in validation_issues)
                            + "\nPlease resolve these with me before re-proposing."
                        ),
                    })
                    continue

                for k, v in turn_mapping.items():
                    if k in accumulated_mapping and accumulated_mapping[k] != v:
                        logger.info(
                            "Override conflict: '%s' was '%s', now '%s' (using latest)",
                            k, accumulated_mapping[k], v,
                        )
                accumulated_mapping.update(turn_mapping)
                logger.debug("Accumulated mapping: %s", json.dumps(accumulated_mapping))
                print(f"✅  Mapping saved. ({len(accumulated_mapping)} override(s) so far)\n")

                # Check whether the user also included a new instruction alongside
                # the confirmation (e.g. "yes and part 14 should be part 22").
                # If so, strip leading confirmation words and forward the rest.
                extra = re.sub(
                    r"^(yes|yep|yeah|yup|sure|ok|okay|correct|confirmed?|good|great)"
                    r"[\s,;.!]*(?:and\s*)?",
                    "",
                    user,
                    flags=re.IGNORECASE,
                ).strip()

                acc_summary = (
                    ", ".join(f"{k} → {v}" for k, v in accumulated_mapping.items())
                    if accumulated_mapping else "none yet"
                )
                if extra:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Confirmed. Saved mappings so far: {acc_summary}. "
                            f"Additional instruction: {extra}"
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Confirmed. Saved mappings so far: {acc_summary}. "
                            "Ask if there are more changes, or the user can say 'done'."
                        ),
                    })
                continue

            if decision == "no":
                messages.append({"role": "user", "content": "Rejected. Ask me again."})
                continue

            messages.append({"role": "user", "content": user})

    except LLMCancelled:
        print("\n── LLM call aborted by user. ──\n")
        return accumulated_mapping if accumulated_mapping else None

    return accumulated_mapping


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
    """
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

    mode_label = "Reconfiguration" if mode == "reconfig" else "Motion Sequence"
    print(f"\n── Mode: {mode_label} ──\n")

    try:
        _run_conversation_loop(client, messages, mode, scene)
    except LLMCancelled:
        print("\n── LLM call aborted by user. ──\n")


def _run_conversation_loop(
    client: OpenAI,
    messages: List[Dict[str, str]],
    mode: str,
    scene: Dict[str, Any],
) -> None:
    """Inner loop extracted so LLMCancelled can be caught cleanly."""
    from Orchestration.session_handler import apply_and_save_config  # only orchestration fn still needed here

    pending_sequence: Optional[List[List]] = None
    pending_changes: Optional[Dict[str, Any]] = None
    accumulated_changes: Dict[str, Any] = {}

    while True:
        assistant_text = chat(client, messages, scene=scene)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        if "SWITCH_TO_RECONFIG" in assistant_text:
            logger.info("Role conflict — returning to mode selection.")
            return

        pending_sequence = _try_extract_sequence(assistant_text, messages)
        pending_changes = _try_extract_changes(assistant_text, messages)

        if pending_sequence is False or pending_changes is False:
            pending_sequence = None if pending_sequence is False else pending_sequence
            pending_changes = None if pending_changes is False else pending_changes
            continue

        if pending_changes and "priority" in pending_changes:
            state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
            color_map = {
                e["part"]: (e.get("color") or "").lower()
                for e in state.get("predicates", {}).get("color", [])
            }
            ambiguities = detect_priority_ambiguity(pending_changes["priority"], color_map)
            if ambiguities:
                msg = format_priority_ambiguities(ambiguities)
                logger.debug("Priority ambiguity — asking LLM to clarify")
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

        if pending_changes:
            cap_warning = _validate_capacity(pending_changes)
            if cap_warning:
                logger.debug("Capacity check failed")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Before I confirm — there is a capacity problem:\n\n"
                        f"{cap_warning}\n\n"
                        "Do NOT output a changes block yet. "
                        "Ask the user how to resolve this."
                    ),
                })
                pending_changes = None
                continue

        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        if is_finish(user_input):
            _finalize_session(client, accumulated_changes, pending_sequence, pending_changes)
            return

        has_pending = pending_sequence is not None or pending_changes is not None

        if has_pending:
            pending_summary = _describe_pending(pending_sequence, pending_changes)
            decision, user_input = resolve_pending_reply(
                client, settings.model, user_input, pending_summary,
            )
        else:
            decision = "other"

        if decision == "yes":
            accumulated_changes = _confirm_pending(
                client, messages, accumulated_changes,
                pending_sequence, pending_changes,
            )
            pending_sequence = None
            pending_changes = None

            # user_input may contain an extra instruction from "yes but <extra>"
            # (resolve_pending_reply returns the extra part as user_input).
            # If so, skip "Anything else?" and feed it directly into the loop.
            extra_from_yes = user_input if (user_input and not is_yes(user_input)) else ""

            if extra_from_yes:
                user_input = extra_from_yes
            else:
                print("ASSISTANT:\nAnything else? If not, type or press 'done'.\n")
                user_input = input("YOU: ").strip()
            if is_finish(user_input):
                if accumulated_changes:
                    save_changes(accumulated_changes)
                    apply_and_save_config(accumulated_changes)
                else:
                    if CONFIGURATION_PATH.exists():
                        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
                        save_config_to_memory(state)
                print("\n── Configuration complete. ──\n")
                return
            if user_input:
                ambiguity_result = detect_ambiguity(
                    client, settings.model, user_input, scene, mode,
                    conversation_history=messages,
                )
                if ambiguity_result:
                    hint = format_ambiguity_hint(ambiguity_result)
                    if hint:
                        messages.append({"role": "system", "content": hint})
                        logger.debug("Ambiguity detected — guiding LLM to ask targeted question")
                messages.append({"role": "user", "content": user_input})
            continue

        if decision == "no":
            pending_sequence = None
            pending_changes = None
            messages.append({
                "role": "user",
                "content": "Rejected. Discard that proposal and ask what I want to change.",
            })
            continue

        ambiguity_result = detect_ambiguity(
            client, settings.model, user_input, scene, mode,
            conversation_history=messages,
        )
        if ambiguity_result:
            hint = format_ambiguity_hint(ambiguity_result)
            if hint:
                messages.append({"role": "system", "content": hint})
                logger.debug("Ambiguity detected — guiding LLM to ask targeted question")

        messages.append({"role": "user", "content": user_input})