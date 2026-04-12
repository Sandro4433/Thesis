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
        "- To reassign to existing identity: {\"<name_in_image>\": \"<old_Part_N>\"}",
        "  (key = what image shows, value = true identity from old config)",
        "- For new part with auto-assigned ID: {\"<name_in_image>\": \"new\"}",
        "- For new part with specific ID:      {\"<name_in_image>\": \"Part_8\"}",
        "  (use any Part_N not already taken — the system validates this)",
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


# ── Tool definition for structured mapping output ────────────────────────────

_SUBMIT_MAPPING_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_mapping",
        "description": (
            "Submit the part identity mapping overrides. Keys are the part "
            "names visible in the image (e.g. 'Part_6'), values are either "
            "an existing old part name (e.g. 'Part_1') for reassignment, "
            "the literal string 'new' for auto-assigned ID, or a specific "
            "Part_N (e.g. 'Part_8') for a custom new ID. "
            "When proposing overrides for user confirmation, include them "
            "here — confirmed mappings are accumulated by the system. "
            "Submit {} when you have no overrides to propose (e.g. first "
            "message, after confirmation, or while asking clarifications)."
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
                        "name (reassignment), a new Part_N not in old config "
                        "(custom new ID), or 'new' (auto-assign)."
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

    Defensive fallback: if the LLM puts a mapping in its message text but
    leaves the mapping field empty (a common compliance failure), we extract
    the mapping from the message text.
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

        # ── Defensive fallback ───────────────────────────────────────
        # If the mapping field is empty but the message text contains
        # a JSON object with Part_* keys (the LLM described a mapping
        # in text but forgot to put it in the tool field), extract it.
        if not clean and msg_text:
            fallback = _extract_mapping_from_text(msg_text)
            if fallback:
                print("  [Note: mapping extracted from message text (tool field was empty)]")
                clean = fallback

        return msg_text, clean

    # Fallback: if the model returned plain text instead of a tool call
    text = (choice.message.content or "").strip()
    return text, {}


def _extract_mapping_from_text(text: str) -> Dict[str, str]:
    """
    Try to extract a Part_* mapping from the LLM's message text.

    Looks for JSON-like patterns such as:
      {"Part_5": "Part_3"}
      {"Part_7": "new", "Part_4": "Part_5"}

    Returns a validated mapping dict, or {} if nothing found.
    """
    import re
    # Find JSON objects in the text that contain Part_ keys
    pattern = r'\{[^{}]*"Part_\d+"[^{}]*\}'
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            candidate = json.loads(match)
            if isinstance(candidate, dict):
                clean: Dict[str, str] = {}
                for k, v in candidate.items():
                    if (isinstance(k, str) and isinstance(v, str)
                            and k.startswith("Part_")):
                        clean[k] = v
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
    """
    Validate a proposed mapping batch against the current scene state and
    any previously accumulated overrides.

    Checks performed:
      1. Format: values must be "new", or match Part_N.
      2. Displacement: if the target ID is currently held by another part
         (via auto-match or accumulated mapping) and this proposal doesn't
         also reassign that displaced part, flag it.
      3. Duplicate targets: two entries in the same proposal (or across
         accumulated + proposal) mapping to the same target ID.
      4. Custom new ID collision: a Part_N that isn't in the old config
         but IS already assigned as a target elsewhere.

    Returns a list of human-readable issue strings.  Empty list = all OK.
    """
    from Configuration_Module.Update_Scene import _match_parts_by_position

    issues: List[str] = []
    old_parts = set(old_state.get("objects", {}).get("parts", []))

    # Build the current ID assignment picture:
    # image_label → assigned_identity
    # Start with auto-matches (image shows old name → identity is old name)
    auto_matched, _new, _missing = _match_parts_by_position(old_state, fresh_state)
    # image_rename_map: {fresh_vision_name: image_label}
    # auto_matched: [(old_name, fresh_name), ...]

    # current_assignments: {image_label: identity}
    # For auto-matched parts, the image label IS the old name (identity).
    current_assignments: Dict[str, str] = {}
    for old_name, fresh_name in auto_matched:
        img_label = image_rename_map.get(fresh_name, old_name)
        current_assignments[img_label] = old_name

    # Layer accumulated overrides on top
    for img_key, target in accumulated.items():
        current_assignments[img_key] = target

    # Build reverse: identity → image_label (who currently holds each ID)
    identity_holders: Dict[str, str] = {}
    for img_label, identity in current_assignments.items():
        if identity != "new":
            identity_holders[identity] = img_label

    # ── Check each entry in the proposal ────────────────────────────────
    # Collect all targets in this proposal for duplicate detection
    proposal_targets: Dict[str, List[str]] = {}  # target → [keys]

    for img_key, target in proposal.items():
        # 1. Format check
        if target != "new" and not re.match(r'^Part_\d+$', target):
            issues.append(
                f"'{img_key}' → '{target}': invalid format "
                f"(must be Part_N or 'new')."
            )
            continue

        if target == "new":
            continue

        # Track for duplicate detection
        proposal_targets.setdefault(target, []).append(img_key)

        # 2. Displacement check: is this target ID currently held by a
        #    DIFFERENT image-label that is NOT also being reassigned in
        #    this proposal?
        if target in identity_holders:
            current_holder = identity_holders[target]
            if current_holder != img_key:
                # Someone else holds this ID. Is that someone being
                # reassigned in this same proposal?
                if current_holder not in proposal:
                    issues.append(
                        f"'{img_key}' → '{target}': ID '{target}' is "
                        f"currently assigned to '{current_holder}'. "
                        f"What should happen to '{current_holder}'? "
                        f"(reassign it to a different ID, or mark it "
                        f"as 'new'?)"
                    )

        # 4. Custom new ID collision with accumulated targets
        if target not in old_parts:
            acc_targets = set(accumulated.values()) - {"new"}
            if target in acc_targets:
                # Find who already claimed it
                claimer = next(
                    (k for k, v in accumulated.items() if v == target), "?"
                )
                issues.append(
                    f"'{img_key}' → '{target}': ID '{target}' was "
                    f"already assigned to '{claimer}' in a previous "
                    f"mapping."
                )

    # 3. Duplicate targets within this proposal
    for target, keys in proposal_targets.items():
        if len(keys) > 1:
            issues.append(
                f"Duplicate target: {', '.join(keys)} all map to "
                f"'{target}'. Each part needs a unique ID."
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

    Called by session_handler.run_update_pipeline() AFTER vision and
    auto-matching are already done.

    The dialogue follows an accumulate-then-apply pattern:
      1. LLM asks if auto-matches are correct.
      2. User describes a change → LLM proposes mapping → user confirms
         → mapping is accumulated (not yet applied).
      3. LLM asks "Any more changes?"
      4. User describes more → repeat from 2, or says "done" → all
         accumulated mappings are applied at once.

    Parameters
    ----------
    old_state        : previous configuration state
    fresh_state      : fresh vision scan state
    image_rename_map : {fresh_vision_name: image_label} from auto-matching

    Returns
    -------
    mapping : dict of user-confirmed overrides, None if cancelled,
              or the string "__RECAPTURE__" if the user wants a new scan.
    """
    from session_handler import RECAPTURE_SENTINEL

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
                "CONVERSATION FLOW — HARDCODED STRUCTURE:\n"
                "1. FIRST MESSAGE: Ask whether the auto-matched part IDs look "
                "correct. If not, ask the user to explain what changed. "
                "Submit an empty mapping {}. Example: 'Do the auto-matched "
                "part IDs look correct? If not, tell me what changed.'\n"
                "2. When the user describes a change, ask any needed "
                "clarification questions (one per turn, submit mapping {}).\n"
                "3. Once you understand the change, propose the mapping and "
                "ask 'Confirm?'. You MUST include the proposed overrides in "
                "the mapping field — the system reads the mapping field, NOT "
                "your message text. If the mapping field is empty, NOTHING "
                "gets saved regardless of what your message says.\n"
                "4. If the user confirms, respond ONLY with: "
                "'Saved. Any more changes? (say \"done\" to finish)' "
                "and submit an empty mapping {}.\n"
                "5. If the user rejects, respond ONLY with: "
                "'What should I change?' and submit an empty mapping {}.\n"
                "6. If the user describes another change, repeat from step 2.\n"
                "7. The user will say 'done' when finished — you will NOT see "
                "that message (the system handles it). Do NOT ask the user to "
                "say done yourself after step 4 — the prompt already tells "
                "them.\n\n"
                "EXAMPLES OF CORRECT TOOL CALLS AT STEP 3:\n"
                "User says 'Part_5 was replaced with Part_3':\n"
                "  message: 'Part_5 in the image is actually Part_3. "
                "Mapping: {\"Part_5\": \"Part_3\"}. Confirm?'\n"
                "  mapping: {\"Part_5\": \"Part_3\"}\n\n"
                "User says 'Part_7 is new and Part_4 is actually Part_5':\n"
                "  message: 'Part_7 is a new part and Part_4 should be "
                "Part_5. Confirm?'\n"
                "  mapping: {\"Part_7\": \"new\", \"Part_4\": \"Part_5\"}\n\n"
                "User says 'Part_7 is a new part, call it Part_8':\n"
                "  message: 'Part_7 will be assigned as Part_8. Confirm?'\n"
                "  mapping: {\"Part_7\": \"Part_8\"}\n\n"
                "User says 'Part_2 and Part_6 were swapped':\n"
                "  message: 'Cross-reassigning Part_2 and Part_6. Confirm?'\n"
                "  mapping: {\"Part_2\": \"Part_6\", \"Part_6\": \"Part_2\"}\n\n"
                "⚠ WRONG (mapping field empty while proposing):\n"
                "  message: 'Mapping: {\"Part_5\": \"Part_3\"}. Confirm?'\n"
                "  mapping: {}     ← THIS SAVES NOTHING. NEVER DO THIS.\n\n"
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
                "that states the concrete numbers. Do NOT silently accept "
                "claims that conflict with the physical evidence. Do NOT "
                "block the user — if they confirm after your question, "
                "proceed with their answer.\n\n"
                "CONSTRAINT — UNIQUE IDs AND DISPLACEMENT:\n"
                "Every part must have a unique ID. The system validates your "
                "proposed mapping before accepting it. It will reject the "
                "proposal if:\n"
                "- An ID is already held by another part and your proposal "
                "doesn't also reassign that displaced part.\n"
                "- Two entries map to the same target ID.\n"
                "- A value doesn't match Part_N format.\n"
                "When the system rejects a proposal, it tells you exactly "
                "what the conflict is. Ask the user how to resolve it. "
                "For example: 'Part_5 is currently assigned to the part in "
                "Container_2_Pos_3. If you want this part to be Part_5, what "
                "should happen to the current Part_5?' Then re-propose with "
                "BOTH reassignments in one mapping.\n\n"
                "IMAGE NAMES: The image shows auto-matched parts with their "
                "old-config names and new detections with fresh vision names. "
                "The mapping uses these image-visible names as keys. "
                "Use the names exactly as they appear in the analysis.\n\n"
                "MAPPING DIRECTION — KEY RULE:\n"
                "The mapping key is the name shown in the image, the value "
                "is the identity you want that physical part to have. "
                "Three cases:\n"
                "  {\"Part_5\": \"Part_3\"} — reassign: image's Part_5 is "
                "really Part_3 (must exist in old config)\n"
                "  {\"Part_7\": \"new\"} — auto-assign: system picks next "
                "available Part_N\n"
                "  {\"Part_7\": \"Part_8\"} — custom new ID: Part_8 must not "
                "already exist and must follow Part_N format\n"
                "If the user asks for a specific ID (e.g. 'call it Part_8'), "
                "use the custom new ID form. If they just say 'it's new', "
                "use 'new'. Never reverse key and value.\n\n"
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
                "genuinely new. Include them as {\"<n>\": \"new\"} for auto-"
                "assigned ID, or {\"<n>\": \"Part_8\"} if the user requests "
                "a specific ID.\n\n"
                "HANDLING SPATIAL REFERENCES: If the user refers to positions "
                "(left, right, top, bottom), consult the SLOT-BY-SLOT "
                "COMPARISON table to identify which part they mean. Remember "
                "the axis convention: LARGER X = LEFT, LARGER Y = LOWER."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]

    print("\n── Update Scene Dialogue ──\n")

    # Accumulated mapping: all confirmed overrides collected across turns.
    # Only applied once the user says "done".
    accumulated_mapping: Dict[str, str] = {}

    while True:
        msg_text, turn_mapping = _chat_update(client, messages)
        print(f"ASSISTANT:\n{msg_text}\n")
        if turn_mapping:
            print(f"  [Proposed mapping: {json.dumps(turn_mapping)}]")

        # Record the assistant turn in conversation history.
        messages.append({"role": "assistant", "content": msg_text})

        user = input("YOU: ").strip()

        # ── "done" / empty → apply all accumulated overrides and exit ────
        if not user or user.lower() in ("done", "skip"):
            if accumulated_mapping:
                print(f"\n  Applying mapping: {json.dumps(accumulated_mapping)}")
            else:
                print("\n  No overrides — accepting auto-matches only.")
            return accumulated_mapping

        # ── Recapture request ────────────────────────────────────────────
        if user == RECAPTURE_SENTINEL:
            print("\n  Recapture requested — restarting vision …\n")
            return RECAPTURE_SENTINEL

        # ── "yes" → validate, accumulate, and let the LLM ask about more.
        if is_yes(user):
            if turn_mapping:
                validation_issues = _validate_mapping_proposal(
                    turn_mapping, accumulated_mapping,
                    old_state, fresh_state, image_rename_map,
                )
                if validation_issues:
                    # Bounce back to the LLM to resolve with the user
                    messages.append({
                        "role": "user",
                        "content": (
                            "The system detected issues with the proposed "
                            "mapping:\n"
                            + "\n".join(f"- {i}" for i in validation_issues)
                            + "\nPlease resolve these with me before "
                            "re-proposing."
                        ),
                    })
                    continue

                # No issues — accumulate
                conflicts = []
                for k, v in turn_mapping.items():
                    if k in accumulated_mapping and accumulated_mapping[k] != v:
                        conflicts.append(
                            f"  '{k}' was previously mapped to "
                            f"'{accumulated_mapping[k]}', now to '{v}'"
                        )
                if conflicts:
                    print("  ⚠  Override conflict (using latest):")
                    for c in conflicts:
                        print(c)
                accumulated_mapping.update(turn_mapping)
                print(f"  [Accumulated mapping: {json.dumps(accumulated_mapping)}]")
            # Tell the LLM the user confirmed — it should ask about more changes.
            messages.append({
                "role": "user",
                "content": (
                    "Confirmed. (The mapping has been saved. "
                    "Ask if there are more changes.)"
                ),
            })
            continue

        # ── "no" → reject the proposed mapping ──────────────────────────
        if is_no(user):
            messages.append({"role": "user", "content": "Rejected. Ask me again."})
            continue

        # ── anything else → pass through to the LLM ─────────────────────
        messages.append({"role": "user", "content": user})

    # Should not be reached, but for safety
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

            print("ASSISTANT:\nAnything else? If not, type or press 'done'.\n")
            user_input = input("YOU: ").strip()
            if is_finish(user_input):
                if accumulated_changes:
                    sh.save_changes(accumulated_changes)
                    sh.apply_and_save_config(accumulated_changes)
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