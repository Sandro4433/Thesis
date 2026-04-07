"""
API_Main.py — LLM-driven workspace configuration and motion planning.

This is the main session orchestrator. It handles:
  - Mode selection (reconfig / motion / execute)
  - Scene loading (vision / memory)
  - The LLM conversation loop
  - Confirmation, rejection, and session lifecycle

Implementation details (parsing, prompts, change management, scene processing)
are delegated to sibling modules.
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

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON, CONFIGURATION_JSON

LLM_INPUT_PATH = Path(LLM_INPUT_JSON.resolve())
CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
OUTPUT_DIR = Path(LLM_RESPONSE_JSON.resolve()).parent
SEQUENCE_PATH = OUTPUT_DIR / "sequence.json"
CHANGES_PATH = OUTPUT_DIR / "workspace_changes.json"
MEMORY_DIR = PROJECT_DIR / "Memory"

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


# ── File I/O ─────────────────────────────────────────────────────────────────

def save_sequence(sequence: List[List]) -> Path:
    SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENCE_PATH.write_text(
        json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return SEQUENCE_PATH


def save_changes(changes: Dict[str, Any]) -> Path:
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(
        json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return CHANGES_PATH


# ── LLM helpers ──────────────────────────────────────────────────────────────

def chat(client: OpenAI, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature
    )
    return (resp.choices[0].message.content or "").strip()


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


# ── Config application ───────────────────────────────────────────────────────

def _apply_and_save_config(accumulated_changes: Dict[str, Any]) -> None:
    from Configuration_Module.Apply_Config_Changes import apply_changes
    import io, contextlib

    if not CONFIGURATION_PATH.exists():
        print(f"⚠  configuration.json not found — cannot apply changes.")
        return

    scene = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

    # Suppress any verbose output from apply_changes
    with contextlib.redirect_stdout(io.StringIO()):
        updated = apply_changes(scene, accumulated_changes)

    tmp = str(CONFIGURATION_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(CONFIGURATION_PATH)
    print("✅  Configuration updated.")

    _refresh_annotated_image(updated)


def _refresh_annotated_image(state: Dict[str, Any]) -> None:
    """Redraw latest_image.png with current part names and FRAGILE labels."""
    file_exchange = PROJECT_DIR / "File_Exchange"
    base_path = file_exchange / "latest_image_base.png"
    pmap_path = file_exchange / "latest_pixel_map.json"
    out_path = file_exchange / "latest_image.png"

    if not base_path.exists() or not pmap_path.exists():
        return

    try:
        import cv2
        from Vision_Module.pipeline import annotate_parts

        img = cv2.imread(str(base_path))
        if img is None:
            return

        pixel_map = json.loads(pmap_path.read_text(encoding="utf-8"))
        preds = state.get("predicates", {})
        fragile_set = {
            e["part"] for e in preds.get("fragility", [])
            if e.get("fragility") == "fragile"
        }

        annotate_parts(img, pixel_map, fragile_set=fragile_set)
        cv2.imwrite(str(out_path), img)
        print("✅  Image updated.")
    except Exception as exc:
        print(f"  ⚠  Image refresh failed: {exc}")


# ── Pre-session setup ────────────────────────────────────────────────────────

def _pick_from_list(prompt: str, options: List[str]) -> int:
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Your choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Please enter a number between 1 and {len(options)}.")


def select_mode() -> str:
    from Vision_Module.config import USE_PDDL_PLANNER

    planner_label = "PDDL planner" if USE_PDDL_PLANNER else "LLM dialogue"
    print("\n" + "=" * 60)
    print("  Robot Configuration")
    print(f"  Sequence planner: {planner_label}  (toggle USE_PDDL_PLANNER in config.py)")
    print("=" * 60)

    idx = _pick_from_list("\nWhat do you want to do?", [
        "Workspace reconfiguration  (change attributes, roles, recipes)",
        f"Motion sequence planning   ({planner_label})",
        "Execute robot motion       (run current sequence.json)",
        "Exit",
    ])
    return ["reconfig", "motion", "execute", "exit"][idx]


def select_reconfig_source() -> str:
    idx = _pick_from_list("\nHow do you want to load the scene?", [
        "Fresh scan — new picture, start from scratch (no memory)",
        "Fresh scan — new picture, keep high-level config from memory",
        "Memory    — use existing configuration.json as-is",
    ])
    return ["reconfig_fresh", "reconfig_update", "reconfig_memory"][idx]


def _run_vision_subprocess() -> None:
    import subprocess as _sp

    vision_main_path = PROJECT_DIR / "Vision_Module" / "Vision_Main.py"
    print("\nStarting vision module …")
    result = _sp.run([sys.executable, str(vision_main_path)], cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"Vision_Main subprocess exited with code {result.returncode}.")
    print("Vision complete.\n")


def select_scene() -> dict:
    options = [
        "Live vision  (capture new image with camera)",
        f"Current configuration.json  ({CONFIGURATION_PATH})",
    ]
    idx = _pick_from_list("\nWhich scene do you want to use?", options)

    if idx == 0:
        try:
            _run_vision_subprocess()
        except RuntimeError as e:
            print(f"\n❌  Vision failed: {e}\n")
            sys.exit(1)
        if not CONFIGURATION_PATH.exists():
            print(f"ERROR: Vision module did not produce {CONFIGURATION_PATH}")
            sys.exit(1)
        print("Loaded fresh scene from vision.")
    else:
        if not CONFIGURATION_PATH.exists():
            print(f"ERROR: configuration.json not found at {CONFIGURATION_PATH.resolve()}")
            sys.exit(1)
        print(f"Using stored configuration: {CONFIGURATION_PATH}")

    state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    return slim_scene(state)


# ── Update Scene dialogue ────────────────────────────────────────────────────

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


def _run_update_dialogue(client: OpenAI) -> None:
    from Configuration_Module.Update_Scene import (
        prepare_update, apply_update_mapping, redraw_image_with_auto_matches,
    )

    # prepare_update() runs vision internally and returns both old and fresh states.
    # Do NOT call _run_vision_subprocess() here — it would overwrite configuration.json
    # before prepare_update() reads it, losing high-level config (fragility, roles, etc.).
    try:
        old_state, fresh_state = prepare_update()
    except RuntimeError as e:
        print(f"\n❌  Vision failed: {e}\n")
        return

    if old_state is None or fresh_state is None:
        print("⚠  Update aborted (missing state files).")
        return

    # Re-annotate the image with auto-matched names so the user sees
    # old-config IDs (where auto-matching succeeded) during the dialogue,
    # not the arbitrary sequential IDs from the vision detector.
    image_rename_map = redraw_image_with_auto_matches(old_state, fresh_state)

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
        # We store the tool call as a regular assistant message so
        # subsequent turns see the text naturally.
        messages.append({"role": "assistant", "content": msg_text})

        user = input("YOU: ").strip()
        if not user or is_yes(user) or user.lower() in ("done", "skip"):
            if mapping:
                print(f"\n  Applying mapping: {json.dumps(mapping)}")
            else:
                print("\n  No overrides — accepting auto-matches only.")
            break
        if is_no(user):
            messages.append({"role": "user", "content": "Rejected. Ask me again."})
            continue
        messages.append({"role": "user", "content": user})

    apply_update_mapping(old_state, fresh_state, mapping, image_rename_map)
    print("✅  Update complete.\n")


# ── Conflict resolution helper ───────────────────────────────────────────────

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


# ── Session runner ───────────────────────────────────────────────────────────

def run_session(client: OpenAI, mode: str) -> None:
    from Vision_Module.config import USE_PDDL_PLANNER

    # Special-case: execute stored sequence
    if mode == "execute":
        _run_robot_execution()
        return

    # Special-case: PDDL planning for motion mode
    if mode == "motion" and USE_PDDL_PLANNER:
        _run_pddl_sequence()
        return

    # Load scene based on mode
    scene = _load_scene(client, mode)
    if scene is None:
        return

    # Build initial messages
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
        assistant_text = chat(client, messages)
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
                    save_changes(accumulated_changes)
                    _apply_and_save_config(accumulated_changes)
                    print("✅  Changes saved.")
                print("\n── Configuration complete. ──\n")
                return
            if user_input:
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
        messages.append({"role": "user", "content": user_input})


# ── Session helper functions ─────────────────────────────────────────────────

def _load_scene(client: OpenAI, mode: str) -> Optional[dict]:
    """Load the scene for the current mode. Returns slim scene dict or None on failure."""
    if mode == "reconfig":
        sub = select_reconfig_source()

        if sub == "reconfig_fresh":
            try:
                _run_vision_subprocess()
            except RuntimeError as e:
                print(f"\n❌  Vision failed: {e}\n")
                return None
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found")
                return None
            print("Loaded fresh scene from vision.")

        elif sub == "reconfig_update":
            _run_update_dialogue(client)
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found")
                return None

        else:  # reconfig_memory
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found at {CONFIGURATION_PATH.resolve()}")
                return None
            print(f"Loaded scene from: {CONFIGURATION_PATH}")

        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        return slim_scene(state)
    else:
        return select_scene()


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
    if pending_sequence is not None:
        save_sequence(pending_sequence)
        print("✅  Sequence saved.")

    if pending_changes is not None:
        accumulated = _handle_conflicts(client, accumulated, pending_changes)

    if accumulated:
        save_changes(accumulated)
        _apply_and_save_config(accumulated)
        print("✅  Changes saved.")

    print("\n── Configuration complete. ──\n")


# ── Robot execution ──────────────────────────────────────────────────────────

def _run_robot_execution() -> None:
    import subprocess as _sp

    run_script = PROJECT_DIR / "run_execute.py"
    if not run_script.exists():
        print(f"❌  run_execute.py not found: {run_script.resolve()}")
        return

    print("\n── Launching robot execution ──\n")
    result = _sp.run([sys.executable, str(run_script)], cwd=str(PROJECT_DIR))

    if result.returncode != 0:
        print(f"⚠  Execution subprocess exited with code {result.returncode}")
    else:
        print("\n── Execution finished. ──\n")


def _run_pddl_sequence() -> None:
    from Vision_Module.config import USE_PDDL_PLANNER
    from pddl_planner import plan_sequence

    if not CONFIGURATION_PATH.exists():
        print(f"❌ configuration.json not found: {CONFIGURATION_PATH.resolve()}")
        return

    state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

    print("\n── PDDL Planner ──")
    print(f"  operation_mode : {state.get('workspace', {}).get('operation_mode', 'not set')}")
    print(f"  kit_recipe     : {state.get('predicates', {}).get('kit_recipe', [])}")
    print(f"  priority       : {state.get('predicates', {}).get('priority', [])}\n")

    sequence = plan_sequence(
        state, output_path=str(SEQUENCE_PATH), keep_pddl=True,
    )

    if sequence is None:
        print("❌ PDDL planning failed. Check roles, recipes, and Fast Downward installation.")
    else:
        print(f"\n✅  Sequence written → {SEQUENCE_PATH.resolve()}")
        print(json.dumps(sequence, indent=2))

    print("\n── PDDL planning complete. ──\n")


# ── Main entry point ─────────────────────────────────────────────────────────

def main():
    client = OpenAI()
    while True:
        mode = select_mode()
        if mode == "exit":
            print("\n👋  Goodbye!\n")
            return
        run_session(client, mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋  Exiting.\n")