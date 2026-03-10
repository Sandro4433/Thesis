from __future__ import annotations

from pathlib import Path
import sys
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4.1"

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON

LLM_INPUT_PATH = Path(LLM_INPUT_JSON.resolve())
OUTPUT_DIR     = Path(LLM_RESPONSE_JSON.resolve()).parent
SEQUENCE_PATH  = OUTPUT_DIR / "sequence.json"
CHANGES_PATH   = OUTPUT_DIR / "workspace_changes.json"
MEMORY_DIR     = PROJECT_DIR / "Memory"

# Fenced-block regexes
SEQUENCE_BLOCK_RE = re.compile(r"```sequence\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CHANGES_BLOCK_RE  = re.compile(r"```changes\s*(.*?)\s*```",  re.DOTALL | re.IGNORECASE)

# Valid attribute values
VALID_ROLE  = {"input", "output"}
VALID_SIZE  = {None, "large"}
VALID_COLOR = {"Blue", "Red"}


# ── Block parsing ─────────────────────────────────────────────────────────────

def extract_sequence_block(text: str) -> List[List]:
    """
    Parses a ```sequence``` fenced block.
    Format: JSON array of 2- or 3-element entries.
      2 elements: [pick_name, place_name]              → standard size, gripper 0.05
      3 elements: [pick_name, place_name, 0.06]        → large size, gripper 0.06
    """
    m = SEQUENCE_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```sequence``` block found.")
    data = json.loads(m.group(1).strip())
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Sequence block must be a non-empty JSON array.")
    for i, entry in enumerate(data):
        if not isinstance(entry, list) or len(entry) not in (2, 3):
            raise ValueError(
                f"Entry {i} must be [pick_name, place_name] or "
                f"[pick_name, place_name, 0.06], got: {entry!r}"
            )
        if not isinstance(entry[0], str) or not isinstance(entry[1], str):
            raise ValueError(f"Entry {i}: pick_name and place_name must be strings, got: {entry!r}")
        if not entry[0].strip() or not entry[1].strip():
            raise ValueError(f"Entry {i}: names must not be empty.")
        if len(entry) == 3:
            if not isinstance(entry[2], (int, float)) or entry[2] <= 0:
                raise ValueError(
                    f"Entry {i}: gripper_close_width must be a positive number, got: {entry[2]!r}"
                )
    return data


def extract_changes_block(text: str) -> Dict[str, Dict[str, Any]]:
    m = CHANGES_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```changes``` block found.")
    data = json.loads(m.group(1).strip())
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("Changes block must be a non-empty JSON object.")
    for obj_name, attrs in data.items():
        if not isinstance(obj_name, str) or not obj_name.strip():
            raise ValueError(f"Object name must be a non-empty string, got: {obj_name!r}")
        if not isinstance(attrs, dict) or len(attrs) == 0:
            raise ValueError(f"Attributes for '{obj_name}' must be a non-empty object.")
        for attr, val in attrs.items():
            if attr == "Role" and val not in VALID_ROLE:
                raise ValueError(f"'{obj_name}'.Role must be 'input' or 'output', got: {val!r}")
            if attr == "Size" and val not in VALID_SIZE:
                raise ValueError(f"'{obj_name}'.Size must be null or 'large', got: {val!r}")
            if attr == "Color" and val not in VALID_COLOR:
                raise ValueError(f"'{obj_name}'.Color must be 'Blue' or 'Red', got: {val!r}")
            if attr not in ("Role", "Size", "Color"):
                raise ValueError(
                    f"'{obj_name}': unknown attribute '{attr}'. "
                    f"Allowed: Role, Size, Color. Never use dotted paths like 'child_part.Size'."
                )
    return data


def merge_changes(
    accumulated: Dict[str, Dict[str, Any]],
    new_block: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged = {obj: dict(attrs) for obj, attrs in accumulated.items()}
    for obj_name, attrs in new_block.items():
        if obj_name not in merged:
            merged[obj_name] = {}
        merged[obj_name].update(attrs)
    return merged


# ── File saving ───────────────────────────────────────────────────────────────

def save_sequence(sequence: List[List[str]]) -> Path:
    SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENCE_PATH.write_text(json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8")
    return SEQUENCE_PATH


def save_changes(changes: Dict[str, Dict[str, Any]]) -> Path:
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
    return CHANGES_PATH


# ── LLM helpers ───────────────────────────────────────────────────────────────

def chat(client: OpenAI, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()


def is_finish(text: str) -> bool:
    t = text.strip().lower()
    return any(x in t for x in [
        "finish", "finalize", "done", "end", "export",
        "save", "write", "last step", "that's all", "thats all", "quit", "exit",
    ])


def is_yes(text: str) -> bool:
    t = text.strip().lower()
    return any(x in t for x in [
        "yes", "y", "ok", "okay", "confirm", "confirmed", "sure",
        "go ahead", "do it", "looks good", "correct",
    ])


def is_no(text: str) -> bool:
    t = text.strip().lower()
    return any(x in t for x in ["no", "n", "nope", "cancel", "reject", "wrong", "redo"])


def is_reconfig_request(text: str) -> bool:
    """Detect when user agrees to switch to reconfiguration mode."""
    t = text.strip().lower()
    return any(x in t for x in [
        "yes", "y", "ok", "okay", "sure", "go ahead", "switch", "reconfigure",
        "reconfig", "change", "fix", "correct",
    ])


# ── Scene pre-processing ──────────────────────────────────────────────────────

def slim_scene(scene: dict) -> dict:
    """Strip pos/quat before sending to the LLM to save tokens."""
    skip = {"pos", "quat"}

    def slim_entry(e: dict) -> dict:
        return {k: v for k, v in e.items() if k not in skip}

    out: dict = {}
    if "slots" in scene:
        out["slots"] = {}
        for name, slot in scene["slots"].items():
            s = slim_entry(slot)
            if "child_part" in s and isinstance(s["child_part"], dict):
                s["child_part"] = slim_entry(s["child_part"])
            out["slots"][name] = s
    if "parts" in scene:
        out["parts"] = {k: slim_entry(v) for k, v in scene["parts"].items()}
    return out


# ── System prompts ────────────────────────────────────────────────────────────

_COMMON_RULES = """\
──────────────────────────────────────────────────────────────
AMBIGUITY RULES
──────────────────────────────────────────────────────────────
- Never guess. Never infer. Ask.
- If a request matches multiple objects, list candidates (name + one attribute)
  and ask which to include.
- If a part has no valid name in the JSON, skip it and warn the user.
- Do NOT re-print the full scene JSON when asking clarification questions.
"""

_CHANGES_BLOCK_RULES = """\
──────────────────────────────────────────────────────────────
OUTPUT BLOCK — WORKSPACE CHANGES
──────────────────────────────────────────────────────────────
Output ONLY the attributes that are actually changing — not the full scene.
If an attribute is changed more than once, only the latest value will be kept.

```changes
{
  "<object_name>": {"<attribute>": <value>},
  "<object_name>": {"<attribute>": <value>, "<attribute>": <value>}
}
```

Allowed attributes and values:
  Role  (slots only)      : "input" | "output"
  Size  (slots and parts) : null | "large"
  Color (parts only)      : "Blue" | "Red"

CRITICAL FORMAT RULES:
- <object_name> must be a verbatim name from the INPUT JSON (e.g. "Container_3_Pos_2").
- <attribute> must be EXACTLY one of: Role, Size, Color — nothing else.
- NEVER use dotted paths like "child_part.Size". Use "Size" directly.
  When you set Size or Color on a slot name, the system automatically redirects
  it to the child_part inside that slot. You do not need to navigate into it.
- NEVER output the full scene JSON.
- null means reset to default (no size / no role).
"""


def build_system_prompt(mode: str) -> str:
    if mode == "reconfig":
        return """\
You are a robot workspace configurator. Your job is to track user-specified
attribute changes to workspace objects and output a structured changes block.

You will receive an INPUT JSON describing the current scene:
  - "slots": named workspace positions (Kit_* or Container_*).
    A slot with a non-null "child_part" means a part is sitting there.
    Slots have: Role (null | "input" | "output"), Size (null | "large"), child_part.
  - "parts": standalone parts not in any slot (Part_*).
    Parts have: Color ("Blue" | "Red"), Size (null | "large").

──────────────────────────────────────────────────────────────
WORKFLOW
──────────────────────────────────────────────────────────────
1. Give a SHORT scene summary: slots with their current Role/Size, parts with Color/Size.
2. Ask what attributes the user wants to change.
3. If truly ambiguous, ask ONE focused clarification question. Otherwise skip straight to step 4.
4. Once you understand the request: output the changes block immediately and ask "Confirm?"
   Do NOT ask "is that correct?" or "just to confirm" before proposing the block.
   The block IS the confirmation step — one round only.

""" + _CHANGES_BLOCK_RULES + """
──────────────────────────────────────────────────────────────
CONFIRMATION
──────────────────────────────────────────────────────────────
- After proposing changes, always ask: "Confirm?"
- If confirmed → changes are saved and applied to produce a new config.json.
  Ask if there is more to change.
- If rejected → discard and ask what to change.

""" + _COMMON_RULES

    else:  # motion
        return """\
You are a robot task planner. Your job is to translate natural-language task
descriptions into an ordered sequence of pick-and-place actions for a robot arm.

You will receive an INPUT JSON describing the current scene:
  - "slots": named workspace positions (Kit_* or Container_*).
    A slot with a non-null "child_part" means a part is sitting there.
    Slots have: Role (null | "input" | "output"), child_part.
  - "parts": standalone parts not in any slot (Part_*).
    Parts have: Color ("Blue" | "Red"), Size (null | "large").

──────────────────────────────────────────────────────────────
ROLE RESTRICTIONS — ENFORCE STRICTLY
──────────────────────────────────────────────────────────────
Every slot and embedded child_part may have a Role attribute:
  - Role = "input"  → the slot/part CAN be picked FROM, CANNOT be placed INTO.
  - Role = "output" → the slot/part CAN be placed INTO, CANNOT be picked FROM.
  - Role = null     → no restriction; can be used for either pick or place.

Before proposing any sequence, check every pick and place target against these rules.
If a conflict is found:
  1. Do NOT output a sequence block.
  2. Clearly explain which object has a conflicting role and why it blocks the action.
  3. Tell the user that the role must be changed via workspace reconfiguration first.
  4. Ask: "Would you like to switch to reconfiguration mode to fix this?"
  5. If the user says yes → output exactly this marker on its own line:
       SWITCH_TO_RECONFIG
     The system will automatically return to the mode selection menu.

──────────────────────────────────────────────────────────────
GRIPPER WIDTH
──────────────────────────────────────────────────────────────
The gripper close width depends on the Size of the object being PICKED:
  - Size = null  (standard) → gripper_close_width = 0.05
  - Size = "large"          → gripper_close_width = 0.06

The gripper open width is always 0.075 and must NOT be included in the sequence.

──────────────────────────────────────────────────────────────
WORKFLOW
──────────────────────────────────────────────────────────────
1. Give a SHORT scene summary: part counts per color, empty slots, roles.
2. Ask what task the user wants ("sort", "move", "assemble", etc.).
3. Check role restrictions for all involved objects BEFORE proposing anything.
4. If truly ambiguous, ask ONE focused clarification question. Otherwise skip straight to step 5.
5. Once you understand the request: output the sequence block immediately and ask "Confirm?"
   Do NOT ask "is that correct?" or "just to confirm" before proposing the block.
   The block IS the confirmation step — one round only.

──────────────────────────────────────────────────────────────
OUTPUT BLOCK — SEQUENCE
──────────────────────────────────────────────────────────────
Write a short plain-text summary, then:

```sequence
[
  ["<pick_name>", "<place_name>"],
  ["<pick_name>", "<place_name>", 0.06]
]
```

GRIPPER WIDTH — only add the 3rd element when the picked object has Size = "large":
  - Size = null  (standard) → 2 elements: ["pick", "place"]          ← no width needed
  - Size = "large"          → 3 elements: ["pick", "place", 0.06]    ← must include 0.06

Examples:
  ["Part_Red_Nr_1",  "Kit_0_Pos_1"]        ← Red part, standard size
  ["Part_Blue_Nr_1", "Kit_0_Pos_2", 0.06]  ← Blue part, large → include 0.06

Other rules:
- Parts in slots   → pick = child_part name  (e.g. "Part_Blue_Nr_1")
- Standalone parts → pick = top-level key     (e.g. "Part_Blue_Nr_3")
- Destination      → place = slot key         (e.g. "Kit_0_Pos_1")
- NEVER use coordinates or invented names.
- NEVER include a sequence with role violations — explain and ask about reconfiguration instead.
- If no valid empty destination slots exist, say so before proposing anything.

──────────────────────────────────────────────────────────────
CONFIRMATION
──────────────────────────────────────────────────────────────
- After proposing a sequence, always ask: "Confirm?"
- If confirmed → sequence is saved. Ask if there is more to plan.
- If rejected → discard and ask what to change.

""" + _COMMON_RULES


# ── Pre-session setup (mode + source selection) ───────────────────────────────

def _pick_from_list(prompt: str, options: List[str]) -> int:
    """Print numbered options and return the 0-based index of the user's choice."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("Your choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Please enter a number between 1 and {len(options)}.")


def select_mode() -> str:
    """Ask the user what they want to do. Returns 'reconfig', 'motion', 'execute', or 'exit'."""
    print("\n" + "=" * 60)
    print("  Robot Configuration")
    print("=" * 60)
    idx = _pick_from_list("\nWhat do you want to do?", [
        "Workspace reconfiguration  (change slot/part attributes)",
        "Motion sequence planning   (pick-and-place task)",
        "Execute robot motion       (run current sequence.json)",
        "Exit",
    ])
    return ["reconfig", "motion", "execute", "exit"][idx]


def select_scene() -> tuple[dict, Path]:
    """
    Ask the user which baseline config to use.
    Returns (scene_dict, baseline_path).
    If live vision is chosen the vision module is called first.
    Memory files are slimmed (pos/quat stripped) before being sent to the LLM.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    memory_files: List[Path] = sorted(MEMORY_DIR.glob("*.json"))

    options = ["Live vision  (capture new image with camera)"]
    for f in memory_files:
        options.append(f"Memory: {f.name}")

    idx = _pick_from_list("\nWhich baseline config do you want to use?", options)

    if idx == 0:
        print("\nStarting vision module …")
        from Vision_Module.Vision_Main import main as vision_main
        vision_main()
        if not LLM_INPUT_PATH.exists():
            print(f"ERROR: Vision module did not produce {LLM_INPUT_PATH}")
            sys.exit(1)
        scene = json.loads(LLM_INPUT_PATH.read_text(encoding="utf-8"))
        baseline_path = LLM_INPUT_PATH
        print(f"Loaded scene from vision: {LLM_INPUT_PATH}")
    else:
        chosen = memory_files[idx - 1]
        raw = json.loads(chosen.read_text(encoding="utf-8"))
        scene = slim_scene(raw)   # strip pos/quat just like llm_input
        baseline_path = chosen
        print(f"Loaded scene from memory: {chosen}")

    return scene, baseline_path


# ── Config application helper ─────────────────────────────────────────────────

def _apply_and_save_config(
    baseline_path: Path,
    accumulated_changes: Dict[str, Dict[str, Any]],
) -> None:
    from Configuration_Module.Apply_Config_Changes import apply_changes

    baseline_scene = json.loads(baseline_path.read_text(encoding="utf-8"))
    updated = apply_changes(baseline_scene, accumulated_changes)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    config_path = MEMORY_DIR / "config.json"
    tmp = str(config_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(config_path)
    print(f"✅  config.json saved → {config_path.resolve()}\n")


# ── Session (one mode + scene selection) ─────────────────────────────────────

def run_session(client: OpenAI, mode: str) -> None:
    """
    Runs one full mode session (reconfig, motion, or execute).
    For 'execute': calls Robot_Main directly and returns.
    For reconfig/motion: all confirmed changes are accumulated in memory and
    only written to disk when the user types 'done'.
    """

    # ── Option 3: Execute robot motion ────────────────────────────────────────
    if mode == "execute":
        seq_path = SEQUENCE_PATH
        if not seq_path.exists():
            print(f"\n⚠  No sequence.json found at {seq_path.resolve()}")
            print("   Plan a motion sequence first (option 2).\n")
            return
        print(f"\n▶  Executing sequence from: {seq_path.resolve()}")
        from Execution_Module.Robot_Main import main as robot_main
        robot_main()
        print("\n── Execution complete. ──\n")
        return
    scene, baseline_path = select_scene()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(mode)},
        {
            "role": "user",
            "content": (
                "INPUT SCENE JSON (sole source of truth — use these names verbatim):\n"
                + json.dumps(slim_scene(scene), indent=2, ensure_ascii=False)
            ),
        },
        {
            "role": "user",
            "content": "Please give your scene summary and ask what task I want.",
        },
    ]

    pending_sequence: Optional[List[List[str]]] = None
    pending_changes:  Optional[Dict[str, Dict[str, Any]]] = None
    # All confirmed changes for this session — only written to disk on 'done'
    accumulated_changes: Dict[str, Dict[str, Any]] = {}

    mode_label = "Reconfiguration" if mode == "reconfig" else "Motion Sequence"
    print(f"\n── Mode: {mode_label}  |  Baseline: {baseline_path.name} ──\n")

    while True:
        # ── LLM turn ──────────────────────────────────────────────────────────
        assistant_text = chat(client, messages)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        # Detect role-conflict redirect → return to outer loop (mode selection)
        if "SWITCH_TO_RECONFIG" in assistant_text:
            print("  [Role conflict detected — returning to mode selection for reconfiguration.]\n")
            print("── Returning to main menu ──\n")
            return

        # Detect sequence proposal
        try:
            pending_sequence = extract_sequence_block(assistant_text)
            print(f"  [Sequence proposal detected: {len(pending_sequence)} step(s)]\n")
        except Exception as e:
            if "```sequence" in (assistant_text or ""):
                print(f"  [WARNING: sequence block found but failed to parse — {e}]")
                print(f"  [Asking LLM to correct its output format …]\n")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your sequence block failed to parse with this error: " + str(e) + "\n"
                        "Each entry must be 2 or 3 elements:\n"
                        "  [\"pick\", \"place\"]        ← standard size (no width needed)\n"
                        "  [\"pick\", \"place\", 0.06]  ← large size only\n"
                        "Please rewrite the sequence block with the correct format."
                    ),
                })
                continue

        # Detect changes proposal
        try:
            pending_changes = extract_changes_block(assistant_text)
            objs  = len(pending_changes)
            attrs = sum(len(v) for v in pending_changes.values())
            print(f"  [Changes proposal detected: {attrs} attribute(s) across {objs} object(s)]\n")
        except Exception as e:
            if "```changes" in (assistant_text or ""):
                print(f"  [WARNING: changes block found but failed to parse — {e}]")
                print(f"  [Asking LLM to correct its output format …]\n")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your changes block failed to parse with this error: " + str(e) + "\n"
                        "IMPORTANT: Use only flat attribute keys: Role, Size, or Color. "
                        "NEVER use dotted paths like 'child_part.Size'. "
                        "Use the slot name as the key (e.g. 'Container_3_Pos_2') and "
                        "'Size' as the attribute. The system redirects Size/Color to the "
                        "child_part automatically. Please rewrite the changes block correctly."
                    ),
                })
                continue

        # ── User turn ─────────────────────────────────────────────────────────
        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        # ── Done — write everything and return to mode selection ──────────────
        if is_finish(user_input):
            # Merge any unconfirmed pending blocks before writing
            if pending_sequence is not None:
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Sequence saved ({len(pending_sequence)} pairs) → {saved.resolve()}")
                print(json.dumps(pending_sequence, indent=2, ensure_ascii=False))
            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)

            # Write all accumulated changes + config in one go
            if accumulated_changes:
                saved = save_changes(accumulated_changes)
                total = sum(len(v) for v in accumulated_changes.values())
                print(f"\n✅  All workspace changes saved ({total} attribute(s)) → {saved.resolve()}")
                print(json.dumps(accumulated_changes, indent=2, ensure_ascii=False))
                _apply_and_save_config(baseline_path, accumulated_changes)

            print("\n── Session complete. ──\n")
            return  # back to outer loop → new mode selection

        # ── Confirmation handling ─────────────────────────────────────────────
        has_pending = pending_sequence is not None or pending_changes is not None

        if has_pending and is_yes(user_input):
            if pending_sequence is not None:
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Sequence confirmed & saved ({len(pending_sequence)} step(s)) → {saved.resolve()}\n")
                # Add to history silently so LLM has context, but don't let it respond
                messages.append({"role": "user",    "content": "Confirmed the sequence."})
                messages.append({"role": "assistant","content": "Sequence saved."})
                pending_sequence = None

            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
                attrs = sum(len(v) for v in pending_changes.values())
                total = sum(len(v) for v in accumulated_changes.values())
                print(f"✅  Changes noted ({attrs} new, {total} total). "
                      f"Will be written to config.json when done.\n")
                messages.append({"role": "user",    "content": "Confirmed the changes."})
                messages.append({"role": "assistant","content": "Changes noted."})
                pending_changes = None

            # Skip LLM — go straight back to user input
            user_input = input("Anything else? (or type 'done')\nYOU: ").strip()
            if not user_input or not is_finish(user_input):
                if user_input:
                    messages.append({"role": "user", "content": user_input})
                continue
            # user typed 'done' — fall through to the done-handler below

        if has_pending and is_no(user_input):
            print("  [Proposal rejected.]\n")
            pending_sequence = None
            pending_changes  = None
            messages.append({
                "role": "user",
                "content": "Rejected. Discard that proposal. Ask what I want to change or redo.",
            })
            continue

        # ── Normal chat turn ───────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    client = OpenAI()
    while True:
        mode = select_mode()
        if mode == "exit":
            print("\n👋  Exiting. Goodbye!\n")
            return
        run_session(client, mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋  Exiting planner.\n")