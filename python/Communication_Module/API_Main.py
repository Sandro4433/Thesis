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

INPUT_PATH   = Path(LLM_INPUT_JSON.resolve())
OUTPUT_DIR   = Path(LLM_RESPONSE_JSON.resolve()).parent
SEQUENCE_PATH = OUTPUT_DIR / "sequence.json"
CHANGES_PATH  = OUTPUT_DIR / "workspace_changes.json"

# Fenced-block regexes
SEQUENCE_BLOCK_RE = re.compile(r"```sequence\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CHANGES_BLOCK_RE  = re.compile(r"```changes\s*(.*?)\s*```",  re.DOTALL | re.IGNORECASE)

# Valid attribute values — enforced at parse time
VALID_ROLE   = {"input", "output"}
VALID_SIZE   = {None, "large"}
VALID_COLOR  = {"Blue", "Red"}


# ── Sequence block parsing ────────────────────────────────────────────────────

def extract_sequence_block(text: str) -> List[List[str]]:
    """
    Parses a ```sequence``` fenced block.
    Format: JSON array of [pick_name, place_name] string pairs.
    """
    m = SEQUENCE_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```sequence``` block found.")

    data = json.loads(m.group(1).strip())

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Sequence block must be a non-empty JSON array.")
    for i, pair in enumerate(data):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Pair {i} must be [pick_name, place_name], got: {pair!r}")
        if not isinstance(pair[0], str) or not isinstance(pair[1], str):
            raise ValueError(f"Pair {i}: both elements must be strings, got: {pair!r}")
        if not pair[0].strip() or not pair[1].strip():
            raise ValueError(f"Pair {i}: names must not be empty.")
    return data


# ── Changes block parsing ─────────────────────────────────────────────────────

def extract_changes_block(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses a ```changes``` fenced block.

    Format: JSON object mapping object names to their changed attributes.
      {
        "Kit_0_Pos_1":     {"Role": "input"},
        "Part_Blue_Nr_3":  {"Color": "Red", "Size": "large"}
      }

    Valid values:
      Role  (slots only) : "input" | "output"
      Size  (parts only) : null | "large"
      Color (parts only) : "Blue" | "Red"
    """
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
                raise ValueError(f"'{obj_name}': unknown attribute '{attr}'. Allowed: Role, Size, Color.")

    return data


def merge_changes(
    accumulated: Dict[str, Dict[str, Any]],
    new_block: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Merges new_block into accumulated. Latest value for each object/attribute wins.
    """
    merged = {obj: dict(attrs) for obj, attrs in accumulated.items()}
    for obj_name, attrs in new_block.items():
        if obj_name not in merged:
            merged[obj_name] = {}
        merged[obj_name].update(attrs)
    return merged


# ── File saving ───────────────────────────────────────────────────────────────

def save_sequence(sequence: List[List[str]]) -> Path:
    SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENCE_PATH.write_text(
        json.dumps(sequence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return SEQUENCE_PATH


def save_changes(changes: Dict[str, Dict[str, Any]]) -> Path:
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(
        json.dumps(changes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return CHANGES_PATH


# ── LLM helpers ───────────────────────────────────────────────────────────────

def chat(client: OpenAI, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a robot task planner. Your job is to:
  (A) Plan ordered pick-and-place sequences for a robot arm.
  (B) Track user-specified attribute changes to workspace objects.

You will receive an INPUT JSON describing the current scene:
  - "slots": named workspace positions (Kit_* or Container_*).
    A slot with a non-null "child_part" means a part is sitting there.
    Slots have: Role (null | "input" | "output"), child_part.
  - "parts": standalone parts not in any slot (Part_*).
    Parts have: Color ("Blue" | "Red"), Size (null | "large").

──────────────────────────────────────────────────────────────
WORKFLOW
──────────────────────────────────────────────────────────────
1. Give a SHORT scene summary: part counts per color, empty slots, current roles.
2. Ask what the user wants to do (move parts, change attributes, or both).
3. Ask ONE focused clarification question at a time until fully unambiguous.
4. Once clear: propose the output blocks (sequence and/or changes as needed).
5. Always ask for confirmation — never skip it.

──────────────────────────────────────────────────────────────
OUTPUT BLOCK A — SEQUENCE
──────────────────────────────────────────────────────────────
Use when the user wants pick-and-place actions.

Write a short plain-text summary, then:

```sequence
[
  ["<pick_name>", "<place_name>"],
  ["<pick_name>", "<place_name>"]
]
```

Rules:
- Parts in slots   → pick = child_part name  (e.g. "Part_Blue_Nr_1")
- Standalone parts → pick = top-level key     (e.g. "Part_Blue_Nr_3")
- Destination      → place = slot key         (e.g. "Kit_0_Pos_1")
- NEVER use coordinates, floats, or invented names.

──────────────────────────────────────────────────────────────
OUTPUT BLOCK B — WORKSPACE CHANGES
──────────────────────────────────────────────────────────────
Use when the user specifies attribute changes.
Output ONLY the attributes that are actually changing — not the full scene.
If an attribute is changed more than once in the conversation, only the
latest value will be kept.

```changes
{
  "<object_name>": {"<attribute>": <value>},
  "<object_name>": {"<attribute>": <value>, "<attribute>": <value>}
}
```

Allowed attributes and values:
  Role  (slots only) : "input" | "output"
  Size  (parts only) : null | "large"
  Color (parts only) : "Blue" | "Red"

Rules:
- Object names must be taken verbatim from the INPUT JSON.
- Only include objects whose attributes are actually changing.
- NEVER output the full scene JSON.
- null means reset to default (no size / no role).

──────────────────────────────────────────────────────────────
CONFIRMATION
──────────────────────────────────────────────────────────────
- After proposing any block(s), always ask: "Confirm?"
- If the user confirms → the blocks are saved. Ask if there is more to do.
- If the user rejects → discard and ask what to change.
- Both blocks can appear in the same response if the user's request involves
  both moving parts AND changing attributes.

──────────────────────────────────────────────────────────────
AMBIGUITY RULES
──────────────────────────────────────────────────────────────
- Never guess. Never infer. Ask.
- If a request matches multiple objects, list candidates (name + one attribute)
  and ask which to include.
- If no empty destination slots exist, say so before proposing a sequence.
- If a part has no valid name in the JSON, skip it and warn the user.
- Do NOT re-print the full scene JSON when asking clarification questions.
"""


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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    scene_json = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    client = OpenAI()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "INPUT SCENE JSON (sole source of truth — use these names verbatim):\n"
                + json.dumps(slim_scene(scene_json), indent=2, ensure_ascii=False)
            ),
        },
        {
            "role": "user",
            "content": "Please give your scene summary and ask what task I want.",
        },
    ]

    pending_sequence: Optional[List[List[str]]] = None
    pending_changes:  Optional[Dict[str, Dict[str, Any]]] = None
    # Accumulates all confirmed changes across the session (latest value per attr wins)
    accumulated_changes: Dict[str, Dict[str, Any]] = {}

    print("\n" + "=" * 60)
    print("  Robot Sequence Planner  |  type 'done' to save & exit")
    print("=" * 60 + "\n")

    while True:
        # ── LLM turn ──────────────────────────────────────────────────────────
        assistant_text = chat(client, messages)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        # Detect sequence proposal
        try:
            pending_sequence = extract_sequence_block(assistant_text)
            print(f"  [Sequence proposal detected: {len(pending_sequence)} pair(s)]\n")
        except Exception:
            pass

        # Detect changes proposal
        try:
            pending_changes = extract_changes_block(assistant_text)
            objs = len(pending_changes)
            attrs = sum(len(v) for v in pending_changes.values())
            print(f"  [Changes proposal detected: {attrs} attribute(s) across {objs} object(s)]\n")
        except Exception:
            pass

        # ── User turn ─────────────────────────────────────────────────────────
        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        # ── Save & exit ───────────────────────────────────────────────────────
        if is_finish(user_input):
            if pending_sequence is not None:
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Sequence saved ({len(pending_sequence)} pairs) → {saved.resolve()}")
                print(json.dumps(pending_sequence, indent=2, ensure_ascii=False))
            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
            if accumulated_changes:
                saved = save_changes(accumulated_changes)
                print(f"\n✅  Workspace changes saved ({sum(len(v) for v in accumulated_changes.values())} attribute(s)) → {saved.resolve()}")
                print(json.dumps(accumulated_changes, indent=2, ensure_ascii=False))
            print("\n👋  Exiting planner.\n")
            return

        # ── Confirmation handling ─────────────────────────────────────────────
        has_pending = pending_sequence is not None or pending_changes is not None

        if has_pending and is_yes(user_input):
            confirm_parts = []

            if pending_sequence is not None:
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Sequence confirmed & saved ({len(pending_sequence)} pairs) → {saved.resolve()}\n")
                confirm_parts.append(f"Sequence saved ({len(pending_sequence)} pairs).")
                pending_sequence = None

            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
                saved = save_changes(accumulated_changes)
                attrs = sum(len(v) for v in pending_changes.values())
                print(f"✅  Changes confirmed & saved ({attrs} attribute(s)) → {saved.resolve()}\n")
                confirm_parts.append(f"Workspace changes saved ({attrs} attribute(s)).")
                pending_changes = None

            messages.append({
                "role": "user",
                "content": (
                    "Confirmed. " + " ".join(confirm_parts) +
                    " Ask if I want to plan another task, make more attribute changes, or if we are done."
                ),
            })
            continue

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


if __name__ == "__main__":
    main()