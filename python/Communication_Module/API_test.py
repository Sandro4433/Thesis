from __future__ import annotations

from pathlib import Path
import sys
import json
import re
from typing import Dict, List, Optional

from openai import OpenAI

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4.1"

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON

INPUT_PATH  = Path(LLM_INPUT_JSON.resolve())
OUTPUT_PATH = Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json"

# Regex to extract a ```sequence ... ``` fenced block from LLM output
SEQUENCE_BLOCK_RE = re.compile(r"```sequence\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# ── Sequence block parsing ────────────────────────────────────────────────────

def extract_sequence_block(text: str) -> List[List[str]]:
    """
    Extracts and validates a fenced sequence block from LLM text.

    Compact format: a JSON array of [pick_name, place_name] string pairs.
      [
        ["Part_Blue_Nr_3", "Kit_0_Pos_1"],
        ["Part_Blue_Nr_4", "Kit_0_Pos_2"]
      ]

    - pair[0] = pick location name  (must exist in positions.json)
    - pair[1] = place location name (must exist in positions.json)
    Both must be verbatim name strings from the INPUT JSON — never coordinates.
    """
    m = SEQUENCE_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```sequence``` block found in LLM response.")

    raw = m.group(1).strip()
    data = json.loads(raw)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Sequence block must be a non-empty JSON array.")
    for i, pair in enumerate(data):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"Pair {i} must be a 2-element array [pick_name, place_name], got: {pair!r}"
            )
        if not isinstance(pair[0], str) or not isinstance(pair[1], str):
            raise ValueError(
                f"Pair {i}: both elements must be name strings, got: {pair!r}"
            )
        if not pair[0].strip() or not pair[1].strip():
            raise ValueError(f"Pair {i}: location names must not be empty.")
    return data


def save_sequence(sequence: List[List[str]]) -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(sequence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return OUTPUT_PATH


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


# ── System prompt (rules embedded — no external rules.txt needed) ─────────────

SYSTEM_PROMPT = """\
You are a robot task planner. Your job is to translate natural-language task
descriptions into an ordered sequence of pick-and-place actions for a robot arm.

You will receive an INPUT JSON describing the current scene. It contains:
  - "slots": named workspace positions. A slot with a non-null "child_part" field
    means a part is sitting there. Use child_part["name"] as the pick location.
  - "parts": standalone parts not in any slot. Use their top-level key as the
    pick location (e.g. "Part_Blue_Nr_3").
  - Empty slots (child_part: null) are valid place/destination targets.

──────────────────────────────────────────────────────────────
WORKFLOW
──────────────────────────────────────────────────────────────
1. Give a SHORT scene summary: part counts per color, available empty slots.
2. Ask what task the user wants ("sort", "move", "assemble", etc.).
3. Ask ONE focused clarification question at a time until fully unambiguous.
4. Once clear: scan JSON, collect matching parts, propose the full sequence.
5. Always ask "Confirm this sequence?" — never skip confirmation.

──────────────────────────────────────────────────────────────
SEQUENCE OUTPUT FORMAT
──────────────────────────────────────────────────────────────
Write a short plain-text summary, then this exact fenced block:

```sequence
[
  ["<pick_name>", "<place_name>"],
  ["<pick_name>", "<place_name>"]
]
```

Then ask: "Confirm this sequence?"

CRITICAL FORMAT RULES:
- Every string MUST be a name taken verbatim from the INPUT JSON.
  Parts in slots   → use child_part["name"]  (e.g. "Part_Blue_Nr_1")
  Standalone parts → use the top-level key    (e.g. "Part_Blue_Nr_3")
  Destination slot → use the slot's key       (e.g. "Kit_0_Pos_1")
- NEVER output coordinates, floats, or invented names.
- NEVER output the full scene JSON.
- NEVER skip the confirmation step.

──────────────────────────────────────────────────────────────
AMBIGUITY RULES
──────────────────────────────────────────────────────────────
- If a request matches multiple objects, list only the candidates
  (name + one distinguishing attribute) and ask which to include.
- If a part has no name in the JSON, skip it and warn the user.
- If no empty destination slots exist, say so before proposing anything.
- Never guess. Never infer. Ask.
"""


# ── Scene pre-processing ──────────────────────────────────────────────────────

def slim_scene(scene: dict) -> dict:
    """
    Strip pos/quat from every entry before sending to the LLM.
    This cuts token usage significantly while preserving all names,
    colors, and structural info the model needs.
    """
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

    print("\n" + "=" * 60)
    print("  Robot Sequence Planner  |  type 'done' to save & exit")
    print("=" * 60 + "\n")

    while True:
        # ── LLM turn ──────────────────────────────────────────────────────────
        assistant_text = chat(client, messages)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        # Detect a sequence proposal in this response
        try:
            pending_sequence = extract_sequence_block(assistant_text)
            print(f"  [Sequence proposal detected: {len(pending_sequence)} pair(s)]\n")
        except Exception:
            pass  # no block — normal conversation turn

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
            print("\n👋  Exiting planner.\n")
            return

        # ── Confirmation handling ─────────────────────────────────────────────
        if pending_sequence is not None:
            if is_yes(user_input):
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Confirmed & saved ({len(pending_sequence)} pairs) → {saved.resolve()}\n")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Confirmed. Sequence saved ({len(pending_sequence)} pairs). "
                        "Ask if I want to plan another task or if we are done."
                    ),
                })
                pending_sequence = None
                continue

            if is_no(user_input):
                print("  [Sequence rejected.]\n")
                pending_sequence = None
                messages.append({
                    "role": "user",
                    "content": "Rejected. Discard that sequence. Ask what I want to change or redo.",
                })
                continue

        # ── Normal chat turn ───────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})


if __name__ == "__main__":
    main()