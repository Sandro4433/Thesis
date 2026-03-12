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

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON, POSITIONS_JSON

LLM_INPUT_PATH = Path(LLM_INPUT_JSON.resolve())
POSITIONS_PATH = Path(POSITIONS_JSON.resolve())
OUTPUT_DIR     = Path(LLM_RESPONSE_JSON.resolve()).parent
SEQUENCE_PATH  = OUTPUT_DIR / "sequence.json"
CHANGES_PATH   = OUTPUT_DIR / "workspace_changes.json"
MEMORY_DIR     = PROJECT_DIR / "Memory"

# Fenced-block regexes
SEQUENCE_BLOCK_RE = re.compile(r"```sequence\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CHANGES_BLOCK_RE  = re.compile(r"```changes\s*(.*?)\s*```",  re.DOTALL | re.IGNORECASE)

# Valid attribute values
VALID_ROLE  = {"input", "output", None}
VALID_SIZE  = {None, "large"}
VALID_COLOR = {"Blue", "Red", "blue", "red"}


# ── Block parsing ─────────────────────────────────────────────────────────────

def extract_sequence_block(text: str) -> List[List]:
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
            raise ValueError(f"Entry {i}: pick_name and place_name must be strings.")
        if not entry[0].strip() or not entry[1].strip():
            raise ValueError(f"Entry {i}: names must not be empty.")
        if len(entry) == 3:
            if not isinstance(entry[2], (int, float)) or entry[2] <= 0:
                raise ValueError(
                    f"Entry {i}: gripper_close_width must be a positive number, got: {entry[2]!r}"
                )
    return data


def extract_changes_block(text: str) -> Dict[str, Any]:
    m = CHANGES_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```changes``` block found.")
    data = json.loads(m.group(1).strip())
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("Changes block must be a non-empty JSON object.")

    for obj_name, attrs in data.items():
        if not isinstance(obj_name, str) or not obj_name.strip():
            raise ValueError(f"Object name must be a non-empty string, got: {obj_name!r}")

        # Special top-level keys are list/dict values, not attribute dicts
        if obj_name in ("workspace", "priority", "kit_recipe", "part_compatibility"):
            continue

        if not isinstance(attrs, dict) or len(attrs) == 0:
            raise ValueError(f"Attributes for '{obj_name}' must be a non-empty object.")

        for attr, val in attrs.items():
            attr_lower = attr.lower()
            if attr_lower == "role" and val not in VALID_ROLE:
                raise ValueError(f"'{obj_name}'.role must be 'input', 'output', or null.")
            if attr_lower == "size" and val not in VALID_SIZE:
                raise ValueError(f"'{obj_name}'.size must be null or 'large'.")
            if attr_lower == "color" and val not in VALID_COLOR:
                raise ValueError(f"'{obj_name}'.color must be 'Blue' or 'Red'.")
            if attr_lower not in ("role", "size", "color", "Role", "Size", "Color"):
                raise ValueError(
                    f"'{obj_name}': unknown attribute '{attr}'. "
                    f"Allowed: role, size, color."
                )
    return data


def merge_changes(
    accumulated: Dict[str, Any],
    new_block: Dict[str, Any],
) -> Dict[str, Any]:
    merged = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
               for k, v in accumulated.items()}
    for key, value in new_block.items():
        if key in ("priority", "kit_recipe", "part_compatibility") and isinstance(value, list):
            merged[key] = value          # list keys replace entirely
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


# ── File saving ───────────────────────────────────────────────────────────────

def save_sequence(sequence: List[List]) -> Path:
    SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENCE_PATH.write_text(json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8")
    return SEQUENCE_PATH


def save_changes(changes: Dict[str, Any]) -> Path:
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8")
    return CHANGES_PATH


# ── LLM helpers ───────────────────────────────────────────────────────────────

def chat(client: OpenAI, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return (resp.choices[0].message.content or "").strip()


def _contains_word(text: str, words) -> bool:
    for w in words:
        if re.search(r'\b' + re.escape(w) + r'\b', text):
            return True
    return False


def is_finish(text: str) -> bool:
    t = text.strip().lower()
    return _contains_word(t, [
        "finish", "finalize", "done", "end", "export",
        "save", "write", "quit", "exit",
    ]) or any(x in t for x in ["last step", "that's all", "thats all"])


def is_yes(text: str) -> bool:
    t = text.strip().lower()
    return _contains_word(t, [
        "yes", "ok", "okay", "confirm", "confirmed", "sure", "correct",
    ]) or any(x in t for x in ["go ahead", "do it", "looks good"]) or t == "y"


def is_no(text: str) -> bool:
    t = text.strip().lower()
    return _contains_word(t, [
        "no", "nope", "cancel", "reject", "wrong", "redo",
    ]) or t == "n"


# ── Scene helpers (new PDDL-friendly structure) ───────────────────────────────

def slim_scene(state: dict) -> dict:
    """
    Produce the LLM-facing view from the new PDDL-friendly positions.json.

    Returns:
    {
        "workspace": {"operation_mode": ..., "batch_size": ...},
        "slots": {
            "Kit_0_Pos_1": {
                "role": "output",          ← propagated from parent receptacle
                "child_part": null
            },
            "Container_3_Pos_1": {
                "role": "input",
                "child_part": {"name": "Part_Blue_Nr_1", "color": "blue", "size": "large"}
            }
        },
        "parts": {}                        ← standalone parts (not in a slot)
    }
    """
    preds        = state.get("predicates", {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects", {})

    # role per receptacle
    role_map: Dict[str, Optional[str]] = {
        e["object"]: e.get("role")
        for e in preds.get("role", [])
    }

    # part attributes
    color_map = {e["part"]: e.get("color") for e in preds.get("color", [])}
    size_map  = {e["part"]: e.get("size")  for e in preds.get("size",  [])}

    # part → slot mapping
    part_in_slot: Dict[str, str] = {
        e["part"]: e["slot"] for e in preds.get("at", [])
    }

    # build slot view
    slots_view: Dict[str, Any] = {}
    for slot_name in objs.get("slots", []):
        parent = slot_belongs.get(slot_name)
        role   = role_map.get(parent) if parent else None
        slots_view[slot_name] = {"role": role, "child_part": None}

    # embed parts into their slots
    for part_name, slot_name in part_in_slot.items():
        if slot_name not in slots_view:
            continue
        slots_view[slot_name]["child_part"] = {
            "name":  part_name,
            "color": color_map.get(part_name),
            "size":  size_map.get(part_name),
        }

    # standalone parts (in objects.parts but not in any slot)
    in_slot_set = set(part_in_slot.keys())
    parts_view: Dict[str, Any] = {
        p: {"color": color_map.get(p), "size": size_map.get(p)}
        for p in objs.get("parts", [])
        if p not in in_slot_set
    }

    return {
        "workspace": state.get("workspace", {"operation_mode": None, "batch_size": None}),
        "slots":     slots_view,
        "parts":     parts_view,
    }


def apply_sequence_to_scene(state: dict, sequence: list) -> dict:
    """
    Update the PDDL-friendly state after the robot has executed a sequence.
    For each [pick_name, place_name, ...]:
      - Removes (part, source_slot) from predicates.at
      - Adds source_slot to predicates.slot_empty
      - Removes dest_slot from predicates.slot_empty
      - Adds (part, dest_slot) to predicates.at
      - Updates metric[part] position to match metric[dest_slot]
    Returns an updated deep copy.
    """
    import copy
    state = copy.deepcopy(state)
    preds = state.setdefault("predicates", {})
    metric = state.setdefault("metric", {})

    at_list:    List[Dict] = preds.setdefault("at", [])
    empty_list: List[str]  = preds.setdefault("slot_empty", [])

    for entry in sequence:
        pick_name  = entry[0]   # part name
        place_name = entry[1]   # destination slot name

        # find which slot the part is currently in
        source_slot: Optional[str] = None
        for item in at_list:
            if item["part"] == pick_name:
                source_slot = item["slot"]
                break

        if source_slot is None:
            print(f"⚠  apply_sequence: '{pick_name}' not found in predicates.at — skipping.")
            continue

        if place_name not in {e["slot"] for e in at_list} | set(empty_list):
            print(f"⚠  apply_sequence: destination slot '{place_name}' unknown — skipping.")
            continue

        # update predicates.at
        at_list[:] = [i for i in at_list if i["part"] != pick_name]
        at_list.append({"part": pick_name, "slot": place_name})

        # update slot_empty
        if source_slot not in empty_list:
            empty_list.append(source_slot)
        if place_name in empty_list:
            empty_list.remove(place_name)

        # update metric for the moved part
        dest_metric = metric.get(place_name, {})
        if dest_metric and pick_name in metric:
            for key in ("pos", "quat", "orientation"):
                if key in dest_metric:
                    metric[pick_name][key] = dest_metric[key]

    return state


# ── Config application helper ─────────────────────────────────────────────────

def _apply_and_save_config(accumulated_changes: Dict[str, Any]) -> None:
    from Configuration_Module.Apply_Config_Changes import apply_changes  # type: ignore

    if not POSITIONS_PATH.exists():
        print(f"⚠  positions.json not found at {POSITIONS_PATH.resolve()} — cannot apply changes.")
        return

    scene   = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    updated = apply_changes(scene, accumulated_changes)

    tmp = str(POSITIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(POSITIONS_PATH)
    print(f"✅  positions.json updated → {POSITIONS_PATH.resolve()}\n")


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

```changes
{
  "<receptacle_or_part_name>": {"<attribute>": <value>}
}
```

Allowed keys and values:
  RECEPTACLE name (Kit_*, Container_*)   → "role": "input" | "output" | null
  PART name (Part_*)                     → "size": null | "large"
                                         → "color": "Blue" | "Red"
  "workspace"                            → {"operation_mode": "sorting"|"kitting", "batch_size": N}
  "priority"                             → [{"color": "blue", "order": 1}, ...]
  "kit_recipe"                           → [{"kit": "Kit_0", "color": "blue", "quantity": 2}, ...]
  "part_compatibility"                   → [{"part_color": "blue", "allowed_in": ["Kit_0"]}, ...]

CRITICAL FORMAT RULES:
- Use the RECEPTACLE name (e.g. "Container_3", "Kit_0") for role changes,
  NOT individual slot names.
- Use the PART name (e.g. "Part_Blue_Nr_1") for size/color changes.
- Never invent names. Use verbatim names from the INPUT JSON.
- null means reset to default.

Example:
```changes
{
  "Container_3": {"role": "input"},
  "Kit_0": {"role": "output"},
  "Part_Blue_Nr_1": {"size": "large"},
  "workspace": {"operation_mode": "kitting"},
  "kit_recipe": [
    {"kit": "Kit_0", "color": "blue", "quantity": 2},
    {"kit": "Kit_0", "color": "red",  "quantity": 1}
  ],
  "priority": [{"color": "blue", "order": 1}, {"color": "red", "order": 2}]
}
```
"""


def build_system_prompt(mode: str) -> str:
    if mode == "reconfig":
        return """\
You are a robot workspace configurator. Your job is to track user-specified
attribute changes to workspace objects and output a structured changes block.

You will receive an INPUT JSON describing the current scene with:
  - "workspace": current operation_mode and batch_size.
  - "slots": named workspace positions (Kit_* or Container_*).
    Each slot has: role (null|"input"|"output"), child_part {name, color, size}.
    The role shown is the role of the parent receptacle (container or kit),
    applied to all its slots.
  - "parts": standalone parts not currently in any slot.

──────────────────────────────────────────────────────────────
WORKFLOW
──────────────────────────────────────────────────────────────
1. Give a SHORT scene summary: receptacles with their role, parts with color/size,
   current operation_mode.
2. Ask what attributes the user wants to change.
3. If truly ambiguous, ask ONE focused clarification question. Otherwise go to step 4.
4. Once you understand the request: output the changes block and ask "Confirm?"

""" + _CHANGES_BLOCK_RULES + """\

──────────────────────────────────────────────────────────────
CONFIRMATION
──────────────────────────────────────────────────────────────
- After proposing changes, always ask: "Confirm?"
- If confirmed → changes are saved. Ask if there is more to change.
- If rejected → discard and ask what to change.

""" + _COMMON_RULES

    else:  # motion
        return """\
You are a robot task planner. Your job is to translate natural-language task
descriptions into an ordered sequence of pick-and-place actions.

You will receive an INPUT JSON describing the current scene with:
  - "workspace": current operation_mode and batch_size.
  - "slots": named workspace positions (Kit_* or Container_*).
    Each slot has: role (null|"input"|"output"), child_part {name, color, size}.
  - "parts": standalone parts not in any slot.

──────────────────────────────────────────────────────────────
ROLE RESTRICTIONS — ENFORCE STRICTLY
──────────────────────────────────────────────────────────────
  - role = "input"  → CAN be picked FROM, CANNOT be placed INTO.
  - role = "output" → CAN be placed INTO, CANNOT be picked FROM.
  - role = null     → no restriction; either pick or place is allowed.

If a conflict exists, do NOT output a sequence. Explain the conflict and ask:
"Would you like to switch to reconfiguration mode to fix this?"
If yes → output exactly: SWITCH_TO_RECONFIG

──────────────────────────────────────────────────────────────
GRIPPER WIDTH
──────────────────────────────────────────────────────────────
  - size = null (standard) → gripper_close_width = 0.05 (omit from sequence)
  - size = "large"         → gripper_close_width = 0.06 (include as 3rd element)

──────────────────────────────────────────────────────────────
WORKFLOW
──────────────────────────────────────────────────────────────
1. Give a SHORT scene summary: part counts, roles, operation_mode.
2. Ask what task the user wants.
3. Check roles for all objects BEFORE proposing anything.
4. Ask ONE clarification if truly ambiguous. Otherwise go to step 5.
5. Output the sequence block and ask "Confirm?"

──────────────────────────────────────────────────────────────
OUTPUT BLOCK — SEQUENCE
──────────────────────────────────────────────────────────────
```sequence
[
  ["<pick_name>", "<place_name>"],
  ["<pick_name>", "<place_name>", 0.06]
]
```

pick_name  = part name from child_part.name  (e.g. "Part_Blue_Nr_1")
           = standalone part key             (e.g. "Part_Red_Nr_2")
place_name = destination slot key           (e.g. "Kit_0_Pos_1")

NEVER use slot names as pick targets. NEVER invent names.

──────────────────────────────────────────────────────────────
CONFIRMATION
──────────────────────────────────────────────────────────────
- After proposing a sequence, always ask: "Confirm?"
- If confirmed → sequence is saved. Ask if there is more to plan.
- If rejected → discard and ask what to change.

""" + _COMMON_RULES


# ── Pre-session setup ─────────────────────────────────────────────────────────

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
    from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
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


def select_scene() -> dict:
    options = [
        "Live vision  (capture new image with camera)",
        f"Current positions.json  ({POSITIONS_PATH})",
    ]
    idx = _pick_from_list("\nWhich scene do you want to use?", options)

    if idx == 0:
        print("\nStarting vision module …")
        from Vision_Module.Vision_Main import main as vision_main  # type: ignore
        vision_main()
        if not POSITIONS_PATH.exists():
            print(f"ERROR: Vision module did not produce {POSITIONS_PATH}")
            sys.exit(1)
        print("Loaded fresh scene from vision.")
    else:
        if not POSITIONS_PATH.exists():
            print(f"ERROR: positions.json not found at {POSITIONS_PATH.resolve()}")
            sys.exit(1)
        print(f"Loaded scene from: {POSITIONS_PATH}")

    state = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    return slim_scene(state)


# ── Main session loop ─────────────────────────────────────────────────────────

def run_session(client: OpenAI, mode: str) -> None:

    # ── Option 3: Execute robot motion ────────────────────────────────────────
    if mode == "execute":
        if not SEQUENCE_PATH.exists():
            print(f"\n⚠  No sequence.json found at {SEQUENCE_PATH.resolve()}")
            print("   Plan a motion sequence first (option 2).\n")
            return

        sequence = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
        print(f"\n▶  Executing {len(sequence)} step(s) from: {SEQUENCE_PATH.resolve()}")

        from Execution_Module.Robot_Main import main as robot_main  # type: ignore
        robot_main()
        print("\n── Execution complete. ──\n")

        if not POSITIONS_PATH.exists():
            print("⚠  positions.json not found — cannot update after execution.")
            return

        state   = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
        updated = apply_sequence_to_scene(state, sequence)

        tmp = str(POSITIONS_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)
        Path(tmp).replace(POSITIONS_PATH)
        print(f"✅  positions.json updated → {POSITIONS_PATH.resolve()}\n")
        return

    # ── Option 2 (PDDL path): skip LLM dialogue, run planner directly ─────────
    if mode == "motion":
        from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
        if USE_PDDL_PLANNER:
            _run_pddl_sequence()
            return

    # ── Options 1 & 2 (LLM path) ─────────────────────────────────────────────
    scene = select_scene()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(mode)},
        {
            "role": "user",
            "content": (
                "INPUT SCENE JSON (sole source of truth — use these names verbatim):\n"
                + json.dumps(scene, indent=2, ensure_ascii=False)
            ),
        },
        {
            "role": "user",
            "content": "Please give your scene summary and ask what task I want.",
        },
    ]

    pending_sequence: Optional[List[List]] = None
    pending_changes:  Optional[Dict[str, Any]] = None
    accumulated_changes: Dict[str, Any] = {}

    mode_label = "Reconfiguration" if mode == "reconfig" else "Motion Sequence"
    print(f"\n── Mode: {mode_label} ──\n")

    while True:
        assistant_text = chat(client, messages)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        if "SWITCH_TO_RECONFIG" in assistant_text:
            print("  [Role conflict — returning to mode selection.]\n")
            return

        try:
            pending_sequence = extract_sequence_block(assistant_text)
            print(f"  [Sequence proposal: {len(pending_sequence)} step(s)]\n")
        except Exception as e:
            if "```sequence" in (assistant_text or ""):
                print(f"  [WARNING: sequence block parse error — {e}]")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your sequence block failed to parse: {e}\n"
                        "Each entry must be [\"pick\", \"place\"] or [\"pick\", \"place\", 0.06].\n"
                        "Please rewrite the sequence block."
                    ),
                })
                continue

        try:
            pending_changes = extract_changes_block(assistant_text)
            attrs = sum(len(v) if isinstance(v, dict) else 1 for v in pending_changes.values())
            print(f"  [Changes proposal: {len(pending_changes)} key(s), ~{attrs} value(s)]\n")
        except Exception as e:
            if "```changes" in (assistant_text or ""):
                print(f"  [WARNING: changes block parse error — {e}]")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your changes block failed to parse: {e}\n"
                        "Use receptacle names (e.g. 'Container_3') for role changes.\n"
                        "Use part names (e.g. 'Part_Blue_Nr_1') for size/color changes.\n"
                        "Please rewrite the changes block."
                    ),
                })
                continue

        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        if is_finish(user_input):
            if pending_sequence is not None:
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Sequence saved ({len(pending_sequence)} pairs) → {saved.resolve()}")
            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
            if accumulated_changes:
                saved = save_changes(accumulated_changes)
                print(f"\n✅  Changes saved → {saved.resolve()}")
                _apply_and_save_config(accumulated_changes)
            print("\n── Session complete. ──\n")
            return

        has_pending = pending_sequence is not None or pending_changes is not None

        if has_pending and is_yes(user_input):
            if pending_sequence is not None:
                saved = save_sequence(pending_sequence)
                print(f"\n✅  Sequence confirmed ({len(pending_sequence)} step(s)) → {saved.resolve()}\n")
                messages.append({"role": "user",      "content": "Confirmed the sequence."})
                messages.append({"role": "assistant",  "content": "Sequence saved."})
                pending_sequence = None

            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
                total = sum(len(v) if isinstance(v, dict) else 1 for v in accumulated_changes.values())
                print(f"✅  Changes noted ({total} total). Will be written on 'done'.\n")
                messages.append({"role": "user",      "content": "Confirmed the changes."})
                messages.append({"role": "assistant",  "content": "Changes noted."})
                pending_changes = None

            user_input = input("Anything else? (or type 'done')\nYOU: ").strip()
            if is_finish(user_input):
                if accumulated_changes:
                    saved = save_changes(accumulated_changes)
                    print(f"\n✅  Changes saved → {saved.resolve()}")
                    _apply_and_save_config(accumulated_changes)
                print("\n── Session complete. ──\n")
                return
            if user_input:
                messages.append({"role": "user", "content": user_input})
            continue

        if has_pending and is_no(user_input):
            print("  [Proposal rejected.]\n")
            pending_sequence = None
            pending_changes  = None
            messages.append({
                "role": "user",
                "content": "Rejected. Discard that proposal and ask what I want to change.",
            })
            continue

        messages.append({"role": "user", "content": user_input})


def _run_pddl_sequence() -> None:
    """
    PDDL planning path for motion mode.
    Loads positions.json, runs the PDDL planner, saves sequence.json.
    """
    from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
    from pddl_planner import plan_sequence  # type: ignore

    if not POSITIONS_PATH.exists():
        print(f"❌ positions.json not found: {POSITIONS_PATH.resolve()}")
        return

    state = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))

    print("\n── PDDL Planner ──")
    print(f"  operation_mode : {state.get('workspace', {}).get('operation_mode', 'not set')}")
    print(f"  kit_recipe     : {state.get('predicates', {}).get('kit_recipe', [])}")
    print(f"  priority       : {state.get('predicates', {}).get('priority', [])}\n")

    sequence = plan_sequence(
        state,
        output_path=str(SEQUENCE_PATH),
        keep_pddl=True,
    )

    if sequence is None:
        print("❌ PDDL planning failed. Check roles, recipes, and Fast Downward installation.")
    else:
        print(f"\n✅  Sequence written → {SEQUENCE_PATH.resolve()}")
        print(json.dumps(sequence, indent=2))

    print("\n── PDDL planning complete. ──\n")


# ── Main entry point ──────────────────────────────────────────────────────────

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