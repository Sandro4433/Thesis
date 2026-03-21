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

from paths import LLM_INPUT_JSON, LLM_RESPONSE_JSON, CONFIGURATION_JSON

LLM_INPUT_PATH     = Path(LLM_INPUT_JSON.resolve())
CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
OUTPUT_DIR         = Path(LLM_RESPONSE_JSON.resolve()).parent
SEQUENCE_PATH  = OUTPUT_DIR / "sequence.json"
CHANGES_PATH   = OUTPUT_DIR / "workspace_changes.json"
MEMORY_DIR     = PROJECT_DIR / "Memory"

# Fenced-block regexes
SEQUENCE_BLOCK_RE = re.compile(r"```sequence\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CHANGES_BLOCK_RE  = re.compile(r"```changes\s*(.*?)\s*```",  re.DOTALL | re.IGNORECASE)
MAPPING_BLOCK_RE  = re.compile(r"```mapping\s*(.*?)\s*```",  re.DOTALL | re.IGNORECASE)

# Valid attribute values
VALID_ROLE      = {"input", "output", None}
VALID_COLOR     = {"Blue", "Red", "Green", "blue", "red", "green"}
VALID_FRAGILITY = {"normal", "fragile", None}


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
                f"[pick_name, place_name, 0.05], got: {entry!r}"
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
            if attr_lower == "color" and val not in VALID_COLOR:
                raise ValueError(f"'{obj_name}'.color must be 'Blue', 'Red', or 'Green'.")
            if attr_lower == "fragility" and val not in VALID_FRAGILITY:
                raise ValueError(f"'{obj_name}'.fragility must be 'normal' or 'fragile'.")
            if attr_lower not in ("role", "color", "fragility",
                                  "Role", "Color", "Fragility"):
                raise ValueError(
                    f"'{obj_name}': unknown attribute '{attr}'. "
                    f"Allowed: role, color, fragility."
                )
    return data


def extract_mapping_block(text: str) -> Dict[str, str]:
    """Parse ```mapping``` block from LLM response → {fresh_name: old_name|"new"}."""
    m = MAPPING_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```mapping``` block found.")
    data = json.loads(m.group(1).strip())
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("Mapping block must be a non-empty JSON object.")
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"Mapping entries must be string→string, got {k!r}→{v!r}")
        if not k.startswith("Part_"):
            raise ValueError(f"Keys must be Part_* names from the new scan, got {k!r}")
        if v != "new" and not v.startswith("Part_"):
            raise ValueError(f"Values must be old Part_* names or 'new', got {v!r}")
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
    Produce the LLM-facing view from the new PDDL-friendly configuration.json.

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
                "child_part": {"name": "Part_1", "color": "blue"}
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

    # part → slot mapping
    part_in_slot: Dict[str, str] = {
        e["part"]: e["slot"] for e in preds.get("at", [])
    }

    # fragility lookup (built once, used for both slots and standalone parts)
    frag_map: Dict[str, str] = {
        e["part"]: e.get("fragility", "normal")
        for e in preds.get("fragility", [])
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
            "name":      part_name,
            "color":     color_map.get(part_name),
            "fragility": frag_map.get(part_name, "normal"),
        }

    # standalone parts (in objects.parts but not in any slot)
    in_slot_set = set(part_in_slot.keys())
    parts_view: Dict[str, Any] = {
        p: {
            "color":     color_map.get(p),
            "fragility": frag_map.get(p, "normal"),
        }
        for p in objs.get("parts", [])
        if p not in in_slot_set
    }

    return {
        "workspace": state.get("workspace", {"operation_mode": None, "batch_size": None}),
        "slots":     slots_view,
        "parts":     parts_view,
    }


# apply_sequence_to_scene removed — see Configuration_Module/Apply_Sequence_Changes.py


# ── Config application helper ─────────────────────────────────────────────────

def _apply_and_save_config(accumulated_changes: Dict[str, Any]) -> None:
    from Configuration_Module.Apply_Config_Changes import apply_changes  # type: ignore

    if not CONFIGURATION_PATH.exists():
        print(f"⚠  configuration.json not found at {CONFIGURATION_PATH.resolve()} — cannot apply changes.")
        return

    scene   = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    updated = apply_changes(scene, accumulated_changes)

    tmp = str(CONFIGURATION_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(CONFIGURATION_PATH)
    print(f"✅  Configuration updated.")

    _refresh_annotated_image(updated)


def _refresh_annotated_image(state: Dict[str, Any]) -> None:
    """Redraw latest_image.png with current part names and FRAGILE labels."""
    file_exchange = PROJECT_DIR / "File_Exchange"
    base_path  = file_exchange / "latest_image_base.png"
    pmap_path  = file_exchange / "latest_pixel_map.json"
    out_path   = file_exchange / "latest_image.png"

    if not base_path.exists() or not pmap_path.exists():
        return

    try:
        import cv2
        from Vision_Module.pipeline import annotate_parts  # type: ignore

        img = cv2.imread(str(base_path))
        if img is None:
            return

        pixel_map = json.loads(pmap_path.read_text(encoding="utf-8"))

        # Build fragility lookup from current config state
        preds = state.get("predicates", {})

        fragile_set: set = set()
        for entry in preds.get("fragility", []):
            if entry.get("fragility") == "fragile":
                fragile_set.add(entry["part"])

        annotate_parts(img, pixel_map, fragile_set=fragile_set)
        cv2.imwrite(str(out_path), img)
        print("✅  Image updated.")
    except Exception as exc:
        print(f"  ⚠  Image refresh failed: {exc}")


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
  PART name (Part_*)                     → "color": "Blue" | "Red" | "Green"
                                         → "fragility": "normal" | "fragile"
  "workspace"                            → {"operation_mode": "sorting"|"kitting", "batch_size": N}
  "priority"                             → [{"color": "blue", "order": 1}, ...]
  "kit_recipe"                           → [{"kit": "Kit_0", "color": "blue", "quantity": 2}, ...]
  "part_compatibility"                   → [{"part_color": "blue", "allowed_in": ["Container_1"]}, ...]

CRITICAL FORMAT RULES:
- Use the RECEPTACLE name (e.g. "Container_3", "Kit_0") for role changes,
  NOT individual slot names.
- Use the PART name (e.g. "Part_1") for color changes.
- Never invent names. Use verbatim names from the INPUT JSON.
- null means reset to default.

Example (kitting):
```changes
{
  "Container_3": {"role": "input"},
  "Kit_0": {"role": "output"},
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
You are a robot workspace configurator.

COMMUNICATION RULES — FOLLOW STRICTLY:
- Be extremely concise. No filler, no greetings, no repetition.
- Never restate the scene JSON or repeat information the user already has.
- Do NOT output a scene summary. The scene description panel handles that.
- Your FIRST message must be ONLY: "What would you like to change?"
- Ask at most ONE clarification question per turn. No multi-part questions.
- When outputting a changes block, add ONLY "Confirm?" after it. No explanation.
- After confirmation, respond ONLY with: "Anything else?"
- After rejection, respond ONLY with: "What should I change?"

You will receive an INPUT JSON with:
  - "workspace": operation_mode, batch_size
  - "slots": Kit_*/Container_* positions with role and child_part
  - "parts": standalone parts

SORTING INFERENCE (apply automatically when user says "sort"):
A — Infer destination containers from existing same-color contents.
B — Source = receptacles with mixed colors → role="input".
C — Always emit: source roles, destination roles, workspace, part_compatibility.
D — If a container has mixed colors, ask which color it should receive.

""" + _CHANGES_BLOCK_RULES + """

CONFIRMATION:
- After proposing changes: output block + "Confirm?"
- Confirmed → "Anything else?"
- Rejected → "What should I change?"

""" + _COMMON_RULES

    else:  # motion
        return """\
You are a robot task planner.

COMMUNICATION RULES — FOLLOW STRICTLY:
- Be extremely concise. No filler, no greetings, no repetition.
- Never restate the scene JSON or repeat information the user already has.
- Do NOT output a scene summary. The scene description panel handles that.
- Your FIRST message must be ONLY: "What task do you want to execute?"
- Ask at most ONE clarification question per turn.
- When outputting a sequence block, add ONLY "Confirm?" after it.
- After confirmation: "Anything else?"
- After rejection: "What should I change?"

You will receive an INPUT JSON with:
  - "workspace": operation_mode, batch_size
  - "slots": Kit_*/Container_* positions with role and child_part
  - "parts": standalone parts

ROLE RESTRICTIONS:
  - role="input" → pick FROM only.  role="output" → place INTO only.  null → either.
  If conflict: explain briefly + "Switch to reconfiguration mode?" → if yes: SWITCH_TO_RECONFIG

GRIPPER WIDTH:
  - All parts use standard gripper width 0.05 (omit from sequence entries).

OUTPUT:
```sequence
[["<pick>", "<place>"], ["<pick>", "<place>"]]
```
pick = part name, place = slot name. Never use slots as pick targets.

CONFIRMATION:
- Output block + "Confirm?"
- Confirmed → "Anything else?"
- Rejected → "What should I change?"

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


def select_reconfig_source() -> str:
    idx = _pick_from_list("\nHow do you want to load the scene?", [
        "Fresh scan — new picture, start from scratch (no memory)",
        "Fresh scan — new picture, keep high-level config from memory",
        "Memory    — use existing configuration.json as-is",
    ])
    return ["reconfig_fresh", "reconfig_update", "reconfig_memory"][idx]


def _run_vision_subprocess() -> None:
    """Run Vision_Main in a clean subprocess (avoids libapriltag segfault)."""
    import subprocess as _sp
    vision_main_path = PROJECT_DIR / "Vision_Module" / "Vision_Main.py"
    print("\nStarting vision module …")
    result = _sp.run([sys.executable, str(vision_main_path)], cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        raise RuntimeError(
            f"Vision_Main subprocess exited with code {result.returncode}."
        )
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
        print(f"Loaded scene from: {CONFIGURATION_PATH}")

    state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    return slim_scene(state)


# ── Scene update dialogue (LLM-guided) ──────────────────────────────────────

def _build_update_system_prompt() -> str:
    return """\
You resolve part identities after a new camera scan.

COMMUNICATION RULES — FOLLOW STRICTLY:
- Be extremely concise. No filler, no greetings, no preamble.
- Never repeat the scene JSONs or the auto-match analysis back to the user.
- If ALL parts are auto-matched with no ambiguities: immediately output the
  mapping block + "Confirm?" — nothing else.
- If there ARE ambiguities: list ONLY the unresolved parts (one line each),
  then ask ONE focused question. Example:
    "Part_3 (blue) detected at Part_5's old position. Part_5 was red.
     Is this the same part (moved) or a new part?"
- After each answer, either ask the next ambiguity or output the mapping.
- After the mapping block, add ONLY "Confirm?" — no summary, no explanation.

CONTEXT:
- Receptacles have AprilTags → names are stable, handled automatically.
- Parts are detected by colour/shape → names can change between scans.
- Auto-match analysis is provided. Trust it for confirmed matches.

MAPPING RULES:
- Keys = ALL Part_* names from the NEW scan.
- Values = old Part_* name (identity preserved) or "new" (addition).
- Parts absent from both keys and values → removed.
- Swaps: two parts exchanged positions. Map each detection to its correct old identity.

OUTPUT:
```mapping
{"<new_name>": "<old_name_or_new>", ...}
```
Then: "Confirm?"
"""


def _run_update_dialogue(client: OpenAI) -> None:
    """
    LLM-guided scene update dialogue.

    1. Captures a new camera image (via prepare_update).
    2. Presents the old scene, new scan, and auto-match analysis to the LLM.
    3. The LLM converses with the user to resolve ambiguous part identities.
    4. Once confirmed, applies the mapping and saves.
    """
    from Configuration_Module.Update_Scene import (       # type: ignore
        prepare_update, build_update_context, apply_update_mapping,
    )

    print("\n── Scene update ──\n")

    try:
        old_state, fresh_state = prepare_update()
    except Exception:
        return          # error already printed by prepare_update

    old_scene = slim_scene(old_state)
    new_scene = slim_scene(fresh_state)
    context   = build_update_context(old_state, fresh_state)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_update_system_prompt()},
        {
            "role": "user",
            "content": (
                "OLD SCENE:\n"
                + json.dumps(old_scene, indent=2, ensure_ascii=False)
                + "\n\nNEW SCAN:\n"
                + json.dumps(new_scene, indent=2, ensure_ascii=False)
                + "\n\nAUTO-MATCH:\n"
                + context
                + "\n\nResolve identities and output the mapping."
            ),
        },
    ]

    pending_mapping: Optional[Dict[str, str]] = None

    while True:
        assistant_text = chat(client, messages)
        print(f"\nASSISTANT:\n{assistant_text}\n")
        messages.append({"role": "assistant", "content": assistant_text})

        # ── try to extract mapping block ──────────────────────────────────────
        try:
            pending_mapping = extract_mapping_block(assistant_text)

            # Validate completeness: every fresh part must be in the mapping
            fresh_parts = set(fresh_state.get("objects", {}).get("parts", []))
            mapped_keys = set(pending_mapping.keys())
            missing_keys = fresh_parts - mapped_keys
            extra_keys   = mapped_keys - fresh_parts

            if missing_keys:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Mapping is missing these parts from the new scan: "
                        f"{sorted(missing_keys)}. Please include ALL parts."
                    ),
                })
                pending_mapping = None
                continue
            if extra_keys:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Mapping includes parts not in the new scan: "
                        f"{sorted(extra_keys)}. Please fix."
                    ),
                })
                pending_mapping = None
                continue

            # Validate: old names in values must actually exist in old config
            old_parts = set(old_state.get("objects", {}).get("parts", []))
            for fresh_name, target in pending_mapping.items():
                if target != "new" and target not in old_parts:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"'{target}' is not a part in the old config. "
                            f"Old parts are: {sorted(old_parts)}. Please fix."
                        ),
                    })
                    pending_mapping = None
                    break
            if pending_mapping is None:
                continue

        except ValueError as e:
            if "```mapping" in (assistant_text or ""):
                print(f"  [WARNING: mapping block parse error — {e}]")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your mapping block failed to parse: {e}\n"
                        "Please rewrite it."
                    ),
                })
                pending_mapping = None
                continue
        except Exception:
            pass

        # ── user input ────────────────────────────────────────────────────────
        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        if is_finish(user_input) and pending_mapping is None:
            print("\n── Scene update cancelled. Old config restored. ──\n")
            return

        if pending_mapping is not None and (is_yes(user_input) or is_finish(user_input)):
            apply_update_mapping(old_state, fresh_state, pending_mapping)
            print("Loaded fresh scene from vision.")   # signal for GUI
            return

        if pending_mapping is not None and is_no(user_input):
            pending_mapping = None
            messages.append({
                "role": "user",
                "content": (
                    "Mapping rejected. Ask what needs to change and "
                    "produce a corrected mapping."
                ),
            })
            continue

        messages.append({"role": "user", "content": user_input})


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

        from Configuration_Module.Apply_Sequence_Changes import apply_and_save  # type: ignore
        updated = apply_and_save(CONFIGURATION_PATH, sequence, save_memory=True)

        if updated:
            from Configuration_Module.Update_Scene import run_post_execution_rescan  # type: ignore
            run_post_execution_rescan(updated)
        return

    # ── Option 2 (PDDL path): skip LLM dialogue, run planner directly ─────────
    if mode == "motion":
        from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
        if USE_PDDL_PLANNER:
            _run_pddl_sequence()
            return

    # ── Option 1: Reconfiguration — choose scene source ──────────────────────
    if mode == "reconfig":
        sub = select_reconfig_source()

        if sub == "reconfig_fresh":
            try:
                _run_vision_subprocess()
            except RuntimeError as e:
                print(f"\n❌  Vision failed: {e}\n")
                return
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found at {CONFIGURATION_PATH.resolve()}")
                return
            print("Loaded fresh scene from vision.")

        elif sub == "reconfig_update":
            _run_update_dialogue(client)
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found at {CONFIGURATION_PATH.resolve()}")
                return

        else:  # reconfig_memory
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found at {CONFIGURATION_PATH.resolve()}")
                return
            print(f"Loaded scene from: {CONFIGURATION_PATH}")

        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        scene = slim_scene(state)

    else:
        # ── Option 2 (LLM motion path) — still uses select_scene() ──────────
        scene = select_scene()

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
        except Exception as e:
            if "```sequence" in (assistant_text or ""):
                print(f"  [WARNING: sequence block parse error — {e}]")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your sequence block failed to parse: {e}\n"
                        "Each entry must be [\"pick\", \"place\"] or [\"pick\", \"place\", 0.05].\n"
                        "Please rewrite the sequence block."
                    ),
                })
                continue

        try:
            pending_changes = extract_changes_block(assistant_text)
        except Exception as e:
            if "```changes" in (assistant_text or ""):
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
                continue

        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        if is_finish(user_input):
            if pending_sequence is not None:
                save_sequence(pending_sequence)
                print("✅  Sequence saved.")
            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
            if accumulated_changes:
                save_changes(accumulated_changes)
                _apply_and_save_config(accumulated_changes)
                print("✅  Changes saved.")
            print("\n── Session complete. ──\n")
            return

        has_pending = pending_sequence is not None or pending_changes is not None

        if has_pending and is_yes(user_input):
            if pending_sequence is not None:
                save_sequence(pending_sequence)
                print(f"✅  Sequence confirmed.\n")
                messages.append({"role": "user",      "content": "Confirmed the sequence."})
                messages.append({"role": "assistant",  "content": "Sequence saved."})
                pending_sequence = None

            if pending_changes is not None:
                accumulated_changes = merge_changes(accumulated_changes, pending_changes)
                print(f"✅  Changes noted.\n")
                messages.append({"role": "user",      "content": "Confirmed the changes."})
                messages.append({"role": "assistant",  "content": "Changes noted."})
                pending_changes = None

            user_input = input("Anything else?\nYOU: ").strip()
            if is_finish(user_input):
                if accumulated_changes:
                    save_changes(accumulated_changes)
                    _apply_and_save_config(accumulated_changes)
                    print("✅  Changes saved.")
                print("\n── Session complete. ──\n")
                return
            if user_input:
                messages.append({"role": "user", "content": user_input})
            continue

        if has_pending and is_no(user_input):
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
    Loads configuration.json, runs the PDDL planner, saves sequence.json.
    """
    from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
    from pddl_planner import plan_sequence  # type: ignore

    if not CONFIGURATION_PATH.exists():
        print(f"❌ configuration.json not found: {CONFIGURATION_PATH.resolve()}")
        return

    state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

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