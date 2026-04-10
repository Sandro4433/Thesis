"""
session_handler.py — Pipeline orchestrator.

Coordinates the four modules (Vision, Communication, Configuration, Execution)
without containing any LLM logic itself.  All LLM interaction is delegated to
Communication_Module.API_Main.

Responsibilities:
  - Mode selection and menu UI
  - Spawning the vision subprocess
  - Running the update-scene pipeline (vision → auto-match → LLM dialogue → merge)
  - Launching robot execution and PDDL planning
  - Scene loading (fresh / update / memory)
  - Config application and image refresh after changes

The GUI and Main.py import from here instead of reaching into API_Main for
orchestration.
"""
from __future__ import annotations

import json
import subprocess as _sp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Project root setup ───────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import CONFIGURATION_JSON, LLM_RESPONSE_JSON

CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
OUTPUT_DIR         = Path(LLM_RESPONSE_JSON.resolve()).parent
SEQUENCE_PATH      = OUTPUT_DIR / "sequence.json"
CHANGES_PATH       = OUTPUT_DIR / "workspace_changes.json"
MEMORY_DIR         = PROJECT_DIR / "Memory"


# ═══════════════════════════════════════════════════════════════════════════════
# Menu / UI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pick_from_list(prompt: str, options: List[str]) -> int:
    """Present a numbered menu and return the 0-based index of the choice."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Vision subprocess
# ═══════════════════════════════════════════════════════════════════════════════

def run_vision() -> None:
    """Spawn Vision_Main in a clean subprocess."""
    vision_main_path = PROJECT_DIR / "Vision_Module" / "Vision_Main.py"
    print("\nStarting vision module …")
    result = _sp.run([sys.executable, str(vision_main_path)], cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        raise RuntimeError(
            f"Vision_Main subprocess exited with code {result.returncode}."
        )
    print("Vision complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Config application & image refresh
# ═══════════════════════════════════════════════════════════════════════════════

def apply_and_save_config(accumulated_changes: Dict[str, Any]) -> None:
    """Apply accumulated LLM changes to configuration.json and redraw image."""
    from Configuration_Module.Apply_Config_Changes import apply_changes
    import io, contextlib

    if not CONFIGURATION_PATH.exists():
        print("⚠  configuration.json not found — cannot apply changes.")
        return

    scene = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

    with contextlib.redirect_stdout(io.StringIO()):
        updated = apply_changes(scene, accumulated_changes)

    tmp = str(CONFIGURATION_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    Path(tmp).replace(CONFIGURATION_PATH)
    print("✅  Configuration updated.")

    refresh_annotated_image(updated)


def refresh_annotated_image(state: Dict[str, Any]) -> None:
    """Redraw latest_image.png with current part names and FRAGILE labels."""
    file_exchange = PROJECT_DIR / "File_Exchange"
    base_path = file_exchange / "latest_image_base.png"
    pmap_path = file_exchange / "latest_pixel_map.json"
    out_path  = file_exchange / "latest_image.png"

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


# ═══════════════════════════════════════════════════════════════════════════════
# Scene loading
# ═══════════════════════════════════════════════════════════════════════════════

def select_scene() -> dict:
    """Menu for motion-planning mode: live vision or stored config."""
    from Communication_Module.scene_helpers import slim_scene

    options = [
        "Live vision  (capture new image with camera)",
        f"Current configuration.json  ({CONFIGURATION_PATH})",
    ]
    idx = _pick_from_list("\nWhich scene do you want to use?", options)

    if idx == 0:
        try:
            run_vision()
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


def load_scene(client: OpenAI, mode: str) -> Optional[dict]:
    """Load the scene for the current mode. Returns slim scene dict or None."""
    from Communication_Module.scene_helpers import slim_scene

    if mode == "reconfig":
        sub = select_reconfig_source()

        if sub == "reconfig_fresh":
            try:
                run_vision()
            except RuntimeError as e:
                print(f"\n❌  Vision failed: {e}\n")
                return None
            if not CONFIGURATION_PATH.exists():
                print("ERROR: configuration.json not found")
                return None
            print("Loaded fresh scene from vision.")

        elif sub == "reconfig_update":
            run_update_pipeline(client)
            if not CONFIGURATION_PATH.exists():
                print("ERROR: configuration.json not found")
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


# ═══════════════════════════════════════════════════════════════════════════════
# Update Scene pipeline  (vision → auto-match → LLM dialogue → merge → save)
# ═══════════════════════════════════════════════════════════════════════════════

# Sentinel returned by run_update_dialogue when the user requests a recapture
RECAPTURE_SENTINEL = "__RECAPTURE__"


def run_update_pipeline(client: OpenAI) -> None:
    """
    Full update-scene pipeline:
      1. Run vision subprocess (via prepare_update)
      2. Auto-match parts and re-annotate image
      3. Hand off to the LLM for user dialogue (API_Main)
      4. Apply confirmed mapping and save

    If the user requests a recapture during the dialogue, steps 1-3 are
    repeated with a fresh vision scan while keeping the original old_state.
    """
    from Configuration_Module.Update_Scene import (
        prepare_update,
        prepare_recapture,
        apply_update_mapping,
        redraw_image_with_auto_matches,
    )
    from Communication_Module.API_Main import run_update_dialogue

    # ── Step 1: initial vision + state capture ─────────────────────────────
    try:
        old_state, fresh_state = prepare_update()
    except RuntimeError as e:
        print(f"\n❌  Vision failed: {e}\n")
        return

    if old_state is None or fresh_state is None:
        print("⚠  Update aborted (missing state files).")
        return

    # ── Recapture loop ─────────────────────────────────────────────────────
    while True:
        # ── Step 2: auto-match + image re-annotation ───────────────────
        image_rename_map = redraw_image_with_auto_matches(old_state, fresh_state)

        # ── Step 3: LLM dialogue ───────────────────────────────────────
        mapping = run_update_dialogue(
            client, old_state, fresh_state, image_rename_map,
        )

        # ── Recapture requested? ───────────────────────────────────────
        if mapping == RECAPTURE_SENTINEL:
            print("\n── Recapturing image … ──\n")
            try:
                fresh_state = prepare_recapture(old_state)
            except RuntimeError as e:
                print(f"\n❌  Recapture failed: {e}\n")
                print("Continuing with previous scan.\n")
            continue  # loop back to step 2

        # ── Step 4: apply mapping and save ─────────────────────────────
        if mapping is not None:
            apply_update_mapping(old_state, fresh_state, mapping, image_rename_map)
            print("✅  Update complete.\n")
        else:
            print("⚠  Update dialogue was cancelled.\n")
        break


# ═══════════════════════════════════════════════════════════════════════════════
# Robot execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_execution() -> None:
    """Spawn robot execution in a subprocess."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# PDDL planning
# ═══════════════════════════════════════════════════════════════════════════════

def run_pddl_sequence() -> None:
    """Run the PDDL planner on the current configuration."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# File I/O helpers (used by the LLM conversation loop in API_Main)
# ═══════════════════════════════════════════════════════════════════════════════

def save_sequence(sequence: List[List]) -> Path:
    SEQUENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENCE_PATH.write_text(
        json.dumps(sequence, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return SEQUENCE_PATH


def save_changes(changes: Dict[str, Any]) -> Path:
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(
        json.dumps(changes, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return CHANGES_PATH


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level session runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_session(client: OpenAI, mode: str) -> None:
    """
    Top-level dispatcher.  Routes to the correct pipeline based on mode,
    then hands off to API_Main only for the LLM conversation loop.
    """
    from Vision_Module.config import USE_PDDL_PLANNER

    # ── Execute ──────────────────────────────────────────────────────────
    if mode == "execute":
        run_execution()
        return

    # ── PDDL planning (motion mode with PDDL enabled) ───────────────────
    if mode == "motion" and USE_PDDL_PLANNER:
        run_pddl_sequence()
        return

    # ── Load scene (handles fresh / update / memory internally) ──────────
    scene = load_scene(client, mode)
    if scene is None:
        return

    # ── Hand off to the LLM conversation loop ────────────────────────────
    from Communication_Module.API_Main import run_conversation
    run_conversation(client, mode, scene)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
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