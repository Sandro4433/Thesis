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

# Ensure project root is on sys.path when imported or run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI

from Core.paths import (
    PROJECT_DIR, CONFIGURATION_PATH, SEQUENCE_PATH, CHANGES_PATH,
    MEMORY_DIR, WORKSPACE_DIR, save_atomic, save_to_memory,
)


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
    """Apply accumulated LLM changes to configuration.json, save to Memory, and redraw image."""
    from Configuration_Module.apply_config_changes import apply_changes
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

    # Save timestamped copy to Memory/
    save_config_to_memory(updated)

    refresh_annotated_image(updated)


def save_config_to_memory(state: Dict[str, Any]) -> Path:
    """Save a timestamped configuration to Memory/. Delegates to paths.save_to_memory."""
    dest = save_to_memory(state, label="session")
    print(f"✅  State archived → {dest.name}")
    return dest


def list_memory_configs() -> List[Dict[str, str]]:
    """List all configuration files in Memory/, sorted newest first.

    Returns a list of dicts with keys: name, path, date, time.
    Recognises:
      - configuration_{label}_YYYYMMDD_HHMMSS.json (current convention)
      - configuration_DDMMYYYY_HHMM.json           (legacy convention)
      - Any other .json file (uses file mtime)
    """
    import re as _re

    if not MEMORY_DIR.exists():
        return []

    # Current convention:  configuration_{label}_YYYYMMDD_HHMMSS.json
    pat_new = _re.compile(
        r"^configuration_[a-z_]+_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.json$"
    )
    # Legacy convention:  configuration_DDMMYYYY_HHMM.json
    pat_old = _re.compile(
        r"^configuration_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})\.json$"
    )
    configs = []
    for f in sorted(MEMORY_DIR.iterdir(), reverse=True):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        m_new = pat_new.match(f.name)
        m_old = pat_old.match(f.name)
        if m_new:
            yyyy, mm, dd, hh, mi, ss = m_new.groups()
            configs.append({
                "name": f.name,
                "path": str(f),
                "date": f"{dd}.{mm}.{yyyy}",
                "time": f"{hh}:{mi}:{ss}",
                "sort_key": f"{yyyy}{mm}{dd}{hh}{mi}{ss}",
            })
        elif m_old:
            dd, mm, yyyy, hh, mi = m_old.groups()
            configs.append({
                "name": f.name,
                "path": str(f),
                "date": f"{dd}.{mm}.{yyyy}",
                "time": f"{hh}:{mi}",
                "sort_key": f"{yyyy}{mm}{dd}{hh}{mi}00",
            })
        else:
            mtime = f.stat().st_mtime
            from datetime import datetime as _dt
            dt = _dt.fromtimestamp(mtime)
            configs.append({
                "name": f.name,
                "path": str(f),
                "date": dt.strftime("%d.%m.%Y"),
                "time": dt.strftime("%H:%M"),
                "sort_key": dt.strftime("%Y%m%d%H%M%S"),
            })

    configs.sort(key=lambda x: x["sort_key"], reverse=True)
    return configs


def load_config_from_memory(path: str) -> bool:
    """Load a configuration from Memory/ into the active configuration.json.
    Returns True on success."""
    src = Path(path)
    if not src.exists():
        print(f"⚠  File not found: {src}")
        return False
    try:
        state = json.loads(src.read_text(encoding="utf-8"))
        save_atomic(CONFIGURATION_PATH, state)
        return True
    except Exception as exc:
        print(f"⚠  Failed to load config: {exc}")
        return False


def refresh_annotated_image(state: Dict[str, Any]) -> None:
    """Redraw latest_image.png with current part names and FRAGILE labels."""
    workspace_dir = WORKSPACE_DIR
    base_path = workspace_dir / "latest_image_base.png"
    pmap_path = workspace_dir / "latest_pixel_map.json"
    out_path  = workspace_dir / "latest_image.png"

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
            # The Done button sets the cancel event to interrupt in-flight LLM
            # calls, but in the update pipeline "done" means "accept and move on"
            # rather than "abort the whole session".  If the event is still set
            # here (from auto-commit or any other path), the reconfiguration
            # conversation that follows would be killed on its very first LLM
            # call.  Clear it now so the conversation starts cleanly.
            try:
                from Communication_Module import api_main as _api_mod
                if _api_mod._cancel_event is not None:
                    _api_mod._cancel_event.clear()
            except Exception:
                pass
            if not CONFIGURATION_PATH.exists():
                print("ERROR: configuration.json not found")
                return None

        else:  # reconfig_memory
            if not CONFIGURATION_PATH.exists():
                print(f"ERROR: configuration.json not found at {CONFIGURATION_PATH.resolve()}")
                return None

        state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        return slim_scene(state)
    else:
        return select_scene()


# ═══════════════════════════════════════════════════════════════════════════════
# Update Scene pipeline  (vision → auto-match → LLM dialogue → merge → save)
# ═══════════════════════════════════════════════════════════════════════════════


def _image_ok(text: str) -> Optional[bool]:
    """Interpret the user's response to 'Is the new image okay?'

    Returns True  → image accepted
            False → wants recapture
            None  → ambiguous (ask again)
    """
    t = text.strip().lower()
    if not t:
        return None

    # Positive signals
    pos = {"yes", "y", "ye", "yep", "yeah", "yea", "yup", "ja", "jo",
           "ok", "okay", "sure", "correct", "fine", "good", "great",
           "perfect", "looks good", "looks fine", "looks ok", "looks okay",
           "that works", "go ahead", "do it", "proceed", "accept",
           "confirmed", "confirm", "alright"}
    if t in pos or any(t.startswith(p) for p in ("looks good", "looks fine",
                                                   "looks ok", "that works")):
        return True

    # Negative signals
    neg = {"no", "n", "nah", "nope", "nej", "wrong", "bad", "redo",
           "again", "retake", "recapture", "another", "retry", "re-take",
           "not ok", "not okay", "not good", "no good"}
    if t in neg or any(t.startswith(p) for p in ("not ok", "not good",
                                                   "no good", "take another")):
        return False

    return None


def run_update_pipeline(client: OpenAI) -> None:
    """
    Full update-scene pipeline:
      1. Run vision subprocess (via prepare_update)
      2. Auto-match parts and re-annotate image
      2b. Ask user if the image looks okay; recapture if not
      3. Hand off to the LLM for user dialogue (API_Main)
      4. Apply confirmed mapping and save

    IMAGE CONSISTENCY GUARANTEE
    ───────────────────────────
    redraw_image_with_auto_matches() writes latest_image.png to disk early
    (before the user has confirmed anything) so the GUI can show it.  If the
    pipeline exits at any point before apply_update_mapping() commits the new
    configuration, the image on disk would no longer match configuration.json.

    To prevent that, this function always restores the annotated image to
    match the current configuration.json on any early exit — whether the user
    typed "done", pressed Done, or the LLM call was aborted.
    """
    from Configuration_Module.update_scene import (
        prepare_update,
        prepare_recapture,
        apply_update_mapping,
        redraw_image_with_auto_matches,
    )
    from Communication_Module.api_main import run_update_dialogue

    # ── Step 1: initial vision + state capture ─────────────────────────────
    try:
        old_state, fresh_state = prepare_update()
    except RuntimeError as e:
        print(f"\n❌  Vision failed: {e}\n")
        return

    if old_state is None or fresh_state is None:
        print("⚠  Update aborted (missing state files).")
        return

    # Track whether we committed a new config + image successfully.
    _committed = False

    # Back up latest_image.png BEFORE Step 2 overwrites it with auto-matched
    # annotations.  On any early exit we restore this backup so the image
    # stays in sync with configuration.json (already restored to old_state
    # by prepare_update).
    #
    # We cannot use refresh_annotated_image(old_state) for this: that function
    # reads latest_pixel_map.json (fresh scan — fresh part names) and draws
    # those names verbatim.  The old config has different names, so image and
    # config would still be out of sync after "restoration".  A straight file
    # copy is the only reliable way to get back the pre-update image.
    import shutil as _shutil
    _fe         = PROJECT_DIR / "workspace"
    _img_path   = _fe / "latest_image.png"
    _img_backup = _fe / "latest_image_pre_update.png"

    if _img_path.exists():
        try:
            _shutil.copy2(str(_img_path), str(_img_backup))
        except Exception as _bk_err:
            print(f"  ⚠  Could not back up image before update: {_bk_err}")
            _img_backup = None
    else:
        _img_backup = None

    try:
        # ── Image confirmation + recapture loop ────────────────────────────
        while True:
            # ── Step 2: auto-match + image re-annotation ───────────────
            image_rename_map = redraw_image_with_auto_matches(old_state, fresh_state)

            # ── Step 2b: ask user if the captured image is acceptable ──
            print("\nASSISTANT:\nA new image has been captured. "
                  "Does the image look okay, or should I take another picture? "
                  "(Press 'done' to accept the image and apply auto-matching "
                  "without further dialogue.)\n")

            while True:
                answer = input("YOU: ").strip()
                if not answer:
                    continue
                # "done" = accept the image and commit using auto-match only,
                # skipping the LLM override dialogue entirely.
                if answer.lower() in ("done", "exit", "quit"):
                    print("── Applying auto-match and updating config … ──\n")
                    apply_update_mapping(old_state, fresh_state, {}, image_rename_map)
                    _committed = True
                    print("✅  Update complete.\n")
                    return
                # "cancel" = abort entirely, restore old image + config
                if answer.lower() == "cancel":
                    print("⚠  Update cancelled.\n")
                    return          # finally block restores the old image
                verdict = _image_ok(answer)
                if verdict is True:
                    break
                elif verdict is False:
                    # ── Recapture ──────────────────────────────────────
                    print("\n── Recapturing image … ──\n")
                    try:
                        fresh_state = prepare_recapture(old_state)
                    except RuntimeError as e:
                        print(f"\n❌  Recapture failed: {e}\n")
                        print("Continuing with previous scan.\n")
                    break  # break inner loop, outer loop will re-run step 2
                else:
                    # Ambiguous — ask again
                    print("\nASSISTANT:\nSorry, I didn't understand. "
                          "Is the image okay? (yes / no)\n")

            if verdict is True:
                break  # image accepted — proceed to LLM dialogue
            # verdict is False → outer loop repeats with new fresh_state

        # ── Step 3: LLM dialogue ───────────────────────────────────────
        mapping = run_update_dialogue(
            client, old_state, fresh_state, image_rename_map,
        )

        # ── Step 4: apply mapping and save ─────────────────────────────
        if mapping is not None:
            apply_update_mapping(old_state, fresh_state, mapping, image_rename_map)
            _committed = True
            print("✅  Update complete.\n")
        else:
            print("⚠  Update dialogue was cancelled.\n")

    finally:
        if not _committed:
            # The update was cancelled before a new config was committed.
            # configuration.json was already restored to old_state by
            # prepare_update().  Restore the image to match it.
            if _img_backup is not None and _img_backup.exists():
                try:
                    _shutil.copy2(str(_img_backup), str(_img_path))
                except Exception as _re_err:
                    print(f"  ⚠  Could not restore image after cancelled update: {_re_err}")
            else:
                # No backup (image didn't exist before this session) — just
                # redraw from the base image using the old config names.
                refresh_annotated_image(old_state)
        # Always clean up the temp backup regardless of outcome.
        if _img_backup is not None:
            try:
                _img_backup.unlink(missing_ok=True)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Robot execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_execution() -> None:
    """Spawn robot execution in a subprocess."""
    run_script = PROJECT_DIR / "Orchestration" / "run_execute.py"
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
    from Planning_Module.pddl_planner import plan_sequence

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
    from Communication_Module.api_main import run_conversation
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