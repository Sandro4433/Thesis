# Update_Scene.py
# Scene Update Module — replaces the physical workspace state with a fresh
# camera scan while preserving all high-level configuration attributes.
#
# Design principle
# ────────────────
# Vision is the sole source of truth for the physical state.
# If vision did not see a part, that part is gone — full stop.
#
# Physical state (always taken from fresh vision scan):
#   objects, slot_belongs_to,
#   predicates: at, slot_empty, color, size,
#   metric
#
# High-level config (always preserved from memory):
#   workspace: operation_mode, batch_size
#   predicates: role, priority, kit_recipe, part_compatibility

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import CONFIGURATION_JSON  # type: ignore

CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
MEMORY_DIR         = PROJECT_DIR / "Memory"


# ── helpers ───────────────────────────────────────────────────────────────────

def _empty_state() -> Dict[str, Any]:
    return {
        "workspace": {"operation_mode": None, "batch_size": None},
        "objects":   {"kits": [], "containers": [], "parts": [], "slots": []},
        "slot_belongs_to": {},
        "predicates": {
            "at": [], "slot_empty": [], "role": [],
            "color": [], "size": [],
            "priority": [], "kit_recipe": [], "part_compatibility": [],
        },
        "metric": {},
    }


def _save_atomic(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ── vision subprocess ─────────────────────────────────────────────────────────

def _run_vision() -> None:
    """
    Spawn Vision_Main in a clean subprocess so that libapriltag never shares
    a process with the native libraries already loaded by API_Main.
    """
    vision_main_path = PROJECT_DIR / "Vision_Module" / "Vision_Main.py"
    print("\nStarting vision module …")
    result = subprocess.run(
        [sys.executable, str(vision_main_path)],
        cwd=str(PROJECT_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Vision_Main subprocess exited with code {result.returncode}."
        )
    if not CONFIGURATION_PATH.exists():
        raise RuntimeError("Vision module did not produce configuration.json.")
    print("Vision complete.\n")


# ── core merge ────────────────────────────────────────────────────────────────

def _apply_high_level(fresh: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start from a deep copy of the fresh physical state, then overlay the
    high-level configuration attributes from memory.

    Role entries are filtered to receptacles that still exist in the fresh
    scan, so stale roles for removed containers/kits are automatically dropped.
    New receptacles get a null role by default (same as a brand-new scan).
    """
    result    = copy.deepcopy(fresh)
    mem_ws    = memory.get("workspace", {})
    mem_preds = memory.get("predicates", {})
    res_preds = result.setdefault("predicates", {})

    # ── workspace high-level attrs ────────────────────────────────────────────
    result.setdefault("workspace", {})
    result["workspace"]["operation_mode"] = mem_ws.get("operation_mode")
    result["workspace"]["batch_size"]     = mem_ws.get("batch_size")

    # ── role: carry over for receptacles that still exist ────────────────────
    fresh_receptacles = set(
        result.get("objects", {}).get("kits", []) +
        result.get("objects", {}).get("containers", [])
    )
    res_preds["role"] = [
        e for e in mem_preds.get("role", [])
        if e.get("object") in fresh_receptacles
    ]
    # null-role entry for any brand-new receptacle not present in memory
    existing = {e["object"] for e in res_preds["role"]}
    for rec in sorted(fresh_receptacles - existing):
        res_preds["role"].append({"object": rec, "role": None})

    # ── list predicates: carry over wholesale ─────────────────────────────────
    res_preds["priority"]           = mem_preds.get("priority",           [])
    res_preds["kit_recipe"]         = mem_preds.get("kit_recipe",         [])
    res_preds["part_compatibility"] = mem_preds.get("part_compatibility",  [])

    return result


# ── terminal summary ──────────────────────────────────────────────────────────

def _print_summary(memory: Dict[str, Any], merged: Dict[str, Any]) -> None:
    m_parts = set(memory.get("objects", {}).get("parts", []))
    f_parts = set(merged.get("objects", {}).get("parts", []))
    m_slots = set(memory.get("objects", {}).get("slots", []))
    f_slots = set(merged.get("objects", {}).get("slots", []))

    lines = []
    for p in sorted(f_parts - m_parts):
        lines.append(f"  + {p}  [new part detected]")
    for p in sorted(m_parts - f_parts):
        lines.append(f"  - {p}  [no longer detected]")
    for s in sorted(f_slots - m_slots):
        lines.append(f"  + {s}  [new slot detected]")
    for s in sorted(m_slots - f_slots):
        lines.append(f"  - {s}  [slot no longer detected]")
    if not lines:
        lines.append("  (no structural changes — positions refreshed)")

    print("── Scene Update Summary ──")
    for line in lines:
        print(line)
    print()


# ── main entry point ──────────────────────────────────────────────────────────

def run_update_session(client: Any) -> None:
    """
    Run a full scene update:
      1. Load memory state (high-level config is preserved from here).
      2. Run Vision_Main in a clean subprocess → fresh physical state on disk.
      3. Overlay high-level attributes from memory onto the fresh state.
      4. Save configuration.json and archive to Memory/.

    The `client` parameter is accepted for API_Main compatibility but is not
    used — this function requires no LLM interaction.
    """
    # Load memory before vision overwrites the file on disk
    memory_state: Dict[str, Any] = (
        json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        if CONFIGURATION_PATH.exists()
        else _empty_state()
    )

    try:
        _run_vision()
    except Exception as exc:
        print(f"\n❌  Vision failed: {exc}\n")
        # Restore memory so the file is not left in a partial state
        _save_atomic(CONFIGURATION_PATH, memory_state)
        return

    fresh_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

    # Vision is physical truth; memory supplies high-level config
    merged = _apply_high_level(fresh_state, memory_state)

    _print_summary(memory_state, merged)

    _save_atomic(CONFIGURATION_PATH, merged)
    print(f"✅  configuration.json updated → {CONFIGURATION_PATH.resolve()}")

    try:
        from Configuration_Module.Apply_Sequence_Changes import save_to_memory  # type: ignore
        mem_path = save_to_memory(merged, label="scene_update")
        print(f"✅  State archived → {mem_path.resolve()}\n")
    except ImportError:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        arc_path = MEMORY_DIR / f"configuration_scene_update_{ts}.json"
        _save_atomic(arc_path, merged)
        print(f"✅  State archived → {arc_path.resolve()}\n")

    print("── Scene update complete. ──\n")