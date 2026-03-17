# Update_Scene.py
# Scene Update Module — replaces the physical workspace state with a fresh
# camera scan while preserving all high-level configuration attributes.
#
# Design principle
# ────────────────
# Vision is the sole source of truth for the physical state.
# Part identity is preserved across scans by XY position matching:
#   if a freshly detected part is within POSITION_MATCH_THRESHOLD_M of an
#   old part, it inherits the old part's ID (and therefore all high-level
#   attributes like fragility).
# Parts that cannot be matched are presented to the user, who specifies
# whether each is a new addition, a removal, or a swap/replacement.
#
# Physical state (always taken from fresh vision scan):
#   objects, slot_belongs_to,
#   predicates: at, slot_empty, color, size,
#   metric
#
# High-level config (always preserved from memory):
#   workspace: operation_mode, batch_size
#   predicates: role, priority, kit_recipe, part_compatibility, fragility

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from paths import CONFIGURATION_JSON  # type: ignore

CONFIGURATION_PATH = Path(CONFIGURATION_JSON.resolve())
MEMORY_DIR         = PROJECT_DIR / "Memory"

# Parts within this XY distance (metres) of their previous position are
# considered the same physical part and keep their old ID.
POSITION_MATCH_THRESHOLD_M = 0.025


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
            "fragility": [],
        },
        "metric": {},
    }


def _save_atomic(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _part_xy(metric: Dict[str, Any], name: str) -> Optional[Tuple[float, float]]:
    """Extract XY position for a part from the metric section."""
    pos = metric.get(name, {}).get("pos")
    if pos and len(pos) >= 2:
        return (float(pos[0]), float(pos[1]))
    return None


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


# ── position-based part identity matching ─────────────────────────────────────

def _match_parts_by_position(
    mem_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    threshold_m: float = POSITION_MATCH_THRESHOLD_M,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Match freshly detected parts to memory parts by XY proximity.

    Returns
    -------
    matched : list of (mem_name, fresh_name) — position-confirmed pairs
    new_parts : list of fresh_name — no matching memory part
    missing_parts : list of mem_name — no matching fresh part
    """
    mem_parts   = mem_state.get("objects", {}).get("parts", [])
    fresh_parts = fresh_state.get("objects", {}).get("parts", [])
    mem_metric   = mem_state.get("metric", {})
    fresh_metric = fresh_state.get("metric", {})

    mem_pos   = {n: _part_xy(mem_metric, n)   for n in mem_parts}
    mem_pos   = {k: v for k, v in mem_pos.items() if v is not None}
    fresh_pos = {n: _part_xy(fresh_metric, n) for n in fresh_parts}
    fresh_pos = {k: v for k, v in fresh_pos.items() if v is not None}

    # Build candidate pairs within threshold, sorted closest-first
    candidates: List[Tuple[float, str, str]] = []
    for fn, fp in fresh_pos.items():
        for mn, mp in mem_pos.items():
            d = math.hypot(fp[0] - mp[0], fp[1] - mp[1])
            if d <= threshold_m:
                candidates.append((d, mn, fn))
    candidates.sort()

    # Greedy assignment (closest first, no double-use)
    used_mem: set   = set()
    used_fresh: set = set()
    matched: List[Tuple[str, str]] = []

    for _d, mn, fn in candidates:
        if mn in used_mem or fn in used_fresh:
            continue
        matched.append((mn, fn))
        used_mem.add(mn)
        used_fresh.add(fn)

    new_parts     = [n for n in fresh_parts if n not in used_fresh]
    missing_parts = [n for n in mem_parts   if n not in used_mem]

    return matched, new_parts, missing_parts


# ── user interaction for unmatched parts ──────────────────────────────────────

def _interact_part_changes(
    matched: List[Tuple[str, str]],
    new_parts: List[str],
    missing_parts: List[str],
    mem_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Show the user what changed and ask about unmatched parts.

    Returns
    -------
    extra_renames    : {fresh_name: mem_name} for user-confirmed replacements
    confirmed_removed: list of mem_names the user confirmed as removed
    confirmed_new    : list of fresh_names the user confirmed as additions
    """
    mem_preds   = mem_state.get("predicates", {})
    fresh_preds = fresh_state.get("predicates", {})

    mem_colors   = {e["part"]: e.get("color", "?") for e in mem_preds.get("color", [])}
    fresh_colors = {e["part"]: e.get("color", "?") for e in fresh_preds.get("color", [])}
    mem_frag     = {e["part"]: e.get("fragility")  for e in mem_preds.get("fragility", [])}

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n── Part Identity Matching ──")
    if matched:
        for mem_name, fresh_name in matched:
            color = fresh_colors.get(fresh_name, "?")
            extras = []
            if mem_name in mem_frag:
                extras.append(str(mem_frag[mem_name]))
            tag = f", {', '.join(extras)}" if extras else ""
            print(f"  ✓  {mem_name} ({color}{tag}) — position matched")

    if not missing_parts and not new_parts:
        print("\n  All parts matched. No changes needed.\n")
        return {}, [], []

    if missing_parts:
        print()
        for mp in missing_parts:
            color = mem_colors.get(mp, "?")
            frag  = f", {mem_frag[mp]}" if mp in mem_frag else ""
            print(f"  ✗  {mp} ({color}{frag}) — not detected")

    if new_parts:
        print()
        for np_name in new_parts:
            color = fresh_colors.get(np_name, "?")
            print(f"  +  {np_name} ({color}) — newly detected")

    print()

    # ── trivial cases (no interaction needed) ─────────────────────────────────
    if missing_parts and not new_parts:
        print("No new parts detected. Missing parts are removed.\n")
        return {}, list(missing_parts), []

    if new_parts and not missing_parts:
        print("No parts missing. New detections are additions.\n")
        return {}, [], list(new_parts)

    # ── both missing and new → ask user about each new part ───────────────────
    remaining_missing = list(missing_parts)
    extra_renames: Dict[str, str] = {}
    confirmed_new: List[str] = []

    for np_name in new_parts:
        if not remaining_missing:
            confirmed_new.append(np_name)
            continue

        np_color  = fresh_colors.get(np_name, "?")
        n_options = len(remaining_missing)

        print(f"'{np_name}' ({np_color}) — does it replace a missing part?")
        for i, mp in enumerate(remaining_missing, 1):
            mc   = mem_colors.get(mp, "?")
            frag = f", {mem_frag[mp]}" if mp in mem_frag else ""
            print(f"  [{i}] Yes — replaces {mp} ({mc}{frag})")
        print(f"  [{n_options + 1}] No — it's a new part")

        while True:
            raw = input("Choice: ").strip().lower()
            if raw in ("done", "cancel", "skip"):
                confirmed_new.append(np_name)
                break
            if raw.isdigit():
                choice = int(raw)
                if 1 <= choice <= n_options:
                    replaced = remaining_missing[choice - 1]
                    extra_renames[np_name] = replaced
                    remaining_missing.remove(replaced)
                    print(f"  → {np_name} inherits identity of {replaced}\n")
                    break
                elif choice == n_options + 1:
                    confirmed_new.append(np_name)
                    print(f"  → {np_name} is a new addition\n")
                    break
            print(f"  Please enter a number between 1 and {n_options + 1}.")

    confirmed_removed = list(remaining_missing)
    if confirmed_removed:
        print(f"Removed: {', '.join(confirmed_removed)}\n")

    return extra_renames, confirmed_removed, confirmed_new


# ── part renaming ─────────────────────────────────────────────────────────────

def _build_rename_map(
    matched: List[Tuple[str, str]],
    extra_renames: Dict[str, str],
    fresh_parts: List[str],
) -> Dict[str, str]:
    """
    Build {fresh_name: final_name} for EVERY part in the fresh state.

    - Position-matched parts → inherit mem_name
    - User-confirmed replacements → inherit mem_name
    - Truly new parts → next available Part_N (to avoid collisions)
    """
    rename_map: Dict[str, str] = {}

    # From position matching
    for mem_name, fresh_name in matched:
        rename_map[fresh_name] = mem_name

    # From user-specified replacements
    for fresh_name, mem_name in extra_renames.items():
        rename_map[fresh_name] = mem_name

    # Collect all reserved part numbers (names that are already assigned)
    reserved_numbers: set = set()
    for final_name in rename_map.values():
        if final_name.startswith("Part_"):
            try:
                reserved_numbers.add(int(final_name.split("_", 1)[1]))
            except ValueError:
                pass

    # Assign new IDs to truly new parts
    next_num = max(reserved_numbers, default=0) + 1
    for fp in fresh_parts:
        if fp not in rename_map:
            while next_num in reserved_numbers:
                next_num += 1
            rename_map[fp] = f"Part_{next_num}"
            reserved_numbers.add(next_num)
            next_num += 1

    return rename_map


def _rename_parts_in_state(
    state: Dict[str, Any],
    rename_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Rename parts throughout the state dict.
    Uses a two-pass (current→tmp, tmp→final) to avoid collisions when
    names are swapped (e.g. fresh Part_1→old Part_2, fresh Part_2→old Part_1).
    """
    # Filter to only effective renames (skip identity mappings)
    effective = {k: v for k, v in rename_map.items() if k != v}
    if not effective:
        return state

    # Build two rename passes
    tmp_map:   Dict[str, str] = {}
    final_map: Dict[str, str] = {}
    for i, (current, desired) in enumerate(effective.items()):
        tmp = f"__tmp_{i}__"
        tmp_map[current]  = tmp
        final_map[tmp]    = desired

    for rmap in (tmp_map, final_map):
        _apply_rename_pass(state, rmap)

    # Re-sort for consistency
    state["objects"]["parts"] = sorted(state["objects"]["parts"])
    return state


def _apply_rename_pass(state: Dict[str, Any], rmap: Dict[str, str]) -> None:
    """Single rename pass across all part-referencing sections."""
    # objects.parts
    parts = state.get("objects", {}).get("parts", [])
    for i, p in enumerate(parts):
        if p in rmap:
            parts[i] = rmap[p]

    # predicates with "part" key
    preds = state.get("predicates", {})
    for pred_key in ("at", "color", "size", "fragility"):
        for entry in preds.get(pred_key, []):
            if entry.get("part") in rmap:
                entry["part"] = rmap[entry["part"]]

    # metric keys
    metric = state.get("metric", {})
    for old_name in list(rmap.keys()):
        if old_name in metric:
            metric[rmap[old_name]] = metric.pop(old_name)


# ── core merge ────────────────────────────────────────────────────────────────

def _apply_high_level(fresh: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start from a deep copy of the fresh physical state, then overlay the
    high-level configuration attributes from memory.

    Part IDs in the fresh state have already been stabilised via position
    matching and renaming, so fragility is carried over by name.

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

    # ── fragility: carry over by name (stable IDs via position matching) ──────
    # Parts have already been renamed to their old IDs where matched, so a
    # simple name-based filter keeps fragility for all surviving parts and
    # drops it for parts that were genuinely removed.
    fresh_parts_set = set(result.get("objects", {}).get("parts", []))
    res_preds["fragility"] = [
        entry for entry in mem_preds.get("fragility", [])
        if entry.get("part") in fresh_parts_set
    ]

    return result


# ── terminal summary ──────────────────────────────────────────────────────────

def _print_summary(memory: Dict[str, Any], merged: Dict[str, Any]) -> None:
    m_parts = set(memory.get("objects", {}).get("parts", []))
    f_parts = set(merged.get("objects", {}).get("parts", []))
    m_slots = set(memory.get("objects", {}).get("slots", []))
    f_slots = set(merged.get("objects", {}).get("slots", []))

    lines = []
    for p in sorted(f_parts - m_parts):
        lines.append(f"  + {p}  [new part]")
    for p in sorted(m_parts - f_parts):
        lines.append(f"  - {p}  [removed]")
    for s in sorted(f_slots - m_slots):
        lines.append(f"  + {s}  [new slot detected]")
    for s in sorted(m_slots - f_slots):
        lines.append(f"  - {s}  [slot no longer detected]")
    if not lines:
        lines.append("  (no structural changes — positions refreshed)")

    print("── Final Scene Summary ──")
    for line in lines:
        print(line)
    print()


# ── main entry point ──────────────────────────────────────────────────────────

def run_update_session(client: Any) -> None:
    """
    Run a full scene update:
      1. Load memory state (high-level config is preserved from here).
      2. Run Vision_Main in a clean subprocess → fresh physical state on disk.
      3. Match part identities by position; ask user about unmatched parts.
      4. Rename parts in fresh state to preserve old IDs.
      5. Overlay high-level attributes from memory onto the fresh state.
      6. Save configuration.json and archive to Memory/.

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

    # ── step 3: position-based part identity matching ─────────────────────────
    matched, new_parts, missing_parts = _match_parts_by_position(
        memory_state, fresh_state,
    )

    # ── step 4: user resolves unmatched parts ─────────────────────────────────
    extra_renames, confirmed_removed, confirmed_new = _interact_part_changes(
        matched, new_parts, missing_parts, memory_state, fresh_state,
    )

    # ── step 5: rename parts in fresh state ───────────────────────────────────
    all_fresh_parts = fresh_state.get("objects", {}).get("parts", [])
    rename_map = _build_rename_map(matched, extra_renames, all_fresh_parts)
    _rename_parts_in_state(fresh_state, rename_map)

    # ── step 6: overlay high-level config from memory ─────────────────────────
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