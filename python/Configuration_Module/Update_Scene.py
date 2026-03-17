# Update_Scene.py
# Scene Update Module — takes a fresh camera scan and uses an LLM dialogue
# (driven by API_Main._run_update_dialogue) to resolve how freshly detected
# parts correspond to parts in the previous configuration.
#
# Design principle
# ────────────────
# Vision is the sole source of truth for the PHYSICAL state (positions,
# colours, sizes).  Part IDENTITY is preserved by matching fresh detections
# to old parts — first automatically (by position + colour), then via a
# user-facing LLM conversation for any ambiguous cases.
#
# The three public functions form a pipeline:
#   prepare_update()      → run vision, return (old_state, fresh_state)
#   build_update_context() → compute diff for the LLM prompt
#   apply_update_mapping() → apply the confirmed mapping, merge, and save
#
# Physical state (always taken from fresh vision scan):
#   objects, slot_belongs_to,
#   predicates: at, slot_empty, color, size,
#   metric
#
# High-level config (always preserved from old config):
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
FILE_EXCHANGE      = PROJECT_DIR / "File_Exchange"

# Parts within this XY distance (metres) AND with the same colour are
# considered the same physical part and keep their old ID automatically.
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


# ── position + colour matching ────────────────────────────────────────────────

def _match_parts_by_position(
    mem_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    threshold_m: float = POSITION_MATCH_THRESHOLD_M,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Match freshly detected parts to memory parts by XY proximity AND colour.

    A match requires BOTH:
      - XY distance ≤ threshold_m
      - colours agree (or at least one is unknown/missing)

    This prevents wrong auto-matches when parts are swapped: a red part
    at a blue part's old position will NOT be auto-matched.

    Returns
    -------
    matched       : list of (mem_name, fresh_name) — confirmed pairs
    new_parts     : list of fresh_name — no matching memory part
    missing_parts : list of mem_name — no matching fresh part
    """
    mem_parts    = mem_state.get("objects", {}).get("parts", [])
    fresh_parts  = fresh_state.get("objects", {}).get("parts", [])
    mem_metric   = mem_state.get("metric", {})
    fresh_metric = fresh_state.get("metric", {})
    mem_preds    = mem_state.get("predicates", {})
    fresh_preds  = fresh_state.get("predicates", {})

    mem_colors   = {e["part"]: e.get("color") for e in mem_preds.get("color", [])}
    fresh_colors = {e["part"]: e.get("color") for e in fresh_preds.get("color", [])}

    mem_pos   = {n: _part_xy(mem_metric, n) for n in mem_parts}
    mem_pos   = {k: v for k, v in mem_pos.items() if v is not None}
    fresh_pos = {n: _part_xy(fresh_metric, n) for n in fresh_parts}
    fresh_pos = {k: v for k, v in fresh_pos.items() if v is not None}

    # Build candidate pairs: within threshold AND colour-compatible
    candidates: List[Tuple[float, str, str]] = []
    for fn, fp in fresh_pos.items():
        for mn, mp in mem_pos.items():
            d = math.hypot(fp[0] - mp[0], fp[1] - mp[1])
            if d <= threshold_m:
                fc = fresh_colors.get(fn)
                mc = mem_colors.get(mn)
                if fc and mc and fc != mc:
                    continue        # colour mismatch → skip
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


# ── part renaming ─────────────────────────────────────────────────────────────

def _rename_parts_in_state(
    state: Dict[str, Any],
    rename_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Rename parts throughout the state dict.
    Uses a two-pass (current→tmp, tmp→final) to avoid collisions when
    names are swapped (e.g. fresh Part_1→old Part_2, fresh Part_2→old Part_1).
    """
    effective = {k: v for k, v in rename_map.items() if k != v}
    if not effective:
        return state

    tmp_map:   Dict[str, str] = {}
    final_map: Dict[str, str] = {}
    for i, (current, desired) in enumerate(effective.items()):
        tmp = f"__tmp_{i}__"
        tmp_map[current] = tmp
        final_map[tmp]   = desired

    for rmap in (tmp_map, final_map):
        _apply_rename_pass(state, rmap)

    state["objects"]["parts"] = sorted(state["objects"]["parts"])
    return state


def _apply_rename_pass(state: Dict[str, Any], rmap: Dict[str, str]) -> None:
    """Single rename pass across all part-referencing sections."""
    parts = state.get("objects", {}).get("parts", [])
    for i, p in enumerate(parts):
        if p in rmap:
            parts[i] = rmap[p]

    preds = state.get("predicates", {})
    for pred_key in ("at", "color", "size", "fragility"):
        for entry in preds.get(pred_key, []):
            if entry.get("part") in rmap:
                entry["part"] = rmap[entry["part"]]

    metric = state.get("metric", {})
    for old_name in list(rmap.keys()):
        if old_name in metric:
            metric[rmap[old_name]] = metric.pop(old_name)


# ── high-level config merge ──────────────────────────────────────────────────

def _apply_high_level(fresh: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start from a deep copy of the fresh physical state, then overlay the
    high-level configuration attributes from memory.

    Part IDs in the fresh state must already be stabilised (renamed) before
    calling this, so fragility is carried over by name.
    """
    result    = copy.deepcopy(fresh)
    mem_ws    = memory.get("workspace", {})
    mem_preds = memory.get("predicates", {})
    res_preds = result.setdefault("predicates", {})

    # workspace
    result.setdefault("workspace", {})
    result["workspace"]["operation_mode"] = mem_ws.get("operation_mode")
    result["workspace"]["batch_size"]     = mem_ws.get("batch_size")

    # role — carry over for receptacles that still exist
    fresh_receptacles = set(
        result.get("objects", {}).get("kits", []) +
        result.get("objects", {}).get("containers", [])
    )
    res_preds["role"] = [
        e for e in mem_preds.get("role", [])
        if e.get("object") in fresh_receptacles
    ]
    existing = {e["object"] for e in res_preds["role"]}
    for rec in sorted(fresh_receptacles - existing):
        res_preds["role"].append({"object": rec, "role": None})

    # list predicates — carry over wholesale
    res_preds["priority"]           = mem_preds.get("priority",           [])
    res_preds["kit_recipe"]         = mem_preds.get("kit_recipe",         [])
    res_preds["part_compatibility"] = mem_preds.get("part_compatibility",  [])

    # fragility — carry over by name (IDs are stable at this point)
    fresh_parts_set = set(result.get("objects", {}).get("parts", []))
    res_preds["fragility"] = [
        entry for entry in mem_preds.get("fragility", [])
        if entry.get("part") in fresh_parts_set
    ]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (called by API_Main._run_update_dialogue)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_update() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the vision module and return both old and fresh states.

    The old configuration is restored on disk immediately after the fresh
    state is loaded into memory, so nothing is permanently changed until
    apply_update_mapping() is called.
    """
    memory_state: Dict[str, Any] = (
        json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        if CONFIGURATION_PATH.exists()
        else _empty_state()
    )

    try:
        _run_vision()  # writes fresh state to CONFIGURATION_PATH
    except Exception as exc:
        print(f"\n❌  Vision failed: {exc}\n")
        # Restore old config so the file is not corrupted
        _save_atomic(CONFIGURATION_PATH, memory_state)
        raise

    fresh_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

    # Restore old config on disk — the fresh state lives only in memory
    # until the user confirms the mapping.
    _save_atomic(CONFIGURATION_PATH, memory_state)

    return memory_state, fresh_state


def build_update_context(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
) -> str:
    """
    Compute the structural diff between old and fresh states and return a
    human-readable analysis for the LLM prompt.

    Includes:
      - Receptacle changes (AprilTag-based, automatic)
      - Auto-matched parts (position + colour)
      - Unmatched new detections
      - Missing old parts
      - High-level attributes that should be preserved
    """
    matched, new_parts, missing_parts = _match_parts_by_position(
        old_state, fresh_state,
    )

    old_preds   = old_state.get("predicates", {})
    fresh_preds = fresh_state.get("predicates", {})

    old_colors   = {e["part"]: e.get("color", "?") for e in old_preds.get("color", [])}
    fresh_colors = {e["part"]: e.get("color", "?") for e in fresh_preds.get("color", [])}
    old_sizes    = {e["part"]: e.get("size", "standard") for e in old_preds.get("size", [])}
    fresh_sizes  = {e["part"]: e.get("size", "standard") for e in fresh_preds.get("size", [])}
    old_frag     = {e["part"]: e.get("fragility") for e in old_preds.get("fragility", [])}

    old_at   = {e["part"]: e["slot"] for e in old_preds.get("at", [])}
    fresh_at = {e["part"]: e["slot"] for e in fresh_preds.get("at", [])}

    # Receptacle diff
    old_recs = set(
        old_state.get("objects", {}).get("kits", []) +
        old_state.get("objects", {}).get("containers", [])
    )
    fresh_recs = set(
        fresh_state.get("objects", {}).get("kits", []) +
        fresh_state.get("objects", {}).get("containers", [])
    )

    lines = ["RECEPTACLE CHANGES (AprilTag-based, automatic):"]
    for r in sorted(old_recs & fresh_recs):
        lines.append(f"  {r} — present in both scans")
    for r in sorted(fresh_recs - old_recs):
        lines.append(f"  + {r} — newly detected")
    for r in sorted(old_recs - fresh_recs):
        lines.append(f"  - {r} — no longer detected")

    # Auto-matched parts
    if matched:
        lines.append("\nAUTO-MATCHED PARTS (position + colour confirmed):")
        for mem_name, fresh_name in matched:
            fc   = fresh_colors.get(fresh_name, "?")
            fs   = fresh_sizes.get(fresh_name, "standard")
            slot = fresh_at.get(fresh_name, "standalone")
            frag = f", fragile" if mem_name in old_frag else ""
            lines.append(
                f"  New scan '{fresh_name}' → Old '{mem_name}' "
                f"({fc}, {fs}{frag}) in {slot}"
            )

    # Unmatched new detections
    if new_parts:
        lines.append("\nNEW DETECTIONS (no matching old part at same position+colour):")
        for np_name in new_parts:
            fc   = fresh_colors.get(np_name, "?")
            fs   = fresh_sizes.get(np_name, "standard")
            slot = fresh_at.get(np_name, "standalone")
            lines.append(f"  '{np_name}' ({fc}, {fs}) in {slot}")

    # Missing old parts
    if missing_parts:
        lines.append("\nMISSING FROM NEW SCAN:")
        for mp in missing_parts:
            oc   = old_colors.get(mp, "?")
            os   = old_sizes.get(mp, "standard")
            slot = old_at.get(mp, "standalone")
            frag = f", fragile" if mp in old_frag else ""
            lines.append(f"  '{mp}' ({oc}, {os}{frag}) was in {slot}")

    # High-level attributes summary
    if old_frag:
        lines.append("\nHIGH-LEVEL ATTRIBUTES TO PRESERVE (from old config):")
        for part_name, frag_val in sorted(old_frag.items()):
            lines.append(f"  {part_name}: fragility={frag_val}")

    if not new_parts and not missing_parts:
        lines.append("\nAll parts auto-matched successfully. No ambiguities.")

    return "\n".join(lines)


def apply_update_mapping(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    """
    Apply the LLM-produced identity mapping to produce the final merged state.

    Parameters
    ----------
    old_state   : previous configuration (source of high-level attributes)
    fresh_state : fresh vision scan (source of physical state)
    mapping     : {fresh_part_name: old_part_name | "new"}

    Steps:
      1. Build a rename map from the mapping
      2. Rename parts in a deep copy of the fresh state
      3. Overlay high-level config from old state (roles, fragility, etc.)
      4. Save to disk and archive to Memory/
    """
    result = copy.deepcopy(fresh_state)

    # ── collect all old part numbers so new IDs don't collide ────────────────
    used_numbers: set = set()
    for old_name in old_state.get("objects", {}).get("parts", []):
        if old_name.startswith("Part_"):
            try:
                used_numbers.add(int(old_name.split("_", 1)[1]))
            except ValueError:
                pass

    # ── build rename map ──────────────────────────────────────────────────────
    rename_map: Dict[str, str] = {}

    for fresh_name, target in mapping.items():
        if target != "new":
            rename_map[fresh_name] = target
            if target.startswith("Part_"):
                try:
                    used_numbers.add(int(target.split("_", 1)[1]))
                except ValueError:
                    pass

    # Assign new sequential IDs for "new" parts
    next_num = max(used_numbers, default=0) + 1
    for fresh_name, target in mapping.items():
        if target == "new":
            while next_num in used_numbers:
                next_num += 1
            rename_map[fresh_name] = f"Part_{next_num}"
            used_numbers.add(next_num)
            next_num += 1

    _rename_parts_in_state(result, rename_map)

    # ── overlay high-level config from memory ─────────────────────────────────
    merged = _apply_high_level(result, old_state)

    # ── summary ───────────────────────────────────────────────────────────────
    m_parts = set(old_state.get("objects", {}).get("parts", []))
    f_parts = set(merged.get("objects", {}).get("parts", []))
    m_slots = set(old_state.get("objects", {}).get("slots", []))
    f_slots = set(merged.get("objects", {}).get("slots", []))

    lines = []
    for p in sorted(f_parts - m_parts):
        lines.append(f"  + {p}  [new part]")
    for p in sorted(m_parts - f_parts):
        lines.append(f"  - {p}  [removed]")
    for s in sorted(f_slots - m_slots):
        lines.append(f"  + {s}  [new slot]")
    for s in sorted(m_slots - f_slots):
        lines.append(f"  - {s}  [slot removed]")
    if not lines:
        lines.append("  (no structural changes — positions refreshed)")

    print("── Final Scene Summary ──")
    for line in lines:
        print(line)
    print()

    # ── save ──────────────────────────────────────────────────────────────────
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

    # ── redraw annotated image with updated IDs + fragility ───────────────
    _redraw_annotated_image(rename_map, merged)

    return merged


# ── image re-annotation ──────────────────────────────────────────────────────

def _redraw_annotated_image(
    rename_map: Dict[str, str],
    merged_state: Dict[str, Any],
) -> None:
    """
    Redraw latest_image.png using the base image (tags+slots only) and the
    pixel map saved by Vision_Main, applying the new part names from
    rename_map and adding FRAGILE labels from the merged state.

    Files read:
      File_Exchange/latest_image_base.png   — clean image (no part labels)
      File_Exchange/latest_pixel_map.json   — pixel coords per detected part

    File written:
      File_Exchange/latest_image.png        — final annotated image
    """
    base_path  = FILE_EXCHANGE / "latest_image_base.png"
    pmap_path  = FILE_EXCHANGE / "latest_pixel_map.json"
    out_path   = FILE_EXCHANGE / "latest_image.png"

    if not base_path.exists() or not pmap_path.exists():
        print("  ⚠  Base image or pixel map not found — skipping image re-annotation.")
        return

    try:
        import cv2
        from Vision_Module.pipeline import annotate_parts  # type: ignore

        img = cv2.imread(str(base_path))
        if img is None:
            print("  ⚠  Could not load base image — skipping re-annotation.")
            return

        pixel_map = json.loads(pmap_path.read_text(encoding="utf-8"))

        # Build fragile set from merged state
        fragile_set: set = set()
        for entry in merged_state.get("predicates", {}).get("fragility", []):
            if entry.get("fragility") == "fragile":
                fragile_set.add(entry["part"])

        # Apply rename_map to pixel annotations
        updated_annotations = []
        for p in pixel_map:
            old_name = p["name"]
            new_name = rename_map.get(old_name, old_name)
            updated_annotations.append({
                **p,
                "name": new_name,
            })

        annotate_parts(img, updated_annotations, fragile_set=fragile_set)

        cv2.imwrite(str(out_path), img)
        print(f"✅  Annotated image updated → {out_path}")

    except Exception as exc:
        print(f"  ⚠  Image re-annotation failed: {exc}")


# ── post-execution automated rescan ──────────────────────────────────────────

# More generous threshold for post-execution matching: the planner moved parts
# to known slots, so the predicted and actual positions should be very close.
_POST_EXEC_THRESHOLD_M = 0.035


def run_post_execution_rescan(config_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a fresh camera image after execution and merge it with the
    post-execution config file.  The config is the source of truth for
    part identity — no user interaction is needed.

    Flow:
      1. Save current config to disk (restore point).
      2. Run vision → fresh state.
      3. Match fresh detections to config parts by position+colour.
      4. Auto-build mapping: matched → config ID, unmatched fresh → ignored,
         missing config parts → kept with predicted metrics.
      5. Rename, overlay high-level attributes, save, redraw image.

    Returns the final merged state.
    """
    print("\n── Post-execution vision rescan ──\n")

    # Ensure config is on disk before vision runs (vision overwrites it)
    _save_atomic(CONFIGURATION_PATH, config_state)

    try:
        _run_vision()
    except Exception as exc:
        print(f"  ⚠  Vision rescan failed: {exc}")
        # Restore config so the post-execution state is not lost
        _save_atomic(CONFIGURATION_PATH, config_state)
        return config_state

    fresh_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))

    # Restore config on disk — we merge in memory first
    _save_atomic(CONFIGURATION_PATH, config_state)

    # ── match fresh detections to config parts ────────────────────────────
    matched, new_fresh, missing_config = _match_parts_by_position(
        config_state, fresh_state, threshold_m=_POST_EXEC_THRESHOLD_M,
    )

    n_matched = len(matched)
    n_ignored = len(new_fresh)
    n_kept    = len(missing_config)
    print(f"  Matched: {n_matched}  |  "
          f"Vision-only (ignored): {n_ignored}  |  "
          f"Config-only (kept): {n_kept}")

    # ── build automatic rename map ────────────────────────────────────────
    # matched: (config_name, fresh_name) → rename fresh_name → config_name
    # new_fresh: ignored (vision noise / irrelevant detection)
    # missing_config: kept — their metrics stay from the config prediction

    rename_map: Dict[str, str] = {}
    for config_name, fresh_name in matched:
        rename_map[fresh_name] = config_name

    # Remove unmatched fresh parts from the fresh state (they're noise)
    fresh_parts = fresh_state.get("objects", {}).get("parts", [])
    matched_fresh_names = {fn for _, fn in matched}
    parts_to_remove = [p for p in fresh_parts if p not in matched_fresh_names]

    if parts_to_remove:
        # Remove from objects.parts
        fresh_state["objects"]["parts"] = [
            p for p in fresh_parts if p in matched_fresh_names
        ]
        # Remove from predicates
        fresh_preds = fresh_state.get("predicates", {})
        for pred_key in ("at", "color", "size", "fragility"):
            fresh_preds[pred_key] = [
                e for e in fresh_preds.get(pred_key, [])
                if e.get("part") not in parts_to_remove
            ]
        # Remove from slot_empty (parts leaving frees slots)
        # Actually — just keep slot_empty as vision detected it
        # Remove from metric
        fresh_metric = fresh_state.get("metric", {})
        for p in parts_to_remove:
            fresh_metric.pop(p, None)

    # Rename matched fresh parts to config IDs
    _rename_parts_in_state(fresh_state, rename_map)

    # ── inject missing config parts back (vision didn't see them) ─────────
    # These keep their config-predicted metric data.
    config_metric = config_state.get("metric", {})
    config_preds  = config_state.get("predicates", {})
    config_at     = {e["part"]: e["slot"] for e in config_preds.get("at", [])}
    config_colors = {e["part"]: e.get("color") for e in config_preds.get("color", [])}
    config_sizes  = {e["part"]: e.get("size")  for e in config_preds.get("size", [])}

    fresh_preds  = fresh_state.get("predicates", {})
    fresh_metric = fresh_state.get("metric", {})

    for mp in missing_config:
        # Add to objects.parts
        if mp not in fresh_state["objects"]["parts"]:
            fresh_state["objects"]["parts"].append(mp)

        # Add to predicates.at (if it was in a slot)
        if mp in config_at:
            fresh_preds.setdefault("at", []).append(
                {"part": mp, "slot": config_at[mp]}
            )
            # Remove slot from slot_empty
            slot = config_at[mp]
            if slot in fresh_preds.get("slot_empty", []):
                fresh_preds["slot_empty"].remove(slot)

        # Add to predicates.color / size
        if mp in config_colors:
            fresh_preds.setdefault("color", []).append(
                {"part": mp, "color": config_colors[mp]}
            )
        if mp in config_sizes:
            fresh_preds.setdefault("size", []).append(
                {"part": mp, "size": config_sizes[mp]}
            )

        # Add config-predicted metric
        if mp in config_metric:
            fresh_metric[mp] = config_metric[mp]

        print(f"  ⚠  {mp} not detected by vision — kept with predicted position")

    fresh_state["objects"]["parts"] = sorted(fresh_state["objects"]["parts"])

    # ── overlay high-level config ─────────────────────────────────────────
    merged = _apply_high_level(fresh_state, config_state)

    # ── summary ───────────────────────────────────────────────────────────
    print()
    c_parts = set(config_state.get("objects", {}).get("parts", []))
    m_parts = set(merged.get("objects", {}).get("parts", []))
    if c_parts == m_parts:
        print("  All parts accounted for — positions refreshed from vision.")
    else:
        for p in sorted(m_parts - c_parts):
            print(f"  + {p}  [unexpected]")
        for p in sorted(c_parts - m_parts):
            print(f"  - {p}  [lost]")
    print()

    # ── save ──────────────────────────────────────────────────────────────
    _save_atomic(CONFIGURATION_PATH, merged)
    print(f"✅  configuration.json updated → {CONFIGURATION_PATH.resolve()}")

    try:
        from Configuration_Module.Apply_Sequence_Changes import save_to_memory  # type: ignore
        mem_path = save_to_memory(merged, label="post_exec_rescan")
        print(f"✅  State archived → {mem_path.resolve()}\n")
    except ImportError:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        arc_path = MEMORY_DIR / f"configuration_post_exec_{ts}.json"
        _save_atomic(arc_path, merged)
        print(f"✅  State archived → {arc_path.resolve()}\n")

    # ── redraw image with confirmed IDs + fragility ───────────────────────
    _redraw_annotated_image(rename_map, merged)

    print("── Post-execution rescan complete. ──\n")
    return merged