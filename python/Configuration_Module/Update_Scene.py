# Update_Scene.py
# Scene Update Module — takes a fresh camera scan and uses an LLM dialogue
# (driven by API_Main._run_update_dialogue) to resolve how freshly detected
# parts correspond to parts in the previous configuration.
#
# Design principle
# ────────────────
# Vision is the sole source of truth for the PHYSICAL state (positions,
# colours).  Part IDENTITY is preserved by matching fresh detections
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
#   predicates: at, slot_empty, color,
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
            "color": [],
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
    Match freshly detected parts to memory parts by slot identity OR
    XY proximity, combined with colour compatibility.

    Matching is done in two passes:

    1. **Slot-based** (highest priority): if a memory part and a fresh part
       occupy the same named slot (e.g. ``Container_1_Pos_2``) and their
       colours are compatible, they are matched immediately.  This handles
       the case where a container/kit was physically moved — the parts
       inside stay in the same slot but their absolute XY coordinates
       change, so pure position matching would fail.

    2. **XY-based** (fallback for standalone parts or unmatched slotted
       parts): memory and fresh parts within *threshold_m* metres AND
       with compatible colours are matched greedily by distance.

    A colour is "compatible" when both colours are known and equal, or
    when at least one side is unknown/missing.

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

    # ── Pass 1: slot-based matching ──────────────────────────────────────
    # Build slot→part lookups from the "at" predicate.
    mem_at:   Dict[str, str] = {}   # slot → mem_part
    fresh_at: Dict[str, str] = {}   # slot → fresh_part
    for e in mem_preds.get("at", []):
        mem_at[e["slot"]] = e["part"]
    for e in fresh_preds.get("at", []):
        fresh_at[e["slot"]] = e["part"]

    used_mem:   set = set()
    used_fresh: set = set()
    matched: List[Tuple[str, str]] = []

    for slot, mn in mem_at.items():
        fn = fresh_at.get(slot)
        if fn is None:
            continue                # slot is empty in fresh scan
        if mn in used_mem or fn in used_fresh:
            continue                # already matched (shouldn't happen, but safe)
        # Colour compatibility check
        fc = fresh_colors.get(fn)
        mc = mem_colors.get(mn)
        if fc and mc and fc != mc:
            continue                # colour mismatch → don't match by slot
        matched.append((mn, fn))
        used_mem.add(mn)
        used_fresh.add(fn)

    # ── Pass 2: XY-based matching (for remaining unmatched parts) ────────
    mem_pos   = {n: _part_xy(mem_metric, n) for n in mem_parts if n not in used_mem}
    mem_pos   = {k: v for k, v in mem_pos.items() if v is not None}
    fresh_pos = {n: _part_xy(fresh_metric, n) for n in fresh_parts if n not in used_fresh}
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
    names are swapped.
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
    """
    Single rename pass across all part-referencing sections.

    Handles any predicate that references parts, regardless of key name:
      - "part" field  (used by at, color, fragility, and any future per-part predicate)
      - "part_name" field (used by priority and part_compatibility entries)
    """
    # ── objects.parts list ────────────────────────────────────────────────
    parts = state.get("objects", {}).get("parts", [])
    for i, p in enumerate(parts):
        if p in rmap:
            parts[i] = rmap[p]

    # ── all predicates: rename "part" and "part_name" fields ─────────────
    preds = state.get("predicates", {})
    for _pred_key, entries in preds.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("part") in rmap:
                entry["part"] = rmap[entry["part"]]
            if entry.get("part_name") in rmap:
                entry["part_name"] = rmap[entry["part_name"]]

    # ── metric ───────────────────────────────────────────────────────────
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
    calling this, so attributes are carried over by name.

    High-level predicates (everything except the physical predicates produced
    by vision: at, slot_empty, color) are copied from the old config and
    filtered to remove references to parts/receptacles that no longer exist.
    """
    result    = copy.deepcopy(fresh)
    mem_ws    = memory.get("workspace", {})
    mem_preds = memory.get("predicates", {})
    res_preds = result.setdefault("predicates", {})

    fresh_parts_set = set(result.get("objects", {}).get("parts", []))
    fresh_receptacles = set(
        result.get("objects", {}).get("kits", []) +
        result.get("objects", {}).get("containers", [])
    )
    fresh_slots = set(result.get("objects", {}).get("slots", []))

    # ── workspace ────────────────────────────────────────────────────────
    result.setdefault("workspace", {})
    result["workspace"]["operation_mode"] = mem_ws.get("operation_mode")
    result["workspace"]["batch_size"]     = mem_ws.get("batch_size")

    # ── role — carry over for receptacles that still exist ───────────────
    res_preds["role"] = [
        e for e in mem_preds.get("role", [])
        if e.get("object") in fresh_receptacles
    ]
    existing = {e["object"] for e in res_preds["role"]}
    for rec in sorted(fresh_receptacles - existing):
        res_preds["role"].append({"object": rec, "role": None})

    # ── kit_recipe — no part references, carry over wholesale ────────────
    res_preds["kit_recipe"] = mem_preds.get("kit_recipe", [])

    # ── predicates with part references — filter out stale entries ───────
    # Physical predicates (at, slot_empty, color) come from the fresh state
    # and are already correct.  Everything else is high-level config from
    # the old state and needs stale-reference filtering.
    #
    # An entry is stale if it references a part, receptacle, or slot that
    # no longer exists in the fresh state.

    _PHYSICAL_PREDS = {"at", "slot_empty", "color", "role", "kit_recipe"}

    for pred_key, mem_entries in mem_preds.items():
        if pred_key in _PHYSICAL_PREDS:
            continue
        if not isinstance(mem_entries, list):
            continue

        filtered = []
        for entry in mem_entries:
            if not isinstance(entry, dict):
                filtered.append(entry)
                continue

            # Check all fields that might reference scene objects
            stale = False

            # "part" field (used by fragility and any future per-part predicate)
            if "part" in entry and entry["part"] not in fresh_parts_set:
                stale = True

            # "part_name" field (used by priority and part_compatibility)
            if "part_name" in entry and entry["part_name"] not in fresh_parts_set:
                stale = True

            # "source" / "destination" fields (priority — reference receptacles)
            if "source" in entry and entry["source"] not in fresh_receptacles:
                stale = True
            if "destination" in entry and entry["destination"] not in fresh_receptacles:
                stale = True

            # "allowed_in" / "not_allowed_in" (part_compatibility — lists of receptacles)
            for list_key in ("allowed_in", "not_allowed_in"):
                if list_key in entry and isinstance(entry[list_key], list):
                    entry[list_key] = [
                        r for r in entry[list_key]
                        if r in fresh_receptacles
                    ]

            # "target_slot" (part_compatibility — single slot reference)
            if "target_slot" in entry and entry["target_slot"] not in fresh_slots:
                stale = True

            if not stale:
                filtered.append(entry)

        res_preds[pred_key] = filtered

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
    image_rename_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Compute the structural diff between old and fresh states and return a
    human-readable analysis for the LLM prompt.

    The GUI image has been re-annotated by redraw_image_with_auto_matches()
    so auto-matched parts show their old-config name and new detections get
    unique non-colliding IDs.  This context uses the same image-visible names
    so the LLM, the user, and the image are all consistent.

    Parameters
    ----------
    image_rename_map : {fresh_vision_name: image_label} returned by
                       redraw_image_with_auto_matches(). If None, image-visible
                       names are computed locally (backward compat).
    """
    matched, new_parts, missing_parts = _match_parts_by_position(
        old_state, fresh_state,
    )

    old_preds   = old_state.get("predicates", {})
    fresh_preds = fresh_state.get("predicates", {})

    old_colors   = {e["part"]: e.get("color", "?") for e in old_preds.get("color", [])}
    fresh_colors = {e["part"]: e.get("color", "?") for e in fresh_preds.get("color", [])}
    old_frag     = {e["part"]: e.get("fragility") for e in old_preds.get("fragility", [])}

    old_at   = {e["part"]: e["slot"] for e in old_preds.get("at", [])}
    fresh_at = {e["part"]: e["slot"] for e in fresh_preds.get("at", [])}

    # Image-visible name: use the map from redraw_image_with_auto_matches
    # if provided, otherwise build locally for backward compatibility.
    if image_rename_map is not None:
        img_name = image_rename_map  # fresh_name → image label
    else:
        img_name = {}
        for mem_name, fresh_name in matched:
            img_name[fresh_name] = mem_name
        for np_name in new_parts:
            img_name[np_name] = np_name

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

    # ── Auto-matched parts ────────────────────────────────────────────────
    if matched:
        lines.append(
            "\nAUTO-MATCHED PARTS (same slot + same colour, or same position + same colour → assumed same part):"
        )
        lines.append(
            "  (The image shows these with their old-config name.)"
        )
        for mem_name, fresh_name in matched:
            fc   = fresh_colors.get(fresh_name, "?")
            slot = fresh_at.get(fresh_name, "standalone")
            frag = f", fragile" if mem_name in old_frag else ""
            lines.append(f"  '{mem_name}' ({fc}{frag}) in {slot}")

    # ── Unmatched new detections ─────────────────────────────────────────
    if new_parts:
        lines.append("\nNEW DETECTIONS (no matching old part at same slot/position+colour):")
        for np_name in new_parts:
            np_img = img_name.get(np_name, np_name)
            fc   = fresh_colors.get(np_name, "?")
            slot = fresh_at.get(np_name, "standalone")
            lines.append(f"  '{np_img}' ({fc}) in {slot}")

    # ── Missing old parts ────────────────────────────────────────────────
    if missing_parts:
        lines.append("\nMISSING FROM NEW SCAN:")
        for mp in missing_parts:
            oc   = old_colors.get(mp, "?")
            slot = old_at.get(mp, "standalone")
            frag = f", fragile" if mp in old_frag else ""
            lines.append(f"  '{mp}' ({oc}{frag}) was in {slot}")

    # ── Slot-level comparison table ──────────────────────────────────────
    old_slot_to_part   = {s: p for p, s in old_at.items()}
    fresh_slot_to_part = {s: p for p, s in fresh_at.items()}
    all_slots = sorted(set(list(old_at.values()) + list(fresh_at.values())))

    if all_slots:
        lines.append("\nSLOT-BY-SLOT COMPARISON (old config → current image):")
        for slot in all_slots:
            op       = old_slot_to_part.get(slot, "empty")
            fp_fresh = fresh_slot_to_part.get(slot, "empty")
            fp_img   = img_name.get(fp_fresh, fp_fresh) if fp_fresh != "empty" else "empty"
            oc = f"/{old_colors.get(op, '?')}" if op != "empty" else ""
            fc = f"/{fresh_colors.get(fp_fresh, '?')}" if fp_fresh != "empty" else ""
            of = ",fragile" if op in old_frag else ""
            change = ""
            if op == "empty" and fp_fresh != "empty":
                change = " ← NEW OCCUPANT"
            elif op != "empty" and fp_fresh == "empty":
                change = " ← NOW EMPTY"
            elif op != "empty" and fp_fresh != "empty":
                if old_colors.get(op, "?") != fresh_colors.get(fp_fresh, "?"):
                    change = " ← COLOR CHANGED"
            lines.append(
                f"  {slot}: {op}{oc}{of} → {fp_img}{fc}{change}"
            )

    # ── Move hypotheses ──────────────────────────────────────────────────
    # Uses image-visible names (mem_name for auto-matched parts).
    if missing_parts:
        move_hyps: List[str] = []
        for mp in missing_parts:
            mp_color = old_colors.get(mp, "?")
            mp_slot  = old_at.get(mp, "?")
            frag = ", fragile" if mp in old_frag else ""

            for mem_name, fresh_name in matched:
                if mem_name == mp:
                    continue
                fc = fresh_colors.get(fresh_name, "?")
                if fc != mp_color or mp_color == "?":
                    continue
                fresh_slot = fresh_at.get(fresh_name, "?")
                matched_old_slot = old_at.get(mem_name, "?")
                if matched_old_slot == mp_slot:
                    continue
                # Image shows mem_name — use that in the override syntax
                move_hyps.append(
                    f"  '{mem_name}' in {fresh_slot} could actually be "
                    f"'{mp}'({mp_color}{frag}) moved from {mp_slot}. "
                    f"If so, override: {{\"{mem_name}\": \"{mp}\"}}"
                )

        if move_hyps:
            lines.append(
                "\nPOSSIBLE MOVES — the auto-matcher assigns identity by "
                "slot+colour, so a part moved to a different slot gets the "
                "wrong identity if that slot previously held a same-colour part:"
            )
            lines.extend(move_hyps)
            lines.append(
                "  → Ask the user which interpretation is correct before "
                "accepting the auto-match."
            )

    # ── High-level attributes to preserve ────────────────────────────────
    if old_frag:
        lines.append("\nHIGH-LEVEL ATTRIBUTES TO PRESERVE (from old config):")
        for part_name, frag_val in sorted(old_frag.items()):
            lines.append(f"  {part_name}: fragility={frag_val}")

    if not new_parts and not missing_parts:
        lines.append("\nAll parts auto-matched successfully. No ambiguities.")

    # ── Inventory summary (for sanity-checking user claims) ──────────────
    old_parts_list = old_state.get("objects", {}).get("parts", [])
    fresh_parts_list = fresh_state.get("objects", {}).get("parts", [])

    # Count by colour
    from collections import Counter
    old_color_counts = Counter(old_colors.get(p, "unknown") for p in old_parts_list)
    fresh_color_counts = Counter(fresh_colors.get(p, "unknown") for p in fresh_parts_list)
    all_colors_seen = sorted(set(list(old_color_counts.keys()) + list(fresh_color_counts.keys())))

    lines.append(f"\nINVENTORY SUMMARY (use to sanity-check user claims):")
    lines.append(f"  Total parts:  old={len(old_parts_list)}, current={len(fresh_parts_list)}, "
                 f"delta={len(fresh_parts_list) - len(old_parts_list):+d}")
    for c in all_colors_seen:
        oc = old_color_counts.get(c, 0)
        fc = fresh_color_counts.get(c, 0)
        delta = fc - oc
        if delta != 0:
            lines.append(f"  {c}: old={oc}, current={fc} (delta={delta:+d})")
        else:
            lines.append(f"  {c}: old={oc}, current={fc} (unchanged)")

    return "\n".join(lines)


def apply_update_mapping(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    mapping: Dict[str, str],
    image_rename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Apply the combined identity mapping (auto-matches + LLM overrides) to
    produce the final merged state.

    Parameters
    ----------
    old_state   : previous configuration (source of high-level attributes)
    fresh_state : fresh vision scan (source of physical state)
    mapping     : {image_visible_name: old_part_name | "new"} — LLM/user overrides.
                  Keys use the names shown in the GUI image: old-config names
                  for auto-matched parts, preliminary new IDs for new detections.
    image_rename_map : {fresh_vision_name: image_label} returned by
                       redraw_image_with_auto_matches(). Used to translate
                       image-visible mapping keys back to fresh-scan names.
                       If None, falls back to auto-match-only translation.
    """
    result = copy.deepcopy(fresh_state)

    # ── compute auto-matches ──────────────────────────────────────────────────
    auto_matched, new_parts, missing_parts = _match_parts_by_position(
        old_state, fresh_state,
    )

    # ── translate image-visible mapping keys → fresh-scan names ──────────────
    # The image shows renamed labels (old names for matched, new IDs for new
    # detections).  The mapping from the LLM uses these image labels as keys.
    # We need the original fresh-scan names for the rename logic below.
    if image_rename_map:
        # Build reverse: image_label → fresh_name
        img_to_fresh: Dict[str, str] = {v: k for k, v in image_rename_map.items()}
    else:
        # Fallback: only auto-match translation (no new-detection translation)
        img_to_fresh = {}
        for old_name, fresh_name in auto_matched:
            img_to_fresh[old_name] = fresh_name

    translated_mapping: Dict[str, str] = {}
    for img_key, target in mapping.items():
        if img_key in img_to_fresh:
            translated_mapping[img_to_fresh[img_key]] = target
        else:
            # Key is already a fresh-scan name or unknown — pass through
            translated_mapping[img_key] = target
    mapping = translated_mapping

    # ── collect all old part numbers so new IDs don't collide ────────────────
    used_numbers: set = set()
    for old_name in old_state.get("objects", {}).get("parts", []):
        if old_name.startswith("Part_"):
            try:
                used_numbers.add(int(old_name.split("_", 1)[1]))
            except ValueError:
                pass

    # ── build full rename map: auto-matches first, then user overrides ───────
    rename_map: Dict[str, str] = {}

    # Start with auto-matches (fresh_name → old_name)
    for old_name, fresh_name in auto_matched:
        rename_map[fresh_name] = old_name

    # Layer user/LLM mapping on top — these override auto-matches
    # First, find which old_names the user is claiming for different fresh parts
    user_claimed_old: Dict[str, str] = {}  # old_name → fresh_name (from user)
    for fresh_name, target in mapping.items():
        if target != "new":
            user_claimed_old[target] = fresh_name

    # Displace any auto-match that conflicts with a user override
    for fresh_name_auto, old_name_auto in list(rename_map.items()):
        if old_name_auto in user_claimed_old:
            user_fresh = user_claimed_old[old_name_auto]
            if user_fresh != fresh_name_auto:
                # User is assigning this old_name to a DIFFERENT fresh part
                # → displace the auto-match (this fresh part will get a new ID)
                del rename_map[fresh_name_auto]
                if fresh_name_auto not in mapping:
                    new_parts.append(fresh_name_auto)

    # Apply user overrides
    for fresh_name, target in mapping.items():
        if target != "new":
            rename_map[fresh_name] = target
            if target.startswith("Part_"):
                try:
                    used_numbers.add(int(target.split("_", 1)[1]))
                except ValueError:
                    pass

    # Also track numbers from auto-matches
    for old_name in rename_map.values():
        if old_name.startswith("Part_"):
            try:
                used_numbers.add(int(old_name.split("_", 1)[1]))
            except ValueError:
                pass

    # Assign new sequential IDs for "new" parts and displaced auto-matches
    all_new = [fn for fn, tgt in mapping.items() if tgt == "new"]
    # Also include fresh parts not in any mapping (auto-match or user)
    all_fresh = set(fresh_state.get("objects", {}).get("parts", []))
    unmapped = [p for p in sorted(all_fresh) if p not in rename_map and p not in all_new]
    all_new = all_new + [p for p in new_parts if p not in all_new] + unmapped

    next_num = max(used_numbers, default=0) + 1
    for fresh_name in all_new:
        if fresh_name in rename_map:
            continue  # already assigned
        while next_num in used_numbers:
            next_num += 1
        rename_map[fresh_name] = f"Part_{next_num}"
        used_numbers.add(next_num)
        next_num += 1

    # ── validate: no duplicate target names ──────────────────────────────────
    target_counts: Dict[str, List[str]] = {}
    for fresh, old in rename_map.items():
        target_counts.setdefault(old, []).append(fresh)
    for target, sources in target_counts.items():
        if len(sources) > 1:
            print(f"  ⚠  ID collision: {sources} all map to '{target}' — "
                  f"keeping first, reassigning others")
            for extra in sources[1:]:
                while next_num in used_numbers:
                    next_num += 1
                rename_map[extra] = f"Part_{next_num}"
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
    print("✅  Configuration updated.")

    try:
        from Configuration_Module.Apply_Sequence_Changes import save_to_memory  # type: ignore
        mem_path = save_to_memory(merged, label="scene_update")
        print("✅  State archived.")
    except ImportError:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        arc_path = MEMORY_DIR / f"configuration_scene_update_{ts}.json"
        _save_atomic(arc_path, merged)
        print("✅  State archived.")

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
    pixel map saved by Vision_Main, applying the new part names from rename_map
    and adding FRAGILE labels from the merged state.
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

        # Build fragility lookup from merged state
        preds = merged_state.get("predicates", {})

        fragile_set: set = set()
        for entry in preds.get("fragility", []):
            if entry.get("fragility") == "fragile":
                fragile_set.add(entry["part"])

        # Apply rename_map to pixel annotations
        updated_annotations = []
        for p in pixel_map:
            old_name = p["name"]
            new_name = rename_map.get(old_name, old_name)
            entry = {**p, "name": new_name}
            updated_annotations.append(entry)

        annotate_parts(img, updated_annotations, fragile_set=fragile_set)

        cv2.imwrite(str(out_path), img)
        print("✅  Image updated.")

    except Exception as exc:
        print(f"  ⚠  Image re-annotation failed: {exc}")


def redraw_image_with_auto_matches(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
) -> Dict[str, str]:
    """
    Re-annotate latest_image.png with auto-matched part names BEFORE the
    LLM dialogue starts.  This way the user sees old-config IDs (for parts
    that auto-matched) and unique new IDs for genuinely new detections,
    rather than the arbitrary sequential IDs assigned by the vision detector.

    Called from _run_update_dialogue() right after prepare_update().

    Returns the rename map {fresh_vision_name: image_label} so that
    build_update_context() can use the same image-visible names.
    """
    matched, new_parts, _missing = _match_parts_by_position(
        old_state, fresh_state,
    )

    # Build preliminary rename map: auto-matches → old names
    rename_map: Dict[str, str] = {}
    for old_name, fresh_name in matched:
        rename_map[fresh_name] = old_name

    # Assign new detections unique IDs that don't collide with any old name
    # or any auto-matched name.
    used_numbers: set = set()
    for name in old_state.get("objects", {}).get("parts", []):
        if name.startswith("Part_"):
            try:
                used_numbers.add(int(name.split("_", 1)[1]))
            except ValueError:
                pass

    next_num = max(used_numbers, default=0) + 1
    for fresh_name in new_parts:
        while next_num in used_numbers:
            next_num += 1
        rename_map[fresh_name] = f"Part_{next_num}"
        used_numbers.add(next_num)
        next_num += 1

    # Build fragility set from old config so FRAGILE labels appear
    # on auto-matched parts that had fragility in the previous config
    old_preds = old_state.get("predicates", {})
    old_frag = {
        e["part"] for e in old_preds.get("fragility", [])
        if e.get("fragility") == "fragile"
    }

    fragile_set: set = set()
    for fresh_name, old_name in rename_map.items():
        if old_name in old_frag:
            fragile_set.add(old_name)

    _redraw_annotated_image(rename_map, {
        "predicates": {
            "fragility": [
                {"part": p, "fragility": "fragile"} for p in fragile_set
            ]
        }
    })

    return rename_map


# ── post-execution automated rescan ──────────────────────────────────────────

_POST_EXEC_THRESHOLD_M = 0.035


def run_post_execution_rescan(config_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a fresh camera image after execution and merge it with the
    post-execution config file.  The config is the source of truth for
    part identity — no user interaction is needed.
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
    rename_map: Dict[str, str] = {}
    for config_name, fresh_name in matched:
        rename_map[fresh_name] = config_name

    # Remove unmatched fresh parts from the fresh state (they're noise)
    fresh_parts = fresh_state.get("objects", {}).get("parts", [])
    matched_fresh_names = {fn for _, fn in matched}
    parts_to_remove = [p for p in fresh_parts if p not in matched_fresh_names]

    if parts_to_remove:
        fresh_state["objects"]["parts"] = [
            p for p in fresh_parts if p in matched_fresh_names
        ]
        fresh_preds = fresh_state.get("predicates", {})
        for pred_key in ("at", "color", "fragility"):
            fresh_preds[pred_key] = [
                e for e in fresh_preds.get(pred_key, [])
                if e.get("part") not in parts_to_remove
            ]
        fresh_metric = fresh_state.get("metric", {})
        for p in parts_to_remove:
            fresh_metric.pop(p, None)

    # Rename matched fresh parts to config IDs
    _rename_parts_in_state(fresh_state, rename_map)

    # ── inject missing config parts back (vision didn't see them) ─────────
    config_metric = config_state.get("metric", {})
    config_preds  = config_state.get("predicates", {})
    config_at     = {e["part"]: e["slot"] for e in config_preds.get("at", [])}
    config_colors = {e["part"]: e.get("color") for e in config_preds.get("color", [])}

    fresh_preds  = fresh_state.get("predicates", {})
    fresh_metric = fresh_state.get("metric", {})

    for mp in missing_config:
        if mp not in fresh_state["objects"]["parts"]:
            fresh_state["objects"]["parts"].append(mp)

        if mp in config_at:
            fresh_preds.setdefault("at", []).append(
                {"part": mp, "slot": config_at[mp]}
            )
            slot = config_at[mp]
            if slot in fresh_preds.get("slot_empty", []):
                fresh_preds["slot_empty"].remove(slot)

        if mp in config_colors:
            fresh_preds.setdefault("color", []).append(
                {"part": mp, "color": config_colors[mp]}
            )

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
    print("✅  Configuration updated.")

    try:
        from Configuration_Module.Apply_Sequence_Changes import save_to_memory  # type: ignore
        mem_path = save_to_memory(merged, label="post_exec_rescan")
        print("✅  State archived.")
    except ImportError:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        arc_path = MEMORY_DIR / f"configuration_post_exec_{ts}.json"
        _save_atomic(arc_path, merged)
        print("✅  State archived.")

    # ── redraw image with confirmed IDs + fragility ───────────────────────
    _redraw_annotated_image(rename_map, merged)

    print("── Post-execution rescan complete. ──\n")
    return merged