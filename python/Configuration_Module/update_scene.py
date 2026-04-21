"""
update_scene.py — Scene update module.

Takes a fresh camera scan and uses an LLM dialogue (driven by
``api_main.run_update_dialogue``) to resolve how freshly detected parts
correspond to parts in the previous configuration.

Design principle
----------------
Vision is the sole source of truth for the PHYSICAL state (positions,
colours).  Part IDENTITY is preserved by matching fresh detections to old
parts — first automatically (by position + colour), then via a user-facing
LLM conversation for any ambiguous cases.

The three public functions form a pipeline::

    prepare_update()       → run vision, return (old_state, fresh_state)
    build_update_context() → compute diff for the LLM prompt
    apply_update_mapping() → apply the confirmed mapping, merge, and save

Physical state (always taken from fresh vision scan):
    objects, slot_belongs_to, predicates: at, slot_empty, color, metric

High-level config (always preserved from old config):
    workspace: operation_mode, batch_size
    predicates: role, priority, kit_recipe, part_compatibility, fragility
"""
from __future__ import annotations

import copy
import json
import logging
import math
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from Core.config import settings
from Core.paths import (
    CONFIGURATION_PATH,
    WORKSPACE_DIR,
    PROJECT_DIR,
    empty_state,
    save_atomic,
    save_to_memory,
)

logger = logging.getLogger(__name__)

WORKSPACE = WORKSPACE_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

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
    a process with the native libraries already loaded by api_main.
    """
    vision_main_path = PROJECT_DIR / "Vision_Module" / "Vision_Main.py"
    print("\nStarting vision module …")
    result = subprocess.run([sys.executable, str(vision_main_path)], cwd=str(PROJECT_DIR))
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
    threshold_m: Optional[float] = None,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Match freshly detected parts to memory parts by slot identity OR
    XY proximity, combined with colour compatibility.

    Returns
    -------
    matched : [(old_name, fresh_name), ...]
    new_parts : [fresh_name, ...]   — fresh parts with no match in memory
    missing_parts : [old_name, ...]  — memory parts not found in fresh scan
    """
    if threshold_m is None:
        threshold_m = settings.position_match_threshold_m

    mem_metric = mem_state.get("metric", {})
    fresh_metric = fresh_state.get("metric", {})
    mem_preds = mem_state.get("predicates", {})
    fresh_preds = fresh_state.get("predicates", {})

    mem_parts = mem_state.get("objects", {}).get("parts", [])
    fresh_parts = fresh_state.get("objects", {}).get("parts", [])

    mem_colors: Dict[str, str] = {
        e["part"]: (e.get("color") or "").lower() for e in mem_preds.get("color", [])
    }
    fresh_colors: Dict[str, str] = {
        e["part"]: (e.get("color") or "").lower() for e in fresh_preds.get("color", [])
    }

    mem_at: Dict[str, str] = {e["part"]: e["slot"] for e in mem_preds.get("at", [])}
    fresh_at: Dict[str, str] = {e["part"]: e["slot"] for e in fresh_preds.get("at", [])}
    slot_to_mem: Dict[str, str] = {slot: part for part, slot in mem_at.items()}

    def _colors_compatible(c1: str, c2: str) -> bool:
        return not c1 or not c2 or c1 == c2

    matched: List[Tuple[str, str]] = []
    unmatched_fresh = list(fresh_parts)
    unmatched_mem = list(mem_parts)

    # Pass 1: slot-based matching
    for fresh_name in list(unmatched_fresh):
        slot = fresh_at.get(fresh_name)
        if slot and slot in slot_to_mem:
            old_name = slot_to_mem[slot]
            if old_name in unmatched_mem:
                fc = fresh_colors.get(fresh_name, "")
                mc = mem_colors.get(old_name, "")
                if _colors_compatible(fc, mc):
                    matched.append((old_name, fresh_name))
                    unmatched_fresh.remove(fresh_name)
                    unmatched_mem.remove(old_name)

    # Pass 2: XY-based matching
    fresh_xy: Dict[str, Optional[Tuple[float, float]]] = {
        p: _part_xy(fresh_metric, p) for p in unmatched_fresh
    }
    mem_xy: Dict[str, Optional[Tuple[float, float]]] = {
        p: _part_xy(mem_metric, p) for p in unmatched_mem
    }

    candidates: List[Tuple[float, str, str]] = []
    for fresh_name in unmatched_fresh:
        fxy = fresh_xy.get(fresh_name)
        if fxy is None:
            continue
        for old_name in unmatched_mem:
            mxy = mem_xy.get(old_name)
            if mxy is None:
                continue
            fc = fresh_colors.get(fresh_name, "")
            mc = mem_colors.get(old_name, "")
            if not _colors_compatible(fc, mc):
                continue
            dist = math.hypot(fxy[0] - mxy[0], fxy[1] - mxy[1])
            if dist <= threshold_m:
                candidates.append((dist, old_name, fresh_name))

    candidates.sort(key=lambda x: x[0])
    for dist, old_name, fresh_name in candidates:
        if old_name in unmatched_mem and fresh_name in unmatched_fresh:
            matched.append((old_name, fresh_name))
            unmatched_fresh.remove(fresh_name)
            unmatched_mem.remove(old_name)

    return matched, unmatched_fresh, unmatched_mem


def _rename_parts_in_state(
    state: Dict[str, Any],
    rename_map: Dict[str, str],
) -> None:
    """Rename parts in-place according to rename_map {old_name: new_name}."""
    if not rename_map:
        return

    objs = state.get("objects", {})
    objs["parts"] = [rename_map.get(p, p) for p in objs.get("parts", [])]

    preds = state.get("predicates", {})
    for pred in ("at", "color", "fragility"):
        preds[pred] = [
            {**e, "part": rename_map.get(e["part"], e["part"])}
            for e in preds.get(pred, [])
        ]

    metric = state.get("metric", {})
    for old, new in rename_map.items():
        if old in metric and old != new:
            metric[new] = metric.pop(old)


def _apply_high_level(
    fresh_state: Dict[str, Any],
    old_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay high-level config from old_state onto fresh_state."""
    result = copy.deepcopy(fresh_state)
    result["workspace"] = copy.deepcopy(
        old_state.get("workspace", {"operation_mode": None, "batch_size": None})
    )
    old_preds = old_state.get("predicates", {})
    preds = result.setdefault("predicates", {})
    for key in ("role", "priority", "kit_recipe", "part_compatibility", "fragility"):
        preds[key] = copy.deepcopy(old_preds.get(key, []))
    return result


# ── pipeline entry points ─────────────────────────────────────────────────────

def prepare_update() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run vision and load both the old state (from memory) and fresh state.

    Returns ``(old_state, fresh_state)``.
    """
    if CONFIGURATION_PATH.exists():
        old_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    else:
        old_state = empty_state()

    save_atomic(CONFIGURATION_PATH, old_state)
    _run_vision()
    fresh_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    save_atomic(CONFIGURATION_PATH, old_state)

    return old_state, fresh_state


def prepare_recapture(old_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run vision again to get a fresh scan, keeping old_state as the reference.

    Called when the user rejects the initial image and requests a retake.
    Returns the new fresh_state dict.
    """
    save_atomic(CONFIGURATION_PATH, old_state)
    _run_vision()
    fresh_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    # Restore old_state so configuration.json stays consistent until mapping is applied
    save_atomic(CONFIGURATION_PATH, old_state)
    return fresh_state


def build_update_context(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    image_rename_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Compute an auto-match summary for the LLM prompt.

    Parameters
    ----------
    image_rename_map:
        ``{fresh_vision_name: image_label}`` — maps internal fresh names to
        the labels shown on the annotated image shown to the user.
    """
    if image_rename_map is None:
        image_rename_map = {}

    matched, new_parts, missing_parts = _match_parts_by_position(old_state, fresh_state)

    old_preds = old_state.get("predicates", {})
    fresh_preds = fresh_state.get("predicates", {})
    old_colors: Dict[str, str] = {
        e["part"]: (e.get("color") or "unknown") for e in old_preds.get("color", [])
    }
    fresh_colors: Dict[str, str] = {
        e["part"]: (e.get("color") or "unknown") for e in fresh_preds.get("color", [])
    }

    lines: List[str] = ["AUTO-MATCHED PARTS (accepted automatically):"]
    for old_name, fresh_name in matched:
        img_label = image_rename_map.get(fresh_name, fresh_name)
        oc = old_colors.get(old_name, "unknown")
        fc = fresh_colors.get(fresh_name, "unknown")
        color_note = f" (was {oc}, now {fc})" if oc != fc and oc != "unknown" else f" ({fc})"
        lines.append(f"  {img_label} → {old_name}{color_note}")

    if not matched:
        lines.append("  (none)")

    lines.append("")
    lines.append("NEW DETECTIONS (no matching old part found):")
    for fresh_name in new_parts:
        img_label = image_rename_map.get(fresh_name, fresh_name)
        fc = fresh_colors.get(fresh_name, "unknown")
        lines.append(f"  {img_label} ({fc}) — new part")
    if not new_parts:
        lines.append("  (none)")

    lines.append("")
    lines.append("MISSING FROM FRESH SCAN (old parts not detected):")
    for old_name in missing_parts:
        oc = old_colors.get(old_name, "unknown")
        lines.append(f"  {old_name} ({oc}) — not detected")
    if not missing_parts:
        lines.append("  (none)")

    return "\n".join(lines)


def apply_update_mapping(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
    confirmed_mapping: Dict[str, str],
    image_rename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Apply the LLM-confirmed override mapping and save the merged state.

    Returns the final merged state.
    """
    if image_rename_map is None:
        image_rename_map = {}

    matched_auto, new_parts_fresh, _ = _match_parts_by_position(old_state, fresh_state)

    # Build rename map: fresh_name → final_identity
    rename_map: Dict[str, str] = {}
    for old_name, fresh_name in matched_auto:
        img_label = image_rename_map.get(fresh_name, fresh_name)
        if img_label in confirmed_mapping:
            rename_map[fresh_name] = confirmed_mapping[img_label]
        else:
            rename_map[fresh_name] = old_name

    # Assign new parts
    used = set(rename_map.values()) | set(old_state.get("objects", {}).get("parts", []))
    used_nums = set()
    for name in used:
        if name.startswith("Part_"):
            try:
                used_nums.add(int(name.split("_", 1)[1]))
            except ValueError:
                pass

    next_num = max(used_nums, default=0) + 1
    for fresh_name in new_parts_fresh:
        img_label = image_rename_map.get(fresh_name, fresh_name)
        if img_label in confirmed_mapping:
            target = confirmed_mapping[img_label]
            if target == "new":
                while next_num in used_nums:
                    next_num += 1
                rename_map[fresh_name] = f"Part_{next_num}"
                used_nums.add(next_num)
                next_num += 1
            else:
                rename_map[fresh_name] = target
        else:
            while next_num in used_nums:
                next_num += 1
            rename_map[fresh_name] = f"Part_{next_num}"
            used_nums.add(next_num)
            next_num += 1

    result = copy.deepcopy(fresh_state)
    old_parts_in_rename = {v for v in rename_map.values() if v.startswith("Part_")}

    fresh_parts_total = len(fresh_state.get("objects", {}).get("parts", []))
    if fresh_parts_total > 0 and not old_parts_in_rename:
        logger.warning(
            "All %d fresh part(s) received new IDs — no position match found.",
            fresh_parts_total,
        )

    _rename_parts_in_state(result, rename_map)
    merged = _apply_high_level(result, old_state)

    # Summary
    m_parts = set(old_state.get("objects", {}).get("parts", []))
    f_parts = set(merged.get("objects", {}).get("parts", []))
    for p in sorted(f_parts - m_parts):
        print(f"  + {p}  [new part]")
    for p in sorted(m_parts - f_parts):
        print(f"  - {p}  [removed]")
    if m_parts == f_parts:
        print("  (no structural changes — positions refreshed)")

    save_atomic(CONFIGURATION_PATH, merged)
    print("✅  Configuration updated.")
    save_to_memory(merged, label="update")
    print("✅  State archived.")

    _redraw_annotated_image(rename_map, merged)
    return merged


# ── image re-annotation ──────────────────────────────────────────────────────

def _redraw_annotated_image(
    rename_map: Dict[str, str],
    merged_state: Dict[str, Any],
) -> None:
    """Redraw latest_image.png with updated part names and fragility labels."""
    base_path = WORKSPACE / "latest_image_base.png"
    pmap_path = WORKSPACE / "latest_pixel_map.json"
    out_path = WORKSPACE / "latest_image.png"

    if not base_path.exists() or not pmap_path.exists():
        logger.debug("Base image or pixel map not found — skipping re-annotation.")
        return

    try:
        import cv2
        from Vision_Module.pipeline import annotate_parts  # type: ignore

        img = cv2.imread(str(base_path))
        if img is None:
            logger.warning("Could not load base image — skipping re-annotation.")
            return

        pixel_map = json.loads(pmap_path.read_text(encoding="utf-8"))
        preds = merged_state.get("predicates", {})
        fragile_set = {
            e["part"] for e in preds.get("fragility", [])
            if e.get("fragility") == "fragile"
        }

        updated_annotations = [
            {**p, "name": rename_map.get(p["name"], p["name"])}
            for p in pixel_map
        ]
        annotate_parts(img, updated_annotations, fragile_set=fragile_set)
        cv2.imwrite(str(out_path), img)
        print("✅  Image updated.")

    except Exception as exc:
        logger.warning("Image re-annotation failed: %s", exc)


def redraw_image_with_auto_matches(
    old_state: Dict[str, Any],
    fresh_state: Dict[str, Any],
) -> Dict[str, str]:
    """
    Re-annotate latest_image.png with auto-matched part names before the LLM
    dialogue starts.

    Returns ``{fresh_vision_name: image_label}`` so that
    ``build_update_context`` can use the same image-visible names.
    """
    matched, new_parts, _ = _match_parts_by_position(old_state, fresh_state)

    rename_map: Dict[str, str] = {fresh_name: old_name for old_name, fresh_name in matched}

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

    old_preds = old_state.get("predicates", {})
    old_frag = {
        e["part"] for e in old_preds.get("fragility", [])
        if e.get("fragility") == "fragile"
    }
    fragile_set = {new_name for fresh, new_name in rename_map.items() if new_name in old_frag}

    _redraw_annotated_image(
        rename_map,
        {"predicates": {"fragility": [{"part": p, "fragility": "fragile"} for p in fragile_set]}},
    )
    return rename_map


def run_post_execution_rescan(config_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a fresh camera image after execution and merge it with the
    post-execution config file without user interaction.
    """
    print("\n── Post-execution vision rescan ──\n")
    save_atomic(CONFIGURATION_PATH, config_state)

    try:
        _run_vision()
    except Exception as exc:
        logger.warning("Vision rescan failed: %s", exc)
        save_atomic(CONFIGURATION_PATH, config_state)
        return config_state

    fresh_state = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    save_atomic(CONFIGURATION_PATH, config_state)

    matched, new_fresh, missing_config = _match_parts_by_position(
        config_state, fresh_state,
        threshold_m=settings.position_match_threshold_m * 0.875,  # tighter post-exec
    )

    rename_map = {fresh_name: config_name for config_name, fresh_name in matched}

    # Remove unmatched fresh parts
    matched_fresh = {fn for _, fn in matched}
    parts_to_remove = [
        p for p in fresh_state.get("objects", {}).get("parts", [])
        if p not in matched_fresh
    ]
    if parts_to_remove:
        fresh_state["objects"]["parts"] = [
            p for p in fresh_state["objects"]["parts"] if p in matched_fresh
        ]
        fresh_preds = fresh_state.get("predicates", {})
        for pred_key in ("at", "color", "fragility"):
            fresh_preds[pred_key] = [
                e for e in fresh_preds.get(pred_key, [])
                if e.get("part") not in parts_to_remove
            ]
        for p in parts_to_remove:
            fresh_state.get("metric", {}).pop(p, None)

    _rename_parts_in_state(fresh_state, rename_map)

    # Inject missing config parts
    config_preds = config_state.get("predicates", {})
    config_at = {e["part"]: e["slot"] for e in config_preds.get("at", [])}
    config_colors = {e["part"]: e.get("color") for e in config_preds.get("color", [])}
    fresh_preds = fresh_state.get("predicates", {})
    fresh_metric = fresh_state.get("metric", {})
    config_metric = config_state.get("metric", {})

    for mp in missing_config:
        if mp not in fresh_state["objects"]["parts"]:
            fresh_state["objects"]["parts"].append(mp)
        if mp in config_at:
            fresh_preds.setdefault("at", []).append({"part": mp, "slot": config_at[mp]})
            slot = config_at[mp]
            if slot in fresh_preds.get("slot_empty", []):
                fresh_preds["slot_empty"].remove(slot)
        if mp in config_colors:
            fresh_preds.setdefault("color", []).append({"part": mp, "color": config_colors[mp]})
        if mp in config_metric:
            fresh_metric[mp] = config_metric[mp]
        logger.warning("%s not detected by vision — kept with predicted position", mp)

    fresh_state["objects"]["parts"] = sorted(fresh_state["objects"]["parts"])
    merged = _apply_high_level(fresh_state, config_state)

    save_atomic(CONFIGURATION_PATH, merged)
    print("✅  Configuration updated.")
    save_to_memory(merged, label="post_exec_rescan")
    print("✅  State archived.")
    _redraw_annotated_image(rename_map, merged)
    print("── Post-execution rescan complete. ──\n")
    return merged
