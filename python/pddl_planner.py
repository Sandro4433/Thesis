# pddl_planner.py
#
# Full PDDL planning pipeline:
#   state (configuration.json) → domain.pddl + problem.pddl → planner → sequence.json
#
# Two planner backends:
#
#   Fast Downward (recommended, set FAST_DOWNWARD_PATH in config.py)
#     Uses PDDL 2.1 with action costs and numeric fluents.
#     Priority ordering is enforced as a HARD CONSTRAINT in the domain:
#     a part at priority level K cannot be picked until all parts at levels
#     1…K-1 have been picked.  This is academically standard and requires no
#     post-processing.
#     Install: git clone https://github.com/aibasel/downward && python build.py
#
#     Does not support numeric fluents, so priority is enforced by
#     post-sorting the plan (soft enforcement).

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

GRIPPER_CLOSE_STANDARD = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# PDDL domain builders
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Goal generators
# ─────────────────────────────────────────────────────────────────────────────

def _generate_goal(state: Dict[str, Any], mode: str) -> List[str]:
    if mode == "kitting":
        return _goal_kitting(state)
    return _goal_sorting(state)


def _goal_sorting(state: Dict[str, Any]) -> List[str]:
    """
    Generate sorting goals from part_compatibility rules.
    
    Supports both color-based and part-specific rules:
      - {"part_color": "blue", "allowed_in": ["Container_1"]}
      - {"part_name": "Part_3", "allowed_in": ["Container_2"]}
      - {"part_name": "Part_3", "allowed_in": ["Container_2"], "target_slot": "Container_2_Pos_1"}
    
    Uses TWO-PASS approach:
      Pass 1: Process inclusion rules to build allowed receptacles
      Pass 2: Apply exclusion-only rules to remove receptacles
    
    Receptacle priority from priority list is used to determine which container to fill first.
    """
    preds        = state.get("predicates",    {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects",       {})
    inputs, outputs = _role_map(preds)
    color_map       = _color_map(preds)
    pkey            = _priority_key(preds)
    rkey            = _destination_priority_key(preds)

    parts_in_slot = {e["part"] for e in preds.get("at", [])}
    parts_in_input = [
        e["part"] for e in preds.get("at", [])
        if slot_belongs.get(e["slot"], "") in inputs
    ]
    standalone    = [p for p in objs.get("parts", []) if p not in parts_in_slot]
    parts_to_move = parts_in_input + standalone

    # Build lookups for full additive scoring
    frag_map = {e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])}
    part_to_receptacle: Dict[str, str] = {}
    for entry in preds.get("at", []):
        parent = slot_belongs.get(entry["slot"], "")
        part_to_receptacle[entry["part"]] = parent

    # Sort by full additive score descending: higher score = assigned first
    def _sort_key(p):
        color = color_map.get(p, "")
        score = _compute_part_score(p, color, preds,
                                    part_to_receptacle=part_to_receptacle,
                                    fragility_map=frag_map)
        return (-score, pkey(color))

    parts_to_move.sort(key=_sort_key)

    compat_rules = preds.get("part_compatibility", [])
    
    # Build fragility lookup
    frag_map = {e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])}
    
    # Separate inclusion rules from pure exclusion rules
    inclusion_rules = []
    exclusion_rules = []
    
    for rule in compat_rules:
        has_inclusion = "allowed_in" in rule or "allowed_in_role" in rule
        has_exclusion = "not_allowed_in" in rule
        
        if has_exclusion and not has_inclusion:
            exclusion_rules.append(rule)
        elif has_inclusion:
            inclusion_rules.append(rule)
    
    # Build info map: part → {allowed_recs, target_slot}
    part_info: Dict[str, Dict[str, Any]] = {}
    color_to_allowed: Dict[str, set] = {}
    
    # PASS 1: Process inclusion rules
    for rule in inclusion_rules:
        allowed_recs = set(rule.get("allowed_in", []))
        
        # Handle allowed_in_role
        if "allowed_in_role" in rule:
            role = (rule["allowed_in_role"] or "").lower()
            if role == "output":
                allowed_recs.update(outputs)
            elif role == "input":
                allowed_recs.update(inputs)
        
        # Apply inline exclusions
        if "not_allowed_in" in rule:
            allowed_recs -= set(rule["not_allowed_in"])
        
        # Filter to only output receptacles
        allowed_recs &= outputs
        
        if not allowed_recs:
            continue
        
        target_slot = rule.get("target_slot")
        
        # Part-specific rule
        if "part_name" in rule:
            part_name = rule["part_name"]
            if part_name not in part_info:
                part_info[part_name] = {"allowed_recs": set(), "target_slot": None}
            part_info[part_name]["allowed_recs"].update(allowed_recs)
            if target_slot:
                part_info[part_name]["target_slot"] = target_slot
        
        # Color-based rule
        if "part_color" in rule:
            c = (rule["part_color"] or "").lower()
            color_to_allowed.setdefault(c, set()).update(allowed_recs)
    
    # PASS 2: Process pure exclusion rules
    for rule in exclusion_rules:
        excluded_recs = set(rule.get("not_allowed_in", []))
        
        # Determine which parts this exclusion applies to
        if "part_name" in rule:
            # Specific part
            part_name = rule["part_name"]
            if part_name in part_info:
                part_info[part_name]["allowed_recs"] -= excluded_recs
        
        if "part_color" in rule:
            # Color-based exclusion
            c = (rule["part_color"] or "").lower()
            if c in color_to_allowed:
                color_to_allowed[c] -= excluded_recs
        
        if "part_fragility" in rule:
            # Fragility-based exclusion — apply to all matching parts
            frag = (rule["part_fragility"] or "").lower()
            matching_parts = [p for p in parts_to_move if frag_map.get(p, "normal").lower() == frag]
            for part in matching_parts:
                if part in part_info:
                    part_info[part]["allowed_recs"] -= excluded_recs
                # Also check color-based and remove from those
                part_color = color_map.get(part, "")
                if part_color in color_to_allowed:
                    # Need to track per-part exclusions, build part_info entry
                    if part not in part_info:
                        part_info[part] = {"allowed_recs": color_to_allowed[part_color].copy(), "target_slot": None}
                    part_info[part]["allowed_recs"] -= excluded_recs

    empty_by_receptacle: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in outputs:
            empty_by_receptacle.setdefault(parent, []).append(s)

    goals: List[str] = []
    for part in parts_to_move:
        color = color_map.get(part, "")
        
        # Check part-specific rules first
        if part in part_info:
            info = part_info[part]
            target_slot = info.get("target_slot")
            allowed_recs = info.get("allowed_recs", set())
            
            # If target_slot is specified, use it directly
            if target_slot:
                slot_parent = slot_belongs.get(target_slot, "")
                if slot_parent in allowed_recs:
                    free_slots = empty_by_receptacle.get(slot_parent, [])
                    if target_slot in free_slots:
                        goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(target_slot)})")
                        free_slots.remove(target_slot)
                    else:
                        print(f"  ⚠  target slot '{target_slot}' is not empty — skipping {part}")
                continue
            
            # No target_slot — auto-assign (sorted by receptacle priority)
            valid = sorted([r for r in allowed_recs if empty_by_receptacle.get(r)], key=rkey)
        # Then check color-based rules
        elif color_to_allowed and color in color_to_allowed:
            valid = sorted([r for r in color_to_allowed[color] if empty_by_receptacle.get(r)], key=rkey)
        # No rules defined — allow any output (sorted by receptacle priority)
        elif not compat_rules:
            valid = sorted([r for r in outputs if empty_by_receptacle.get(r)], key=rkey)
        else:
            print(f"  ⚠  no part_compatibility rule for '{part}' (color={color}) — skipped.")
            continue
        
        if not valid:
            print(f"  ⚠  no available output slot for '{part}' (color={color}) — skipped.")
            continue
        
        receptacle = valid[0]
        slot = empty_by_receptacle[receptacle].pop(0)
        goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(slot)})")
    return goals


def _goal_kitting(state: Dict[str, Any]) -> List[str]:
    """
    Generate kitting goals from kit_recipe OR part_compatibility.
    
    If kit_recipe is defined, uses that (color-based selection).
    If only part_compatibility is defined, uses that (supports part_name for specific parts).
    
    Recipe format supports two modes:
      - Kit-specific: {"kit": "Kit_1", "color": "blue", "quantity": 2}
        → applies only to Kit_1
      - Universal: {"color": "blue", "quantity": 2}
        → applies to ALL output kits
    
    Universal recipes are expanded to all kits with role=output.
    
    Receptacle priority (from priority list with "kit"/"container" or legacy "destination"/"receptacle" key):
      - [{"kit": "Kit_1", "order": 1}, {"kit": "Kit_2", "order": 2}]
      - Lower order = higher priority (filled first)
      - This naturally achieves "finish Kit_1 before Kit_2" behavior
    """
    preds        = state.get("predicates",    {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects",       {})
    inputs, outputs = _role_map(preds)
    color_map    = _color_map(preds)
    pkey         = _priority_key(preds)
    rkey         = _destination_priority_key(preds)

    # Build available parts (in input receptacles or standalone)
    available_parts: List[str] = []
    parts_in_slot = {e["part"] for e in preds.get("at", [])}
    for e in preds.get("at", []):
        if slot_belongs.get(e["slot"], "") in inputs:
            available_parts.append(e["part"])
    for p in objs.get("parts", []):
        if p not in parts_in_slot:
            available_parts.append(p)

    kit_set     = set(objs.get("kits", []))
    # Only consider kits that are marked as output, sorted by priority
    output_kits = sorted([k for k in kit_set if k in outputs], key=rkey)

    # ── batch_size: limit how many kits are filled in one planning cycle ──
    workspace_cfg = state.get("workspace", {})
    batch_size = workspace_cfg.get("batch_size")
    if batch_size is not None and isinstance(batch_size, int) and batch_size > 0:
        output_kits = output_kits[:batch_size]
    
    empty_slots: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in kit_set:
            empty_slots.setdefault(parent, []).append(s)

    used_parts: set = set()
    goals: List[str] = []

    raw_recipes = preds.get("kit_recipe", [])
    compat_rules = preds.get("part_compatibility", [])

    # ── Build part → kit eligibility from ALL compatibility rules ──
    # This replaces the old exclusion-only map with a full compatible-pairs set
    # that respects BOTH inclusion rules (part_color+part_fragility→allowed_in)
    # and exclusion rules (not_allowed_in).
    # Without this, inclusion rules like {"part_color": "red", "part_fragility": "normal",
    # "allowed_in_role": "output"} would implicitly exclude fragile-red parts from goals
    # but _goal_kitting would still assign them (bug: unsolvable PDDL).

    # Build fragility lookup for matching
    frag_map = {e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])}

    # Build the set of (part, receptacle) pairs that are compatible
    # using the same two-pass logic as _build_compat_init
    _eligible_pairs: Optional[set] = None  # None means "no rules → all compatible"

    if compat_rules:
        _all_parts_set = set(available_parts)
        _output_kits_set = set(output_kits)

        _inclusion_rules = []
        _exclusion_rules = []
        for rule in compat_rules:
            has_incl = "allowed_in" in rule or "allowed_in_role" in rule
            has_excl = "not_allowed_in" in rule
            if has_excl and not has_incl:
                _exclusion_rules.append(rule)
            else:
                _inclusion_rules.append(rule)

        _eligible_pairs = set()

        # PASS 1: inclusion rules
        for rule in _inclusion_rules:
            matching = _all_parts_set.copy()
            if "part_color" in rule:
                c = (rule["part_color"] or "").lower()
                matching = {p for p in matching if color_map.get(p, "").lower() == c}
            if "part_fragility" in rule:
                f = (rule["part_fragility"] or "").lower()
                matching = {p for p in matching if frag_map.get(p, "normal").lower() == f}
            if "part_name" in rule:
                matching &= {rule["part_name"]}

            recs: set = set()
            if "allowed_in" in rule:
                recs = set(rule["allowed_in"]) & _output_kits_set
            elif "allowed_in_role" in rule:
                role = (rule["allowed_in_role"] or "").lower()
                if role == "output":
                    recs = _output_kits_set.copy()
            if "not_allowed_in" in rule:
                recs -= set(rule["not_allowed_in"])

            for p in matching:
                for r in recs:
                    _eligible_pairs.add((p, r))

        # If no inclusion rules produced pairs, default to all-compatible
        if not _eligible_pairs and _exclusion_rules:
            for p in _all_parts_set:
                for r in _output_kits_set:
                    _eligible_pairs.add((p, r))

        # PASS 2: pure exclusion rules override
        for rule in _exclusion_rules:
            matching = _all_parts_set.copy()
            if "part_color" in rule:
                c = (rule["part_color"] or "").lower()
                matching = {p for p in matching if color_map.get(p, "").lower() == c}
            if "part_fragility" in rule:
                f = (rule["part_fragility"] or "").lower()
                matching = {p for p in matching if frag_map.get(p, "normal").lower() == f}
            if "part_name" in rule:
                matching &= {rule["part_name"]}
            excluded = set(rule.get("not_allowed_in", []))
            for p in matching:
                for r in excluded:
                    _eligible_pairs.discard((p, r))

    # ── If we have kit_recipe, use color-based selection ──
    if raw_recipes:
        # Build lookups for full additive scoring
        _part_to_rec: Dict[str, str] = {}
        for entry in preds.get("at", []):
            parent = slot_belongs.get(entry["slot"], "")
            _part_to_rec[entry["part"]] = parent

        # Group available parts by color
        available_by_color: Dict[str, List[str]] = {}
        for p in available_parts:
            c = color_map.get(p, "unknown")
            available_by_color.setdefault(c, []).append(p)

        # Expand universal recipes (no "kit" field) to all output kits
        expanded_recipes: List[Dict[str, Any]] = []
        
        for recipe in raw_recipes:
            kit_field = (recipe.get("kit") or "").strip()
            if kit_field:
                # Kit-specific recipe — use as-is
                expanded_recipes.append(recipe)
            else:
                # Universal recipe — expand to all output kits (in priority order)
                for kit in output_kits:
                    expanded_recipes.append({
                        "kit": kit,
                        "color": recipe.get("color"),
                        "quantity": recipe.get("quantity"),
                        "size": recipe.get("size"),
                    })
        
        # Sort expanded recipes by kit priority first, then by color priority
        # This fills higher-priority kits first (achieving "finish Kit_1 before Kit_2")
        expanded_recipes.sort(key=lambda r: (rkey(r.get("kit", "")), pkey((r.get("color") or "").lower())))

        for recipe in expanded_recipes:
            kit   = recipe.get("kit", "")
            color = (recipe.get("color") or "").lower()
            qty   = int(recipe.get("quantity", 0))
            free_slots = empty_slots.get(kit, [])

            # Filter candidates: not used, and compatible with this kit
            if _eligible_pairs is not None:
                candidates = [
                    p for p in available_by_color.get(color, [])
                    if p not in used_parts and (p, kit) in _eligible_pairs
                ]
            else:
                candidates = [
                    p for p in available_by_color.get(color, [])
                    if p not in used_parts
                ]
            # Sort by full additive score: fragility, source, part_name all contribute
            # Higher score = picked first; negate for ascending sort
            candidates.sort(key=lambda p: -_compute_part_score(
                p, color_map.get(p, ""), preds,
                part_to_receptacle=_part_to_rec,
                fragility_map=frag_map,
            ))

            for part, slot in zip(candidates[:qty], free_slots[:qty]):
                goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(slot)})")
                used_parts.add(part)
                free_slots.remove(slot)

    # ── If we have part_compatibility (and no kit_recipe), use compatibility-based selection ──
    elif compat_rules:
        # TWO-PASS APPROACH:
        # Pass 1: Build allowed_kits from inclusion rules
        # Pass 2: Apply exclusions from exclusion-only rules
        
        # Build fragility lookup
        frag_map_local = {e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])}
        
        # Separate inclusion rules from pure exclusion rules
        inclusion_rules = []
        exclusion_rules = []
        
        for rule in compat_rules:
            has_inclusion = "allowed_in" in rule or "allowed_in_role" in rule
            has_exclusion = "not_allowed_in" in rule
            
            if has_exclusion and not has_inclusion:
                exclusion_rules.append(rule)
            elif has_inclusion:
                inclusion_rules.append(rule)
        
        # Build compatibility map: part → (allowed_kits, target_slot or None)
        part_info: Dict[str, Dict[str, Any]] = {}
        
        # PASS 1: Process inclusion rules
        for rule in inclusion_rules:
            # Determine which parts this rule applies to
            matching_parts: set = set(available_parts)
            
            if "part_name" in rule:
                matching_parts &= {rule["part_name"]}
            if "part_color" in rule:
                color = (rule["part_color"] or "").lower()
                matching_parts = {p for p in matching_parts if color_map.get(p, "").lower() == color}
            if "part_fragility" in rule:
                frag = (rule["part_fragility"] or "").lower()
                matching_parts = {p for p in matching_parts if frag_map_local.get(p, "normal").lower() == frag}
            
            # Determine which kits this rule allows
            allowed_kits: set = set()
            if "allowed_in" in rule:
                allowed_kits = set(rule["allowed_in"]) & set(output_kits)
            elif "allowed_in_role" in rule:
                role = (rule["allowed_in_role"] or "").lower()
                if role == "output":
                    allowed_kits = set(output_kits)
            
            # Apply inline exclusions (not_allowed_in combined with allowed_in)
            if "not_allowed_in" in rule:
                allowed_kits -= set(rule["not_allowed_in"])
            
            # Check for target_slot
            target_slot = rule.get("target_slot")
            
            # Record info for each matching part
            for part in matching_parts:
                if part not in part_info:
                    part_info[part] = {"allowed_kits": set(), "target_slot": None}
                part_info[part]["allowed_kits"].update(allowed_kits)
                if target_slot:
                    part_info[part]["target_slot"] = target_slot
        
        # PASS 2: Process pure exclusion rules (remove from allowed_kits)
        for rule in exclusion_rules:
            # Determine which parts this exclusion applies to
            matching_parts: set = set(available_parts)
            
            if "part_name" in rule:
                matching_parts &= {rule["part_name"]}
            if "part_color" in rule:
                color = (rule["part_color"] or "").lower()
                matching_parts = {p for p in matching_parts if color_map.get(p, "").lower() == color}
            if "part_fragility" in rule:
                frag = (rule["part_fragility"] or "").lower()
                matching_parts = {p for p in matching_parts if frag_map_local.get(p, "normal").lower() == frag}
            
            # Get excluded kits
            excluded_kits = set(rule.get("not_allowed_in", []))
            
            # REMOVE these kits from each matching part's allowed_kits
            for part in matching_parts:
                if part in part_info:
                    part_info[part]["allowed_kits"] -= excluded_kits
        
        # Generate goals for each part, sorted by full additive score
        _part_to_rec_compat: Dict[str, str] = {}
        for entry in preds.get("at", []):
            parent = slot_belongs.get(entry["slot"], "")
            _part_to_rec_compat[entry["part"]] = parent

        sorted_parts = sorted(part_info.keys(), key=lambda p: (
            -_compute_part_score(p, color_map.get(p, ""), preds,
                                 part_to_receptacle=_part_to_rec_compat,
                                 fragility_map=frag_map_local),
            pkey(color_map.get(p, "")),
        ))
        for part in sorted_parts:
            info = part_info[part]
            if part in used_parts:
                continue
            
            target_slot = info.get("target_slot")
            allowed_kits = info.get("allowed_kits", set())
            
            # If target_slot is specified, use it directly
            if target_slot:
                # Verify the slot is empty
                slot_parent = slot_belongs.get(target_slot, "")
                if slot_parent in allowed_kits:
                    free_slots_for_kit = empty_slots.get(slot_parent, [])
                    if target_slot in free_slots_for_kit:
                        goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(target_slot)})")
                        used_parts.add(part)
                        free_slots_for_kit.remove(target_slot)
                    else:
                        print(f"  ⚠  target slot '{target_slot}' is not empty — skipping {part}")
                continue
            
            # No target_slot — auto-assign to first available slot (sorted by receptacle priority)
            for kit in sorted(allowed_kits, key=rkey):
                free_slots_for_kit = empty_slots.get(kit, [])
                if free_slots_for_kit:
                    slot = free_slots_for_kit.pop(0)
                    goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(slot)})")
                    used_parts.add(part)
                    break

    if not goals:
        return _goal_sorting(state)
    return goals


# ─────────────────────────────────────────────────────────────────────────────
# Planner backends
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_downward(
    domain_path: str,
    problem_path: str,
    plan_path: str,
    fd_path: str,
) -> Optional[str]:
    """Run Fast Downward with A* + LM-Cut (cost-optimal)."""
    cmd = [
        sys.executable, fd_path,
        "--plan-file", plan_path,
        domain_path,
        problem_path,
        "--search", "astar(ff())",
    ]
    print(f"  command        : {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print(f"❌ Fast Downward not found at: {fd_path}")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Fast Downward timed out (120 s).")
        return None

    if result.returncode not in (0, 10, 11):
        print(f"❌ Fast Downward failed (exit {result.returncode}):")
        print(f"  stdout: {result.stdout or '(empty)'}")
        print(f"  stderr: {result.stderr or '(empty)'}")
        return None

    if not os.path.exists(plan_path):
        print("❌ Fast Downward did not produce a plan file.")
        print(result.stdout[-2000:])
        return None

    with open(plan_path) as f:
        plan_text = f.read()

    if not plan_text.strip():
        print("❌ Fast Downward plan file is empty.")
        return None

    return plan_text


# ─────────────────────────────────────────────────────────────────────────────
# Plan → sequence
# ─────────────────────────────────────────────────────────────────────────────

def parse_plan_to_sequence(plan_text: str, state: Dict[str, Any]) -> List[List]:
    """
    Convert planner output to sequence.json format:
      [[pick_name, place_name, gripper_close_width], ...]

    All parts use the standard gripper width (GRIPPER_CLOSE_STANDARD).

    Handles Fast Downward format  (with optional [COST] suffix):
      (pick-priority-1 part_3 ws_slot_part_3 workspace_floor) [1]
      (place part_3 container_1_pos_2 container_1) [0]

    Also handles compact concatenated format (action+args joined by underscores):
      (pick-priority-1_part_3_ws_slot_part_3_workspace_floor)
    """
    preds = state.get("predicates", {})
    objs  = state.get("objects",    {})

    part_names: List[str] = objs.get("parts", [])
    slot_names: List[str] = objs.get("slots", [])
    virtual_slot_names    = [f"ws_slot_{p}" for p in part_names]

    all_names: List[str] = part_names + slot_names
    ci_map     = {n.lower(): n for n in all_names}
    all_ext    = all_names + virtual_slot_names
    ci_map_ext = {n.lower(): n for n in all_ext}
    sorted_ext = sorted(all_ext, key=lambda n: len(n), reverse=True)

    virtual_to_part: Dict[str, str] = {
        f"ws_slot_{p}".lower(): p for p in part_names
    }

    pending: Dict[str, str] = {}
    sequence: List[List]    = []

    for line in plan_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        # Strip Fast Downward cost annotation: "(action ...) [N]"
        if line.endswith("]") and "[" in line:
            line = line[:line.rfind("[")].strip()
        line = line.strip("()")

        tokens = line.split()
        if len(tokens) >= 3:
            action_raw = tokens[0].lower()
            is_pick    = action_raw.startswith("pick")
            is_place   = action_raw == "place" or action_raw.startswith("place-")
            part_lower = tokens[1].lower()
            slot_lower = tokens[2].lower()

            if is_pick:
                part_orig = ci_map.get(part_lower, tokens[1])
                pending[part_lower] = part_orig
                continue

            if is_place and part_lower in pending:
                pick_name = pending.pop(part_lower)
                slot_orig = ci_map.get(slot_lower, tokens[2])
                sequence.append([pick_name, slot_orig])
                continue

        # ── Compact concatenated format (fallback) ───────────────────────────
        line_lower = line.lower()
        if line_lower.startswith("pick"):
            action    = "pick"
            remainder = line_lower[4:].lstrip("-_")
            # strip "priority-K" or "no-priority" infix
            for prefix in (["no-priority", "no_priority"]
                           + [f"priority-{k}" for k in range(1, 20)]
                           + [f"priority_{k}" for k in range(1, 20)]):
                if remainder.startswith(prefix):
                    remainder = remainder[len(prefix):].lstrip("-_")
                    break
        elif line_lower.startswith("place"):
            action    = "place"
            remainder = line_lower[5:].lstrip("-_")
            # strip "rec-priority-K", "no-rec-priority", "kit-priority-K", "kit-no-priority", "container" infix
            for prefix in (["no-rec-priority", "no_rec_priority",
                            "kit-no-priority", "kit_no_priority", "container"]
                           + [f"rec-priority-{k}" for k in range(1, 20)]
                           + [f"rec_priority_{k}" for k in range(1, 20)]
                           + [f"kit-priority-{k}" for k in range(1, 20)]
                           + [f"kit_priority_{k}" for k in range(1, 20)]):
                if remainder.startswith(prefix):
                    remainder = remainder[len(prefix):].lstrip("-_")
                    break
        else:
            continue

        found: List[str] = []
        pos = 0
        while pos < len(remainder) and len(found) < 3:
            matched = False
            for name in sorted_ext:
                nl = name.lower()
                if remainder[pos:].startswith(nl):
                    found.append(ci_map_ext.get(nl, name))
                    pos += len(nl) + 1
                    matched = True
                    break
            if not matched:
                pos += 1

        if len(found) < 2:
            continue

        part_orig  = found[0]
        slot_token = found[1]
        part_lower = part_orig.lower()

        if action == "pick":
            pending[part_lower] = part_orig
        elif action == "place" and part_lower in pending:
            pick_name = pending.pop(part_lower)
            if slot_token.lower() in virtual_to_part:
                continue
            sequence.append([pick_name, slot_token])

    return sequence


def plan_sequence(
    state: Dict[str, Any],
    output_path: Optional[str] = None,
    keep_pddl: bool = False,
) -> Optional[List[List]]:
    """
    Full pipeline: state → PDDL files → planner → sequence.

    Uses Fast Downward if FAST_DOWNWARD_PATH is set in config.py
    (PDDL 2.1 action costs — priority is a hard constraint in the domain).
    Requires FAST_DOWNWARD_PATH to be set in Vision_Module/config.py or DOWNWARD_PATH env variable.
    """
    try:
        from Vision_Module.config import FAST_DOWNWARD_PATH  # type: ignore
        fd_path = (FAST_DOWNWARD_PATH or "").strip()
    except ImportError:
        fd_path = ""

    if not fd_path:
        print("❌ FAST_DOWNWARD_PATH is not configured. Set it in Vision_Module/config.py or via the DOWNWARD_PATH env variable.")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        domain_path  = os.path.join(tmpdir, "domain.pddl")
        problem_path = os.path.join(tmpdir, "problem.pddl")
        plan_path    = os.path.join(tmpdir, "plan.txt")

        # ── Fast Downward: PDDL 2.1 action costs ─────────────────────────────
            try:
                problem, num_priorities, num_kit_priorities, kit_scoped = state_to_pddl_problem_costs(state)
            except ValueError as e:
                print(f"❌ PDDL problem generation failed: {e}")
                return None

            domain = build_domain_pddl_costs(num_priorities, num_kit_priorities,
                                             kit_scoped_priority=kit_scoped)

            with open(domain_path,  "w") as f: f.write(domain)
            with open(problem_path, "w") as f: f.write(problem)

            if keep_pddl:
                print("\n── Generated PDDL domain (action costs) ──")
                print(domain)
                print("\n── Generated PDDL problem ──")
                print(problem)
                print("────────────────────────────\n")

            print(f"  backend        : Fast Downward  "
                  f"(PDDL 2.1 action costs, {num_priorities} color priority level(s)"
                  f", {num_kit_priorities} receptacle priority level(s)"
                  f"{', kit-scoped' if kit_scoped else ''})")
            plan_text = run_fast_downward(domain_path, problem_path, plan_path, fd_path)

        if plan_text is None:
            return None

        sequence = parse_plan_to_sequence(plan_text, state)


        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(sequence, f, indent=2)

        return sequence