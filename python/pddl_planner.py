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
#   pyperplan (fallback, pip install pyperplan)
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

# ── pyperplan availability check ──────────────────────────────────────────────
try:
    import pyperplan  # noqa: F401
    _PYPERPLAN_AVAILABLE = True
except ImportError:
    _PYPERPLAN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PDDL domain builders
# ─────────────────────────────────────────────────────────────────────────────

# ── Basic domain (pyperplan / no action costs) ────────────────────────────────

DOMAIN_PDDL_BASIC = """\
(define (domain pick-and-place)
  (:requirements :typing :negative-preconditions)

  (:types
    part slot receptacle - object
    kit container - receptacle
  )

  (:predicates
    (at ?p - part ?s - slot)
    (slot-empty ?s - slot)
    (slot-of ?s - slot ?r - receptacle)
    (role-input ?r - receptacle)
    (role-output ?r - receptacle)
    (compatible ?p - part ?r - receptacle)
    (color-red ?p - part)
    (color-blue ?p - part)
    (color-green ?p - part)
    (hand-empty)
    (holding ?p - part)
  )

  (:action pick
    :parameters (?p - part ?s - slot ?r - receptacle)
    :precondition (and
      (at ?p ?s)
      (slot-of ?s ?r)
      (role-input ?r)
      (hand-empty)
    )
    :effect (and
      (not (at ?p ?s))
      (slot-empty ?s)
      (holding ?p)
      (not (hand-empty))
    )
  )

  (:action place
    :parameters (?p - part ?s - slot ?r - receptacle)
    :precondition (and
      (holding ?p)
      (slot-empty ?s)
      (slot-of ?s ?r)
      (role-output ?r)
      (compatible ?p ?r)
    )
    :effect (and
      (at ?p ?s)
      (not (slot-empty ?s))
      (not (holding ?p))
      (hand-empty)
    )
  )
)
"""


# ── Action-costs domain (Fast Downward) ───────────────────────────────────────

def build_domain_pddl_costs(num_priorities: int, num_kit_priorities: int = 0,
                            kit_scoped_priority: bool = False) -> str:
    """
    Generate a PDDL domain with action costs + derived predicates that enforces
    priority ordering as a hard constraint.

    For each priority level K, a derived predicate (priority-K-available) is
    true while any priority-K part still sits in an input receptacle.
    pick-priority-(K+1) requires (not (priority-K-available)), so the planner
    cannot legally pick a lower-priority part while a higher-priority one
    remains.

    Kit/receptacle priority (num_kit_priorities > 0):
    For each receptacle priority level K, a derived predicate
    (rec_priority_K_needs_filling) is true while any goal slot in a
    receptacle at that level is still empty.  place-rec-priority-(K+1)
    requires (not (rec_priority_K_needs_filling)), so the planner must
    fill higher-priority receptacles before lower-priority ones.
    Works for both kits AND containers.

    Kit-scoped priority (kit_scoped_priority=True):
    When BOTH pick priorities and receptacle priorities are active (e.g.
    kitting with "pick green first" + multiple kits), the pick-priority
    derived predicates are scoped to the currently-fillable kit using a
    (for-kit ?p - part ?r - receptacle) predicate.  This means
    "priority-1-available" only fires for priority-1 parts whose target
    kit still has empty goal slots AND is not blocked by a higher
    receptacle priority.  This avoids deadlock (where globally-required
    priority-1 parts destined for later kits block priority-2 picks for
    the current kit) and achieves the desired "within each kit, pick
    green first" behavior.

    Requirements: :typing :negative-preconditions :existential-preconditions
                  :derived-predicates :action-costs
    All supported by Fast Downward's translator (no :numeric-fluents needed).
    """
    NL = "\n"

    pred_lines = [
        "    (at ?p - part ?s - slot)",
        "    (slot-empty ?s - slot)",
        "    (slot-of ?s - slot ?r - receptacle)",
        "    (role-input ?r - receptacle)",
        "    (role-output ?r - receptacle)",
        "    (compatible ?p - part ?r - receptacle)",
        "    (color-red ?p - part)",
        "    (color-blue ?p - part)",
        "    (color-green ?p - part)",
        "    (hand-empty)",
        "    (holding ?p - part)",
    ]
    for k in range(1, num_priorities + 1):
        pred_lines.append(f"    (priority-{k} ?p - part)")
    pred_lines.append("    (no-priority ?p - part)")
    # priority_K_available derived predicates enforce hard pick ordering.
    # Deadlock from excess stock is prevented upstream: only goal-bound parts
    # receive priority-K tags (see goal_parts filter below), so once all
    # needed priority-K parts are picked, priority_K_available drops to False
    # and priority-(K+1) picks are unblocked immediately.
    for k in range(1, num_priorities + 1):
        pred_lines.append(f"    (priority_{k}_available)")

    # ── Kit-scoped priority predicate ──
    if kit_scoped_priority:
        pred_lines.append("    (for-kit ?p - part ?r - receptacle)")
        pred_lines.append("    (kit-fillable ?r - receptacle)")

    # ── Receptacle priority predicates ──
    if num_kit_priorities > 0:
        pred_lines.append("    (is-goal-slot ?s - slot)")
        for k in range(1, num_kit_priorities + 1):
            pred_lines.append(f"    (rec-priority-{k} ?r - receptacle)")
        for k in range(1, num_kit_priorities + 1):
            pred_lines.append(f"    (rec_priority_{k}_needs_filling)")

    # ── Derived predicates ──
    derived_blocks = []

    # ── Receptacle priority derived predicates (generated FIRST because
    #    kit-scoped pick priority depends on them) ──
    for k in range(1, num_kit_priorities + 1):
        block = (
            f"  (:derived (rec_priority_{k}_needs_filling)" + NL +
            f"    (exists (?s - slot ?r - receptacle)" + NL +
            f"      (and (rec-priority-{k} ?r) (slot-of ?s ?r)" + NL +
            f"           (is-goal-slot ?s) (slot-empty ?s))))" + NL
        )
        derived_blocks.append(block)

    # ── Kit-fillable derived predicates (one per receptacle priority level) ──
    # A kit at rec-priority-K is fillable when it still has empty goal slots
    # AND all higher-priority (lower-K) receptacles are already filled.
    if kit_scoped_priority:
        for k in range(1, num_kit_priorities + 1):
            higher_filled = "".join(
                NL + f"           (not (rec_priority_{j}_needs_filling))"
                for j in range(1, k)
            )
            block = (
                f"  (:derived (kit-fillable ?r - receptacle)" + NL +
                f"    (and (rec-priority-{k} ?r)" + NL +
                f"         (exists (?s - slot)" + NL +
                f"           (and (slot-of ?s ?r) (is-goal-slot ?s) (slot-empty ?s){higher_filled}))))" + NL
            )
            derived_blocks.append(block)

    # ── Pick-priority derived predicates ──
    if kit_scoped_priority and num_kit_priorities > 0:
        # Kit-scoped: priority_K_available only fires for parts whose
        # target kit is currently fillable.  This prevents parts destined
        # for later kits from blocking picks for the current kit.
        for k in range(1, num_priorities + 1):
            block = (
                f"  (:derived (priority_{k}_available)" + NL +
                f"    (exists (?p - part ?s - slot ?r - receptacle ?rk - receptacle)" + NL +
                f"      (and (priority-{k} ?p) (at ?p ?s)" + NL +
                f"           (slot-of ?s ?r) (role-input ?r)" + NL +
                f"           (for-kit ?p ?rk) (kit-fillable ?rk))))" + NL
            )
            derived_blocks.append(block)
    else:
        # Global (original): priority_K_available fires for any priority-K
        # part still in an input receptacle.
        for k in range(1, num_priorities + 1):
            block = (
                f"  (:derived (priority_{k}_available)" + NL +
                f"    (exists (?p - part ?s - slot ?r - receptacle)" + NL +
                f"      (and (priority-{k} ?p) (at ?p ?s)" + NL +
                f"           (slot-of ?s ?r) (role-input ?r))))" + NL
            )
            derived_blocks.append(block)

    # Pick actions — one per priority level with hard blocking preconditions.
    # pick-priority-K requires (not priority_J_available) for all J < K, so the
    # planner cannot legally pick a lower-priority part while a higher-priority
    # one remains in an input receptacle.
    # Safe from deadlock because only goal-bound parts carry priority tags.
    pick_actions = []
    for k in range(1, num_priorities + 1):
        blocking = "".join(
            NL + f"      (not (priority_{j}_available))"
            for j in range(1, k)
        )
        action = (
            f"  (:action pick-priority-{k}" + NL +
            f"    :parameters (?p - part ?s - slot ?r - receptacle)" + NL +
            f"    :precondition (and" + NL +
            f"      (at ?p ?s)" + NL +
            f"      (slot-of ?s ?r)" + NL +
            f"      (role-input ?r)" + NL +
            f"      (hand-empty)" + NL +
            f"      (priority-{k} ?p){blocking}" + NL +
            f"    )" + NL +
            f"    :effect (and" + NL +
            f"      (not (at ?p ?s))" + NL +
            f"      (slot-empty ?s)" + NL +
            f"      (holding ?p)" + NL +
            f"      (not (hand-empty))" + NL +
            f"      (increase (total-cost) {k})" + NL +
            f"    )" + NL +
            f"  )"
        )
        pick_actions.append(action)

    # Unprioritised pick — only fires once all priority levels are exhausted
    all_blocking = "".join(
        NL + f"      (not (priority_{k}_available))"
        for k in range(1, num_priorities + 1)
    )
    pick_actions.append(
        "  (:action pick-no-priority" + NL +
        "    :parameters (?p - part ?s - slot ?r - receptacle)" + NL +
        "    :precondition (and" + NL +
        "      (at ?p ?s)" + NL +
        "      (slot-of ?s ?r)" + NL +
        "      (role-input ?r)" + NL +
        "      (hand-empty)" + NL +
        "      (no-priority ?p)" + all_blocking + NL +
        "    )" + NL +
        "    :effect (and" + NL +
        "      (not (at ?p ?s))" + NL +
        "      (slot-empty ?s)" + NL +
        "      (holding ?p)" + NL +
        "      (not (hand-empty))" + NL +
        f"      (increase (total-cost) {num_priorities + 1})" + NL +
        "    )" + NL +
        "  )"
    )

    # ── Place actions ──
    place_actions = []
    if num_kit_priorities > 0:
        # Split place into per-receptacle-priority actions (hard constraint)
        for k in range(1, num_kit_priorities + 1):
            rec_blocking = "".join(
                NL + f"      (not (rec_priority_{j}_needs_filling))"
                for j in range(1, k)
            )
            place_actions.append(
                f"  (:action place-rec-priority-{k}" + NL +
                f"    :parameters (?p - part ?s - slot ?r - receptacle)" + NL +
                f"    :precondition (and" + NL +
                f"      (holding ?p)" + NL +
                f"      (slot-empty ?s)" + NL +
                f"      (slot-of ?s ?r)" + NL +
                f"      (role-output ?r)" + NL +
                f"      (compatible ?p ?r)" + NL +
                f"      (rec-priority-{k} ?r){rec_blocking}" + NL +
                f"    )" + NL +
                f"    :effect (and" + NL +
                f"      (at ?p ?s)" + NL +
                f"      (not (slot-empty ?s))" + NL +
                f"      (not (holding ?p))" + NL +
                f"      (hand-empty)" + NL +
                f"      (increase (total-cost) 0)" + NL +
                f"    )" + NL +
                f"  )"
            )

        # Place for receptacles with no priority — fires only when all
        # prioritised receptacles are filled
        all_rec_blocking = "".join(
            NL + f"      (not (rec_priority_{k}_needs_filling))"
            for k in range(1, num_kit_priorities + 1)
        )
        place_actions.append(
            "  (:action place-no-rec-priority" + NL +
            "    :parameters (?p - part ?s - slot ?r - receptacle)" + NL +
            "    :precondition (and" + NL +
            "      (holding ?p)" + NL +
            "      (slot-empty ?s)" + NL +
            "      (slot-of ?s ?r)" + NL +
            "      (role-output ?r)" + NL +
            "      (compatible ?p ?r)" + all_rec_blocking + NL +
            "    )" + NL +
            "    :effect (and" + NL +
            "      (at ?p ?s)" + NL +
            "      (not (slot-empty ?s))" + NL +
            "      (not (holding ?p))" + NL +
            "      (hand-empty)" + NL +
            "      (increase (total-cost) 0)" + NL +
            "    )" + NL +
            "  )"
        )
    else:
        # No kit priority — single place action (original behaviour)
        place_actions.append(
            "  (:action place" + NL +
            "    :parameters (?p - part ?s - slot ?r - receptacle)" + NL +
            "    :precondition (and" + NL +
            "      (holding ?p)" + NL +
            "      (slot-empty ?s)" + NL +
            "      (slot-of ?s ?r)" + NL +
            "      (role-output ?r)" + NL +
            "      (compatible ?p ?r)" + NL +
            "    )" + NL +
            "    :effect (and" + NL +
            "      (at ?p ?s)" + NL +
            "      (not (slot-empty ?s))" + NL +
            "      (not (holding ?p))" + NL +
            "      (hand-empty)" + NL +
            "      (increase (total-cost) 0)" + NL +
            "    )" + NL +
            "  )"
        )

    return (
        "(define (domain pick-and-place)" + NL +
        "  (:requirements :typing :negative-preconditions" + NL +
        "                 :existential-preconditions :derived-predicates" + NL +
        "                 :action-costs)" + NL + NL +
        "  (:types" + NL +
        "    part slot receptacle - object" + NL +
        "    kit container - receptacle" + NL +
        "  )" + NL + NL +
        "  (:predicates" + NL +
        NL.join(pred_lines) + NL +
        "  )" + NL + NL +
        "  (:functions" + NL +
        "    (total-cost)" + NL +
        "  )" + NL + NL +
        "".join(derived_blocks) + NL +
        (NL + NL).join(pick_actions) + NL + NL +
        (NL + NL).join(place_actions) + NL +
        ")" + NL
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_pddl_name(name: str) -> str:
    return name.replace(" ", "_")


def _role_map(preds: Dict[str, Any]) -> Tuple[set, set]:
    inputs, outputs = set(), set()
    for entry in preds.get("role", []):
        role = (entry.get("role") or "").lower()
        if role == "input":
            inputs.add(entry["object"])
        elif role == "output":
            outputs.add(entry["object"])
    return inputs, outputs


def _color_map(preds: Dict[str, Any]) -> Dict[str, str]:
    return {e["part"]: (e.get("color") or "").lower() for e in preds.get("color", [])}


def _compute_part_score(part: str, color: str, preds: Dict[str, Any]) -> int:
    """
    Additive priority score for a part.
    Each applicable rule in the priority list contributes its 'order' value as a score.
    Higher total score = higher priority = picked first.

    Rules that contribute:
      - {"color": "green", "order": 2}  → all green parts get +2
      - {"part_name": "Part_3", "order": 1} → Part_3 gets +1

    Example: Part_3 is green.
      green score = 2, part_name score = 1 → total = 3  (picked before other greens or Part_3 alone)
    """
    score = 0
    for e in preds.get("priority", []):
        if "color" in e and (e["color"] or "").lower() == color:
            score += int(e.get("order", 0))
        elif "part_name" in e and e["part_name"] == part:
            score += int(e.get("order", 0))
    return score


def _priority_key(preds: Dict[str, Any]):
    """Sort key for colors: higher score = sorted first (negated for ascending sort)."""
    score = {}
    for e in preds.get("priority", []):
        if "color" in e:
            score[(e["color"] or "").lower()] = int(e.get("order", 0))
    return lambda color: -score.get((color or "").lower(), 0)


def _receptacle_priority_key(preds: Dict[str, Any]):
    """Return sort key for receptacles (kits/containers) based on priority list."""
    order = {}
    for e in preds.get("priority", []):
        if "receptacle" in e:
            order[e["receptacle"]] = e.get("order", 999)
    return lambda rec: (order.get(rec, 999), rec)  # secondary sort by name for stability


# ─────────────────────────────────────────────────────────────────────────────
# Problem generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_standalone_structures(state: Dict[str, Any]) -> Tuple[List[str], List[str], str]:
    preds         = state.get("predicates", {})
    objs          = state.get("objects",    {})
    parts_in_slot = {entry["part"] for entry in preds.get("at", [])}
    standalone    = [p for p in objs.get("parts", []) if p not in parts_in_slot]
    virtual_slots = [f"ws_slot_{_to_pddl_name(p)}" for p in standalone]
    return standalone, virtual_slots, "workspace_floor"


def _build_common_init(state, standalone, virtual_slots, VIRTUAL):
    preds        = state.get("predicates",    {})
    slot_belongs = state.get("slot_belongs_to", {})

    init = ["    (hand-empty)"]

    for entry in preds.get("at", []):
        init.append(f"    (at {_to_pddl_name(entry['part'])} {_to_pddl_name(entry['slot'])})")
    for s in preds.get("slot_empty", []):
        init.append(f"    (slot-empty {_to_pddl_name(s)})")
    for slot_name, receptacle in slot_belongs.items():
        init.append(f"    (slot-of {_to_pddl_name(slot_name)} {_to_pddl_name(receptacle)})")
    for p_orig, vs in zip(standalone, virtual_slots):
        p_pddl = _to_pddl_name(p_orig)
        init.append(f"    (at {p_pddl} {vs})")
        init.append(f"    (slot-of {vs} {VIRTUAL})")
    for entry in preds.get("role", []):
        obj  = _to_pddl_name(entry["object"])
        role = (entry.get("role") or "").lower()
        if role == "input":
            init.append(f"    (role-input {obj})")
        elif role == "output":
            init.append(f"    (role-output {obj})")
    if virtual_slots:
        init.append(f"    (role-input {VIRTUAL})")
    for entry in preds.get("color", []):
        part  = _to_pddl_name(entry["part"])
        color = (entry.get("color") or "").lower()
        if color == "red":
            init.append(f"    (color-red {part})")
        elif color == "blue":
            init.append(f"    (color-blue {part})")
        elif color == "green":
            init.append(f"    (color-green {part})")
    return init


def _build_compat_init(state, kits, containers):
    """
    Build PDDL compatibility predicates from flexible rules.
    
    Rule format supports multiple selectors (AND logic for part selectors):
      Part selectors:
        - part_color: "blue" | "red" | "green"
        - part_fragility: "fragile" | "normal"  
        - part_name: "Part_1" (specific part)
      
      Receptacle selectors (mutually exclusive):
        - allowed_in: ["Container_1", "Kit_2"] — specific receptacles
        - allowed_in_role: "output" | "input" — all receptacles with that role
        - not_allowed_in: ["Kit_1"] — exclusion (applied AFTER all inclusions)
    
    IMPORTANT: Exclusion rules (not_allowed_in without allowed_in/allowed_in_role)
    are processed LAST so they override any inclusions from other rules.
    This ensures "fragile parts not allowed in Kit_1" overrides "green parts allowed in Kit_1"
    even if a part is both green AND fragile.
    """
    preds     = state.get("predicates", {})
    workspace = state.get("workspace",  {})
    objs      = state.get("objects", {})
    mode      = (workspace.get("operation_mode") or "sorting").lower()

    all_parts_raw = objs.get("parts", [])
    all_receptacles = kits + containers

    # Build part lookup maps
    color_to_parts: Dict[str, set] = {}
    for entry in preds.get("color", []):
        c = (entry.get("color") or "").lower()
        color_to_parts.setdefault(c, set()).add(_to_pddl_name(entry["part"]))

    fragility_to_parts: Dict[str, set] = {"normal": set(), "fragile": set()}
    for entry in preds.get("fragility", []):
        f = (entry.get("fragility") or "normal").lower()
        fragility_to_parts.setdefault(f, set()).add(_to_pddl_name(entry["part"]))
    # Parts without explicit fragility are "normal"
    all_parts_pddl = {_to_pddl_name(p) for p in all_parts_raw}
    explicitly_set = fragility_to_parts["normal"] | fragility_to_parts["fragile"]
    fragility_to_parts["normal"] |= (all_parts_pddl - explicitly_set)

    # Build receptacle role map
    inputs, outputs = _role_map(preds)
    role_to_receptacles: Dict[str, set] = {
        "input": {_to_pddl_name(r) for r in inputs},
        "output": {_to_pddl_name(r) for r in outputs},
    }

    compat_entries = preds.get("part_compatibility", [])
    use_compat = bool(compat_entries)

    init: List[str] = []
    
    if use_compat:
        # Separate inclusion rules from pure exclusion rules
        inclusion_rules = []
        exclusion_rules = []
        
        for rule in compat_entries:
            has_inclusion = "allowed_in" in rule or "allowed_in_role" in rule
            has_exclusion = "not_allowed_in" in rule
            
            if has_exclusion and not has_inclusion:
                # Pure exclusion rule — process last
                exclusion_rules.append(rule)
            else:
                # Inclusion rule (may also have not_allowed_in as a filter)
                inclusion_rules.append(rule)
        
        # Track all (part, receptacle) pairs that are compatible
        compatible_pairs: set = set()
        
        # FIRST PASS: Process inclusion rules
        for rule in inclusion_rules:
            # ── Find matching parts (AND logic) ──
            matching_parts = all_parts_pddl.copy()
            
            if "part_color" in rule:
                color = (rule["part_color"] or "").lower()
                matching_parts &= color_to_parts.get(color, set())
            
            if "part_fragility" in rule:
                frag = (rule["part_fragility"] or "").lower()
                matching_parts &= fragility_to_parts.get(frag, set())
            
            if "part_name" in rule:
                part_name = _to_pddl_name(rule["part_name"])
                matching_parts &= {part_name}
            
            if not matching_parts:
                continue
            
            # ── Find matching receptacles ──
            matching_receptacles: set = set()
            
            if "allowed_in" in rule:
                # Specific receptacles
                matching_receptacles = {_to_pddl_name(r) for r in rule["allowed_in"]}
            elif "allowed_in_role" in rule:
                # All receptacles with specified role
                role = (rule["allowed_in_role"] or "").lower()
                matching_receptacles = role_to_receptacles.get(role, set()).copy()
            else:
                # No positive selector — start with all receptacles
                matching_receptacles = set(all_receptacles)
            
            # Apply exclusions (as a filter for this rule)
            if "not_allowed_in" in rule:
                excluded = {_to_pddl_name(r) for r in rule["not_allowed_in"]}
                matching_receptacles -= excluded
            
            # Add compatible pairs
            for part in matching_parts:
                for rec in matching_receptacles:
                    compatible_pairs.add((part, rec))
        
        # If no inclusion rules produced any pairs, default to all-compatible
        # (same as the no-rules case) so that exclusion-only rules work correctly.
        if not compatible_pairs and exclusion_rules:
            for part in all_parts_pddl:
                for rec in all_receptacles:
                    compatible_pairs.add((part, rec))

        # SECOND PASS: Process pure exclusion rules (override inclusions)
        for rule in exclusion_rules:
            # ── Find matching parts (AND logic) ──
            matching_parts = all_parts_pddl.copy()
            
            if "part_color" in rule:
                color = (rule["part_color"] or "").lower()
                matching_parts &= color_to_parts.get(color, set())
            
            if "part_fragility" in rule:
                frag = (rule["part_fragility"] or "").lower()
                matching_parts &= fragility_to_parts.get(frag, set())
            
            if "part_name" in rule:
                part_name = _to_pddl_name(rule["part_name"])
                matching_parts &= {part_name}
            
            if not matching_parts:
                continue
            
            # Get excluded receptacles
            excluded = {_to_pddl_name(r) for r in rule.get("not_allowed_in", [])}
            
            # REMOVE these pairs from compatible_pairs
            for part in matching_parts:
                for rec in excluded:
                    compatible_pairs.discard((part, rec))
        
        # Generate PDDL predicates
        for part, rec in sorted(compatible_pairs):
            init.append(f"    (compatible {part} {rec})")
    else:
        # No rules defined — all parts compatible with all output receptacles
        parts = [_to_pddl_name(p) for p in all_parts_raw]
        for part in parts:
            for rec in all_receptacles:
                init.append(f"    (compatible {part} {rec})")
    
    return init


# ── Basic problem (pyperplan) ─────────────────────────────────────────────────

def state_to_pddl_problem(state: Dict[str, Any]) -> str:
    objs = state.get("objects", {})

    kits       = [_to_pddl_name(k) for k in objs.get("kits",       [])]
    containers = [_to_pddl_name(c) for c in objs.get("containers", [])]
    parts      = [_to_pddl_name(p) for p in objs.get("parts",      [])]
    slots      = [_to_pddl_name(s) for s in objs.get("slots",      [])]

    standalone, virtual_slots, VIRTUAL = _build_standalone_structures(state)
    all_containers = containers + ([VIRTUAL] if virtual_slots else [])
    all_slots      = slots + virtual_slots

    obj_lines: List[str] = []
    if kits:           obj_lines.append("    " + " ".join(kits)            + " - kit")
    if all_containers: obj_lines.append("    " + " ".join(all_containers)  + " - container")
    if parts:          obj_lines.append("    " + " ".join(parts)           + " - part")
    if all_slots:      obj_lines.append("    " + " ".join(all_slots)       + " - slot")

    init  = _build_common_init(state, standalone, virtual_slots, VIRTUAL)
    init += _build_compat_init(state, kits, containers)

    mode = (state.get("workspace", {}).get("operation_mode") or "sorting").lower()
    goal_lines = _generate_goal(state, mode)
    if not goal_lines:
        raise ValueError("Could not generate a PDDL goal. Check roles and operation_mode.")

    return (
        "(define (problem workspace)\n"
        "  (:domain pick-and-place)\n"
        "  (:objects\n"    + "\n".join(obj_lines) + "\n  )\n"
        "  (:init\n"       + "\n".join(init)      + "\n  )\n"
        "  (:goal (and\n"  + "\n".join(f"    {g}" for g in goal_lines) + "\n  ))\n"
        ")\n"
    )


# ── Action-costs problem (Fast Downward) ──────────────────────────────────────

def state_to_pddl_problem_costs(state: Dict[str, Any]) -> Tuple[str, int, int, bool]:
    """
    Generate a PDDL 2.1 problem with numeric fluents and a minimize metric.
    Returns (problem_str, num_priorities, num_kit_priorities, kit_scoped_priority).
    """
    objs  = state.get("objects",    {})
    preds = state.get("predicates", {})
    inputs, outputs = _role_map(preds)

    kits       = [_to_pddl_name(k) for k in objs.get("kits",       [])]
    containers = [_to_pddl_name(c) for c in objs.get("containers", [])]
    parts      = [_to_pddl_name(p) for p in objs.get("parts",      [])]
    slots      = [_to_pddl_name(s) for s in objs.get("slots",      [])]

    standalone, virtual_slots, VIRTUAL = _build_standalone_structures(state)
    all_containers = containers + ([VIRTUAL] if virtual_slots else [])
    all_slots      = slots + virtual_slots

    obj_lines: List[str] = []
    if kits:           obj_lines.append("    " + " ".join(kits)            + " - kit")
    if all_containers: obj_lines.append("    " + " ".join(all_containers)  + " - container")
    if parts:          obj_lines.append("    " + " ".join(parts)           + " - part")
    if all_slots:      obj_lines.append("    " + " ".join(all_slots)       + " - slot")

    init  = _build_common_init(state, standalone, virtual_slots, VIRTUAL)
    init += _build_compat_init(state, kits, containers)

    # ── priority assignments (additive score system) ─────────────────────────
    # Each priority rule contributes its 'order' value as a SCORE (higher = more
    # important, picked first).  A part that matches multiple rules gets the SUM
    # of all applicable scores and is therefore picked before parts that only
    # match one rule.
    #
    # Example ("always pick green first AND prioritise Part_3 and Part_7"):
    #   {"color": "green", "order": 2}       → all green parts get +2
    #   {"part_name": "Part_3", "order": 1}  → Part_3 gets +1 extra
    #   {"part_name": "Part_7", "order": 1}  → Part_7 gets +1 extra
    #
    #   Part_3 (green + named): 2 + 1 = 3  → priority-1 (picked first)
    #   Part_7 (green + named): 2 + 1 = 3  → priority-1 (same level as Part_3)
    #   Other greens:           2           → priority-2
    #   Everything else:        0           → no-priority
    #
    # Output-receptacle entries still use 'order' as RANK (1 = fill first).
    # They are on a separate axis (PLACE ordering) and do not add to pick scores.

    priority_entries = preds.get("priority", [])
    slot_belongs     = state.get("slot_belongs_to", {})
    color_map_raw    = {e["part"]: (e.get("color") or "").lower()
                        for e in preds.get("color", [])}

    # Build part → container lookup for input-receptacle expansion
    part_to_receptacle: Dict[str, str] = {}
    for entry in preds.get("at", []):
        parent = slot_belongs.get(entry["slot"], "")
        part_to_receptacle[entry["part"]] = parent

    # Compute additive score for every part (color + part_name contributions)
    part_scores: Dict[str, int] = {}
    for p_orig in objs.get("parts", []):
        color = color_map_raw.get(p_orig, "")
        part_scores[p_orig] = _compute_part_score(p_orig, color, preds)

    # Input-receptacle entries add their score to all parts currently in that container
    for entry in priority_entries:
        if "receptacle" not in entry:
            continue
        rec_name  = entry["receptacle"]
        if rec_name not in inputs:
            continue
        rec_score = int(entry.get("order", 0))
        for p_orig in objs.get("parts", []):
            if part_to_receptacle.get(p_orig) == rec_name:
                part_scores[p_orig] = part_scores.get(p_orig, 0) + rec_score

    has_pick_order = any(s > 0 for s in part_scores.values())

    # Map distinct non-zero total scores → contiguous PDDL priority levels.
    # Highest score → level 1 (lowest cost to pick), etc.
    distinct_scores = sorted({s for s in part_scores.values() if s > 0}, reverse=True)
    score_to_level: Dict[int, int] = {s: i + 1 for i, s in enumerate(distinct_scores)}
    num_priorities  = len(distinct_scores)

    # ── goal (generated BEFORE priority tags) ────────────────────────────────
    # Priority tags are only applied to goal-bound parts.  Excess stock of a
    # priority colour that the kit recipe doesn't need stays at no-priority and
    # can never block the planner's pick ordering.
    mode = (state.get("workspace", {}).get("operation_mode") or "sorting").lower()
    goal_lines = _generate_goal(state, mode)
    if not goal_lines:
        raise ValueError("Could not generate a PDDL goal. Check roles and operation_mode.")

    import re as _re
    goal_parts: set = set()
    # Build part → target kit mapping from goal lines (for kit-scoped priority)
    goal_part_to_kit: Dict[str, str] = {}
    for g in goal_lines:
        m = _re.search(r'\(at\s+(\S+)\s+(\S+)\)', g)
        if m:
            p_name = m.group(1)
            s_name = m.group(2)
            goal_parts.add(p_name)
            parent = slot_belongs.get(s_name, "")
            if parent:
                goal_part_to_kit[p_name] = _to_pddl_name(parent)

    # Initialise total-cost to zero (required by :action-costs)
    init.append("    (= (total-cost) 0)")

    # ── receptacle (output) priority assignments ──────────────────────────
    # Only OUTPUT receptacle priorities control fill order.
    rec_priority_entries = [e for e in priority_entries if "receptacle" in e]
    rec_to_level: Dict[str, int] = {}
    for e in rec_priority_entries:
        rec_name = _to_pddl_name(e["receptacle"])
        if e["receptacle"] in outputs:
            rec_to_level[rec_name] = int(e["order"])

    workspace_cfg = state.get("workspace", {})
    fill_order = (workspace_cfg.get("fill_order") or "").lower()

    # Compute the maximum pick-priority score of parts compatible with each
    # output receptacle.  Used below to align rec-priority ordering with
    # part pick ordering so the planner never deadlocks (e.g. "sort red
    # first" → red → Container_2, so Container_2 must be rec-priority-1).
    _output_pddl = {_to_pddl_name(r) for r in outputs}
    _color_lower  = {p: (color_map_raw.get(p, "")).lower()
                     for p in objs.get("parts", [])}
    _rec_max_score: Dict[str, int] = {r: 0 for r in _output_pddl}
    for _rule in preds.get("part_compatibility", []):
        _rule_color = (_rule.get("part_color") or "").lower()
        _rule_score = max(
            (part_scores.get(p_orig, 0)
             for p_orig in objs.get("parts", [])
             if _color_lower.get(p_orig) == _rule_color),
            default=0,
        )
        for _rec_orig in _rule.get("allowed_in", []):
            _rec_pddl = _to_pddl_name(_rec_orig)
            if _rec_pddl in _rec_max_score:
                _rec_max_score[_rec_pddl] = max(_rec_max_score[_rec_pddl], _rule_score)

    # Sort output receptacles: when pick-order exists, the receptacle that
    # receives the highest-priority parts should be filled first (rec-priority-1).
    # Without pick-order, fall back to alphabetical (stable, deterministic).
    output_receptacles = sorted(
        list(_output_pddl),
        key=lambda r: (-_rec_max_score.get(r, 0), r) if has_pick_order else r,
    )

    # Auto-assign receptacle priorities when there are multiple output
    # receptacles and no explicit parallel fill order.
    #
    # KEY INVARIANT: two receptacles that share the same max part pick-score
    # must receive the SAME rec-priority level.  Giving them different levels
    # causes deadlock: after picking a priority-1 part destined for the
    # lower-ranked container, the planner cannot place it (the higher-ranked
    # container is not full yet) and cannot put it back — no solution.
    #
    # When pick-order is active we group receptacles by their max part score
    # and assign contiguous levels to the groups (highest score -> level 1).
    # When there is no pick-order we fall back to sequential alphabetical
    # assignment (legacy behaviour, no deadlock risk there).
    if (not rec_to_level
            and len(output_receptacles) >= 2 and fill_order != "parallel"):
        if has_pick_order:
            distinct_rec_scores = sorted(
                {_rec_max_score.get(r, 0) for r in output_receptacles},
                reverse=True,
            )
            _score_to_rec_level = {s: i + 1 for i, s in enumerate(distinct_rec_scores)}
            for rec_name in output_receptacles:
                rec_to_level[rec_name] = _score_to_rec_level[_rec_max_score.get(rec_name, 0)]
        else:
            for idx, rec_name in enumerate(output_receptacles, start=1):
                rec_to_level[rec_name] = idx

    num_kit_priorities = max(rec_to_level.values(), default=0)

    # ── Determine whether to use kit-scoped priority ─────────────────────
    # When kitting with color priorities AND sequential kit filling, scope
    # the pick-priority derived predicates to the currently-fillable kit.
    # This prevents parts destined for later kits from blocking picks for
    # the current kit.
    # Explicit fill_order="sequential" without pick priorities suppresses
    # priorities entirely (legacy behavior).
    _fill_order_early = (workspace_cfg.get("fill_order") or "").lower()
    kit_scoped = (
        has_pick_order
        and num_kit_priorities >= 2
        and mode == "kitting"
        and fill_order != "parallel"
    )

    if _fill_order_early == "sequential" and not has_pick_order:
        # Legacy: explicit sequential without pick priorities → suppress all
        num_priorities = 0
        score_to_level = {}
        has_pick_order = False
        part_scores    = {p: 0 for p in part_scores}

    # Assign PDDL priority-K tags from additive scores (goal-bound parts only)
    for p_orig in objs.get("parts", []):
        p_pddl = _to_pddl_name(p_orig)
        score  = part_scores.get(p_orig, 0)
        if score > 0 and p_pddl in goal_parts:
            init.append(f"    (priority-{score_to_level[score]} {p_pddl})")
        else:
            init.append(f"    (no-priority {p_pddl})")

    # ── for-kit predicates (kit-scoped priority only) ────────────────────
    if kit_scoped:
        for p_pddl, kit_pddl in goal_part_to_kit.items():
            init.append(f"    (for-kit {p_pddl} {kit_pddl})")

    if num_kit_priorities > 0:
        all_receptacles = kits + containers
        for rec_name in all_receptacles:
            level = rec_to_level.get(rec_name)
            if level is not None:
                init.append(f"    (rec-priority-{level} {rec_name})")

    # ── mark goal slots (needed for receptacle priority derived predicates) ──
    if num_kit_priorities > 0:
        slot_belongs = state.get("slot_belongs_to", {})
        all_output_receptacles = set(kits + containers) & {_to_pddl_name(r) for r in outputs}
        for g in goal_lines:
            m = _re.search(r'\(at\s+\S+\s+(\S+)\)', g)
            if m:
                slot_name = m.group(1)
                parent = slot_belongs.get(slot_name, "")
                if _to_pddl_name(parent) in all_output_receptacles:
                    init.append(f"    (is-goal-slot {_to_pddl_name(slot_name)})")

    problem = (
        "(define (problem workspace)\n"
        "  (:domain pick-and-place)\n"
        "  (:objects\n"    + "\n".join(obj_lines) + "\n  )\n"
        "  (:init\n"       + "\n".join(init)      + "\n  )\n"
        "  (:goal (and\n"  + "\n".join(f"    {g}" for g in goal_lines) + "\n  ))\n"
        "  (:metric minimize (total-cost))\n"
        ")\n"
    )
    return problem, num_priorities, num_kit_priorities, kit_scoped


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
    rkey            = _receptacle_priority_key(preds)

    parts_in_slot = {e["part"] for e in preds.get("at", [])}
    parts_in_input = [
        e["part"] for e in preds.get("at", [])
        if slot_belongs.get(e["slot"], "") in inputs
    ]
    standalone    = [p for p in objs.get("parts", []) if p not in parts_in_slot]
    parts_to_move = parts_in_input + standalone

    # Build part-name priority lookup
    _part_prio = {e["part_name"]: int(e.get("order", 0))
                  for e in preds.get("priority", []) if "part_name" in e}
    # Sort by additive score descending: higher score = assigned first
    # _priority_key already returns -score so ascending sort gives descending priority
    parts_to_move.sort(key=lambda p: (-_part_prio.get(p, 0), pkey(color_map.get(p, ""))))

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
    
    Receptacle priority (from priority list with "receptacle" key):
      - [{"receptacle": "Kit_1", "order": 1}, {"receptacle": "Kit_2", "order": 2}]
      - Lower order = higher priority (filled first)
      - This naturally achieves "finish Kit_1 before Kit_2" behavior
    """
    preds        = state.get("predicates",    {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects",       {})
    inputs, outputs = _role_map(preds)
    color_map    = _color_map(preds)
    pkey         = _priority_key(preds)
    rkey         = _receptacle_priority_key(preds)

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
    
    empty_slots: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in kit_set:
            empty_slots.setdefault(parent, []).append(s)

    used_parts: set = set()
    goals: List[str] = []

    raw_recipes = preds.get("kit_recipe", [])
    compat_rules = preds.get("part_compatibility", [])

    # ── Build exclusion map from pure exclusion rules ──
    # Maps part → set of receptacles it's excluded from
    part_exclusions: Dict[str, set] = {}
    
    # Build fragility lookup for matching
    frag_map = {e["part"]: e.get("fragility", "normal") for e in preds.get("fragility", [])}
    
    for rule in compat_rules:
        has_inclusion = "allowed_in" in rule or "allowed_in_role" in rule
        has_exclusion = "not_allowed_in" in rule
        
        if has_exclusion:
            # Find matching parts for this rule
            matching_parts: set = set(available_parts)
            
            if "part_name" in rule:
                matching_parts &= {rule["part_name"]}
            if "part_color" in rule:
                color = (rule["part_color"] or "").lower()
                matching_parts = {p for p in matching_parts if color_map.get(p, "").lower() == color}
            if "part_fragility" in rule:
                frag = (rule["part_fragility"] or "").lower()
                matching_parts = {p for p in matching_parts if frag_map.get(p, "normal").lower() == frag}
            
            excluded_recs = set(rule["not_allowed_in"])
            
            for part in matching_parts:
                part_exclusions.setdefault(part, set()).update(excluded_recs)

    # ── If we have kit_recipe, use color-based selection ──
    if raw_recipes:
        # Build part-name priority lookup for candidate sorting
        _part_priority_map: Dict[str, int] = {}
        for e in preds.get("priority", []):
            if "part_name" in e:
                _part_priority_map[e["part_name"]] = int(e.get("order", 0))

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

            # Filter candidates: not used, not excluded from this kit
            candidates = [
                p for p in available_by_color.get(color, [])
                if p not in used_parts and kit not in part_exclusions.get(p, set())
            ]
            # Sort: parts with explicit part-name priority first (by order),
            # then remaining parts (stable order)
            # Higher score = picked first; negate for ascending sort
            candidates.sort(key=lambda p: -_part_priority_map.get(p, 0))

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
        
        # Generate goals for each part, sorted by part-name priority then color priority
        _pp = {e["part_name"]: int(e.get("order", 0))
               for e in preds.get("priority", []) if "part_name" in e}
        sorted_parts = sorted(part_info.keys(), key=lambda p: (-_pp.get(p, 0), pkey(color_map.get(p, ""))))
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


def run_pyperplan(
    domain_path: str,
    problem_path: str,
    plan_path: str,
) -> Optional[str]:
    """Run pyperplan (A* + hFF). Fallback when Fast Downward is not configured."""
    if not _PYPERPLAN_AVAILABLE:
        print("\n❌ pyperplan is not installed.  pip install pyperplan\n")
        return None

    from pyperplan import grounding, search      # type: ignore
    from pyperplan.pddl.parser import Parser     # type: ignore

    try:
        parser  = Parser(domain_path, problem_path)
        domain  = parser.parse_domain()
        problem = parser.parse_problem(domain)
    except Exception as e:
        print(f"❌ pyperplan parse error: {e}"); return None

    try:
        task = grounding.ground(problem)
    except Exception as e:
        print(f"❌ pyperplan grounding error: {e}"); return None

    try:
        from pyperplan.heuristics.relaxation import hFFHeuristic  # type: ignore
        plan = search.astar_search(task, hFFHeuristic(task))
    except Exception as e:
        print(f"❌ pyperplan search error: {e}"); return None

    if plan is None:
        print("❌ pyperplan found no plan."); return None

    plan_text = "\n".join(f"({op.name})" for op in plan) + "\n"
    with open(plan_path, "w") as f:
        f.write(plan_text)
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

    And pyperplan concatenated format:
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
                sequence.append([pick_name, slot_orig, GRIPPER_CLOSE_STANDARD])
                continue

        # ── pyperplan concatenated format ─────────────────────────────────────
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
            sequence.append([pick_name, slot_token, GRIPPER_CLOSE_STANDARD])

    return sequence


# ── Priority sort (pyperplan fallback only) ───────────────────────────────────

def _sort_sequence_by_priority(sequence: List[List], state: Dict[str, Any]) -> List[List]:
    """Soft priority enforcement for pyperplan (no action costs)."""
    preds     = state.get("predicates", {})
    color_map = _color_map(preds)
    pkey      = _priority_key(preds)
    return sorted(sequence, key=lambda s: pkey(color_map.get(s[0], "")))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def plan_sequence(
    state: Dict[str, Any],
    output_path: Optional[str] = None,
    keep_pddl: bool = False,
) -> Optional[List[List]]:
    """
    Full pipeline: state → PDDL files → planner → sequence.

    Uses Fast Downward if FAST_DOWNWARD_PATH is set in config.py
    (PDDL 2.1 action costs — priority is a hard constraint in the domain).
    Falls back to pyperplan otherwise (priority is post-sorted).
    """
    fd_path: str = ""
    try:
        from Vision_Module.config import FAST_DOWNWARD_PATH  # type: ignore
        fd_path = (FAST_DOWNWARD_PATH or "").strip()
    except ImportError:
        pass

    use_fd = bool(fd_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        domain_path  = os.path.join(tmpdir, "domain.pddl")
        problem_path = os.path.join(tmpdir, "problem.pddl")
        plan_path    = os.path.join(tmpdir, "plan.txt")

        if use_fd:
            # ── Fast Downward: PDDL 2.1 action costs ─────────────────────────
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

        else:
            # ── pyperplan fallback ────────────────────────────────────────────
            try:
                problem = state_to_pddl_problem(state)
            except ValueError as e:
                print(f"❌ PDDL problem generation failed: {e}")
                return None

            with open(domain_path,  "w") as f: f.write(DOMAIN_PDDL_BASIC)
            with open(problem_path, "w") as f: f.write(problem)

            if keep_pddl:
                print("\n── Generated PDDL problem ──")
                print(problem)
                print("────────────────────────────\n")

            print("  backend        : pyperplan  (fallback — no action costs)")
            plan_text = run_pyperplan(domain_path, problem_path, plan_path)

        if plan_text is None:
            return None

        sequence = parse_plan_to_sequence(plan_text, state)

        if not use_fd:
            sequence = _sort_sequence_by_priority(sequence, state)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(sequence, f, indent=2)

        return sequence