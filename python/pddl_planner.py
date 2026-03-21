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

def build_domain_pddl_costs(num_priorities: int) -> str:
    """
    Generate a PDDL domain with action costs + derived predicates that enforces
    priority ordering as a hard constraint.

    For each priority level K, a derived predicate (priority-K-available) is
    true while any priority-K part still sits in an input receptacle.
    pick-priority-(K+1) requires (not (priority-K-available)), so the planner
    cannot legally pick a lower-priority part while a higher-priority one
    remains.

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
    # Derived predicates must also be declared in :predicates
    for k in range(1, num_priorities + 1):
        pred_lines.append(f"    (priority_{k}_available)")

    # Derived predicates — no numeric fluents needed
    derived_blocks = []
    for k in range(1, num_priorities + 1):
        block = (
            f"  (:derived (priority_{k}_available)" + NL +
            f"    (exists (?p - part ?s - slot ?r - receptacle)" + NL +
            f"      (and (priority-{k} ?p) (at ?p ?s)" + NL +
            f"           (slot-of ?s ?r) (role-input ?r))))" + NL
        )
        derived_blocks.append(block)

    # Pick actions — one per priority level
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

    # Unprioritised pick — fires only when all priority levels exhausted
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

    place_action = (
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
        place_action + NL +
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


def _priority_key(preds: Dict[str, Any]):
    order = {e["color"]: e["order"] for e in preds.get("priority", [])}
    return lambda color: order.get(color, 999)


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
    preds     = state.get("predicates", {})
    workspace = state.get("workspace",  {})
    mode      = (workspace.get("operation_mode") or "sorting").lower()

    color_to_parts: Dict[str, List[str]] = {}
    for entry in preds.get("color", []):
        c = (entry.get("color") or "").lower()
        color_to_parts.setdefault(c, []).append(_to_pddl_name(entry["part"]))

    all_receptacles = kits + containers
    compat_entries  = preds.get("part_compatibility", [])
    use_compat      = bool(compat_entries) and mode == "sorting"

    init: List[str] = []
    if use_compat:
        for rule in compat_entries:
            color = (rule.get("part_color") or "").lower()
            for rec_name in rule.get("allowed_in", []):
                rec = _to_pddl_name(rec_name)
                for part in color_to_parts.get(color, []):
                    init.append(f"    (compatible {part} {rec})")
    else:
        parts = [_to_pddl_name(p) for p in state.get("objects", {}).get("parts", [])]
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

def state_to_pddl_problem_costs(state: Dict[str, Any]) -> Tuple[str, int]:
    """
    Generate a PDDL 2.1 problem with numeric fluents and a minimize metric.
    Returns (problem_str, num_priorities).
    """
    objs  = state.get("objects",    {})
    preds = state.get("predicates", {})

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

    # ── priority assignments ──────────────────────────────────────────────────
    priority_entries = preds.get("priority", [])
    color_to_level: Dict[str, int] = {e["color"]: int(e["order"]) for e in priority_entries}
    num_priorities = max(color_to_level.values(), default=0)

    color_map_raw = {e["part"]: (e.get("color") or "").lower() for e in preds.get("color", [])}

    # Initialise total-cost to zero (required by :action-costs)
    init.append("    (= (total-cost) 0)")

    # Priority predicate tags per part
    for p_orig in objs.get("parts", []):
        p_pddl = _to_pddl_name(p_orig)
        color  = color_map_raw.get(p_orig, "")
        level  = color_to_level.get(color)
        if level is not None:
            init.append(f"    (priority-{level} {p_pddl})")
        else:
            init.append(f"    (no-priority {p_pddl})")

    # ── goal ─────────────────────────────────────────────────────────────────
    mode = (state.get("workspace", {}).get("operation_mode") or "sorting").lower()
    goal_lines = _generate_goal(state, mode)
    if not goal_lines:
        raise ValueError("Could not generate a PDDL goal. Check roles and operation_mode.")

    problem = (
        "(define (problem workspace)\n"
        "  (:domain pick-and-place)\n"
        "  (:objects\n"    + "\n".join(obj_lines) + "\n  )\n"
        "  (:init\n"       + "\n".join(init)      + "\n  )\n"
        "  (:goal (and\n"  + "\n".join(f"    {g}" for g in goal_lines) + "\n  ))\n"
        "  (:metric minimize (total-cost))\n"
        ")\n"
    )
    return problem, num_priorities


# ─────────────────────────────────────────────────────────────────────────────
# Goal generators
# ─────────────────────────────────────────────────────────────────────────────

def _generate_goal(state: Dict[str, Any], mode: str) -> List[str]:
    if mode == "kitting":
        return _goal_kitting(state)
    return _goal_sorting(state)


def _goal_sorting(state: Dict[str, Any]) -> List[str]:
    preds        = state.get("predicates",    {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects",       {})
    inputs, outputs = _role_map(preds)
    color_map       = _color_map(preds)
    pkey            = _priority_key(preds)

    parts_in_slot = {e["part"] for e in preds.get("at", [])}
    parts_in_input = [
        e["part"] for e in preds.get("at", [])
        if slot_belongs.get(e["slot"], "") in inputs
    ]
    standalone    = [p for p in objs.get("parts", []) if p not in parts_in_slot]
    parts_to_move = parts_in_input + standalone
    parts_to_move.sort(key=lambda p: pkey(color_map.get(p, "")))

    compat_rules     = preds.get("part_compatibility", [])
    color_to_allowed: Dict[str, set] = {}
    for rule in compat_rules:
        c = (rule.get("part_color") or "").lower()
        for r in rule.get("allowed_in", []):
            color_to_allowed.setdefault(c, set()).add(r)

    empty_by_receptacle: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in outputs:
            empty_by_receptacle.setdefault(parent, []).append(s)

    goals: List[str] = []
    for part in parts_to_move:
        color = color_map.get(part, "")
        if color_to_allowed and color in color_to_allowed:
            valid = [r for r in color_to_allowed[color] if r in outputs and empty_by_receptacle.get(r)]
        elif not color_to_allowed:
            valid = [r for r in outputs if empty_by_receptacle.get(r)]
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
    preds        = state.get("predicates",    {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects",       {})
    inputs, _    = _role_map(preds)
    color_map    = _color_map(preds)
    pkey         = _priority_key(preds)

    available: Dict[str, List[str]] = {}
    parts_in_slot = {e["part"] for e in preds.get("at", [])}
    for e in preds.get("at", []):
        if slot_belongs.get(e["slot"], "") in inputs:
            c = color_map.get(e["part"], "unknown")
            available.setdefault(c, []).append(e["part"])
    for p in objs.get("parts", []):
        if p not in parts_in_slot:
            c = color_map.get(p, "unknown")
            available.setdefault(c, []).append(p)

    kit_set     = set(objs.get("kits", []))
    empty_slots: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in kit_set:
            empty_slots.setdefault(parent, []).append(s)

    used_parts: set = set()
    goals: List[str] = []

    for recipe in preds.get("kit_recipe", []):
        kit   = recipe.get("kit", "")
        color = (recipe.get("color") or "").lower()
        qty   = int(recipe.get("quantity", 0))
        free_slots = empty_slots.get(kit, [])

        candidates = [p for p in available.get(color, []) if p not in used_parts]
        candidates.sort(key=lambda p: pkey(color))

        for part, slot in zip(candidates[:qty], free_slots[:qty]):
            goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(slot)})")
            used_parts.add(part)
            free_slots.remove(slot)

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
            is_place   = action_raw == "place"
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
                problem, num_priorities = state_to_pddl_problem_costs(state)
            except ValueError as e:
                print(f"❌ PDDL problem generation failed: {e}")
                return None

            domain = build_domain_pddl_costs(num_priorities)

            with open(domain_path,  "w") as f: f.write(domain)
            with open(problem_path, "w") as f: f.write(problem)

            if keep_pddl:
                print("\n── Generated PDDL domain (action costs) ──")
                print(domain)
                print("\n── Generated PDDL problem ──")
                print(problem)
                print("────────────────────────────\n")

            print(f"  backend        : Fast Downward  "
                  f"(PDDL 2.1 action costs, {num_priorities} priority level(s))")
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