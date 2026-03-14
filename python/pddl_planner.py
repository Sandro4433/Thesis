# pddl_planner.py
#
# Full PDDL planning pipeline:
#   state (positions.json) → domain.pddl + problem.pddl → Fast Downward → sequence.json
#
# Install Fast Downward:
#   git clone https://github.com/aibasel/downward.git
#   cd downward && python build.py
#   Then set FAST_DOWNWARD_PATH in config.py to the fast-downward script path.
#
# The domain covers both sorting and kitting.
# The problem file is generated automatically from the workspace state.
# Goals are derived from: operation_mode, kit_recipe, priority, role predicates.

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

GRIPPER_CLOSE_STANDARD = 0.05
GRIPPER_CLOSE_LARGE    = 0.06

# ── pyperplan availability check ──────────────────────────────────────────────
try:
    import pyperplan  # noqa: F401
    _PYPERPLAN_AVAILABLE = True
except ImportError:
    _PYPERPLAN_AVAILABLE = False

# ── PDDL domain (written once per planning call) ──────────────────────────────

DOMAIN_PDDL = """\
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
    (large ?p - part)
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


# ── problem generation ────────────────────────────────────────────────────────

def _to_pddl_name(name: str) -> str:
    """
    Sanitise a Python name for use as a PDDL object identifier.
    PDDL names are case-insensitive; keep underscores, strip anything else odd.
    """
    return name.replace(" ", "_")


def state_to_pddl_problem(state: Dict[str, Any]) -> str:
    """Generate a complete PDDL problem string from a positions.json state."""

    objs         = state.get("objects", {})
    preds        = state.get("predicates", {})
    slot_belongs = state.get("slot_belongs_to", {})
    workspace    = state.get("workspace", {})

    kits       = [_to_pddl_name(k) for k in objs.get("kits", [])]
    containers = [_to_pddl_name(c) for c in objs.get("containers", [])]
    parts      = [_to_pddl_name(p) for p in objs.get("parts", [])]
    slots      = [_to_pddl_name(s) for s in objs.get("slots", [])]

    # ── standalone parts need virtual slots ───────────────────────────────────
    # Parts that are not listed in predicates.at have no slot/receptacle, so
    # the pick action's (role-input ?r) precondition can never be satisfied for
    # them.  Fix: give each standalone part a virtual slot that belongs to a
    # single virtual "workspace" container with role-input.  This is transparent
    # to the rest of the domain — the pick action works unchanged.
    parts_in_slot = {entry["part"] for entry in preds.get("at", [])}
    standalone_parts = [
        p for p in objs.get("parts", [])
        if p not in parts_in_slot
    ]

    VIRTUAL_RECEPTACLE = "workspace_floor"
    virtual_slots: List[str] = []       # original names, for slot_belongs_to
    if standalone_parts:
        for p in standalone_parts:
            virtual_slots.append(f"ws_slot_{_to_pddl_name(p)}")

    # ── object declarations ───────────────────────────────────────────────────
    obj_lines: List[str] = []
    if kits:
        obj_lines.append("    " + " ".join(kits) + " - kit")
    # virtual receptacle is a container so the existing type hierarchy works
    all_containers = containers + ([VIRTUAL_RECEPTACLE] if virtual_slots else [])
    if all_containers:
        obj_lines.append("    " + " ".join(all_containers) + " - container")
    if parts:
        obj_lines.append("    " + " ".join(parts) + " - part")
    all_slots = slots + virtual_slots
    if all_slots:
        obj_lines.append("    " + " ".join(all_slots) + " - slot")

    # ── initial state ─────────────────────────────────────────────────────────
    init: List[str] = ["    (hand-empty)"]

    for entry in preds.get("at", []):
        init.append(f"    (at {_to_pddl_name(entry['part'])} {_to_pddl_name(entry['slot'])})")

    for s in preds.get("slot_empty", []):
        init.append(f"    (slot-empty {_to_pddl_name(s)})")

    for slot_name, receptacle in slot_belongs.items():
        init.append(f"    (slot-of {_to_pddl_name(slot_name)} {_to_pddl_name(receptacle)})")

    # virtual slot facts for standalone parts
    for p_orig, vs in zip(standalone_parts, virtual_slots):
        p_pddl  = _to_pddl_name(p_orig)
        init.append(f"    (at {p_pddl} {vs})")
        init.append(f"    (slot-of {vs} {VIRTUAL_RECEPTACLE})")

    for entry in preds.get("role", []):
        obj  = _to_pddl_name(entry["object"])
        role = (entry.get("role") or "").lower()
        if role == "input":
            init.append(f"    (role-input {obj})")
        elif role == "output":
            init.append(f"    (role-output {obj})")

    # virtual workspace receptacle is always role-input
    if virtual_slots:
        init.append(f"    (role-input {VIRTUAL_RECEPTACLE})")

    for entry in preds.get("color", []):
        part  = _to_pddl_name(entry["part"])
        color = (entry.get("color") or "").lower()
        if color == "red":
            init.append(f"    (color-red {part})")
        elif color == "blue":
            init.append(f"    (color-blue {part})")
        elif color == "green":
            init.append(f"    (color-green {part})")

    for entry in preds.get("size", []):
        if (entry.get("size") or "standard").lower() == "large":
            init.append(f"    (large {_to_pddl_name(entry['part'])})")

    # ── part_compatibility → (compatible part receptacle) ─────────────────────
    # Derive mode early — needed to decide whether compat rules apply.
    mode = (workspace.get("operation_mode") or "sorting").lower()

    # Build a color→parts lookup so we can emit one compatible fact per
    # (part, receptacle) pair rather than per (color, receptacle).
    color_to_parts: Dict[str, List[str]] = {}
    for entry in preds.get("color", []):
        c = (entry.get("color") or "").lower()
        color_to_parts.setdefault(c, []).append(_to_pddl_name(entry["part"]))

    all_receptacles = kits + containers
    compat_entries = preds.get("part_compatibility", [])

    # part_compatibility is a sorting-only concept — it constrains which
    # output container each colour goes to during sorting.  In kitting mode
    # it must be ignored so that parts can be placed into kit slots freely.
    use_compat_rules = bool(compat_entries) and mode == "sorting"

    if use_compat_rules:
        for rule in compat_entries:
            color      = (rule.get("part_color") or "").lower()
            allowed_in = rule.get("allowed_in", [])
            for receptacle_name in allowed_in:
                rec = _to_pddl_name(receptacle_name)
                for part in color_to_parts.get(color, []):
                    init.append(f"    (compatible {part} {rec})")
    else:
        # No sorting rules active (or kitting mode) → every part is compatible
        # with every receptacle so the place precondition is never the bottleneck.
        for part in parts:
            for rec in all_receptacles:
                init.append(f"    (compatible {part} {rec})")

    # ── goal ─────────────────────────────────────────────────────────────────
    goal_lines = _generate_goal(state, mode)

    if not goal_lines:
        raise ValueError(
            "Could not generate a PDDL goal from the current workspace state. "
            "Make sure roles (input/output) and either kit_recipe or a "
            "reachable sorting target are configured."
        )

    problem = (
        "(define (problem workspace)\n"
        "  (:domain pick-and-place)\n"
        "  (:objects\n"
        + "\n".join(obj_lines) + "\n"
        "  )\n"
        "  (:init\n"
        + "\n".join(init) + "\n"
        "  )\n"
        "  (:goal (and\n"
        + "\n".join(f"    {g}" for g in goal_lines) + "\n"
        "  ))\n"
        ")\n"
    )
    return problem


# ── goal generators ───────────────────────────────────────────────────────────

def _role_map(preds: Dict[str, Any]) -> Tuple[set, set]:
    """Returns (input_receptacles, output_receptacles)."""
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


def _generate_goal(state: Dict[str, Any], mode: str) -> List[str]:
    if mode == "kitting":
        return _goal_kitting(state)
    return _goal_sorting(state)


def _goal_sorting(state: Dict[str, Any]) -> List[str]:
    """
    Move every part from input receptacles into output receptacles,
    respecting both priority ordering and part_compatibility rules.

    part_compatibility controls which output receptacle each color goes to:
      [{"part_color": "blue", "allowed_in": ["Container_1"]}, ...]

    If no compatibility rules are defined, parts are distributed across any
    available output slot (original behaviour — preserved for backward compat).

    Priority ordering is respected: higher-priority colors are assigned first,
    ensuring they get slots before lower-priority colors when capacity is tight.
    """
    preds        = state.get("predicates", {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects", {})
    inputs, outputs = _role_map(preds)
    color_map       = _color_map(preds)
    pkey            = _priority_key(preds)

    # parts to move: those sitting in input slots + standalone parts
    # (standalone parts are placed on the virtual workspace_floor receptacle
    # which is always role-input, so they are always eligible to pick)
    parts_in_input = [
        e["part"] for e in preds.get("at", [])
        if slot_belongs.get(e["slot"], "") in inputs
    ]
    all_part_names  = objs.get("parts", [])
    parts_in_slot   = {e["part"] for e in preds.get("at", [])}
    standalone      = [p for p in all_part_names if p not in parts_in_slot]
    parts_to_move   = parts_in_input + standalone
    parts_to_move.sort(key=lambda p: pkey(color_map.get(p, "")))

    # build color → allowed output receptacles from part_compatibility
    # e.g. {"blue": {"Container_1"}, "red": {"Container_3"}}
    compat_rules = preds.get("part_compatibility", [])
    color_to_allowed: Dict[str, set] = {}
    for rule in compat_rules:
        c   = (rule.get("part_color") or "").lower()
        for r in rule.get("allowed_in", []):
            color_to_allowed.setdefault(c, set()).add(r)

    # collect empty output slots grouped by parent receptacle
    # e.g. {"Container_1": ["Container_1_Pos_1", ...], ...}
    empty_by_receptacle: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in outputs:
            empty_by_receptacle.setdefault(parent, []).append(s)

    goals: List[str] = []

    for part in parts_to_move:
        color = color_map.get(part, "")

        # determine which output receptacles are valid for this part
        if color_to_allowed and color in color_to_allowed:
            # compatibility rules exist and cover this color — restrict to them
            valid_receptacles = [
                r for r in color_to_allowed[color]
                if r in outputs and empty_by_receptacle.get(r)
            ]
        elif not color_to_allowed:
            # no rules defined at all — any output receptacle is fair game
            valid_receptacles = [
                r for r in outputs
                if empty_by_receptacle.get(r)
            ]
        else:
            # rules exist but this color has no destination defined — skip it
            print(f"  ⚠  _goal_sorting: '{part}' (color={color}) has no part_compatibility rule — skipped.")
            continue

        if not valid_receptacles:
            print(f"  ⚠  _goal_sorting: no available output slot for '{part}' (color={color}) — skipped.")
            continue

        # pick from the first valid receptacle (priority ordering already handled
        # by sorting parts_to_move above)
        receptacle = valid_receptacles[0]
        slot = empty_by_receptacle[receptacle].pop(0)
        goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(slot)})")

    return goals


def _goal_kitting(state: Dict[str, Any]) -> List[str]:
    """
    Fill kits according to kit_recipe, respecting priority and optional size.

    Recipe entry format:
      {"kit": "Kit_0", "color": "blue", "quantity": 2}
      {"kit": "Kit_0", "color": "blue", "size": "large", "quantity": 1}

    When "size" is present, only parts of that size are eligible for that entry.
    When "size" is omitted or null, any size is accepted.
    """
    preds        = state.get("predicates", {})
    slot_belongs = state.get("slot_belongs_to", {})
    objs         = state.get("objects", {})
    inputs, _    = _role_map(preds)
    color_map    = _color_map(preds)
    pkey         = _priority_key(preds)

    # build size lookup: part_name → "standard" | "large"
    size_map: Dict[str, str] = {
        e["part"]: (e.get("size") or "standard").lower()
        for e in preds.get("size", [])
    }

    # available parts per (color, size): from input slots + standalone parts
    # key is (color, size), e.g. ("blue", "large")
    available: Dict[Tuple[str, str], List[str]] = {}
    parts_in_slot = {e["part"] for e in preds.get("at", [])}
    for e in preds.get("at", []):
        if slot_belongs.get(e["slot"], "") in inputs:
            c = color_map.get(e["part"], "unknown")
            s = size_map.get(e["part"], "standard")
            available.setdefault((c, s), []).append(e["part"])
    for p in objs.get("parts", []):
        if p not in parts_in_slot:
            c = color_map.get(p, "unknown")
            s = size_map.get(p, "standard")
            available.setdefault((c, s), []).append(p)

    # empty kit slots per kit
    kit_set     = set(objs.get("kits", []))
    empty_slots: Dict[str, List[str]] = {}
    for s in preds.get("slot_empty", []):
        parent = slot_belongs.get(s, "")
        if parent in kit_set:
            empty_slots.setdefault(parent, []).append(s)

    used_parts: set = set()
    goals: List[str] = []

    for recipe in preds.get("kit_recipe", []):
        kit        = recipe.get("kit", "")
        color      = (recipe.get("color") or "").lower()
        size_req   = (recipe.get("size") or "").lower() or None   # None means any size
        qty        = int(recipe.get("quantity", 0))

        free_slots = empty_slots.get(kit, [])

        if size_req:
            # specific size requested — only match that (color, size) bucket
            candidates = [p for p in available.get((color, size_req), []) if p not in used_parts]
        else:
            # no size constraint — collect all sizes for this color
            candidates = [
                p for (c, s), parts in available.items()
                if c == color
                for p in parts
                if p not in used_parts
            ]
        candidates.sort(key=lambda p: pkey(color))

        for part, slot in zip(candidates[:qty], free_slots[:qty]):
            goals.append(f"(at {_to_pddl_name(part)} {_to_pddl_name(slot)})")
            used_parts.add(part)
            free_slots.remove(slot)

    # fall back to sorting if no recipe was defined
    if not goals:
        return _goal_sorting(state)

    return goals


# ── planner execution ─────────────────────────────────────────────────────────

def run_planner(
    domain_path: str,
    problem_path: str,
    plan_path: str,
) -> Optional[str]:
    """
    Run pyperplan and return the raw plan text, or None on failure.
    Uses A* with the hFF heuristic.
    Install with: pip install pyperplan
    """
    if not _PYPERPLAN_AVAILABLE:
        print(
            "\n❌ pyperplan is not installed.\n"
            "   Install it with:  pip install pyperplan\n"
        )
        return None

    from pyperplan import grounding, search  # type: ignore
    from pyperplan.pddl.parser import Parser             # type: ignore

    try:
        parser  = Parser(domain_path, problem_path)
        domain  = parser.parse_domain()
        problem = parser.parse_problem(domain)
    except Exception as e:
        print(f"❌ pyperplan failed to parse PDDL: {e}")
        return None

    try:
        task = grounding.ground(problem)
    except Exception as e:
        print(f"❌ pyperplan grounding failed: {e}")
        return None

    try:
        from pyperplan.heuristics.relaxation import hFFHeuristic  # type: ignore
        heuristic = hFFHeuristic(task)
        plan = search.astar_search(task, heuristic)
    except Exception as e:
        print(f"❌ pyperplan search failed: {e}")
        return None

    if plan is None:
        print("❌ pyperplan found no plan (problem may be unsolvable).")
        return None

    lines = [f"({op.name})" for op in plan]
    plan_text = "\n".join(lines) + "\n"

    with open(plan_path, "w") as f:
        f.write(plan_text)

    return plan_text


# ── plan → sequence ───────────────────────────────────────────────────────────

def parse_plan_to_sequence(plan_text: str, state: Dict[str, Any]) -> List[List]:
    """
    Convert a plan to sequence.json format:
      [[pick_name, place_name, gripper_close_width], ...]

    Handles both Fast Downward format:
      (pick part_blue_nr_1 container_3_pos_2 container_3)
    And pyperplan format:
      (pick_part_blue_nr_1_container_3_pos_2_container_3)

    Virtual slots (ws_slot_<part>) generated for standalone parts are resolved
    back to the part name as the pick target, since that is what the robot needs.
    """
    preds = state.get("predicates", {})
    objs  = state.get("objects", {})

    # case-insensitive lookup: lowercase → original name
    # Include virtual slot names so the parser can recognise them in the plan
    part_names: List[str] = objs.get("parts", [])
    slot_names: List[str] = objs.get("slots", [])
    virtual_slot_names = [f"ws_slot_{p}" for p in part_names]

    all_names: List[str] = part_names + slot_names
    ci_map = {n.lower(): n for n in all_names}

    # map virtual slot (lowercase) → original part name
    # e.g. "ws_slot_part_blue_nr_1" → "Part_Blue_Nr_1"
    virtual_to_part: Dict[str, str] = {
        f"ws_slot_{p}".lower(): p for p in part_names
    }

    size_map = {
        e["part"]: (e.get("size") or "standard").lower()
        for e in preds.get("size", [])
    }

    pending: Dict[str, str] = {}   # part_lower → original pick name
    sequence: List[List]    = []

    for line in plan_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        line = line.strip("()")

        # ── Fast Downward format: "pick part slot receptacle" ────────────────
        tokens = line.split()
        if len(tokens) >= 3:
            action     = tokens[0].lower()
            part_lower = tokens[1].lower()
            slot_lower = tokens[2].lower()

            if action in ("pick", "place"):
                # For pick: the "slot" token might be a virtual slot — resolve to part
                if action == "pick" and slot_lower in virtual_to_part:
                    part_orig = ci_map.get(part_lower, tokens[1])
                    # pick target is the part itself (standalone, on the floor)
                    pending[part_lower] = part_orig
                elif action == "pick":
                    part_orig = ci_map.get(part_lower, tokens[1])
                    pending[part_lower] = part_orig
                elif action == "place" and part_lower in pending:
                    pick_name  = pending.pop(part_lower)
                    slot_orig  = ci_map.get(slot_lower, tokens[2])
                    part_orig  = ci_map.get(part_lower, tokens[1])
                    size       = size_map.get(part_orig, "standard")
                    gripper    = GRIPPER_CLOSE_LARGE if size == "large" else GRIPPER_CLOSE_STANDARD
                    sequence.append([pick_name, slot_orig, gripper])
                continue

        # ── pyperplan format: "pick_part_blue_nr_1_ws_slot_part_blue_nr_1_..." ─
        # Include virtual slot names in the sorted list for greedy matching
        all_names_extended = all_names + virtual_slot_names
        sorted_names = sorted(all_names_extended, key=lambda n: len(n), reverse=True)
        ci_map_ext   = {n.lower(): n for n in all_names_extended}

        line_lower = line.lower()
        if line_lower.startswith("pick"):
            action = "pick"
            remainder = line_lower[len("pick"):].lstrip("_")
        elif line_lower.startswith("place"):
            action = "place"
            remainder = line_lower[len("place"):].lstrip("_")
        else:
            continue

        # greedily extract known names from the remainder
        found: List[str] = []
        pos = 0
        while pos < len(remainder) and len(found) < 3:
            matched = False
            for name in sorted_names:
                nl = name.lower()
                if remainder[pos:].startswith(nl):
                    found.append(ci_map_ext.get(nl, name))
                    pos += len(nl) + 1   # +1 for underscore separator
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
            # If the slot is a virtual slot, the pick target is the part itself
            pick_target = part_orig
            pending[part_lower] = pick_target
        elif action == "place" and part_lower in pending:
            pick_name = pending.pop(part_lower)
            # slot_token might be a virtual slot name — skip, use real slot
            if slot_token.lower() in virtual_to_part:
                continue   # this shouldn't happen in a place action
            slot_orig = slot_token
            size      = size_map.get(part_orig, "standard")
            gripper   = GRIPPER_CLOSE_LARGE if size == "large" else GRIPPER_CLOSE_STANDARD
            sequence.append([pick_name, slot_orig, gripper])

    return sequence


# ── top-level entry point ─────────────────────────────────────────────────────

def plan_sequence(
    state: Dict[str, Any],
    output_path: Optional[str] = None,
    keep_pddl: bool = False,
) -> Optional[List[List]]:
    """
    Full pipeline: state → PDDL files → pyperplan → sequence.

    Parameters
    ----------
    state        : loaded positions.json dict
    output_path  : if given, saves sequence to this path as JSON
    keep_pddl    : if True, prints the generated PDDL for inspection

    Returns the sequence list, or None if planning failed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        domain_path  = os.path.join(tmpdir, "domain.pddl")
        problem_path = os.path.join(tmpdir, "problem.pddl")
        plan_path    = os.path.join(tmpdir, "plan.txt")

        # write domain
        with open(domain_path, "w") as f:
            f.write(DOMAIN_PDDL)

        # generate and write problem
        try:
            problem = state_to_pddl_problem(state)
        except ValueError as e:
            print(f"❌ PDDL problem generation failed: {e}")
            return None

        with open(problem_path, "w") as f:
            f.write(problem)

        if keep_pddl:
            print("\n── Generated PDDL problem ──")
            print(problem)
            print("────────────────────────────\n")

        # run planner
        plan_text = run_planner(domain_path, problem_path, plan_path)
        if plan_text is None:
            return None

        if keep_pddl:
            print("\n── Plan found ──")
            print(plan_text)
            print("────────────────\n")

        # convert to sequence
        sequence = parse_plan_to_sequence(plan_text, state)
        print(f"✅ PDDL planner: {len(sequence)} step(s) planned.")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sequence, f, indent=2, ensure_ascii=False)
            print(f"   Sequence saved → {output_path}")

        return sequence


# ── standalone helper (debug / manual testing) ───────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from paths import POSITIONS_JSON, LLM_RESPONSE_JSON
    from Vision_Module.config import FAST_DOWNWARD_PATH  # type: ignore

    positions_path = str(POSITIONS_JSON.resolve())
    sequence_path  = str(Path(LLM_RESPONSE_JSON.resolve()).parent / "sequence.json")

    print(f"Loading state from: {positions_path}")
    with open(positions_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    result = plan_sequence(state, output_path=sequence_path, keep_pddl=True)
    if result is None:
        sys.exit(1)
    print(json.dumps(result, indent=2))