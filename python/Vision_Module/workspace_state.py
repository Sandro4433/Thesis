# workspace_state.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _drop_nulls(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            vv = _drop_nulls(v)
            if vv is None or vv == {} or vv == []:
                continue
            out[k] = vv
        return out
    if isinstance(obj, list):
        out = [_drop_nulls(v) for v in obj]
        out = [v for v in out if v is not None and v != {} and v != []]
        return out
    return obj


def save_json_snapshot(path: str, state: Dict[str, Any], pretty: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(state, f, indent=2, ensure_ascii=False)
        else:
            json.dump(state, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


def load_json_snapshot(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"slots": {}, "parts": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"slots": {}, "parts": {}}
    data.setdefault("slots", {})
    data.setdefault("parts", {})
    return data


def entries_to_state(final_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts your flat list into:
      {
        "slots": { "<slot_name>": {pos,quat,orientation,Role,child_part} },
        "parts": { "<part_name>": {pos,quat,orientation,Color,Size,Fragility,Role} }
      }

    - Slot/Part classification is inferred from name prefix (Kit_/Container_/Part_).
    - If slot embeds a child_part, that part is removed from "parts".
    """
    slots: Dict[str, Dict[str, Any]] = {}
    parts: Dict[str, Dict[str, Any]] = {}

    for obj in final_entries:
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue

        if name.startswith("Kit_") or name.startswith("Container_"):
            slot = dict(obj)
            slot.pop("name", None)
            slot.setdefault("child_part", None)
            slots[name] = slot
        elif name.startswith("Part_"):
            part = dict(obj)
            part.pop("name", None)
            parts[name] = part

    # remove embedded parts from parts dict
    for slot in slots.values():
        ch = slot.get("child_part")
        if isinstance(ch, dict):
            ch_name = ch.get("name")
            if isinstance(ch_name, str):
                parts.pop(ch_name, None)

    return {"slots": slots, "parts": parts}


def _strip_for_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a reduced state for LLM planning:
      - Remove quat (q) and orientation (o)
      - Replace pos [x,y,z] -> [x,y]
      - Keep child_part (if embedded) but also stripped the same way
    """
    def strip_obj(o: Any) -> Any:
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                # Drop fields not needed for LLM planning
                if k in ("quat", "orientation"):
                    continue

                if k == "pos":
                    # Keep only X,Y
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        out["pos"] = [v[0], v[1]]
                    continue

                # Recurse
                out[k] = strip_obj(v)
            return out

        if isinstance(o, list):
            return [strip_obj(v) for v in o]

        return o

    # Only process expected top-level shape
    if not isinstance(state, dict):
        return {"slots": {}, "parts": {}}

    slots = state.get("slots", {})
    parts = state.get("parts", {})

    slim = {
        "slots": strip_obj(slots) if isinstance(slots, dict) else {},
        "parts": strip_obj(parts) if isinstance(parts, dict) else {},
    }
    return slim


def state_to_api_payload(
    state: Dict[str, Any],
    compact_keys: bool = True,
    drop_nulls: bool = True,
) -> str:
    """
    Minified payload for LLM planning:
      - strips quat/orientation and Z
      - optional null stripping
      - optional compact keys
      - always minified JSON
    """
    data = _strip_for_llm(state)

    if drop_nulls:
        data = _drop_nulls(data)

    if compact_keys:
        def remap(o: Any) -> Any:
            if isinstance(o, dict):
                out = {}
                for k, v in o.items():
                    nk = k
                    # keep top-level "slots"/"parts" as-is
                    if k == "pos": nk = "p"
                    elif k == "Color": nk = "c"
                    elif k == "Size": nk = "s"
                    elif k == "Fragility": nk = "f"
                    elif k == "Role": nk = "r"
                    elif k == "child_part": nk = "ch"
                    out[nk] = remap(v)
                return out
            if isinstance(o, list):
                return [remap(v) for v in o]
            return o

        data = remap(data)

    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

def save_llm_snapshot(
    path: str,
    state: Dict[str, Any],
    compact_keys: bool = True,
    drop_nulls: bool = True,
    pretty: bool = False,
) -> None:
    """
    Create and save a stripped-down LLM planning JSON file.
    Uses same stripping logic as state_to_api_payload,
    but writes a JSON file instead of returning a string.
    """
    data = _strip_for_llm(state)

    if drop_nulls:
        data = _drop_nulls(data)

    if compact_keys:
        def remap(o: Any) -> Any:
            if isinstance(o, dict):
                out = {}
                for k, v in o.items():
                    nk = k
                    if k == "pos": nk = "p"
                    elif k == "Color": nk = "c"
                    elif k == "Size": nk = "s"
                    elif k == "Fragility": nk = "f"
                    elif k == "Role": nk = "r"
                    elif k == "child_part": nk = "ch"
                    out[nk] = remap(v)
                return out
            if isinstance(o, list):
                return [remap(v) for v in o]
            return o

        data = remap(data)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

    os.replace(tmp, path)