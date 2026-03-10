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
    Return a state for LLM planning that mirrors positions.json but without
    quat and orientation.

    Slot fields:  pos, Role, child_part
    Part fields:  pos, Color, Size        (no Role on parts)

    Nulls are preserved so the LLM always sees every field.
    child_part embedded in a slot is treated as a part (no Role).
    """
    _SLOT_KEEP = {"pos", "Role", "child_part"}
    _PART_KEEP = {"pos", "Color", "Size"}

    def strip_part(p: Any) -> Any:
        if not isinstance(p, dict):
            return p
        return {k: v for k, v in p.items() if k in _PART_KEEP}

    def strip_slot(s: Any) -> Any:
        if not isinstance(s, dict):
            return s
        out: Dict[str, Any] = {}
        for k, v in s.items():
            if k not in _SLOT_KEEP:
                continue
            if k == "child_part" and isinstance(v, dict):
                out[k] = strip_part(v)
            else:
                out[k] = v
        return out

    if not isinstance(state, dict):
        return {"slots": {}, "parts": {}}

    raw_slots = state.get("slots", {})
    raw_parts = state.get("parts", {})

    return {
        "slots": {name: strip_slot(s) for name, s in raw_slots.items()} if isinstance(raw_slots, dict) else {},
        "parts": {name: strip_part(p) for name, p in raw_parts.items()} if isinstance(raw_parts, dict) else {},
    }


def state_to_api_payload(state: Dict[str, Any]) -> str:
    """
    Minified JSON payload for LLM planning.
    Mirrors positions.json but without quat and orientation.
    Slot fields: pos, Role, child_part
    Part fields: pos, Color, Size  (no Role)
    All nulls preserved.
    """
    data = _strip_for_llm(state)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def save_llm_snapshot(
    path: str,
    state: Dict[str, Any],
    pretty: bool = True,
) -> None:
    """
    Save the LLM planning JSON file.
    Mirrors positions.json but without quat and orientation.
    Slot fields: pos, Role, child_part
    Part fields: pos, Color, Size  (no Role)
    All nulls preserved.
    """
    data = _strip_for_llm(state)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

    os.replace(tmp, path)