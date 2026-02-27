# io_jsonl.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def load_jsonl_by_name(path: str) -> Dict[str, Any]:
    """
    Reads JSONL into {name: obj}. If duplicate names exist in file, the last one wins.
    """
    data: Dict[str, Any] = {}
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if isinstance(name, str) and name:
                data[name] = obj
    return data


def write_jsonl_atomic(path: str, objs: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj) + "\n")
    os.replace(tmp, path)


def upsert_jsonl_by_name(path: str, new_objects: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Upsert entries by 'name':
      - if name exists -> overwrite existing entry
      - if name does not exist -> insert new entry
    Returns: (inserted, overwritten)
    """
    existing = load_jsonl_by_name(path)

    inserted = 0
    overwritten = 0
    for obj in new_objects:
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in existing:
            overwritten += 1
        else:
            inserted += 1
        existing[name] = obj

    out = [existing[k] for k in sorted(existing.keys())]
    write_jsonl_atomic(path, out)

    return inserted, overwritten

def rewrite_jsonl_snapshot(path: str, objects: List[Dict[str, Any]]) -> None:
    """
    Completely rewrites the file. Use this when you need deletions to take effect.
    """
    # Optional: keep stable ordering by name
    objs = [o for o in objects if isinstance(o.get("name"), str) and o["name"]]
    objs.sort(key=lambda x: x["name"])
    write_jsonl_atomic(path, objs)