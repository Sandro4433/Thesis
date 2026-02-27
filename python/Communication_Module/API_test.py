from __future__ import annotations

from pathlib import Path
import sys
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Allow running this file directly (by path) as well as importing it from python/Main.py
PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../python
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# -------------------- Config --------------------

MODEL = "gpt-4.1"

# Stable filesystem paths
from paths import LLM_INPUT_JSON, PATCH_LOG_JSON, LLM_RESPONSE_JSON

INPUT_PATH = str(LLM_INPUT_JSON.resolve())
RULES_PATH = str((Path(__file__).resolve().parent / "rules.txt").resolve())

# Where to persist edits across runs (optional but recommended)
PATCH_LOG_PATH = str(PATCH_LOG_JSON.resolve())

# Final output
OUTPUT_PATH = str(LLM_RESPONSE_JSON.resolve())

# LLM patch block format
PATCH_BLOCK_RE = re.compile(r"```patch\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# -------------------- Patch application (deterministic) --------------------


def _unescape_json_pointer(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _split_pointer(pointer: str) -> List[str]:
    if pointer == "" or pointer == "/":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON Pointer (must start with '/'): {pointer}")
    return [_unescape_json_pointer(p) for p in pointer.lstrip("/").split("/") if p != ""]


def _get_parent_and_key(doc: Any, pointer: str, create: bool) -> Tuple[Any, str]:
    parts = _split_pointer(pointer)
    if not parts:
        raise ValueError("Pointer must not target the root.")

    cur = doc
    for part in parts[:-1]:
        if isinstance(cur, list):
            idx = int(part)
            cur = cur[idx]
        elif isinstance(cur, dict):
            if part not in cur:
                if not create:
                    raise KeyError(f"Missing path segment: {part} in {pointer}")
                cur[part] = {}
            cur = cur[part]
        else:
            raise TypeError(f"Cannot traverse into non-container at segment '{part}'")

    return cur, parts[-1]


def apply_patch(doc: Any, patch: Dict[str, Any]) -> None:
    """
    Supported ops:
      - set: set a value at path (creates missing dict segments)
      - delete: delete key / list index at path (no-op if missing key in dict)
      - append: append value to list at path (path points to list)
    """
    op = patch.get("op")
    path = patch.get("path")
    if not isinstance(op, str) or not isinstance(path, str):
        raise ValueError("Patch must include string fields 'op' and 'path'.")

    if op == "set":
        parent, key = _get_parent_and_key(doc, path, create=True)
        value = patch.get("value")
        if isinstance(parent, list):
            parent[int(key)] = value
        else:
            parent[key] = value

    elif op == "delete":
        parent, key = _get_parent_and_key(doc, path, create=False)
        if isinstance(parent, list):
            del parent[int(key)]
        else:
            parent.pop(key, None)

    elif op == "append":
        # path points to a list itself (not a parent/key pair), so we resolve it
        target = resolve_pointer(doc, path)
        if not isinstance(target, list):
            raise ValueError(f"append requires list target at path {path}")
        target.append(patch.get("value"))

    else:
        raise ValueError(f"Unsupported op: {op}")


def resolve_pointer(doc: Any, pointer: str) -> Any:
    parts = _split_pointer(pointer)
    cur = doc
    for part in parts:
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise TypeError(f"Cannot traverse pointer into non-container at '{part}'")
    return cur


def apply_patch_log(base_scene: Dict[str, Any], patch_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    scene = json.loads(json.dumps(base_scene))  # deep copy
    for p in patch_log:
        apply_patch(scene, p)
    return scene


# -------------------- Patch parsing / persistence --------------------


def extract_patch_block(text: str) -> List[Dict[str, Any]]:
    m = PATCH_BLOCK_RE.search(text or "")
    if not m:
        raise ValueError("No ```patch``` block found.")
    data = json.loads(m.group(1).strip())
    if not isinstance(data, list):
        raise ValueError("Patch block must be a JSON array.")
    for p in data:
        if not isinstance(p, dict):
            raise ValueError("Each patch must be a JSON object.")
        if "op" not in p or "path" not in p:
            raise ValueError("Each patch must contain 'op' and 'path'.")
    return data


def load_patch_log() -> List[Dict[str, Any]]:
    p = Path(PATCH_LOG_PATH)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("patch_log.json must contain a JSON array.")
    return data


def save_patch_log(patch_log: List[Dict[str, Any]]) -> None:
    Path(PATCH_LOG_PATH).write_text(
        json.dumps(patch_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# -------------------- LLM helpers --------------------


def is_yes(text: str) -> bool:
    t = text.strip().lower()
    return any(x in t for x in ["yes", "y", "ok", "okay", "confirm", "confirmed", "sure", "go ahead", "do it"])


def is_no(text: str) -> bool:
    t = text.strip().lower()
    return any(x in t for x in ["no", "n", "nope", "cancel", "reject"])


def is_finish(text: str) -> bool:
    t = text.strip().lower()
    return any(
        x in t
        for x in [
            "finish",
            "finalize",
            "done",
            "end",
            "export",
            "save json",
            "write json",
            "last edit",
            "that's all",
            "thats all",
        ]
    )


def chat(client: OpenAI, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# -------------------- Main interactive loop --------------------


def main():
    base_scene = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))
    rules = Path(RULES_PATH).read_text(encoding="utf-8")

    patch_log = load_patch_log()  # persists across runs
    pending_patch: Optional[List[Dict[str, Any]]] = None

    client = OpenAI()

    system_prompt = (
        "You are a scene-edit assistant.\n\n"
        "You will be given a base scene JSON (for reference) and a RULES text.\n"
        "Your job is ONLY to translate the user's natural-language edit requests into an ORDERED list of minimal patch ops.\n\n"
        "CRITICAL:\n"
        "- Never output the full scene JSON.\n"
        "- Never reprint the full input JSON.\n"
        "- Ask clarification questions if the request is ambiguous.\n"
        "- Only when the request is clear, output a patch proposal.\n\n"
        "PATCH OUTPUT FORMAT (when proposing changes):\n"
        "1) A short plain-text summary of the intended changes.\n"
        "2) Then a fenced block exactly like:\n"
        "   ```patch\n"
        "   [ {\"op\":\"set\",\"path\":\"/...\",\"value\":...}, ... ]\n"
        "   ```\n"
        "3) Then ask: 'Confirm these changes?'\n\n"
        "Supported ops:\n"
        "- set: set value at JSON Pointer path\n"
        "- delete: remove key/index at path\n"
        "- append: append to list at path (path points to list)\n\n"
        "Paths must be JSON Pointer strings.\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "RULES:\n" + rules},
        {"role": "user", "content": "Base scene JSON (reference only):\n" + json.dumps(base_scene, ensure_ascii=False)},
        {
            "role": "user",
            "content": (
                f"Current patch_log length: {len(patch_log)}.\n"
                "Start by giving a SHORT scene summary and ask what I want to change."
            ),
        },
    ]

    while True:
        assistant_text = chat(client, messages, temperature=0.2)
        print("\nASSISTANT:\n" + assistant_text + "\n")
        messages.append({"role": "assistant", "content": assistant_text})

        user_input = input("YOU: ").strip()
        if not user_input:
            continue

        # Finish: apply patches and save final JSON
        if is_finish(user_input):
            final_scene = apply_patch_log(base_scene, patch_log)
            Path(OUTPUT_PATH).write_text(
                json.dumps(final_scene, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Saved final JSON to: {Path(OUTPUT_PATH).resolve()}")
            return

        # If we have a pending patch proposal, user can confirm/reject it
        if pending_patch is not None:
            if is_yes(user_input):
                patch_log.extend(pending_patch)  # preserves order; later wins naturally
                save_patch_log(patch_log)
                pending_patch = None

                # Tell the model the patch was accepted, then continue
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Confirmed. The patches were accepted and appended to the patch log "
                            f"(patch_log length is now {len(patch_log)}). "
                            "Continue; I may request more edits."
                        ),
                    }
                )
                continue

            if is_no(user_input):
                pending_patch = None
                messages.append(
                    {"role": "user", "content": "Rejected. Do not apply those changes. Ask what I want to change instead."}
                )
                continue

        # Normal edit request: send to LLM
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Patch log length (already accepted): {len(patch_log)}.\n"
                    "New request:\n"
                    + user_input
                ),
            }
        )

        proposal_text = chat(client, messages, temperature=0.2)
        print("\nASSISTANT:\n" + proposal_text + "\n")
        messages.append({"role": "assistant", "content": proposal_text})

        # If the assistant proposed a patch block, keep it pending until user confirms
        try:
            pending_patch = extract_patch_block(proposal_text)
        except Exception:
            pending_patch = None


if __name__ == "__main__":
    main()
