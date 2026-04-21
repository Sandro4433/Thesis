"""
ambiguity_detection.py — Background ambiguity analysis for user instructions.

Runs a separate, low-temperature LLM call that compares the user's instruction
against the current scene configuration to detect four ambiguity types:

  1. Referential ambiguity  — multiple objects match the user's description
  2. Omission               — a required parameter is missing and cannot be inferred
  3. Implicit constraint    — the instruction implies a configuration change
                              that is not stated directly
  4. Spatial ambiguity      — spatial descriptions admit multiple valid
                              interpretations

When ambiguity is found, the module returns structured hints that are injected
into the conversation as a hidden system-level message so the main LLM can
ask sharper, more targeted clarification questions.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from Communication_Module.capacity_tools import compute_position_labels

logger = logging.getLogger(__name__)

# ── Ambiguity analysis prompt ────────────────────────────────────────────────

_AMBIGUITY_SYSTEM_PROMPT = """\
You are an ambiguity detector for a robot workspace configurator.

You will receive:
  1. A SCENE JSON describing the current workspace (receptacles, parts, slots,
     colors, roles, positions, etc.)
  2. RECENT CONVERSATION HISTORY — the last few turns between user and assistant.
  3. A USER MESSAGE — the latest instruction the operator just typed.
  4. The current MODE: "reconfig" (changing workspace attributes) or "motion"
     (planning pick-and-place sequences).

CRITICAL — CONVERSATION CONTEXT:
  The user message does NOT exist in isolation. You MUST read the conversation
  history to understand what has already been discussed, decided, or clarified.
  If the user's message is a direct answer to a question the assistant asked,
  it is NOT ambiguous — the context makes it clear.
  If information appears missing from the current message but was provided in
  an earlier turn, it is NOT an omission.
  Only flag ambiguity when the FULL conversation context still leaves something
  genuinely unclear.

CRITICAL — SINGLE ATTRIBUTE CHANGES ARE NOT AMBIGUOUS:
  The configurator supports these independent attributes:
    operation_mode, batch_size, role, kit_recipe, priority,
    part_compatibility, fragility, color, fill_order.
  If the user's message clearly maps to changing ONE of these attributes,
  it is NOT ambiguous — even if other attributes seem "missing" or "implied".

Your job is to check for:
  TYPE 1 — REFERENTIAL AMBIGUITY: the user's description matches multiple objects.
  TYPE 2 — OMISSION: a required parameter for a broad setup request is missing.
  TYPE 3 — IMPLICIT CONSTRAINT: a broad request implies unstated changes with
            multiple valid interpretations.
  TYPE 4 — SPATIAL AMBIGUITY: spatial language matches multiple receptacle
            labels.

RESPONSE FORMAT — reply with ONLY a valid JSON object, nothing else:
{
  "has_ambiguity": true/false,
  "ambiguities": [
    {
      "type": "referential" | "omission" | "implicit_constraint" | "spatial",
      "description": "Brief description of what is ambiguous",
      "candidates": ["Container_1", "Container_2"],
      "suggested_question": "A specific, natural-language question to resolve this"
    }
  ]
}

Rules:
- Return {"has_ambiguity": false, "ambiguities": []} when the message is clear.
- Do NOT flag ambiguity for simple confirmations, rejections, or session control.
- Keep suggested_question short, specific, and natural.
- Return at most 3 ambiguities; prioritise the most impactful ones.
"""


def detect_ambiguity(
    client: OpenAI,
    model: str,
    user_message: str,
    scene_json: dict,
    mode: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Run background ambiguity detection on a user message.

    Returns the parsed ambiguity result dict, or None if detection fails
    or finds no ambiguity.  Failures are silent — never blocks the main flow.
    """
    stripped = user_message.strip().lower()
    trivial = {
        "yes", "no", "y", "n", "ok", "okay", "confirm", "confirmed",
        "reject", "cancel", "done", "exit", "quit", "save", "end",
        "sure", "go ahead", "do it", "looks good", "nope", "wrong",
        "redo", "skip", "nothing", "no changes", "no task",
    }
    if stripped in trivial or len(stripped) < 3:
        return None

    pos_labels = compute_position_labels(scene_json)
    pos_section = "\n".join(
        f"  {name}: {label}" for name, label in sorted(pos_labels.items())
    )

    history_section = ""
    if conversation_history:
        recent = [
            m for m in conversation_history if m.get("role") in ("user", "assistant")
        ][-10:]
        if recent:
            history_lines = []
            for m in recent:
                role = "USER" if m["role"] == "user" else "ASSISTANT"
                content = m.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                history_lines.append(f"  {role}: {content}")
            history_section = (
                "RECENT CONVERSATION:\n" + "\n".join(history_lines) + "\n\n"
            )

    user_content = (
        f"MODE: {mode}\n\n"
        f"SCENE JSON:\n{json.dumps(scene_json, indent=2, ensure_ascii=False)}\n\n"
        f"POSITION LABELS (pre-computed, authoritative):\n{pos_section}\n\n"
        f"{history_section}"
        f"USER MESSAGE (latest, evaluate this for ambiguity):\n{user_message}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _AMBIGUITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        raw = (resp.choices[0].message.content or "").strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        result = json.loads(raw)

        if not isinstance(result, dict):
            return None
        if not result.get("has_ambiguity", False):
            return None
        if not result.get("ambiguities"):
            return None

        return result

    except Exception as exc:
        logger.debug("Ambiguity detection skipped: %s", exc)
        return None


def format_ambiguity_hint(result: Dict[str, Any]) -> str:
    """
    Format detected ambiguities into a hidden system-level hint for the main LLM.

    Injected into the conversation as a system instruction so the LLM can ask
    sharper clarification questions.  The user never sees this text directly.
    """
    ambiguities = result.get("ambiguities", [])
    if not ambiguities:
        return ""

    lines = [
        "AMBIGUITY DETECTED IN USER MESSAGE — ask clarification BEFORE proposing "
        "any changes or sequence.",
        "Do NOT reveal this analysis to the user. Instead, ask ONE focused "
        "clarification question.",
        "",
    ]

    type_labels = {
        "referential": "REFERENTIAL — multiple objects match",
        "omission": "OMISSION — required info missing",
        "implicit_constraint": "IMPLICIT CONSTRAINT — unstated side-effect",
        "spatial": "SPATIAL — ambiguous position reference",
    }

    for i, amb in enumerate(ambiguities, 1):
        label = type_labels.get(amb.get("type", ""), amb.get("type", "UNKNOWN").upper())
        lines.append(f"  {i}. [{label}]")
        lines.append(f"     {amb.get('description', '')}")
        candidates = amb.get("candidates", [])
        if candidates:
            lines.append(f"     Candidates: {', '.join(str(c) for c in candidates)}")
        question = amb.get("suggested_question", "")
        if question:
            lines.append(f"     Suggested question: {question}")
        lines.append("")

    lines.append(
        "INSTRUCTIONS: Pick the MOST important ambiguity above and ask the user "
        "ONE clear, specific question to resolve it. Do NOT output a changes or "
        "sequence block until all ambiguities are resolved."
    )

    return "\n".join(lines)
