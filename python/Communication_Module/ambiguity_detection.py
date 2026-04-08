"""
ambiguity_detection.py — Background ambiguity analysis for user instructions.

Runs a separate, low-temperature LLM call that compares the user's instruction
against the current scene configuration to detect four ambiguity types:

  1. Referential ambiguity  — multiple objects match the user's description
  2. Omission               — a required parameter is missing and cannot be inferred
  3. Implicit constraint     — the instruction implies a configuration change
                               that is not stated directly
  4. Spatial ambiguity       — spatial descriptions admit multiple valid
                               interpretations

When ambiguity is found, the module returns structured hints that are injected
into the conversation as a hidden system-level message so the main LLM can
ask sharper, more targeted clarification questions.  The user never sees the
raw ambiguity analysis — only the resulting questions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from Communication_Module.capacity_tools import compute_position_labels


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

Your job is to check whether the user message contains any of the following
ambiguity types when evaluated against the scene:

───────────────────────────────────────────────────────────────
TYPE 1 — REFERENTIAL AMBIGUITY
  The user refers to an object (part, container, kit, slot, color group) in a
  way that matches MULTIPLE objects in the scene.
  Examples:
    • "move the red part" when there are several red parts
    • "use the container on the left" when two containers have similar x-coords
    • "the kit" when there are multiple kits
  Detection: compare the user's noun phrases against object lists in the scene.
  If a description matches 2+ objects, flag it.

───────────────────────────────────────────────────────────────
TYPE 2 — OMISSION
  A required parameter for the operation is missing from the instruction AND
  cannot be reliably inferred from the scene alone.
  Examples:
    • "set up kitting with blue and red" — missing per-kit quantities
    • "sort the parts" — missing which container is input vs output
      (unless it can be inferred from existing contents/roles)
    • "move Part_3" — missing destination
  Detection: map the instruction to the changes/sequence schema. Identify
  required fields that are neither stated nor unambiguously inferable.
  Do NOT flag something as omitted if the scene JSON provides a single
  obvious answer (e.g. only one empty slot, only one kit).

───────────────────────────────────────────────────────────────
TYPE 3 — IMPLICIT CONSTRAINT
  The instruction implies a configuration change that the user did not state
  directly.
  Examples:
    • "sort blue parts into Container_2" implies Container_2 must be output
      and the container holding blue parts must be input — but the user
      didn't say "set Container_2 as output"
    • "fill kits with green parts first" implies a color priority — but the
      user may mean within-kit ordering OR across-kit sweep
    • "put Part_5 in Kit_1" implies kitting mode, input/output roles, and
      possibly a compatibility rule — none stated explicitly
  Detection: identify side-effects or prerequisite changes that the
  instruction requires but does not mention. Only flag when the implied
  change has multiple valid interpretations or could surprise the user.

───────────────────────────────────────────────────────────────
TYPE 4 — SPATIAL AMBIGUITY
  Spatial language ("left", "right", "top", "bottom", "next to", "near",
  "far", "between", "closest") admits multiple valid interpretations given
  the positions in the scene.
  A POSITION_LABELS section is provided below the scene JSON with
  pre-computed relative positions for each receptacle (e.g.
  "Container_3: top-right"). Use ONLY these labels to resolve spatial
  references — do NOT interpret raw xy coordinates yourself.
  If the user's spatial phrase matches exactly one label, there is NO
  spatial ambiguity — report it as resolved.
  Only flag spatial ambiguity when the phrase could plausibly match 2+
  receptacles based on the labels.
  Examples:
    • "the container on the right" when two containers are labeled
      "top-right" and "bottom-right"
    • "the top kit" when two kits are both labeled "top"

───────────────────────────────────────────────────────────────

RESPONSE FORMAT — reply with ONLY a valid JSON object, nothing else:
{
  "has_ambiguity": true/false,
  "ambiguities": [
    {
      "type": "referential" | "omission" | "implicit_constraint" | "spatial",
      "description": "Brief description of what is ambiguous",
      "candidates": ["Container_1", "Container_2"],
      "suggested_question": "A specific, natural-language question to ask the user to resolve this"
    }
  ]
}

Rules:
- Return {"has_ambiguity": false, "ambiguities": []} when the message is clear.
- Do NOT flag ambiguity for things the LLM can unambiguously resolve from the
  scene JSON (e.g. "which container has blue parts" when only one does).
- Do NOT flag ambiguity for simple confirmations, rejections, or session
  control ("yes", "no", "done").
- Do NOT flag omissions for optional parameters (e.g. priority when no
  ordering was mentioned).
- Keep suggested_question short, specific, and natural — the user will see
  a version of it. Reference concrete object names from the scene.
- Return at most 3 ambiguities. Prioritise the most impactful ones.
- The "candidates" field should list the specific objects/values that create
  the ambiguity. Use an empty list if not applicable.
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
    # Skip trivial messages (confirmations, rejections, session control)
    stripped = user_message.strip().lower()
    trivial = {
        "yes", "no", "y", "n", "ok", "okay", "confirm", "confirmed",
        "reject", "cancel", "done", "exit", "quit", "save", "end",
        "sure", "go ahead", "do it", "looks good", "nope", "wrong",
        "redo", "skip", "nothing", "no changes", "no task",
    }
    if stripped in trivial or len(stripped) < 3:
        return None

    # Build the analysis prompt with pre-computed position labels
    pos_labels = compute_position_labels(scene_json)
    pos_section = "\n".join(f"  {name}: {label}" for name, label in sorted(pos_labels.items()))

    # Include recent conversation history (last N user/assistant turns)
    history_section = ""
    if conversation_history:
        # Take the last 10 messages (skip system prompts and scene JSON)
        recent = [
            m for m in conversation_history
            if m.get("role") in ("user", "assistant")
        ][-10:]
        if recent:
            history_lines = []
            for m in recent:
                role = "USER" if m["role"] == "user" else "ASSISTANT"
                # Truncate very long messages (e.g. scene JSON dumps)
                content = m.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                history_lines.append(f"  {role}: {content}")
            history_section = (
                "RECENT CONVERSATION:\n"
                + "\n".join(history_lines)
                + "\n\n"
            )

    user_content = (
        f"MODE: {mode}\n\n"
        f"SCENE JSON:\n{json.dumps(scene_json, indent=2, ensure_ascii=False)}\n\n"
        f"POSITION LABELS (pre-computed, authoritative — use these, not raw xy):\n"
        f"{pos_section}\n\n"
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

        # Strip markdown fences if present
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

    except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
        # Ambiguity detection is best-effort — never block the main flow
        print(f"  [Ambiguity detection skipped: {e}]")
        return None


def format_ambiguity_hint(result: Dict[str, Any]) -> str:
    """
    Format detected ambiguities into a hidden system-level hint for the main LLM.

    This message is injected into the conversation as a system instruction so
    the LLM can ask sharper clarification questions. The user never sees this
    text directly.
    """
    ambiguities = result.get("ambiguities", [])
    if not ambiguities:
        return ""

    lines = [
        "AMBIGUITY DETECTED IN USER MESSAGE — ask clarification BEFORE proposing any changes or sequence.",
        "Do NOT reveal this analysis to the user. Instead, use it to ask ONE focused clarification question.",
        "",
    ]

    for i, amb in enumerate(ambiguities, 1):
        amb_type = amb.get("type", "unknown")
        desc = amb.get("description", "")
        candidates = amb.get("candidates", [])
        question = amb.get("suggested_question", "")

        type_labels = {
            "referential": "REFERENTIAL — multiple objects match",
            "omission": "OMISSION — required info missing",
            "implicit_constraint": "IMPLICIT CONSTRAINT — unstated side-effect",
            "spatial": "SPATIAL — ambiguous position reference",
        }
        label = type_labels.get(amb_type, amb_type.upper())

        lines.append(f"  {i}. [{label}]")
        lines.append(f"     {desc}")
        if candidates:
            lines.append(f"     Candidates: {', '.join(str(c) for c in candidates)}")
        if question:
            lines.append(f"     Suggested question: {question}")
        lines.append("")

    lines.append(
        "INSTRUCTIONS: Pick the MOST important ambiguity above and ask the user "
        "ONE clear, specific question to resolve it. Use the suggested question "
        "as a starting point but phrase it naturally. Do NOT mention that you "
        "detected ambiguity or ran an analysis. Just ask the question as if it's "
        "your own reasoning. Do NOT output a changes or sequence block until all "
        "ambiguities are resolved."
    )

    return "\n".join(lines)