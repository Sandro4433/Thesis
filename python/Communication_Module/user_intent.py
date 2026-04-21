"""
user_intent.py — Classify user replies as yes/no/finish/other.

Provides deterministic regex-based classifiers plus an LLM fallback
for ambiguous replies (typos, non-canonical phrasings).
"""
from __future__ import annotations

import logging
import re
from typing import Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


# ── Regex-based classifiers ──────────────────────────────────────────────────

def _contains_word(text: str, words) -> bool:
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", text):
            return True
    return False


def is_finish(text: str) -> bool:
    t = text.strip().lower()

    if t in ("done", "end", "quit", "exit", "save", "finalize"):
        return True

    if _contains_word(t, ["finished"]):
        m = re.search(r"\b(finished)\b", t)
        if m:
            before = t[: m.start()].strip()
            after = t[m.end() :].strip().lstrip(".,!?").strip()
            if before or after:
                return False
        return True

    finish_phrases = [
        "last step", "that's all", "thats all", "i'm done", "im done",
        "we're done", "were done", "finish up", "wrap up",
    ]
    if any(p in t for p in finish_phrases):
        return True

    if _contains_word(t, ["finish", "finalize"]):
        m = re.search(r"\b(finish|finalize)\b", t)
        if m:
            rest = t[m.end() :].strip().lstrip(".,!?").strip()
            if rest:
                return False
        return True

    return False


def is_yes(text: str) -> bool:
    t = text.strip().lower()
    if t in (
        "yes", "y", "yep", "yeah", "yup", "sure", "ok", "okay",
        "correct", "right", "good", "confirm", "confirmed", "go",
        "go ahead", "proceed", "approve", "approved", "great", "perfect",
        "looks good", "that's right", "thats right", "that's correct",
        "thats correct", "sounds good", "all good", "agreed",
    ):
        return True
    return False


def is_no(text: str) -> bool:
    t = text.strip().lower()
    if t in (
        "no", "n", "nope", "nah", "negative", "reject", "rejected",
        "cancel", "discard", "undo", "back", "wrong", "incorrect",
        "not right", "not correct", "not good", "that's wrong",
        "thats wrong",
    ):
        return True
    return False


# ── LLM fallback classifier ──────────────────────────────────────────────────

def classify_pending_reply(
    client: OpenAI,
    model: str,
    user_input: str,
    pending_summary: str,
) -> str:
    """
    Use LLM to classify an ambiguous reply as 'yes', 'no', or 'other'.

    Called only when the regex classifiers are inconclusive.
    Returns one of: 'yes', 'no', 'other'.
    """
    prompt = (
        f"The assistant just proposed: {pending_summary}\n\n"
        f"The user replied: \"{user_input}\"\n\n"
        "Is the user confirming (yes), rejecting (no), or doing something else (other)?\n"
        "Reply with exactly one word: yes, no, or other."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        if "yes" in answer:
            return "yes"
        if "no" in answer:
            return "no"
        return "other"
    except Exception as exc:
        logger.warning("classify_pending_reply failed: %s", exc)
        return "other"


def resolve_pending_reply(
    client: OpenAI,
    model: str,
    user_input: str,
    pending_summary: str,
) -> Tuple[str, str]:
    """
    Determine what the user means in response to a pending proposal.

    Uses regex fast-path then LLM fallback.  If still genuinely unclear,
    asks the user directly and returns their clarification.

    Returns (decision, effective_user_input) where decision is
    'yes', 'no', or 'other', and effective_user_input may have been
    updated by a clarification exchange.
    """
    if is_yes(user_input):
        return "yes", user_input
    if is_no(user_input):
        return "no", user_input
    if is_finish(user_input):
        return "other", user_input

    decision = classify_pending_reply(client, model, user_input, pending_summary)
    if decision in ("yes", "no"):
        return decision, user_input

    # Genuinely unclear — ask the user to clarify
    print(
        f'\nSorry, I couldn\'t tell if "{user_input}" means yes or no to '
        f"{pending_summary}.\n"
        "Please type 'yes' to confirm, 'no' to reject, "
        "or rephrase your instruction:"
    )
    clarification = input("YOU: ").strip()
    if is_yes(clarification):
        return "yes", clarification
    if is_no(clarification):
        return "no", clarification
    return "other", clarification
