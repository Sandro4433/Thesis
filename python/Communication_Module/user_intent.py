"""
user_intent.py — Classify user replies as yes/no/finish/other.

Extracted from API_Main.py.  Provides deterministic regex-based classifiers
plus an LLM fallback for ambiguous replies (typos, non-canonical phrasings).

Used by the main conversation loop and the update-scene dialogue to decide
whether to confirm, reject, or pass through user input.
"""
from __future__ import annotations

import re

from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
# Regex-based classifiers
# ═══════════════════════════════════════════════════════════════════════════════

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
        # "finished" alone = session end. "finished" inside a longer
        # sentence (e.g. "green should be finished first") = task instruction.
        m = re.search(r"\b(finished)\b", t)
        if m:
            before = t[:m.start()].strip()
            after  = t[m.end():].strip().lstrip(".,!?").strip()
            if before or after:
                return False  # part of a longer instruction
        return True

    finish_phrases = [
        "last step", "that's all", "thats all", "i'm done", "im done",
        "we're done", "were done", "finish up", "wrap up",
    ]
    if any(p in t for p in finish_phrases):
        return True

    # "finish" alone = session end. "finish [something]" = task instruction.
    if _contains_word(t, ["finish", "finalize"]):
        m = re.search(r"\b(finish|finalize)\b", t)
        if m:
            rest = t[m.end():].strip().lstrip(".,!?").strip()
            if rest:
                return False
        return True

    return False


def _is_pure_short(text: str, max_words: int = 5) -> bool:
    """True if the text is short enough to be a pure confirmation/rejection."""
    return len(text.split()) <= max_words


def is_yes(text: str) -> bool:
    t = text.strip().lower()
    if not _is_pure_short(t):
        return False
    return (
        _contains_word(t, ["yes", "ok", "okay", "confirm", "confirmed", "sure", "correct"])
        or any(p in t for p in ["go ahead", "do it", "looks good"])
        or t == "y"
    )


def is_no(text: str) -> bool:
    t = text.strip().lower()
    if not _is_pure_short(t):
        return False
    return (
        _contains_word(t, ["no", "nope", "cancel", "reject", "wrong", "redo"])
        or t == "n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-based classifier (fallback for ambiguous replies)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_pending_reply(
    client: OpenAI,
    model: str,
    user_input: str,
    pending_summary: str,
) -> str:
    """
    Classify the user's reply to a 'Confirm?' prompt.

    `pending_summary` is a short human-readable description of what is pending
    so the classifier can disambiguate replies like "correct" vs "change it".

    Returns one of: "yes", "no", "other", "unclear".
    On any error, returns "unclear" so the caller falls back to asking the
    user directly — we never silently guess wrong on a commit-to-disk action.
    """
    prompt = f"""You classify a user's reply to a yes/no confirmation prompt.

The assistant just proposed: {pending_summary}
Then asked the user: "Confirm?"

The user replied: "{user_input}"

Classify the reply as EXACTLY one of these four labels:

  YES      — user wants to confirm / accept / commit the proposal.
             Includes clear affirmatives and obvious typos/variants:
             "yes", "yep", "yeah", "yup", "ok", "sure", "correct",
             "zes" (typo of yes), "yess", "mhm", "go ahead", "do it", "k", etc.

  NO       — user wants to reject / cancel / redo the proposal.
             Includes clear negatives and obvious typos:
             "no", "nope", "nah", "n0", "cancel", "wrong", "redo", etc.

  OTHER    — user typed a substantive new instruction or follow-up that is
             clearly NOT an attempt at yes/no. E.g. "actually, make container 2
             input instead", "what about green parts?", "change the batch size".

  UNCLEAR  — the reply is short but does NOT clearly mean yes, no, or a new
             instruction. E.g. "maybe", "idk", "hmm", "?", gibberish, or a
             typo too far from any known word to be sure.

Rules:
- If the reply is a plausible typo of yes/no (edit distance 1-2), classify as
  YES or NO, not UNCLEAR. Be generous with obvious typos.
- If you are not confident, return UNCLEAR. UNCLEAR is the SAFE default.
- Never return YES or NO for a reply that contains new content/constraints.

Reply with EXACTLY one word: YES, NO, OTHER, or UNCLEAR."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        # Strip trailing punctuation the model may add
        answer = re.sub(r"[^A-Z]", "", answer)
        if answer == "YES":
            return "yes"
        if answer == "NO":
            return "no"
        if answer == "OTHER":
            return "other"
        return "unclear"
    except Exception as e:
        print(f"  (Could not classify reply: {e} — asking user directly)")
        return "unclear"


def resolve_pending_reply(
    client: OpenAI,
    model: str,
    user_input: str,
    pending_summary: str,
) -> tuple[str, str]:
    """
    Resolve a user reply to a pending 'Confirm?' prompt.

    Pipeline:
      1. Fast regex path — is_yes / is_no catch the common cases for free.
      2. If regex is silent, ask the LLM classifier.
      3. If the classifier returns 'unclear', prompt the user directly in
         the CLI up to 2 times before giving up and treating as 'other'.

    Returns (decision, final_text) where:
      - decision is "yes", "no", or "other"
      - final_text is the text to use if decision == "other"
        (the original user_input, or the clarified reply)
    """
    # 1. Regex fast path
    if is_yes(user_input):
        return "yes", user_input
    if is_no(user_input):
        return "no", user_input

    # 2. LLM classifier for anything regex didn't catch
    verdict = classify_pending_reply(client, model, user_input, pending_summary)
    if verdict in ("yes", "no", "other"):
        return verdict, user_input

    # 3. Unclear — ask the user directly, without touching the main conversation
    current_input = user_input
    for _ in range(2):
        print(
            "ASSISTANT:\n"
            f"I'm not sure I understood \"{current_input}\". "
            "Did you mean to confirm the proposal (yes), reject it (no), "
            "or change something else?\n"
        )
        clarified = input("YOU: ").strip()
        if not clarified:
            continue
        # Regex pass on the clarified reply
        if is_yes(clarified):
            return "yes", clarified
        if is_no(clarified):
            return "no", clarified
        # LLM pass on the clarified reply
        verdict = classify_pending_reply(client, model, clarified, pending_summary)
        if verdict in ("yes", "no"):
            return verdict, clarified
        if verdict == "other":
            return "other", clarified
        # Still unclear — loop once more, then fall through
        current_input = clarified

    # Gave up — treat as a new instruction so the main LLM can respond.
    return "other", current_input
