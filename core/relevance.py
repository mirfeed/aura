"""
core/relevance.py — relevance scoring for incoming Discord messages.

A RelevanceScore is built by summing additive signal weights, then clamping
the result to [0.0, 1.0] and adding a small random noise term.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

FRUSTRATION_WORDS: List[str] = [
    "ugh",
    "wtf",
    "frustrated",
    "annoying",
    "hate",
    "broken",
    "stupid",
    "ridiculous",
    "argh",
    "ffs",
    "why is",
    "doesn't work",
    "not working",
    "keeps failing",
    # Russian
    "блин",
    "бля",
    "ну почему",
    "не работает",
    "сломалось",
    "надоело",
    "раздражает",
    "wtf",
    "кринж",
]

EXCITEMENT_WORDS: List[str] = [
    "omg",
    "wow",
    "amazing",
    "insane",
    "incredible",
    "finally",
    "yesss",
    "hype",
    "let's go",
    "letsgo",
    "🎉",
    "🔥",
    "❤️",
    "holy",
    "no way",
    # Russian
    "вау",
    "офигеть",
    "обалдеть",
    "ого",
    "наконец",
    "ура",
    "топ",
    "красава",
    "огонь",
]

# Question-like patterns (lowercased match against message body)
_QUESTION_PATTERNS: List[str] = [
    # English
    "anyone",
    "someone",
    "help",
    "think",
    "opinion",
    "what do u",
    "what do you",
    # Russian — direct address / questions
    "расскажи", "привет", "кто ты", "как ты", "помоги",
    "что думаешь", "знаешь", "эй", "ау", "слушай",
    # Russian — question words that start sentences
    "что ты", "чем ты", "где ты", "когда ты", "зачем ты",
    "что тебе", "что тебя", "чего ты", "чего тебе",
    "как ты", "какой ты", "какая ты", "какие ты",
    "почему ты", "откуда ты", "куда ты", "о чём",
    "что любишь", "что нравится", "что думаешь",
    "любишь", "нравится", "хочешь", "умеешь", "можешь",
    "расскажи", "объясни", "посоветуй", "помоги",
    "ты знаешь", "ты умеешь", "ты можешь", "ты любишь",
]


@dataclass
class RelevanceResult:
    """
    The output of :func:`score`.

    Attributes
    ----------
    score:
        Final clamped relevance value in [0.0, 1.0].
    breakdown:
        Mapping of signal name → contribution for debugging / logging.
    """

    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


def score(
    content: str,
    bot_name: str,
    mentioned_bot: bool,
    is_reply_to_bot: bool,
    last_bot_spoke: Optional[datetime],
) -> RelevanceResult:
    """
    Compute a relevance score for an incoming message.

    Parameters
    ----------
    content:
        Raw text of the Discord message (not yet lowercased).
    bot_name:
        The bot's display name as configured (used for name-mention check).
    mentioned_bot:
        True if the Discord message contained an @mention of the bot user.
    is_reply_to_bot:
        True if the message is a reply to one of the bot's own messages.
    last_bot_spoke:
        UTC datetime of the bot's last message in this channel, or None.

    Returns
    -------
    RelevanceResult
        Score and per-signal breakdown.
    """
    breakdown: Dict[str, float] = {}
    raw: float = 0.0
    lower = content.lower()

    # Direct @mention
    if mentioned_bot:
        breakdown["direct_mention"] = 0.65
        raw += 0.65

    # Reply to bot
    if is_reply_to_bot:
        breakdown["reply_to_bot"] = 0.55
        raw += 0.55

    # Name mention (only add if not already credited for direct mention).
    # Учитываем и кириллицу: «Аура», «ауре» и т.п. (config.VOICE_NAME_ALIASES).
    from utils.names import mentions_bot_name

    if not mentioned_bot and (bot_name.lower() in lower or mentions_bot_name(content)):
        breakdown["name_mention"] = 0.40
        raw += 0.40

    # Question detected
    has_question_mark = "?" in content
    has_question_word = any(pat in lower for pat in _QUESTION_PATTERNS)
    if has_question_mark or has_question_word:
        breakdown["question_detected"] = 0.30
        raw += 0.30

    # Emotional keywords
    has_frustration = any(w in lower for w in FRUSTRATION_WORDS)
    has_excitement = any(w in lower for w in EXCITEMENT_WORDS)
    if has_frustration or has_excitement:
        breakdown["emotional_keyword"] = 0.20
        raw += 0.20

    # Bot recently active in this channel (within 5 minutes)
    if last_bot_spoke is not None:
        elapsed = (datetime.now(tz=timezone.utc) - last_bot_spoke).total_seconds()
        if elapsed <= 300:
            breakdown["recent_activity"] = 0.15
            raw += 0.15

    # Random noise
    noise = random.uniform(-0.05, 0.05)
    breakdown["noise"] = noise
    raw += noise

    # Clamp
    final = max(0.0, min(1.0, raw))

    return RelevanceResult(score=final, breakdown=breakdown)
