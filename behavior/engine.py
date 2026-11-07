"""
behavior/engine.py — decision engine that determines whether, when, and how
the bot should reply to a given message.

The engine is stateless itself; all mutable state is passed in via
BotState and read from MessageContext / RelevanceResult.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import config
from behavior.state import BotState, Mood
from behavior import schedule
from core.relevance import RelevanceResult

if TYPE_CHECKING:
    # Imported only for static analysis; avoided at runtime to break the
    # engine ↔ message_handler circular dependency.
    from core.message_handler import MessageContext


# ---------------------------------------------------------------------------
# Mood label mapping
# ---------------------------------------------------------------------------

# Maps each Mood value to a human-readable description injected into the
# LLM system prompt so the model can reflect the bot's current affect.
_MOOD_LABELS = {
    Mood.NEUTRAL:   "обычное, ровное",
    Mood.CURIOUS:   "любопытно, хочется расспросить и поспорить",
    Mood.TIRED:     "устала, отвечаешь короче и ленивее",
    Mood.AMUSED:    "весело, тянет подкалывать и шутить",
    Mood.WITHDRAWN: "задумчивая, говоришь меньше обычного",
    Mood.IRRITATED: "слегка не в духе, отвечаешь суше обычного, но без грубости",
    Mood.CHEERFUL:  "хорошее настроение, ты мягче и разговорчивее обычного",
}


def mood_label(state: BotState) -> str:
    """Человеческое описание настроения + причина, если она есть."""
    label = _MOOD_LABELS.get(state.mood, "обычное")
    if state.mood_reason:
        label += f" (потому что {state.mood_reason})"
    if state.energy < 0.35:
        label += "; сил мало"
    return label


@dataclass
class ReplyDecision:
    """
    The outcome of a single call to :meth:`DecisionEngine.decide`.

    Attributes
    ----------
    should_reply:
        True if the bot should send a reply.
    delay_seconds:
        How long to wait (simulated thinking time) before starting to type.
    typing_duration:
        How many seconds to show the typing indicator.
    reason:
        Short human-readable label explaining the decision (for logging).
    mood_label:
        Human-readable description of the bot's current mood, suitable for
        injection into the LLM system prompt via ``context.mood_label``.
    """

    should_reply: bool
    delay_seconds: float
    typing_duration: float
    reason: str
    mood_label: str = ""
    use_thinking: bool = False


class DecisionEngine:
    """
    Probabilistic reply decision engine.

    The engine combines relevance score, bot energy, mood modifiers, and
    streak dampening to compute a final reply probability, then rolls
    against it.

    No mutable state is stored on the engine itself.  Construct once and
    reuse freely across messages.
    """

    def decide(
        self,
        context: "MessageContext",
        relevance: RelevanceResult,
        state: BotState,
    ) -> ReplyDecision:
        """
        Decide whether to reply to the message described by *context*.

        Parameters
        ----------
        context:
            Structured representation of the incoming Discord message.
        relevance:
            Relevance scoring result from :mod:`core.relevance`.
        state:
            Current mutable bot state; **mutated** by this method.

        Returns
        -------
        ReplyDecision
            Contains the decision, timing values, reason label, and the
            current mood description for prompt injection.
        """
        # Resolve mood label once — used in both early-exit and normal paths.
        mood_label_ = mood_label(state)

        # Always update state counters regardless of final decision.
        state.record_seen()

        # Compute shared timing values used whether or not we reply.
        delay_seconds = self._compute_delay(state, context)
        typing_duration = self._compute_typing_duration()

        # Determine whether the message warrants extended reasoning.
        use_thinking = self._should_think(context)

        # --- Hard pass: DM channel → always respond, no mention required ---
        if context.is_dm:
            state.random_mood_shift()
            state.record_reply(context.channel_id, context.author_id)
            return ReplyDecision(
                should_reply=True,
                delay_seconds=delay_seconds,
                typing_duration=typing_duration,
                reason="dm_always_reply",
                mood_label=mood_label_,
                use_thinking=use_thinking,
            )

        # --- Hard pass: direct mention or reply to bot → always respond ---
        is_direct = (
            "direct_mention" in relevance.breakdown
            or "reply_to_bot" in relevance.breakdown
            or "name_mention" in relevance.breakdown  # «Аура, ...» — её позвали по имени
        )
        # Неприятным людям она отвечает не всегда, даже если зовут её прямо.
        if (is_direct or getattr(context, "addressed_to_me", None)) and self._snubs(context):
            state.random_mood_shift()
            return ReplyDecision(
                should_reply=False,
                delay_seconds=delay_seconds,
                typing_duration=typing_duration,
                reason="disliked_ignore",
                mood_label=mood_label_,
                use_thinking=False,
            )

        if is_direct:
            state.random_mood_shift()
            state.record_reply(context.channel_id, context.author_id)
            return ReplyDecision(
                should_reply=True,
                delay_seconds=delay_seconds,
                typing_duration=typing_duration,
                reason="direct_mention_or_reply",
                mood_label=mood_label_,
                use_thinking=use_thinking,
            )

        # --- Hard pass: продолжение разговора ---
        # Тот же человек пишет снова вскоре после её ответа, и проверка
        # (core.message_handler) решила, что пишут именно ей.
        if getattr(context, "addressed_to_me", None):
            state.random_mood_shift()
            state.record_reply(context.channel_id, context.author_id)
            return ReplyDecision(
                should_reply=True,
                delay_seconds=delay_seconds,
                typing_duration=typing_duration,
                reason="conversation_followup",
                mood_label=mood_label_,
                use_thinking=use_thinking,
            )

        # --- Hard gate: cooldown ---
        if state.is_on_cooldown(context.channel_id):
            state.random_mood_shift()
            return ReplyDecision(
                should_reply=False,
                delay_seconds=delay_seconds,
                typing_duration=typing_duration,
                reason="cooldown",
                mood_label=mood_label_,
                use_thinking=False,
            )

        # --- Build probability ---
        # BASE_RESPONSE_PROBABILITY is the operator-set starting point.
        # relevance.score and state.energy shift it up or down from there,
        # so the configured value is actually honoured instead of ignored.
        base_prob: float = config.BASE_RESPONSE_PROBABILITY
        # Relevance shifts probability: above 0.5 boosts, below 0.5 reduces.
        base_prob += (relevance.score - 0.5) * 0.4
        # Energy scales down when the bot is tired.
        base_prob *= state.energy

        # Streak dampening — avoid dominating the conversation.
        if state.message_streak >= 3:
            base_prob *= 0.4

        # Mood multipliers.
        if state.mood == Mood.WITHDRAWN:
            base_prob *= 0.3
        elif state.mood == Mood.CURIOUS:
            base_prob = min(1.0, base_prob * 1.3)

        # Boredom nudge — bot hasn't spoken in a while.
        if state.messages_seen_since_last_reply > 10:
            base_prob = min(1.0, base_prob + 0.1)

        # Relationship modifier — close friends get more engagement,
        # strangers get slightly less.
        rel = context.relationship_score
        if rel > 0.6:
            base_prob = min(1.0, base_prob + 0.10)
        elif rel < 0.1:
            base_prob = max(0.0, base_prob - 0.05)

        # Отношение к человеку: тем, кто ей неприятен, она отвечает реже.
        aff = getattr(context, "affinity", 0.0)
        if aff <= -0.4:
            base_prob *= 0.6
        elif aff >= 0.5:
            base_prob = min(1.0, base_prob + 0.1)
        if state.mood == Mood.IRRITATED:
            base_prob *= 0.8
        # Распорядок дня: утром сонная и отвечает реже, ночью — её время.
        base_prob *= schedule.reply_factor()

        # Clamp to valid probability range.
        final_prob = max(0.0, min(1.0, base_prob))

        # --- Roll ---
        roll = random.random()
        should_reply = roll < final_prob

        rel_tag = f"rel={rel:.2f}"
        reason = (
            f"probability_pass [{rel_tag}]"
            if should_reply
            else f"probability_fail [{rel_tag}]"
        )

        # --- Mutate state ---
        state.random_mood_shift()
        if should_reply:
            state.record_reply(context.channel_id, context.author_id)

        return ReplyDecision(
            should_reply=should_reply,
            delay_seconds=delay_seconds,
            typing_duration=typing_duration,
            reason=reason,
            mood_label=mood_label_,
            use_thinking=use_thinking if should_reply else False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snubs(context: "MessageContext") -> bool:
        """Игнор неприятного человека: с отношением -2.5 — в ~25% случаев, с -10 — в ~70%."""
        aff = getattr(context, "affinity", 0.0)
        if aff > -0.25 or getattr(context, "is_dm", False):
            return False
        chance = config.DISLIKED_IGNORE_CHANCE + (min(0.0, aff + 0.25) * -0.6)
        return random.random() < min(0.9, chance)

    def _should_think(self, context: "MessageContext") -> bool:
        """
        Determine whether the message is complex enough to warrant
        extended model reasoning (think=True in the Ollama payload).

        Heuristics (any one is sufficient):
        - Message is a question with more than 8 words.
        - Message contains keywords that signal a request for reasoning,
          advice, explanation, or opinion.
        - Message is longer than 80 characters.
        """
        content = context.content.strip()
        word_count = len(content.split())
        is_question = content.endswith("?")
        is_long = len(content) > 80

        _THINKING_KEYWORDS = (
            "почему", "зачем", "как думаешь", "что думаешь",
            "объясни", "расскажи", "посоветуй", "помоги",
            "что лучше", "сравни", "как правильно", "как считаешь",
            "why", "explain", "what do you think", "how do you",
            "should i", "what's better", "help me", "advise",
        )
        lower = content.lower()
        has_keyword = any(kw in lower for kw in _THINKING_KEYWORDS)

        if is_question and word_count > 8:
            return True
        if is_long and (is_question or has_keyword):
            return True
        if has_keyword and word_count > 6:
            return True
        return False

    def _compute_delay(self, state: BotState, context: Any = None) -> float:
        """
        Пауза до начала ответа: заметить сообщение + прочитать его (дольше для длинного и для картинки).
        Уставшая — ленивее; близким отвечает чуть быстрее.
        """
        if schedule.binge_title():
            # Зачиталась: замечает сообщение не сразу, а когда оторвётся от книги.
            lo, hi = config.BINGE_REPLY_DELAY_S
            return random.uniform(lo, hi)
        notice = random.uniform(4.0, 9.0) if state.mood == Mood.TIRED else random.uniform(0.8, 2.5)
        text = getattr(context, "content", "") or ""
        read = min(8.0, len(text) / random.uniform(18.0, 30.0))  # ~20-30 символов в секунду
        if getattr(context, "attachments", None):
            read += random.uniform(1.5, 3.5)  # рассмотреть картинку
        if (getattr(context, "affinity", 0.0) or 0.0) >= 0.5:
            notice *= 0.6
        return notice + read

    def _compute_typing_duration(self) -> float:
        """
        Estimate a realistic typing duration based on a fake word count.

        Assumes roughly 0.055 seconds per word.
        """
        fake_word_count = random.randint(20, 100)
        return fake_word_count * 0.055
