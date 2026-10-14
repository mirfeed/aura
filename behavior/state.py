"""
behavior/state.py — mutable runtime state for the Discord companion bot.

BotState is the single source of truth for the bot's current mood, energy,
channel activity, and reply-streak counters.  It is intentionally *not* a
singleton; the caller constructs one instance at startup and passes it
explicitly wherever it is needed (no global mutable state).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Set, Tuple


class Mood(Enum):
    """Represents the bot's current emotional disposition."""

    NEUTRAL = "neutral"
    CURIOUS = "curious"
    TIRED = "tired"
    AMUSED = "amused"
    WITHDRAWN = "withdrawn"
    IRRITATED = "irritated"
    CHEERFUL = "cheerful"


@dataclass
class BotState:
    """
    Encapsulates all runtime state that influences the bot's behaviour.

    Attributes
    ----------
    mood:
        Current emotional disposition; affects reply probability multipliers.
    energy:
        Float in [0.0, 1.0].  Decays over time; restored when the bot
        chooses to be quiet.
    focus_channel_id:
        Channel the bot is currently paying most attention to, or None.
    last_spoke:
        Maps channel_id → UTC datetime of the most recent reply in that
        channel.
    message_streak:
        Number of consecutive replies sent without a gap.
    messages_seen_since_last_reply:
        Counter that drives the "boredom nudge" in the decision engine.
    """

    mood: Mood = Mood.NEUTRAL
    energy: float = 0.8
    focus_channel_id: Optional[int] = None
    last_spoke: Dict[int, datetime] = field(default_factory=dict)
    message_streak: int = 0
    messages_seen_since_last_reply: int = 0

    # Tracks when the bot last *initiated* a DM with each user (user_id → UTC datetime).
    last_proactive_dm: Dict[int, datetime] = field(default_factory=dict)

    # Почему у неё такое настроение (заполняется рефлексией).
    mood_reason: str = ""
    mood_since: Optional[datetime] = None

    # Set of user IDs that currently have an open DM channel with the bot.
    # Populated lazily in the background loop; used to skip fetch_user calls.
    known_dm_users: Set[int] = field(default_factory=set)

    # С кем она последней разговаривала в канале: channel_id → (author_id, UTC время).
    last_partner: Dict[int, Tuple[str, datetime]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Cooldown helpers
    # ------------------------------------------------------------------

    def is_on_cooldown(self, channel_id: int, seconds: int = 8) -> bool:
        """
        Return True if the bot spoke in *channel_id* fewer than *seconds* ago.

        Parameters
        ----------
        channel_id:
            Discord channel snowflake ID.
        seconds:
            Minimum number of seconds that must have elapsed since the last
            reply before the bot is considered off-cooldown.
        """
        last = self.last_spoke.get(channel_id)
        if last is None:
            return False
        elapsed = (datetime.now(tz=timezone.utc) - last).total_seconds()
        return elapsed < seconds

    def is_followup(self, channel_id: int, author_id: str, window: int) -> bool:
        """True, если она только что отвечала этому же человеку в этом канале."""
        partner = self.last_partner.get(channel_id)
        if partner is None or partner[0] != author_id:
            return False
        return (datetime.now(tz=timezone.utc) - partner[1]).total_seconds() <= window

    # ------------------------------------------------------------------
    # Recording activity
    # ------------------------------------------------------------------

    def record_reply(self, channel_id: int, author_id: Optional[str] = None) -> None:
        """
        Update state to reflect that a reply was just sent in *channel_id*.

        - Sets ``last_spoke[channel_id]`` to the current UTC time.
        - Increments ``message_streak``.
        - Resets ``messages_seen_since_last_reply`` to zero.
        """
        self.last_spoke[channel_id] = datetime.now(tz=timezone.utc)
        if author_id is not None:
            self.last_partner[channel_id] = (author_id, self.last_spoke[channel_id])
        self.message_streak += 1
        self.messages_seen_since_last_reply = 0
        self.decay_energy(0.015)

    def record_seen(self) -> None:
        """
        Increment the counter of messages observed since the last reply.

        Called on every message that passes bot / channel filtering,
        regardless of whether the bot decides to reply.
        """
        self.messages_seen_since_last_reply += 1
        # Серия «ответов подряд» прерывается, когда другие поговорили без неё.
        if self.messages_seen_since_last_reply >= 3:
            self.message_streak = 0

    # ------------------------------------------------------------------
    # Energy management
    # ------------------------------------------------------------------

    def decay_energy(self, amount: float = 0.02) -> None:
        """
        Reduce energy by *amount*, flooring at 0.1.

        Parameters
        ----------
        amount:
            How much energy to subtract (default 0.02).
        """
        self.energy = max(0.1, self.energy - amount)

    def restore_energy(self, amount: float = 0.05) -> None:
        """
        Increase energy by *amount*, ceiling at 1.0.

        Parameters
        ----------
        amount:
            How much energy to add back (default 0.05).
        """
        self.energy = min(1.0, self.energy + amount)

    # ------------------------------------------------------------------
    # Mood
    # ------------------------------------------------------------------

    def random_mood_shift(self) -> None:
        """
        5 % chance on each call to randomly shift the bot's mood.

        When a shift occurs, the new mood is chosen uniformly at random
        from all Mood values (including the current one, meaning an
        apparent "no change" is possible).
        """
        # Настроение «с причиной» держится хотя бы 20 минут.
        if self.mood_reason and self.mood_since is not None:
            if (datetime.now(tz=timezone.utc) - self.mood_since).total_seconds() < 1200:
                return
        if random.random() < 0.02:
            self.mood = random.choice([Mood.NEUTRAL, Mood.CURIOUS, Mood.AMUSED, Mood.TIRED, Mood.CHEERFUL])
            self.mood_reason = ""

    def set_mood(self, mood: str, reason: str = "") -> None:
        """Выставить настроение по итогам рефлексии."""
        try:
            self.mood = Mood(mood)
        except ValueError:
            return
        self.mood_reason = reason.strip()[:120]
        self.mood_since = datetime.now(tz=timezone.utc)

    def tick(self) -> None:
        """Раз в минуту: энергия понемногу восстанавливается."""
        self.restore_energy(0.01)
