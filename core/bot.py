"""
core/bot.py — Discord client subclass.

CompanionBot wires together the Discord event loop with the message handler
and background task loop.  No commands framework is used; all behaviour is
event-driven via discord.Client callbacks.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import discord

import config
from behavior import schedule
from behavior.engine import DecisionEngine
from behavior.state import BotState
from core.message_handler import MessageHandler
from core.reminders import ReminderStore
from memory.bookfinder import BookFinder
from memory.webreader import WebReader
from memory.library import Library
from memory.life import LifeStore
from pipeline.text import TextPipeline
from utils.logger import anonymise_author_id, get_logger

if TYPE_CHECKING:
    from memory.store import MemoryStore
    from voice.manager import VoiceManager

_logger = get_logger("bot.client")

# How often the background task loop fires (seconds).
_BACKGROUND_INTERVAL: int = 60

# Новая просьба передать (а не ответ на «кому?»).
import re as _re

# Слова с большой буквы, которые точно не имена (начало фразы и т.п.).
_NOT_NAMES = {"это", "вот", "ну", "да", "нет", "его", "её", "ее", "она", "он", "они", "мой", "моя", "мне", "а", "и",
              "так", "там", "тут", "ник", "зовут", "скинь", "напиши", "отправь", "передай", "пожалуйста", "спасибо"}
# «скинь ИМ», «отправь ВСЕМ» — не имена, кому именно — из прошлой просьбы.
_RELAY_PRONOUNS = {"им", "всем", "они", "них", "ним", "ему", "ей", "их", "остальным", "всем остальным"}
# «сообщение про печеньку», «то сообщение», «это» — не сам текст, а ссылка на него.
_VAGUE_WHAT_RE = _re.compile(r"^\s*(?:(?:мо[её]|то|это|тот|та|мой)\s+)?(?:сообщени\w*|текст|то|это)?\s*(?:про|о|об|насч[её]т)\b"
                             r"|^\s*(?:мо[её]|то|это)\s+сообщени\w*\s*$", _re.IGNORECASE)

_RELAY_WORDS = _re.compile(
    r"\b(?:напиши|передай|скажи|отправь|отправить|отправишь|черкни|маякни|спроси|скинь|скинуть|перешли|"
    r"переслать|разошли|разослать)\b", _re.IGNORECASE)


class CompanionBot(discord.Client):
    """
    Discord client for the AI companion bot.

    Responsibilities
    ----------------
    * Configures required Intents (message_content, guilds, voice_states).
    * Implements ``on_ready`` to log connection details.
    * Implements ``on_message`` to route messages through the pipeline,
      skipping bots and optional channel whitelisting.
    * Runs a background asyncio task stub every 60 seconds (placeholder for
      Day 6 proactive behaviour).

    Parameters
    ----------
    state:
        Shared :class:`~behavior.state.BotState` instance.
    engine:
        :class:`~behavior.engine.DecisionEngine` instance.
    pipeline:
        :class:`~pipeline.text.TextPipeline` instance.
    memory:
        :class:`~memory.store.MemoryStore` instance.
    """

    def __init__(
        self,
        state: BotState,
        engine: DecisionEngine,
        pipeline: TextPipeline,
        memory: "MemoryStore",
        voice: Optional["VoiceManager"] = None,
    ) -> None:
        """Construct the bot with injected collaborators."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.voice_states = True

        super().__init__(intents=intents)

        self._state = state
        self._engine = engine
        self._pipeline = pipeline
        self._memory = memory
        self._voice = voice

        # MessageHandler is created lazily in on_ready once we know our ID.
        self._message_handler: Optional[MessageHandler] = None
        self._background_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._mind_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._catchup_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._life_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._reminders: Optional[ReminderStore] = None
        self._life: Optional[LifeStore] = None
        self._library: Optional[Library] = None
        self._next_read_min: float = 0.0
        self._last_read_try: float = 0.0
        self._binge_book: Optional[int] = None  # id книги, которую читает взапой
        self._last_book_share: Dict[int, datetime] = {}  # кому и когда последний раз писала о книге
        self._next_life_event_h: float = 0.0
        # Сомневалась, отправлять ли сообщение, и спросила совета: ID советчика → что и кому.
        self._pending_advice: Dict[int, Dict[str, Any]] = {}
        # Когда по чьей просьбе писала другим (для лимита в час): ID просящего → моменты.
        self._relay_log: Dict[int, List[Tuple[datetime, int]]] = {}  # кто просил → (когда, кому)
        # Что именно она отправила по чьей просьбе: ID просящего → [(когда, кому, текст)].
        self._relay_history: Dict[int, List[Any]] = {}
        # Просьба передать, но она не поняла, кому: ID просящего → {"target", "message", "at", "tries"}.
        self._pending_relay: Dict[int, Dict[str, Any]] = {}
        # Последняя внятная просьба переслать от каждого: {"what", "request", "names", "at"} —
        # чтобы «скинь им ещё раз про печеньку» через 20 минут знало, ЧТО и КОМУ.
        self._relay_last: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Discord events
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        """
        Called when the bot has logged in and the Discord cache is ready.

        Logs every connected guild and its text channels.  Instantiates
        the :class:`~core.message_handler.MessageHandler` now that we know
        ``self.user.id``.  Starts the background task loop.
        """
        assert self.user is not None, "on_ready fired but self.user is None"

        self._message_handler = MessageHandler(
            bot_user_id=self.user.id,
            state=self._state,
            engine=self._engine,
            pipeline=self._pipeline,
            memory=self._memory,
        )
        self._message_handler.advice_hook = self._chat_advice_hook
        self._message_handler.relay_hook = self._relay_request
        self._message_handler.relay_facts_hook = self._relay_facts
        if config.REMINDERS_ENABLED:
            if self._reminders is None:
                self._reminders = ReminderStore(lambda: self._memory.social._db)
                await self._reminders.init()
            self._message_handler.reminders = self._reminders
        if config.LIFE_EVENTS_ENABLED and self._life is None:
            self._life = LifeStore(lambda: self._memory.social._db)
            await self._life.init()
            self._pipeline.life = self._life
        if config.READING_ENABLED and self._library is None:
            self._library = Library(lambda: self._memory.social._db, config.LIBRARY_DIR, self._memory.embed)
            await self._library.init()
            await self._library.refresh()
            self._pipeline.library = self._library
        if self._library is not None:
            self._message_handler.library = self._library
            self._message_handler.book_hook = self._find_recommended_book
            self._message_handler.book_link_hook = self._open_book_link

        guild_info: List[str] = []
        for guild in self.guilds:
            channel_names = [c.name for c in guild.text_channels]
            guild_info.append(f"{guild.name} (id={guild.id}, channels={channel_names})")

        if config.TEST_MODE:
            allowed = [g.name for g in self.guilds if g.id in config.TEST_GUILD_IDS]
            _logger.warning("TEST MODE: only these servers + DMs", extra={"data": {"servers": allowed}})
        _logger.info(
            "Bot ready",
            extra={
                "data": {
                    "bot_name": config.BOT_NAME,
                    "user_tag": str(self.user),
                    "guilds": guild_info,
                }
            },
        )

        # Start background loop (on_ready can fire again after a reconnect).
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._background_loop())
        if self._mind_task is None or self._mind_task.done():
            self._mind_task = asyncio.create_task(self._mind_loop())
        # Напоминания, сон / подъём, дни рождения.
        if self._life_task is None or self._life_task.done():
            self._life_task = asyncio.create_task(self._life_loop())
        # Личка, пришедшая, пока её не было в сети.
        if config.DM_CATCHUP and (self._catchup_task is None or self._catchup_task.done()):
            self._catchup_task = asyncio.create_task(self._catch_up_dms())

        # Day 4/5: warm up Whisper + Silero in the background.
        if self._voice is not None and config.VOICE_PRELOAD:
            asyncio.create_task(self._voice.preload())

    async def on_message(self, message: discord.Message) -> None:
        """
        Called for every message visible to the bot.

        Filters applied (in order):
        1. Skip messages from the bot itself.
        2. Skip messages from any other bot.
        3. If ``CHANNEL_WHITELIST`` is non-empty, skip messages from
           channels not in the whitelist.
        4. Delegate to :class:`~core.message_handler.MessageHandler`.

        Parameters
        ----------
        message:
            The incoming Discord message.
        """
        # Guard: handler must be initialised.
        if self._message_handler is None:
            return

        # Skip self.
        if self.user is not None and message.author.id == self.user.id:
            return

        # Skip all bots.
        if message.author.bot:
            return

        # Тестовый режим: чужие серверы — мимо (не читает и не запоминает).
        if not config.guild_allowed(message.guild.id if message.guild else None):
            return

        # DM channels bypass the channel whitelist — personal conversations
        # always reach the handler.
        is_dm = isinstance(message.channel, discord.DMChannel)

        # Channel whitelist filter (guild messages only).
        if not is_dm and config.CHANNEL_WHITELIST and message.channel.id not in config.CHANNEL_WHITELIST:
            return

        # Voice commands: «Аура, зайди в войс» / «Аура, выйди из войса».
        if self._voice is not None and not is_dm and await self._maybe_voice_command(message):
            return

        # Ответ на её вопрос «отправлять ли?» — сначала решить, потом ответить как обычно.
        late_note = ""
        if is_dm and message.author.id in self._pending_advice:
            try:
                late_note = await self._resolve_advice(message)
            except Exception as exc:  # noqa: BLE001
                _logger.error("Advice handling failed", extra={"data": {"error": str(exc)}}, exc_info=True)

        # Delegate to the message handler (swallows its own exceptions).
        try:
            await self._message_handler.handle(message, late_note=late_note)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Unhandled exception in message handler",
                extra={"data": {"error": str(exc)}},
                exc_info=True,
            )

    async def _maybe_voice_command(self, message: discord.Message) -> bool:
        """Return True if the message was a voice join/leave command and was handled."""
        from voice.manager import mentions_bot_name, parse_voice_command

        assert self._voice is not None
        content = message.content or ""
        from utils.names import is_bot_mentioned

        mentioned = self.user is not None and is_bot_mentioned(message, self.user.id)
        replied = (
            isinstance(getattr(message.reference, "resolved", None), discord.Message)
            and self.user is not None
            and message.reference.resolved.author.id == self.user.id  # type: ignore[union-attr]
        )
        if not (mentioned or replied or mentions_bot_name(content)):
            return False
        command = parse_voice_command(content)
        if command is None:
            return False
        try:
            return await self._voice.handle_command(message, command)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Voice command failed",
                extra={"data": {"command": command, "error": str(exc)}},
                exc_info=True,
            )
            return True

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Forward voice state changes to the voice manager (auto-leave etc.)."""
        if self._voice is None:
            return
        try:
            if config.guild_allowed(member.guild.id if member.guild else None):
                await self._voice.on_voice_state_update(member, before, after)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Error in on_voice_state_update",
                extra={"data": {"error": str(exc)}},
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Background task
    # ------------------------------------------------------------------

    async def _background_loop(self) -> None:
        """
        Background task that fires every 60 seconds.

        On each tick, for every user ID listed in config.PROACTIVE_USER_IDS,
        the bot may decide to initiate a DM conversation if:
          - enough time has passed since the last proactive DM to that user
            (config.PROACTIVE_MIN_INTERVAL seconds), and
          - a random roll passes config.PROACTIVE_DM_PROBABILITY.

        The message is generated via the text pipeline with an initiator
        prompt so the model knows it is starting the conversation, not
        responding to one.
        """
        await self.wait_until_ready()
        _logger.info(
            "Background task loop started",
            extra={"data": {"interval_seconds": _BACKGROUND_INTERVAL}},
        )
        while not self.is_closed():
            await asyncio.sleep(_BACKGROUND_INTERVAL)
            _logger.debug(
                "Background tick",
                extra={"data": {"guilds_connected": len(self.guilds)}},
            )
            self._state.tick()
            # Имена и прозвища могли поправить в people_ui — подхватываем без перезапуска.
            try:
                await self._memory.social._reload_names()
            except Exception:  # noqa: BLE001
                pass
            if config.PROACTIVE_USER_IDS:
                await self._maybe_initiate_dms()
            self._expire_advice()
            if config.PROACTIVE_FRIENDS:
                try:
                    await self._maybe_write_to_friends()
                except Exception as exc:  # noqa: BLE001
                    _logger.error("Friend DM failed", extra={"data": {"error": str(exc)}}, exc_info=True)
            if config.PROACTIVE_TRUSTED:
                try:
                    await self._maybe_share_with_trusted()
                except Exception as exc:  # noqa: BLE001
                    _logger.error(
                        "Proactive share failed",
                        extra={"data": {"error": str(exc)}},
                        exc_info=True,
                    )

    async def _mind_loop(self) -> None:
        """
        «Осмысление» разговоров: когда в канале накопились реплики и стало
        тихо, а модель свободна — обновить сводку, факты о людях, её
        обещания, отношение к людям и настроение (memory/social.py).
        """
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(15)
            try:
                if self._pipeline.busy or self._pipeline.seconds_since_last_request < 8:
                    continue
                # Пока в войсе живой разговор — модель нужна для ответов, не для «раздумий».
                # Сидит в войсе — тоже нет: раздумье держит модель ~1.5 мин, и первое «привет» ждало бы его.
                if self._voice is not None and (
                    self._voice.in_voice or self._voice.recently_active(config.VOICE_REFLECT_PAUSE_S)
                ):
                    continue
                channels = await self._memory.social.channels_ready_for_reflection()
                if channels:
                    await self._memory.social.reflect(channels[0], self._pipeline.call_json, self._state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "Reflection failed",
                    extra={"data": {"error": str(exc)}},
                    exc_info=True,
                )

    async def _maybe_initiate_dms(self) -> None:
        """
        Iterate over PROACTIVE_USER_IDS and optionally send an unprompted DM.

        Each user is checked independently; a single tick may initiate DMs
        to multiple users if the probability roll passes for each.
        """
        if self._message_handler is None:
            return

        now = datetime.now(tz=timezone.utc)

        for user_id in config.PROACTIVE_USER_IDS:
            # Cooldown check — don't spam.
            last = self._state.last_proactive_dm.get(user_id)
            if last is not None:
                elapsed = (now - last).total_seconds()
                if elapsed < config.PROACTIVE_MIN_INTERVAL:
                    continue

            # Probability roll.
            if random.random() > config.PROACTIVE_DM_PROBABILITY:
                continue

            try:
                user = await self.fetch_user(user_id)
                dm_channel = await user.create_dm()

                ctx = await self._initiator_context(user, dm_channel)
                response = await self._pipeline.generate_initiator(ctx)

                self._state.last_proactive_dm[user_id] = now
                await self._deliver_or_ask(user, dm_channel, ctx, response)

            except discord.Forbidden:
                _logger.warning(
                    "Cannot send DM — user has DMs disabled",
                    extra={"data": {"user_id": user_id}},
                )
            except discord.NotFound:
                _logger.warning(
                    "User not found for proactive DM",
                    extra={"data": {"user_id": user_id}},
                )
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "Unexpected error in proactive DM",
                    extra={"data": {"user_id": user_id, "error": str(exc)}},
                    exc_info=True,
                )

    async def _initiator_context(self, user: discord.User, dm_channel: discord.DMChannel):
        """Контекст для сообщения, которое она пишет первой (никто ей не писал)."""
        from behavior.engine import mood_label
        from core.message_handler import MessageContext  # local import to avoid circularity

        ctx = MessageContext(
            message_id=0,
            content="",  # no incoming message — bot is initiating
            author_id=anonymise_author_id(user.id),
            raw_author_id=user.id,
            author_display_name=user.display_name,
            channel_id=dm_channel.id,
            guild_id=None,
            timestamp=datetime.now(tz=timezone.utc),
            is_dm=True,
        )
        # Fetch memory snippets to personalise the opener.
        ctx.memory_snippets = await self._memory.recall(ctx)
        ctx.relationship_score = await self._memory.get_relationship_score(ctx.author_id)
        ctx.mood_label = mood_label(self._state)
        ctx.social = await self._memory.social.build_context(
            dm_channel.id, (ctx.author_id, user.display_name, user.id), [], ""
        )
        return ctx

    # ------------------------------------------------------------------
    # Личка, пришедшая, пока её не было в сети
    # ------------------------------------------------------------------

    async def _catch_up_dms(self) -> None:
        """
        После появления в сети: пройтись по личкам знакомых людей и ответить на то,
        что пришло без неё.  Новым считается всё, что новее последней записи этой
        лички в памяти и новее её последнего сообщения там.
        """
        await asyncio.sleep(max(0, config.DM_CATCHUP_DELAY_S))
        if self._message_handler is None or self.user is None:
            return
        if schedule.is_asleep():
            _logger.info("DM catch-up postponed — asleep", extra={"data": {}})
            return
        social = self._memory.social
        user_ids: List[int] = []
        for uid in (
            list(config.OWNER_USER_IDS)
            + list(config.PROACTIVE_USER_IDS)
            + await social.dm_partners()
            + [p[2] for p in await social.trusted_people((config.PROACTIVE_TRUST_MIN - 0.5) / 10)]
        ):
            if uid and uid != self.user.id and uid not in user_ids:
                user_ids.append(uid)

        now = datetime.now(tz=timezone.utc)
        lookback = now - timedelta(hours=config.DM_CATCHUP_LOOKBACK_H)
        found = 0
        for uid in user_ids[:60]:
            try:
                user = self.get_user(uid) or await self.fetch_user(uid)
                if user.bot:
                    continue
                dm = user.dm_channel or await user.create_dm()
                after = lookback
                last = await social.last_line(dm.id)
                if last is not None and last[0] > after:
                    after = last[0]
                recent = [m async for m in dm.history(limit=30, after=after, oldest_first=True)]
                # Только то, что пришло после её последнего сообщения.
                missed: List[discord.Message] = []
                for m in recent:
                    if m.author.id == self.user.id:
                        missed = []
                    elif not m.author.bot:
                        missed.append(m)
                if not missed:
                    continue
                found += 1
                age_h = (now - missed[-1].created_at).total_seconds() / 3600
                reply = age_h <= config.DM_CATCHUP_REPLY_MAX_AGE_H
                _logger.info(
                    "DM catch-up",
                    extra={"data": {"user_id": uid, "missed": len(missed), "reply": reply,
                                    "age_hours": round(age_h, 1)}},
                )
                await self._message_handler.catch_up(missed, reply=reply)
                if reply:
                    await asyncio.sleep(random.uniform(4.0, 12.0))  # не всем сразу в одну секунду
            except (discord.Forbidden, discord.NotFound):
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "DM catch-up failed",
                    extra={"data": {"user_id": uid, "error": str(exc)}},
                    exc_info=True,
                )
        _logger.info("DM catch-up done", extra={"data": {"checked": len(user_ids), "with_new": found}})

    # ------------------------------------------------------------------
    # Сама пишет тем, кому полностью доверяет
    # ------------------------------------------------------------------

    @staticmethod
    def _in_quiet_hours() -> bool:
        spec = (config.PROACTIVE_QUIET_HOURS or "").strip()
        if not spec or "-" not in spec:
            return False
        try:
            start, end = (int(x) % 24 for x in spec.split("-", 1))
        except ValueError:
            return False
        from pipeline.text import _local_now

        hour = _local_now().hour
        return start <= hour < end if start <= end else (hour >= start or hour < end)

    async def _maybe_share_with_trusted(self) -> None:
        """
        Людям с доверием 10/10 она иногда пишет сама: мнение, новость, мысль.

        Условия: не тихие часы; модель и войс свободны; её последнее сообщение в этой
        личке старше PROACTIVE_TRUSTED_MIN_INTERVAL; переписка не идёт прямо сейчас;
        прошлое её сообщение не висит без ответа.  Не больше одного сообщения за тик.
        """
        if self._message_handler is None or self.user is None or self._in_quiet_hours():
            return
        if self._pipeline.busy or self._pipeline.seconds_since_last_request < 30:
            return
        if self._voice is not None and self._voice.recently_active(120):
            return
        social = self._memory.social
        people = [
            p for p in await social.trusted_people((config.PROACTIVE_TRUST_MIN - 0.5) / 10)
            if p[2] not in config.PROACTIVE_USER_IDS and p[2] != self.user.id
        ]
        random.shuffle(people)
        now = datetime.now(tz=timezone.utc)
        for key, name, uid in people:
            if random.random() > config.PROACTIVE_TRUSTED_PROBABILITY:
                continue
            last_sent = self._state.last_proactive_dm.get(uid)
            if last_sent and (now - last_sent).total_seconds() < config.PROACTIVE_TRUSTED_MIN_INTERVAL:
                continue
            try:
                user = self.get_user(uid) or await self.fetch_user(uid)
                if user.bot:
                    continue
                dm = user.dm_channel or await user.create_dm()
                mine = await social.last_bot_line_at(dm.id)
                if mine and (now - mine).total_seconds() < config.PROACTIVE_TRUSTED_MIN_INTERVAL:
                    continue
                last = await social.last_line(dm.id)
                if last is not None:
                    when, is_bot = last
                    if (now - when).total_seconds() < 1800:
                        continue  # переписываются прямо сейчас — это уже не «по своей инициативе»
                    if is_bot and (now - when).total_seconds() < config.PROACTIVE_TRUSTED_UNANSWERED_H * 3600:
                        continue  # не пишет второй раз подряд, пока не ответили

                ctx = await self._initiator_context(user, dm)
                from pipeline.text import _local_now

                tz = _local_now().tzinfo
                history: List[str] = []
                async for m in dm.history(limit=10):
                    if not m.content:
                        continue
                    who = config.BOT_NAME if m.author.id == self.user.id else m.author.display_name
                    history.append(f"[{m.created_at.astimezone(tz).strftime('%d.%m %H:%M')}] {who}: {m.content[:400]}")
                history.reverse()
                ctx.recent_channel_history = history
                news = await social.recent_summaries(24, exclude_channel=dm.id)
                complaints = await social.annoying_dms(
                    48, config.PROACTIVE_COMPLAIN_AFFINITY / 10, exclude_channel=dm.id
                )

                self._state.last_proactive_dm[uid] = now  # даже если промолчит — не дёргать модель каждую минуту
                response = await self._pipeline.generate_share(ctx, news, complaints)
                if not response:
                    _logger.info("Proactive share skipped (nothing to say)", extra={"data": {"user_id": uid}})
                    return
                await self._message_handler._send_with_typing(dm, response, random.uniform(1.5, 3.0))
                await self._memory.log_bot_reply(ctx, response)
                _logger.info(
                    "Proactive share sent",
                    extra={"data": {"user_id": uid, "name": name, "response_length": len(response)}},
                )
                return
            except (discord.Forbidden, discord.NotFound):
                _logger.warning("Cannot DM trusted user", extra={"data": {"user_id": uid}})
                continue

    # ------------------------------------------------------------------
    # Сомневается, отправлять ли, — спрашивает совета у близкого
    # ------------------------------------------------------------------

    async def _deliver_or_ask(self, user: discord.User, dm: discord.DMChannel, ctx: Any, draft: str) -> None:
        """
        Сообщение, которое она сама решила написать *user*.  Сначала прикидывает, стоит ли:
        да — отправляет, нет — молчит, сомневается — спрашивает совета у человека с доверием 10.
        """
        if not draft or self._message_handler is None:
            return
        verdict = "yes"
        if config.PROACTIVE_ASK_ADVICE:
            ctx.recent_channel_history = await self._dm_history(dm, 6)
            verdict = await self._pipeline.doubt_check(ctx, draft)
        _logger.info("Proactive doubt check", extra={"data": {"user_id": user.id, "verdict": verdict}})
        if verdict == "no":
            return
        if verdict == "unsure" and await self._ask_advice(user, dm, draft):
            return
        if verdict == "unsure":
            return  # спросить некого — лучше промолчать
        await self._message_handler._send_with_typing(dm, draft, random.uniform(1.5, 3.0))
        await self._memory.log_bot_reply(ctx, draft)
        _logger.info("Proactive DM sent", extra={"data": {"user_id": user.id, "response_length": len(draft)}})

    async def _ask_advice(self, target: discord.User, target_dm: discord.DMChannel, draft: str) -> bool:
        """Спросить у близкого (доверие 10), отправлять ли черновик.  False — спросить некого."""
        if self.user is None or self._message_handler is None:
            return False
        uid = await self._pick_advisor(exclude=(target.id,))
        if uid is None:
            return False
        try:
            advisor = self.get_user(uid) or await self.fetch_user(uid)
            dm = advisor.dm_channel or await advisor.create_dm()
            actx = await self._initiator_context(advisor, dm)
            actx.recent_channel_history = await self._dm_history(dm, 8)
            text = await self._pipeline.ask_advice_text(actx, target.display_name, draft)
            await self._message_handler._send_with_typing(dm, text, random.uniform(1.5, 3.0))
            await self._memory.log_bot_reply(actx, text)
        except (discord.Forbidden, discord.NotFound):
            return False
        self._pending_advice[uid] = {
            "kind": "send",
            "target_id": target.id,
            "target_name": target.display_name,
            "draft": draft,
            "asked_at": datetime.now(tz=timezone.utc),
        }
        _logger.info("Asked for advice", extra={"data": {"advisor_id": uid, "target_id": target.id}})
        return True

    async def _relay_request(self, message: discord.Message, ctx: Any) -> str:
        """
        «Аура, напиши Паше и Даше, что …».  Выполняет, если доверяет просящему (RELAY_TRUST_MIN).
        Возвращает пометку для её ответа просящему; '' — это не такая просьба или не от доверенного.
        В пометке точно сказано, кому сообщение ушло, а кому нет, — чтобы она не приписывала себе лишнего.
        """
        if self.user is None or self._message_handler is None:
            return ""
        social = self._memory.social
        trust = await social.get_trust(ctx.author_id, message.author.id)
        if trust < (config.RELAY_TRUST_MIN - 0.5) / 10:
            return ""
        now = datetime.now(tz=timezone.utc)
        text_in = message.content or ""
        anonymous = False
        others = [m for m in message.mentions if not m.bot and m.id not in (self.user.id, message.author.id)]

        # Ждала уточнения «кому?» — ищем имена в этом сообщении.
        pending = self._pending_relay.get(message.author.id)
        if pending is not None and (now - pending["at"]).total_seconds() > 1800:
            self._drop_pending_relay(message.author.id)
            pending = None
        what = ""
        request = text_in
        unknown: List[str] = []          # имена, которых она не знает
        targets: Dict[int, str] = {}     # raw_id → имя
        learned: Dict[int, str] = {}     # raw_id → новое прозвище
        if pending is not None:
            found_keys = social.find_names_in_text(text_in, by_sound=True)
            if not others and not found_keys:
                # Похоже на ник/юзернейм (латиница, цифры) или на имя с большой буквы — поискать на сервере.
                for word in _re.findall(r"[\w.!|'-]{3,40}", text_in):
                    if _re.search(r"[a-z0-9]", word.lower()) or (word[0].isupper() and word.lower() not in _NOT_NAMES):
                        hit = await self._find_member(word)
                        if hit is not None:
                            others.append(type("M", (), {"id": hit[0], "display_name": hit[1]})())
            if others or found_keys:
                what, request = pending["message"], pending["request"]
                # Могли заодно уточнить, что передать («им надо скинуть про печеньку»).
                reparsed = await self._pipeline.parse_relay(message.author.display_name, text_in,
                                                            ctx.recent_channel_history)
                if reparsed and reparsed.get("message"):
                    what, request = reparsed["message"], text_in
                anonymous = bool(pending.get("anonymous"))
                for m in others:
                    targets[m.id] = m.display_name
                for key in found_keys:
                    hit = await social.raw_id_for_key(key)
                    if hit is not None:
                        targets[hit[0]] = hit[1]
                # Одно неизвестное имя и один найденный человек — значит, это его прозвище.
                if len(pending["unknown"]) == 1 and len(targets) == 1:
                    learned[next(iter(targets))] = pending["unknown"][0]
                self._drop_pending_relay(message.author.id)
            elif not _RELAY_WORDS.search(text_in):
                pending["tries"] += 1
                names = ", ".join(pending["unknown"])
                if pending["tries"] >= 3:
                    self._drop_pending_relay(message.author.id)
                    return (f"ты так и не поняла, кто такие: {names} — им сообщение НЕ отправлено; "
                            "не говори, что передала, честно скажи, что не поняла, кому")
                return (f"тебя просили передать сообщение, но ты не знаешь, кто такие: {names} — им НЕ отправлено, "
                        "не говори, что передала; попроси написать его ник или юзернейм в Discord")

        if not targets:
            parsed = await self._pipeline.parse_relay(message.author.display_name, text_in,
                                                      ctx.recent_channel_history)
            if parsed is None:
                return ""
            self._message_handler.relay_recent[message.author.id] = time.monotonic()
            what = parsed["message"] or text_in
            anonymous = bool(parsed.get("anonymous"))
            names = [n for n in parsed["targets"] if n.strip().lower() not in _RELAY_PRONOUNS]
            last = self._relay_last.get(message.author.id)
            if last is not None and (now - last["at"]).total_seconds() > config.RELAY_FOLLOWUP_S:
                last = None
            if last is not None:
                # «скинь им ещё раз», «ты про печеньку не отправила» — что и кому, берём из прошлой просьбы.
                if _VAGUE_WHAT_RE.search(what) or len(what) < 6:
                    what, request = last["what"], last["request"]
                if not names and not others:
                    names = list(last["names"])
            for m in others:
                targets[m.id] = m.display_name
            for name in names:
                if name.strip().lower() in ("мне", "меня", "я", "себе"):
                    targets[message.author.id] = message.author.display_name  # «и мне тоже»
                    continue
                hit = await social.raw_id_for_name(name)
                if hit is None:
                    key = social.find_name_in_text(name)
                    hit = await social.raw_id_for_key(key) if key else None
                if hit is None:
                    hit = await self._find_member(name)
                if hit is not None:
                    targets[hit[0]] = hit[1]
                elif not others:
                    unknown.append(name)
        targets.pop(self.user.id, None)
        if what and not _VAGUE_WHAT_RE.search(what):
            self._relay_last[message.author.id] = {
                "what": what, "request": request, "at": now,
                "names": list(targets.values()) + unknown,
            }

        recent = [(t, u) for t, u in self._relay_log.get(message.author.id, []) if (now - t).total_seconds() < 3600]
        people = {u for _, u in recent} | set(targets)
        if targets and len(people) > config.RELAY_PER_HOUR:
            return ("тебя просят написать кому-то ещё, но ты уже много раз за этот час передавала сообщения — "
                    "ничего НЕ отправлено; откажись, мягко")

        sent: List[str] = []
        failed: List[str] = []
        for uid, name in targets.items():
            try:
                target = self.get_user(uid) or await self.fetch_user(uid)
                if target.bot:
                    failed.append(f"{name} (бот)")
                    continue
                dm = target.dm_channel or await target.create_dm()
                tctx = await self._initiator_context(target, dm)
                tctx.recent_channel_history = await self._dm_history(dm, 6)
                text = await self._pipeline.relay_text(tctx, message.author.display_name, what, request,
                                                       anonymous=anonymous)
                await self._message_handler._send_with_typing(dm, text, random.uniform(1.0, 2.5))
                await self._memory.log_bot_reply(tctx, text)
                sent.append(f"{name}: «{text}»")
                self._relay_history.setdefault(message.author.id, []).append((now, name, text))
                recent.append((now, uid))
                if uid in learned:
                    await social.add_alias(uid, learned[uid])
                _logger.info("Relay sent", extra={"data": {"from": message.author.id, "to": uid}})
            except discord.Forbidden:
                failed.append(f"{name} (закрыта личка)")
            except (discord.NotFound, discord.HTTPException):
                failed.append(f"{name} (не получилось)")
        self._relay_log[message.author.id] = recent

        if unknown:
            self._pending_relay[message.author.id] = {
                "unknown": unknown, "message": what, "request": request, "at": now, "tries": 0,
                "anonymous": anonymous,
            }
            self._message_handler.relay_pending_ids.add(message.author.id)
            _logger.info("Relay target unknown", extra={"data": {"from": message.author.id, "unknown": unknown}})

        parts = []
        if sent:
            parts.append("ты написала в личку ТОЛЬКО этим людям — " + "; ".join(sent))
        if failed:
            parts.append("НЕ получилось написать: " + ", ".join(failed))
        if unknown:
            parts.append("ты не знаешь, кто такие: " + ", ".join(unknown)
                         + " — им НЕ отправлено: ты не нашла их ни у себя в памяти, ни на своих серверах. "
                         "Скажи честно и попроси юзернейм в Discord (как в профиле) или ID; если их и так дали — "
                         "значит, у тебя с этим человеком нет общего сервера, а без него в личку не написать")
        if not parts:
            return ""
        return "; ".join(parts) + ". Говори только правду о том, кому написала"

    async def _relay_facts(self, message: discord.Message) -> str:
        """Правда для модели: что она за последний час отправила по просьбам этого человека (или ничего)."""
        if not config.RELAY_ENABLED:
            return ""
        key = anonymise_author_id(message.author.id)
        if await self._memory.social.get_trust(key, message.author.id) < (config.RELAY_TRUST_MIN - 0.5) / 10:
            return ""
        now = datetime.now(tz=timezone.utc)
        recent = [h for h in self._relay_history.get(message.author.id, []) if (now - h[0]).total_seconds() < 3600]
        self._relay_history[message.author.id] = recent
        if recent:
            done = "; ".join(f"{n}: «{t}»" for _, n, t in recent[-5:])
            return (f"по просьбам {message.author.display_name} за последний час ты отправила в личку ТОЛЬКО это — "
                    f"{done}. Никому другому ты ничего не писала; не говори, что отправила то, чего нет")
        return (f"за последний час ты НИКОМУ ничего не отправляла по просьбам {message.author.display_name}. "
                "Если спросят, отправила ли, — честно скажи, что нет")

    async def _find_member(self, name: str) -> Optional[Tuple[int, str]]:
        """
        Человека нет в её памяти — поискать на общих серверах по юзернейму, нику и звучанию
        («suffer9328», «Батыру» → batyr).  Найден один — запомнить с этим прозвищем.
        Подходят несколько — None: лучше переспросить, чем написать не тому.
        """
        from memory.social import name_forms, name_sound

        raw = (name or "").strip().lstrip("@").strip("<>@!")
        if raw.isdigit() and 17 <= len(raw) <= 20:
            # Дали Discord-ID — найти аккаунт напрямую (написать в личку всё равно можно,
            # только если у бота с этим человеком есть общий сервер).
            try:
                user = self.get_user(int(raw)) or await self.fetch_user(int(raw))
            except (discord.NotFound, discord.HTTPException):
                return None
            if user.bot:
                return None
            await self._memory.social.remember_member(anonymise_author_id(user.id), user.id,
                                                      user.display_name, user.name)
            _logger.info("Member found by id", extra={"data": {"user_id": user.id, "name": user.name}})
            return user.id, user.display_name
        if len(raw) < 3 or len(raw) > 40:
            return None
        forms = name_forms(raw)
        sounds = {name_sound(f) for f in forms}
        sounds = {s for s in sounds if len(s) >= 3}
        # query_members ищет по началу юзернейма/ника: само имя, его начало и начало звучания латиницей.
        queries = {raw.lower()} | {f[:3] for f in forms if len(f) >= 3} | {s[:2] for s in sounds}
        found: Dict[int, Tuple[float, str, str]] = {}  # id → (насколько похоже, имя, какой формой нашли)
        await self._scan_members(queries, forms, found)
        if not found and sounds:
            # «Тирозин» → ti…, а ник — Tyrozin: по двум буквам не нашлось — по первой.
            await self._scan_members({s[:1] for s in sounds}, forms, found)
        if not found:
            return None
        ranked = sorted(found.items(), key=lambda kv: kv[1][0], reverse=True)
        if len(ranked) > 1 and ranked[0][1][0] - ranked[1][1][0] < 0.08:
            _logger.info("Member lookup ambiguous", extra={"data": {
                "name": raw, "candidates": [v[1] for _, v in ranked[:4]]}})
            return None
        uid, (score, display, form) = ranked[0]
        alias = raw if score >= 1.0 or not _re.search("[а-яё]", raw.lower()) else form.capitalize()
        await self._memory.social.remember_member(anonymise_author_id(uid), uid, display, alias)
        _logger.info("Member found on server", extra={"data": {"name": raw, "user_id": uid,
                                                              "display": display, "score": round(score, 2)}})
        return uid, display

    async def _scan_members(self, queries: set, forms: List[str],
                            found: Dict[int, Tuple[float, str, str]]) -> None:
        """Участники общих серверов, чьё имя начинается с queries, — оценить похожесть на forms."""
        import difflib

        from memory.social import name_sound

        for guild in self.guilds:
            for q in queries:
                try:
                    members = await guild.query_members(query=q, limit=100)
                except Exception:  # noqa: BLE001  (таймаут, сервер недоступен)
                    continue
                for m in members:
                    if m.bot or (self.user and m.id == self.user.id):
                        continue
                    for label in {m.name, m.display_name, m.global_name or "", m.nick or ""}:
                        if not label:
                            continue
                        lab, lsnd = label.lower(), name_sound(label)
                        for form in forms:
                            fs = name_sound(form)
                            score = 0.0
                            if lab == form:
                                score = 1.0
                            elif len(fs) >= 3 and lsnd == fs:
                                score = 0.95
                            elif len(fs) >= 4 and abs(len(lsnd) - len(fs)) <= 2:
                                r = difflib.SequenceMatcher(None, fs, lsnd).ratio()
                                score = r if r >= 0.84 else 0.0
                            if score > found.get(m.id, (0.0, "", ""))[0]:
                                found[m.id] = (score, m.display_name, form)

    def _drop_pending_relay(self, uid: int) -> None:
        self._pending_relay.pop(uid, None)
        if self._message_handler is not None:
            self._message_handler.relay_pending_ids.discard(uid)

    async def _pick_advisor(self, exclude: tuple = ()) -> Optional[int]:
        """Кого спросить: человек с доверием 10, которого сейчас ни о чём не спрашивает; недавние — первыми."""
        if self.user is None:
            return None
        advisors = [
            p for p in await self._memory.social.trusted_people((config.PROACTIVE_TRUST_MIN - 0.5) / 10)
            if p[2] not in exclude and p[2] != self.user.id and p[2] not in self._pending_advice
        ]
        if not advisors:
            return None
        partners = await self._memory.social.dm_partners()
        advisors.sort(key=lambda p: partners.index(p[2]) if p[2] in partners else len(partners))
        return advisors[0][2]

    async def _chat_advice_hook(self, message: discord.Message, ctx: Any, kind: str, my_reply: str) -> bool:
        """
        Вопрос в общем чате.  kind='ask' — не уверена: спросить совета в личке и пока не отвечать;
        kind='complain' — уже ответила, теперь жалуется близкому.  True — сообщение в личку ушло.
        """
        if self._message_handler is None:
            return False
        if message.author.id in self._pending_advice:
            return False
        uid = await self._pick_advisor(exclude=(message.author.id,))
        if uid is None:
            return False
        try:
            advisor = self.get_user(uid) or await self.fetch_user(uid)
            dm = advisor.dm_channel or await advisor.create_dm()
            actx = await self._initiator_context(advisor, dm)
            actx.recent_channel_history = await self._dm_history(dm, 8)
            text = await self._pipeline.chat_advice_text(
                actx, message.author.display_name, (message.content or "")[:400], kind, my_reply
            )
            await self._message_handler._send_with_typing(dm, text, random.uniform(1.0, 2.5))
            await self._memory.log_bot_reply(actx, text)
        except (discord.Forbidden, discord.NotFound, discord.HTTPException):
            return False
        if kind == "ask":
            self._pending_advice[uid] = {
                "kind": "chat",
                "channel_id": message.channel.id,
                "message_id": message.id,
                "asker": message.author.display_name,
                "question": (message.content or "")[:400],
                "asked_at": datetime.now(tz=timezone.utc),
            }
        _logger.info("Chat advice DM", extra={"data": {"advisor_id": uid, "kind": kind}})
        return True

    async def _answer_in_chat(self, pending: Dict[str, Any], note: str) -> bool:
        """Ответить на тот вопрос из чата (решение уже принято)."""
        if self._message_handler is None:
            return False
        try:
            channel = self.get_channel(pending["channel_id"]) or await self.fetch_channel(pending["channel_id"])
            original = await channel.fetch_message(pending["message_id"])  # type: ignore[union-attr]
        except (discord.Forbidden, discord.NotFound, discord.HTTPException):
            return False
        asyncio.create_task(self._message_handler.handle(original, late_note=note, force_reply=True))
        return True

    async def _resolve_advice(self, message: discord.Message) -> str:
        """Советчик ответил: отправить черновик / ответить в чате или нет.  Возвращает пометку для её ответа."""
        pending = self._pending_advice.get(message.author.id)
        if pending is None or self._message_handler is None:
            return ""
        if pending.get("kind") == "chat":
            asker, question = pending["asker"], pending["question"]
            advice = (message.content or "")[:500]
            verdict = await self._pipeline.classify_chat_advice(advice, asker, question)
            _logger.info("Chat advice received", extra={"data": {"advisor_id": message.author.id, "verdict": verdict}})
            if verdict == "unclear":
                return (f"ты спрашивала совета, отвечать ли {asker} в чате на «{question}», а прямого ответа "
                        "пока нет — ответь на сообщение; можно переспросить, если к месту")
            self._pending_advice.pop(message.author.id, None)
            if verdict == "skip":
                return f"ты спрашивала, отвечать ли {asker} на «{question}»; тебе отсоветовали, и ты не стала отвечать"
            me = message.author.display_name
            ok = await self._answer_in_chat(
                pending, f"ты посоветовалась в личке с {me}, он(а) сказал(а): «{advice}» — учти совет, "
                         f"но отвечай своими словами и не говори, что спрашивала"
            )
            return (f"ты спрашивала, отвечать ли {asker} на «{question}»; тебе посоветовали, и ты сейчас ответишь "
                    "в чате — коротко отреагируй на совет" if ok else
                    f"ты хотела ответить {asker} по совету, но сообщение уже удалили")
        name, draft = pending["target_name"], pending["draft"]
        verdict = await self._pipeline.classify_advice(message.content or "", name, draft)
        _logger.info("Advice received", extra={"data": {"advisor_id": message.author.id, "verdict": verdict}})
        if verdict == "unclear":
            return (f"ты спрашивала, стоит ли написать {name} «{draft}», а прямого ответа пока нет — "
                    "ответь на сообщение; можно переспросить про совет, если к месту")
        self._pending_advice.pop(message.author.id, None)
        if verdict == "skip":
            return f"ты спрашивала, стоит ли написать {name} «{draft}»; тебе отсоветовали, и ты не стала отправлять"
        try:
            target = self.get_user(pending["target_id"]) or await self.fetch_user(pending["target_id"])
            dm = target.dm_channel or await target.create_dm()
            tctx = await self._initiator_context(target, dm)
            await self._message_handler._send_with_typing(dm, draft, random.uniform(1.5, 3.0))
            await self._memory.log_bot_reply(tctx, draft)
            self._state.last_proactive_dm[target.id] = datetime.now(tz=timezone.utc)
            _logger.info("Proactive DM sent after advice", extra={"data": {"user_id": target.id}})
            return f"ты спрашивала, стоит ли написать {name} «{draft}»; тебе посоветовали отправить, и ты отправила"
        except (discord.Forbidden, discord.NotFound):
            return f"ты хотела написать {name}, но у него(неё) закрыта личка — сообщение не ушло"

    def _expire_advice(self) -> None:
        """
        Не ответили вовремя.  Черновик в личку — не отправляет;
        вопрос из чата — решает сама и отвечает.
        """
        now = datetime.now(tz=timezone.utc)
        for uid, p in list(self._pending_advice.items()):
            age = (now - p["asked_at"]).total_seconds()
            if p.get("kind") == "chat":
                if age > config.CHAT_ADVICE_TIMEOUT_MIN * 60:
                    self._pending_advice.pop(uid, None)
                    _logger.info("Chat advice timed out, answering herself", extra={"data": {"advisor_id": uid}})
                    asyncio.create_task(self._answer_in_chat(
                        p, "ты хотела посоветоваться, но ответа не дождалась — решила сама"
                    ))
            elif age > config.PROACTIVE_ADVICE_TIMEOUT_H * 3600:
                self._pending_advice.pop(uid, None)
                _logger.info("Advice timed out, not sending", extra={"data": {"advisor_id": uid}})

    async def _dm_history(self, dm: discord.DMChannel, limit: int) -> List[str]:
        """Последние сообщения лички строками «[дата] Имя: текст», старые сверху."""
        from pipeline.text import _local_now

        tz = _local_now().tzinfo
        lines: List[str] = []
        try:
            async for m in dm.history(limit=limit):
                if m.content:
                    who = config.BOT_NAME if self.user and m.author.id == self.user.id else m.author.display_name
                    lines.append(f"[{m.created_at.astimezone(tz).strftime('%d.%m %H:%M')}] {who}: {m.content[:400]}")
        except (discord.Forbidden, discord.HTTPException):
            pass
        lines.reverse()
        return lines

    async def _maybe_write_to_friends(self) -> None:
        """
        Иногда сама пишет знакомому, кто ей симпатичен и уже писал ей в личку.
        Каждое такое сообщение проходит проверку «стоит ли» (и совет, если сомневается).
        """
        if self._message_handler is None or self.user is None or self._in_quiet_hours():
            return
        if self._pipeline.busy or self._pipeline.seconds_since_last_request < 30:
            return
        if self._voice is not None and self._voice.recently_active(120):
            return
        social = self._memory.social
        partners = set(await social.dm_partners())
        pending_targets = {p.get("target_id") for p in self._pending_advice.values()}
        people = [
            p for p in await social.liked_people(config.PROACTIVE_FRIEND_AFFINITY / 10,
                                                 (config.PROACTIVE_TRUST_MIN - 0.5) / 10)
            if p[2] in partners and p[2] not in config.PROACTIVE_USER_IDS and p[2] not in pending_targets
        ]
        random.shuffle(people)
        now = datetime.now(tz=timezone.utc)
        for _key, _name, uid in people:
            if random.random() > config.PROACTIVE_FRIEND_PROBABILITY:
                continue
            last_sent = self._state.last_proactive_dm.get(uid)
            if last_sent and (now - last_sent).total_seconds() < config.PROACTIVE_FRIEND_MIN_INTERVAL:
                continue
            try:
                user = self.get_user(uid) or await self.fetch_user(uid)
                if user.bot:
                    continue
                dm = user.dm_channel or await user.create_dm()
                mine = await social.last_bot_line_at(dm.id)
                if mine and (now - mine).total_seconds() < config.PROACTIVE_FRIEND_MIN_INTERVAL:
                    continue
                last = await social.last_line(dm.id)
                if last is not None:
                    when, is_bot = last
                    if (now - when).total_seconds() < 1800 or is_bot:
                        continue  # переписываются сейчас или её прошлое сообщение без ответа
                self._state.last_proactive_dm[uid] = now
                ctx = await self._initiator_context(user, dm)
                if last is not None and (now - last[0]).total_seconds() >= config.MISS_AFTER_DAYS * 86400:
                    days = int((now - last[0]).total_seconds() // 86400)
                    ctx.late_note = (f"вы не переписывались {days} дн — ты немного соскучилась; "
                                     "можно так и сказать, легко, без упрёков")
                draft = await self._pipeline.generate_initiator(ctx)
                await self._deliver_or_ask(user, dm, ctx, draft)
                return
            except (discord.Forbidden, discord.NotFound):
                continue

    # ------------------------------------------------------------------
    # Её день: напоминания, сон и подъём, дни рождения
    # ------------------------------------------------------------------

    async def _life_loop(self) -> None:
        await self.wait_until_ready()
        prev_phase = schedule.phase()
        _logger.info("Life loop started", extra={"data": {"phase": prev_phase}})
        while not self.is_closed():
            await asyncio.sleep(20)
            try:
                if self._reminders is not None:
                    for r in await self._reminders.due():
                        await self._reminders.mark_done(r["id"])  # сначала отметить — не повторить при сбое
                        await self._fire_reminder(r)

                phase = schedule.phase()
                if phase != prev_phase:
                    _logger.info("Day phase changed", extra={"data": {"from": prev_phase, "to": phase}})
                    if phase == "asleep":
                        await self._say_goodnight()
                    elif prev_phase == "asleep" and config.DM_CATCHUP:
                        # Проснулась — разобрать личку, пришедшую за ночь.
                        if self._catchup_task is None or self._catchup_task.done():
                            self._catchup_task = asyncio.create_task(self._catch_up_dms())
                    prev_phase = phase

                if config.BIRTHDAYS_ENABLED and phase not in ("asleep",):
                    await self._maybe_congratulate()
                if self._life is not None and phase != "asleep":
                    await self._maybe_life_event()
                if self._library is not None and phase != "asleep":
                    await self._maybe_read()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.error("Life loop tick failed", extra={"data": {"error": str(exc)}}, exc_info=True)

    async def _maybe_read(self) -> None:
        """Свободное время — почитать: кусок книги или несколько страниц комикса."""
        now = datetime.now(tz=timezone.utc)
        binge = bool(schedule.binge_title())
        if self._binge_book and not binge:
            await self._end_binge("время вышло")  # засиделась достаточно — пора отложить
        if not self._next_read_min:
            base = config.BINGE_INTERVAL_MIN if binge else config.READING_INTERVAL_MIN
            self._next_read_min = base * random.uniform(0.67, 1.33)
        last = await self._library.last_session_at()
        if last is not None and (now - last).total_seconds() / 60 < self._next_read_min:
            return
        if time.monotonic() - self._last_read_try < 600:
            return  # прошлая попытка не удалась — не долбить модель каждые 20 секунд
        if self._pipeline.busy or self._pipeline.seconds_since_last_request < (45 if binge else 90):
            return  # она разговаривает — книга подождёт
        if self._voice is not None and (self._voice.in_voice or self._voice.recently_active(300)):
            return  # в войсе с людьми — не до книжки
        if not await self._library.scan() and await self._library.current() is None:
            return  # читать нечего
        book = await self._library.current()
        if book is None:
            return
        self._last_read_try = time.monotonic()
        illus: List[bytes] = []
        prev = await self._library.previous_volume(book)  # следующий том серии — помнит, что было раньше
        if prev:
            book["prev_title"], book["prev_story"] = prev["title"], prev["story"] or ""
        if book.get("url"):
            book = await self._library.ensure_ahead(book, self._web_reader())
            if book["pos"] >= book["total"]:
                return  # новых глав пока нет (или не открылись) — ждём
        try:
            if book["kind"] == "comic":
                pages, pos_to = await asyncio.to_thread(self._library.comic_chunk, book)
                result = await self._pipeline.read_session(book, pages=pages) if pages else None
            else:
                text, pos_to = await asyncio.to_thread(self._library.text_chunk, book)
                illus = await self._library.images_between(book["id"], book["pos"], pos_to)
                result = (await self._pipeline.read_session(book, text=text, illustrations=illus or None)
                          if text else None)
        except Exception as exc:  # noqa: BLE001
            _logger.error("Reading failed", extra={"data": {"book": book["title"], "error": str(exc)}}, exc_info=True)
            return
        if result is None:
            if pos_to >= book["total"]:  # пустой хвост — считать дочитанной
                result = {"summary": "", "story": book["story"], "thought": ""}
            else:
                return
        # Иллюстрации: что на них — в заметки (вспомнит потом); очень понравилась — в любимые.
        loved_pic: Optional[bytes] = None
        for i, ill in enumerate(result.get("illustrations") or []):
            if ill["seen"]:
                await self._library.add_note(book["id"], "illustration", f"Иллюстрация: {ill['seen']}")
            if ill["love"] and book["kind"] != "comic" and i < len(illus):
                loved_pic = loved_pic or illus[i]
                if await self._pipeline.save_book_illustration(illus[i], book["title"], ill["seen"]):
                    _logger.info("Book illustration saved", extra={"data": {"book": book["title"], "seen": ill["seen"]}})
        hooked = int(result.get("hooked") or 0)
        finished = await self._library.save_session(book, pos_to, result["summary"], result["thought"],
                                                    result["story"], hooked)
        self._next_read_min = 0.0
        # Запой: захватило — читает подряд; дочитала или остыла — отпускает.
        if not finished and hooked >= config.BINGE_HOOK and not self._binge_book:
            self._binge_book = book["id"]
            schedule.set_binge(book["title"], now + timedelta(hours=config.BINGE_MAX_H))
            # Без записи в «жизнь»: строка «зачиталась…» в каждом промпте — и она вставляла книгу в любой ответ.
            _logger.info("Binge started", extra={"data": {"book": book["title"], "hooked": hooked}})
        elif self._binge_book and (finished or hooked <= config.BINGE_HOOK - 3 or book["id"] != self._binge_book):
            await self._end_binge("дочитала" if finished else "остыла")
        _logger.info("Reading session", extra={"data": {
            "book": book["title"], "pos": f"{pos_to}/{book['total']}", "thought": result["thought"][:120]}})
        if finished:
            book["story"] = result["story"] or book["story"]
            verdict = await self._pipeline.book_verdict(book)
            if verdict:
                await self._library.set_verdict(book["id"], verdict["opinion"], verdict["rating"])
            # Конспект для долгой памяти (подробность — по тому, как понравилась) и забывание мелочей.
            rating = verdict["rating"] if verdict else 5
            konspekt = await self._pipeline.book_konspekt(book, await self._library.notes_for(book["id"]), rating)
            if konspekt:
                await self._library.store_konspekt(book, konspekt, rating)
            if self._life is not None:
                await self._life.add(f"дочитала «{book['title']}»"
                                     + (f" — {verdict['opinion']}" if verdict and verdict["opinion"] else ""))
            _logger.info("Book finished", extra={"data": {"book": book["title"], "verdict": verdict}})
            if config.READING_SHARE_ENABLED:
                await self._share_reading(book, result, finished=True, verdict=verdict, picture=loved_pic)
        elif config.READING_SHARE_ENABLED and (result.get("share") and hooked >= 6 or loved_pic):
            await self._share_reading(book, result, picture=loved_pic)

    def _web_reader(self) -> WebReader:
        return WebReader(self._pipeline._session, contact=f"https://discord.com/users/{self.user.id}")

    async def _open_book_link(self, message: discord.Message, ctx: Any, url: str) -> None:
        """Скинули ссылку на «читалку» — открыть, положить в закладки и сказать, что получилось."""
        reader = self._web_reader()
        info = await reader.open(url)
        name = ctx.author_display_name
        if info["ok"]:
            book = await self._library.add_web(info["start_url"], info["title"], info["author"], info["kind"], name)
            book = await self._library.ensure_ahead(book, reader, max_chapters=1)
            ok = book["total"] > 0
        else:
            ok = False
        if ok:
            cur = await self._library.current()
            what = "комикс" if info["kind"] == "comic" else "текст"
            when = ("начнёшь прямо сейчас" if cur and cur["id"] == book["id"]
                    else f"возьмёшь следом за «{cur['title']}»" if cur else "скоро начнёшь")
            task = (f"{name} скинул(а) тебе ссылку почитать. Ты открыла: «{info['title']}» на {info['site']} — "
                    f"это {what}, первая глава открылась нормально; ты добавила в закладки и {when}. "
                    "Скажи коротко, в своём стиле. Ты её ещё НЕ читала — не рассказывай, о чём она.")
            fallback = f"открыла «{info['title']}», добавила в закладки 📖"
        else:
            reason = info.get("reason") or "первая глава не открылась"
            task = (f"{name} скинул(а) тебе ссылку почитать, но прочитать не получилось: {reason}. "
                    "Скажи честно и коротко; предложи скинуть файл (epub, fb2, txt, pdf или cbz для комиксов) — "
                    "на многих сайтах есть кнопка «скачать». Не обещай обойти защиту.")
            fallback = "не смогла открыть эту читалку( если есть файл — скинь?"
        _logger.info("Book link", extra={"data": {"url": url, "ok": ok, "info": {k: v for k, v in info.items()
                                                                                 if k != "start_url"}}})
        occ = await self._occasion_ctx(message.channel, message.author)
        text = await self._pipeline.generate_occasion(occ, task, fallback=fallback)
        try:
            await message.reply(text, mention_author=False)
        except discord.HTTPException:
            await message.channel.send(text)
        await self._memory.log_bot_reply(occ, text)

    async def _find_recommended_book(self, message: discord.Message, ctx: Any, req: Dict[str, str]) -> None:
        """Посоветовали книгу — найти в легальных бесплатных библиотеках, положить на полку и сказать."""
        finder = BookFinder(self._pipeline._session, contact=f"https://discord.com/users/{self.user.id}")
        try:
            found = await finder.find(req["title"], req["author"], self._library.dir)
        except Exception as exc:  # noqa: BLE001
            _logger.error("Book search failed", extra={"data": {"error": str(exc)}}, exc_info=True)
            found = None
        name = ctx.author_display_name
        if found:
            book = await self._library.add_file(found["path"], name)
            cur = await self._library.current()
            when = ("начнёшь прямо сейчас" if cur and book and cur["id"] == book["id"]
                    else f"возьмёшь её следом за «{cur['title']}»" if cur else "скоро начнёшь")
            task = (f"{name} посоветовал(а) тебе «{req['title']}». Ты поискала и нашла «{found['title']}»"
                    + (f" ({found['author']})" if found["author"] else "") + f" на {found['source']}, "
                    f"положила на свою полку и {when}. Скажи это коротко, в своём стиле. Ты её ещё НЕ читала — "
                    "не рассказывай, о чём она.")
            fallback = f"нашла «{found['title']}», положила на полку 📚"
        else:
            task = (f"{name} посоветовал(а) тебе «{req['title']}». Ты поискала в бесплатных легальных библиотеках "
                    "(Викитека, Project Gutenberg — там в основном классика, всё, что уже в общественном достоянии) "
                    "и не нашла. Скажи честно и коротко; попроси скинуть файл, если он у человека есть "
                    "(txt, fb2, epub, pdf). Пиратские сайты не предлагай.")
            fallback = f"поискала «{req['title']}» — бесплатно не нашла. если есть файл, скинь?"
        occ = await self._occasion_ctx(message.channel, message.author)
        text = await self._pipeline.generate_occasion(occ, task, fallback=fallback)
        try:
            await message.reply(text, mention_author=False)
        except discord.HTTPException:
            await message.channel.send(text)
        await self._memory.log_bot_reply(occ, text)

    async def _share_reading(self, book: Dict[str, Any], result: Dict[str, Any], finished: bool = False,
                             verdict: Optional[Dict[str, Any]] = None, picture: Optional[bytes] = None) -> None:
        """
        Поделиться впечатлением о прочитанном в личку: тому, кто подарил книгу (он её знает — не спойлер),
        иначе владельцу.  Не чаще READING_SHARE_MIN_H и не поверх своего же неотвеченного сообщения.
        """
        if self.user is None or self._message_handler is None:
            return
        uid: Optional[int] = None
        if book.get("added_by"):
            hit = await self._memory.social.raw_id_for_name(book["added_by"])
            uid = hit[0] if hit else None
        if uid is None and config.OWNER_USER_IDS:
            uid = config.OWNER_USER_IDS[0]
        if uid is None:
            return
        now = datetime.now(tz=timezone.utc)
        last = self._last_book_share.get(uid)
        if not finished and last and (now - last).total_seconds() < config.READING_SHARE_MIN_H * 3600:
            return
        try:
            user = self.get_user(uid) or await self.fetch_user(uid)
            dm = user.dm_channel or await user.create_dm()
        except (discord.NotFound, discord.HTTPException):
            return
        line = await self._memory.social.last_line(dm.id)
        if line is not None and line[1] and (now - line[0]).total_seconds() < 6 * 3600 and not finished:
            return  # её прошлое сообщение ещё без ответа — не заваливать
        gift = f" (её тебе подарил(а) {user.display_name} — он(а) её знает, спойлеров тут нет)" if book.get("added_by") else ""
        if finished:
            task = (f"Ты только что дочитала «{book['title']}»{gift}. Твоё мнение: "
                    f"{(verdict or {}).get('opinion') or result.get('thought')}, оценка {(verdict or {}).get('rating') or '—'}/10. "
                    f"Напиши {user.display_name} в личку — сама, как подруге: что дочитала и что думаешь, коротко и "
                    "эмоционально, одно-два сообщения.")
        else:
            moment = result.get("share") or result.get("summary")
            task = (f"Ты читаешь «{book['title']}»{gift} и только что прочитала: {result.get('summary')}\n"
                    f"Тебе захотелось поделиться: {moment}"
                    + (" (и там была иллюстрация, которая тебе очень понравилась — ты её прикладываешь)" if picture else "")
                    + f"\nНапиши {user.display_name} в личку — сама, без повода с его стороны, как подруге: "
                    "коротко и живо (одно-два сообщения): обязательно скажи, ЧТО именно случилось, или процитируй "
                    "ту самую фразу — не просто «этот момент»; без пересказа всего подряд и без «привет».")
        ctx = await self._occasion_ctx(dm, user)
        if not finished and result.get("share"):
            # То, чем ей захотелось поделиться, — уже готовое сообщение в её стиле (с цитатой и реакцией).
            text = re.sub(r"\s+", " ", result["share"]).strip()  # одним сообщением — цитату не резать
        else:
            text = await self._pipeline.generate_occasion(ctx, task, fallback="")
        if not text:
            return
        self._last_book_share[uid] = now
        await self._message_handler._send_with_typing(dm, text, random.uniform(1.5, 3.0))
        if picture:
            import io as _io
            try:
                await dm.send(file=discord.File(_io.BytesIO(picture), filename="illustration.jpg"))
            except discord.HTTPException:
                pass
        await self._memory.log_bot_reply(ctx, text)
        _logger.info("Reading shared", extra={"data": {"user_id": uid, "book": book["title"], "finished": finished,
                                                       "picture": bool(picture)}})

    async def _end_binge(self, why: str) -> None:
        _logger.info("Binge ended", extra={"data": {"why": why}})
        self._binge_book = None
        schedule.set_binge("", None)
        self._next_read_min = 0.0

    async def _maybe_life_event(self) -> None:
        """Время от времени с ней что-то случается — когда модель свободна."""
        last = self._life.last_at()
        hours = (datetime.now(tz=timezone.utc) - last).total_seconds() / 3600 if last else 1e9
        if not self._next_life_event_h:
            # Случайный интервал вокруг среднего, чтобы события не шли по часам.
            self._next_life_event_h = config.LIFE_EVENT_INTERVAL_H * random.uniform(0.5, 1.5)
        if hours < self._next_life_event_h:
            return
        if self._pipeline.busy or self._pipeline.seconds_since_last_request < 60:
            return
        if self._voice is not None and self._voice.recently_active(120):
            return
        event = await self._pipeline.generate_life_event(self._life.history_lines())
        if event:
            await self._life.add(event)
            self._next_life_event_h = 0.0

    async def _occasion_ctx(self, channel: Any, user: Optional[discord.abc.User] = None) -> Any:
        """Контекст для реплики «по поводу»: канал + последние сообщения в нём."""
        from behavior.engine import mood_label
        from core.message_handler import MessageContext

        is_dm = isinstance(channel, discord.DMChannel)
        if is_dm and user is not None:
            ctx = await self._initiator_context(user, channel)
        else:
            ctx = MessageContext(
                message_id=0, content="", author_id=anonymise_author_id(user.id) if user else "",
                raw_author_id=user.id if user else 0,
                author_display_name=user.display_name if user else "",
                channel_id=channel.id, guild_id=getattr(getattr(channel, "guild", None), "id", None),
                timestamp=datetime.now(tz=timezone.utc), is_dm=is_dm,
            )
            ctx.mood_label = mood_label(self._state)
        try:
            msgs = [m async for m in channel.history(limit=8)]
            ctx.recent_channel_history = [
                f"{config.BOT_NAME if m.author.id == self.user.id else m.author.display_name}: {m.content}"
                for m in reversed(msgs) if m.content
            ]
        except (discord.Forbidden, discord.HTTPException):
            pass
        return ctx

    async def _fire_reminder(self, r: Dict[str, Any]) -> None:
        uid = int(r["user_id"])
        try:
            user = self.get_user(uid) or await self.fetch_user(uid)
        except (discord.NotFound, discord.HTTPException):
            return
        channel: Any = None
        if not r["is_dm"] and r["channel_id"]:
            channel = self.get_channel(int(r["channel_id"]))
            if channel is not None and not config.guild_allowed(getattr(getattr(channel, "guild", None), "id", None)):
                channel = None  # тестовый режим: в чужой сервер не пишет — напомнит в личку
        if channel is None:
            channel = user.dm_channel or await user.create_dm()
        due = datetime.fromisoformat(r["due_at"])
        late_min = (datetime.now(tz=timezone.utc) - due).total_seconds() / 60
        ctx = await self._occasion_ctx(channel, user)
        text = await self._pipeline.generate_occasion(
            ctx,
            f"Ты обещала {r['user_name']} напомнить: «{r['text']}» (просьба была: «{r['request']}»). "
            "Сейчас как раз время. Напомни коротко, в своём стиле, одной-двумя фразами, без «привет»."
            + (f" Ты опоздала на {int(late_min)} мин — можно извиниться мимоходом." if late_min > 30 else ""),
            fallback=f"эй, напоминаю: {r['text']}",
        )
        if not isinstance(channel, discord.DMChannel):
            text = f"{user.mention} {text}"
        try:
            await channel.send(text)
            _logger.info("Reminder fired", extra={"data": {"id": r["id"], "user_id": uid,
                                                           "late_min": round(late_min, 1)}})
            if self._message_handler is not None:
                await self._memory.log_bot_reply(ctx, text)
        except (discord.Forbidden, discord.HTTPException) as exc:
            _logger.warning("Reminder send failed", extra={"data": {"id": r["id"], "error": str(exc)}})

    async def _say_goodnight(self) -> None:
        """Ложится спать — пишет «всё, я спать» там, где только что разговаривала."""
        now = datetime.now(tz=timezone.utc)
        recent = sorted(
            ((cid, t) for cid, t in self._state.last_spoke.items()
             if (now - t).total_seconds() <= config.GOODNIGHT_ACTIVE_MIN * 60),
            key=lambda x: x[1], reverse=True,
        )[:2]
        for cid, _t in recent:
            channel = self.get_channel(cid)
            if channel is None or not config.guild_allowed(getattr(getattr(channel, "guild", None), "id", None)):
                continue
            partner = self._state.last_partner.get(cid)
            user = channel.recipient if isinstance(channel, discord.DMChannel) else None
            ctx = await self._occasion_ctx(channel, user)
            text = await self._pipeline.generate_occasion(
                ctx,
                "Уже под утро, ты ложишься спать. Коротко попрощайся с теми, с кем сейчас переписывалась, "
                "одной фразой, в своём стиле (можно с намёком на то, о чём говорили).",
                fallback="всё, я спать. ночи",
            )
            try:
                await channel.send(text)
                await self._memory.log_bot_reply(ctx, text)
                _logger.info("Goodnight sent", extra={"data": {"channel_id": cid, "partner": bool(partner)}})
            except (discord.Forbidden, discord.HTTPException):
                pass

    async def _maybe_congratulate(self) -> None:
        """День рождения у знакомого — поздравить в личке (раз в год, днём)."""
        from pipeline.text import _local_now

        now = _local_now()
        if now.hour < 12:
            return  # не будить людей поздравлениями
        for key, name, uid, aff in await self._memory.social.birthdays_on(now.strftime("%d.%m"), now.year):
            await self._memory.social.mark_birthday_greeted(key, now.year)
            if aff * 10 < config.BIRTHDAY_MIN_AFFINITY or (self.user and uid == self.user.id):
                continue
            try:
                user = self.get_user(uid) or await self.fetch_user(uid)
                if user.bot:
                    continue
                dm = user.dm_channel or await user.create_dm()
                ctx = await self._occasion_ctx(dm, user)
                text = await self._pipeline.generate_occasion(
                    ctx,
                    f"Сегодня у {name} день рождения — ты помнишь это сама, никто не напоминал. "
                    "Поздравь в личке: тепло и по-своему, без открыточных штампов, можно вспомнить что-то, "
                    "что ты о нём/ней знаешь. Одно-два коротких сообщения.",
                    fallback="с днём рождения! 💎",
                )
                if self._message_handler is not None:
                    await self._message_handler._send_with_typing(dm, text, random.uniform(2.0, 4.0))
                    await self._memory.log_bot_reply(ctx, text)
                _logger.info("Birthday greeted", extra={"data": {"user_id": uid, "name": name}})
            except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
                _logger.warning("Birthday greeting failed", extra={"data": {"user_id": uid, "error": str(exc)}})

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Gracefully shut down the bot.

        Cancels the background task before delegating to the parent
        :meth:`discord.Client.close`.
        """
        _logger.info("Shutting down bot", extra={"data": {}})
        if self._voice is not None:
            try:
                await self._voice.close()
            except Exception:  # noqa: BLE001
                pass
        if self._mind_task is not None and not self._mind_task.done():
            self._mind_task.cancel()
        if self._life_task is not None and not self._life_task.done():
            self._life_task.cancel()
        if self._catchup_task is not None and not self._catchup_task.done():
            self._catchup_task.cancel()
        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        await super().close()
