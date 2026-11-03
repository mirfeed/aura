"""
core/message_handler.py — MessageContext dataclass and MessageHandler class.

MessageHandler sits between the Discord event loop and the rest of the
pipeline.  It:
  1. Builds a MessageContext from a raw discord.Message.
  2. Fetches recent channel history for context.
  3. Populates memory_snippets and relationship_score from MemoryStore.
  4. Calls relevance.score() to assess interest.
  5. Asks the DecisionEngine whether to reply.
  6. Stamps context.mood_label from the decision.
  7. If replying: simulates typing, calls the text pipeline, and sends.
  8. Logs the event to MemoryStore regardless of whether a reply was sent.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import discord

import config
from behavior import schedule
from behavior.engine import DecisionEngine, ReplyDecision
from behavior.state import BotState
from core import relevance as relevance_mod
from core import reminders as reminders_mod
from pipeline.text import TextPipeline, is_provocation
from utils.logger import (
    anonymise_author_id,
    get_logger,
    log_decision,
    log_message_received,
    log_relevance,
    log_reply_sent,
)

if TYPE_CHECKING:
    from memory.store import MemoryStore

_logger = get_logger("bot.message_handler")

# Number of historical messages to pull for context.
_HISTORY_LIMIT: int = config.CHAT_HISTORY_LIMIT

# «Аура, забудь всё обо мне» — стирает факты и мнение о человеке.
_FORGET_RE = re.compile(r"забудь\s+(?:вс[её]\s+)?(?:обо|про|о)\s+мне|забудь\s+меня", re.IGNORECASE)

_REACTIONS = {
    "amused": ["😂", "😏", "👀", "💀"],
    "cheerful": ["😊", "💎", "👀", "✨"],
    "irritated": ["🙄", "😑"],
    "tired": ["😴", "🙄"],
    "default": ["👀", "🤔", "😏", "💎"],
}


# Просьба написать/передать кому-то.
_RELAY_RE = re.compile(
    r"\b(?:напиши|написать|напишешь|передай|передать|передашь|скажи|сказать|скажешь|отправь|черкни|маякни|"
    r"спроси|спросить|спросишь|напугай|напугать|затролль|затроллить|разыграй|разыграть|подколи|поздравь|"
    r"поздравить|позови|позвать|пригласи|пригласить|напомни|напомнить|ответь|ответить|сообщи|сообщить|"
    r"отправить|отправишь|скинь|скинуть|скинешь|перешли|переслать|перешлёшь|перешлешь|разошли|разослать)\b",
    re.IGNORECASE,
)

# Ссылка на «читалку»: слова про чтение в сообщении или известный сайт.
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
_READ_LINK_RE = re.compile(r"чита|почита|прочита|глянь|зацени|ранобэ|ранобе|новелл|манхв|маньхуа|манг|фанф|фик|"
                           r"комикс|книг|глав|тайтл|вебтун|webtoon", re.I)
_READER_HOST_RE = re.compile(r"webtoons\.com|royalroad\.com|wikisource\.org|gutenberg\.org", re.I)
_IMAGE_URL_RE = re.compile(r"\.(?:png|jpe?g|gif|webp)(?:\?|$)|tenor\.com|giphy\.com|youtu", re.I)

# «Аура, почитай …» и вопросы о её книгах.
_READ_CMD_RE = re.compile(
    r"\b(?:почитай|прочитай|начни\s+читать|возьмись\s+за|давай\s+читать|посоветую|советую|рекомендую|"
    r"(?:найди|скачай|поищи)\s+(?:книг|комикс|манг)\w*|зацени\s+(?:книг|комикс|манг)\w*)\b", re.I)
_BOOKS_RE = re.compile(r"книг|комикс|манг|на\s+полке|библиотек|чита(?:ешь|ла|ть)", re.I)

# Она говорит, что отправила/отправит кому-то сообщение.
_SENT_CLAIM_RE = re.compile(
    r"\b(?:отправила|отправлю|скинула|скину|переслала|перешлю|разослала|разошлю|передала|передам|"
    r"раскидала|раскидаю|сообщу|отпишу|написала\s+(?:им|ему|ей|всем)|напишу\s+(?:им|ему|ей|всем)|"
    r"(?:сейчас|щас|ща|скоро)\s+(?:\w+\s+)?сделаю)\b",
    re.IGNORECASE)


def _false_send_claim(text: str) -> bool:
    """Утверждает, что отправила/отправит.  «не отправила», «ничего не скинула» — честно, не в счёт."""
    for m in _SENT_CLAIM_RE.finditer(text or ""):
        before = text[max(0, m.start() - 25):m.start()].lower()
        if not re.search(r"\bне\s+(?:\w+\s+)?$", before):
            return True
    return False

# Похоже на вопрос: знак вопроса или вопросительное слово в начале.
_QUESTION_RE = re.compile(
    r"\?|^\s*(?:а\s+)?(?:что|как|почему|зачем|кто|где|когда|какой|какая|какие|сколько|ты\s+бы|можешь|стоит)\b",
    re.IGNORECASE,
)

_GIF_HOST_RE = re.compile(r"https?://(?:www\.)?(?:tenor\.com|giphy\.com|klipy\.com)/", re.I)


def _embed_media(message: "discord.Message") -> List[str]:
    """
    Картинки из эмбедов (гифки Tenor/Giphy/Klipy, превью ссылок) и стикеров — чтобы она их видела.
    Для гифок-видео (gifv): «видео||превью» — сначала пробуем кадры из видео, не вышло — превью.
    """
    urls: List[str] = []
    for e in message.embeds:
        variants = []
        for media in (e.video, e.image, e.thumbnail):
            url = getattr(media, "proxy_url", None) or getattr(media, "url", None)
            if url:
                variants.append(url)
        if variants:
            urls.append("||".join(variants))
    for st in getattr(message, "stickers", []) or []:
        url = str(getattr(st, "url", "") or "")
        if url and not url.endswith(".json"):
            urls.append(url)
    return urls


@dataclass
class MessageContext:
    """
    A normalised, serialisation-friendly view of a Discord message.

    All fields are plain Python types; no discord.py objects are retained
    after construction so that the dataclass can be passed freely
    throughout the pipeline.

    Attributes
    ----------
    message_id:
        Discord snowflake of the incoming message.
    content:
        Raw text content of the message.
    author_id:
        SHA-256 hash prefix of the author's Discord user ID (anonymised).
    author_display_name:
        The author's server nickname or global display name.
    channel_id:
        Snowflake of the channel the message was sent in.
    guild_id:
        Snowflake of the guild (server), or None for DMs.
    timestamp:
        UTC datetime when the message was created.
    attachments:
        List of attachment URLs included in the message.
    is_reply:
        True if the message is a reply to any earlier message.
    replied_to_bot:
        True if the message is specifically a reply to one of the bot's
        own messages.
    recent_channel_history:
        Up to 5 recent message content strings from the channel, oldest
        first, *excluding* the current message.
    relevance_score:
        Populated after relevance scoring; 0.0 before that step.
    mood_label:
        Human-readable mood description set by the decision engine just
        before the context is passed to the text pipeline.
    memory_snippets:
        Semantically similar past messages retrieved from ChromaDB.
        Injected into the <memory> block of the LLM prompt.
    relationship_score:
        0.0–1.0 score reflecting how often the bot has interacted with
        this author.  Used by the decision engine to modulate reply prob.
    """

    message_id: int
    content: str
    author_id: str  # anonymised hash prefix
    author_display_name: str
    channel_id: int
    guild_id: Optional[int]
    timestamp: datetime
    attachments: List[str] = field(default_factory=list)
    is_reply: bool = False
    replied_to_bot: bool = False
    recent_channel_history: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    mood_label: str = ""
    memory_snippets: List[str] = field(default_factory=list)
    relationship_score: float = 0.0

    # True when the message was received in a DM channel (guild_id is None
    # and the channel is a DMChannel).  In DMs the bot always replies and
    # never requires a mention.
    is_dm: bool = False

    # True when the decision engine determined the message is complex enough
    # to warrant extended model reasoning (think=True in Ollama payload).
    use_thinking: bool = False

    # Discord role IDs of the message author (guild only; empty in DMs).
    # Used by the pipeline to apply role-based warmth modifiers.
    author_role_ids: List[int] = field(default_factory=list)

    # Raw (non-anonymised) numeric author ID.  Used for special-user lookup
    # by config.SPECIAL_USER_IDS without exposing the hash elsewhere.
    raw_author_id: int = 0

    # True when the text came from speech recognition in a voice channel.
    # The pipeline then asks for a short spoken-style reply without emoji.
    is_voice: bool = False

    # Voice only: display names of the people currently in the voice channel
    # (speaker first).  Lets the model keep track of who is present.
    voice_participants: List[str] = field(default_factory=list)
    # Voice: её не звали по имени, она сама решает, вмешаться или промолчать.
    voice_optional: bool = False

    # Social memory block for the prompt (memory.social.SocialContext).
    social: Any = None
    # Her current attitude towards the author, -1..1.
    affinity: float = 0.0
    # Провокация / рейдж-байт (оскорбления, «твою мать…», политота) — отвечает иначе или игнорит.
    provocation: bool = False
    # В этом ответе проявить интерес к собеседнику — спросить после ответа (pipeline.text.choose_length).
    curious: bool = False
    engage: str = ""  # как поддержать разговор: lead / dry / answered / shared (см. pipeline.text)
    # Display name of the person whose message this one replies to.
    reply_to_name: Optional[str] = None
    # Other recent chat participants: (anonymised key, display name, raw id).
    participants: List[Tuple[str, str, int]] = field(default_factory=list)
    # После её последней реплики в канале писал кто-то ещё, кроме автора.
    others_since_bot: bool = False
    # Сообщение обращено к другому человеку (ответ ему или его @упоминание).
    aimed_at_other: bool = False
    # Продолжение разговора с ней: True — пишут ей, False — не ей, None — не проверяли.
    addressed_to_me: Optional[bool] = None
    # Личка, пришедшая, пока её не было в сети: пометка для модели (сколько назад написали).
    late_note: str = ""
    # Что она вспомнила из прочитанных книг к этому сообщению.
    book_memories: List[str] = field(default_factory=list)
    # Что она загуглила перед ответом (запрос и найденное).
    web_query: str = ""
    web_results: str = ""
    # Картинки из её коллекции, которые уходят следом за ответом (пути к файлам).
    outgoing_media: List[Any] = field(default_factory=list)


class MessageHandler:
    """
    Orchestrates the full message-processing pipeline.

    Parameters
    ----------
    bot_user_id:
        The bot's own Discord user ID, used to detect self-replies and
        replies to the bot.
    state:
        Shared mutable bot state; injected at construction time.
    engine:
        Decision engine instance.
    pipeline:
        Text generation pipeline instance.
    memory:
        MemoryStore instance for recall and event logging.
    """

    def __init__(
        self,
        bot_user_id: int,
        state: BotState,
        engine: DecisionEngine,
        pipeline: TextPipeline,
        memory: "MemoryStore",
    ) -> None:
        """Initialise the handler with injected dependencies."""
        self._bot_user_id = bot_user_id
        self._state = state
        self._engine = engine
        self._pipeline = pipeline
        self._memory = memory
        self._send_locks: Dict[int, asyncio.Lock] = {}
        # Бот подключает: (message, ctx, kind, my_reply) -> True, если вопрос ушёл советчику.
        self.advice_hook: Optional[Any] = None
        # Бот подключает: (message, ctx) -> пометка для ответа, если это была просьба написать кому-то.
        self.relay_hook: Optional[Any] = None
        # Бот подключает: (message) -> правда о том, что она отправляла по просьбам этого человека.
        self.relay_facts_hook: Optional[Any] = None
        # Напоминания (core.reminders.ReminderStore); None — выключены.
        self.reminders: Optional[Any] = None
        # Её книжная полка (memory.library.Library); None — чтение выключено.
        self.library: Optional[Any] = None
        # Найти посоветованную книгу в интернете (бот: CompanionBot._find_recommended_book).
        self.book_hook: Optional[Any] = None
        # Ссылка на «читалку» (бот: CompanionBot._open_book_link).
        self.book_link_hook: Optional[Any] = None
        # Кто ждёт, пока она уточнит, кому передать (ведёт бот).
        self.relay_pending_ids: set = set()
        # user_id → monotonic-время последней просьбы переслать (выставляет бот).
        self.relay_recent: Dict[int, float] = {}
        self._last_chat_advice: float = 0.0
        self._bait_log: Dict[int, List[float]] = {}
        # Долгая задержка (зачиталась): сообщения, накопившиеся от того же человека, пока она «не видит».
        # (channel_id, author_id) → {"messages": [Message], "notes": [str], "last_at": monotonic, ...}
        # Заодно и обычная пауза: человек пишет очередью — ждёт, пока допишет, и отвечает разом.
        self._pending_batch: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Она уже пишет ответ этому человеку: что он успел дописать за это время.
        self._inflight: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Когда человек последний раз «печатал…» (событие Discord), monotonic.
        self._typing_at: Dict[Tuple[int, int], float] = {}
        # channel_id → id последнего сообщения в канале (чьего угодно): в личке discord.py его не ведёт,
        # а по нему видно, что её ответ уже не «следом» и нужен reply.
        self._last_msg_id: Dict[int, int] = {}

    def _seen(self, channel_id: int, message_id: int) -> None:
        if message_id > self._last_msg_id.get(channel_id, 0):
            self._last_msg_id[channel_id] = message_id

    def note_typing(self, channel_id: int, user_id: int) -> None:
        """Discord: человек печатает — она подождёт, пока он допишет."""
        self._typing_at[(channel_id, user_id)] = time.monotonic()

    async def _wait_quiet(self, key: Tuple[int, int], pend: Dict[str, Any]) -> None:
        """Ждёт, пока человек допишет: пауза после последнего сообщения и он больше не печатает."""
        deadline = time.monotonic() + config.CHAT_BATCH_MAX_WAIT_S
        while time.monotonic() < deadline and not pend.get("pinged"):
            now = time.monotonic()
            typing = self._typing_at.get(key, 0.0)
            still_typing = typing > pend["last_at"] and now - typing < 10.0  # «печатает…» держится ~10 с
            if now - pend["last_at"] >= config.CHAT_BATCH_QUIET_S and not still_typing:
                return
            await asyncio.sleep(0.5)

    def _channel_lock(self, channel: Any) -> asyncio.Lock:
        """Замок на отправку в канал: ответы уходят по очереди, а не вперемешку."""
        key = getattr(channel, "id", 0)
        lock = self._send_locks.get(key)
        if lock is None:
            lock = self._send_locks[key] = asyncio.Lock()
        return lock

    async def handle(self, message: discord.Message, late_note: str = "", force_reply: bool = False,
                     _followup: Optional[List[discord.Message]] = None) -> None:
        """
        Process a single incoming Discord message end-to-end.

        Parameters
        ----------
        message:
            The raw :class:`discord.Message` from the ``on_message`` event.
        late_note:
            Пометка модели к этому ответу (запоздалая личка, совет близкого и т.п.).
        force_reply:
            Ответить обязательно, без проверок (решение уже принято — например, после совета).
        _followup:
            Сообщения, которые человек дописал, пока она отправляла прошлый ответ (уже записаны
            в ленту), — ответить на них одним сообщением.
        """
        self._seen(message.channel.id, message.id)
        # Спит — ничего не читает.  Личку разберёт, когда проснётся (CompanionBot._catch_up_dms),
        # поэтому в память её сейчас не пишем — иначе она будет считаться прочитанной.
        if schedule.is_asleep() and not force_reply:
            _logger.info("Asleep — message left for later", extra={"data": {
                "message_id": message.id, "dm": message.guild is None}})
            return

        # Build context (including channel history).
        ctx = await self._build_context(message)
        ctx.late_note = late_note
        if _followup:  # контекст — от первого из дописанных; вложения остальных добавит склейка ниже
            ctx.content = "\n".join(m.content for m in _followup if m.content)

        log_message_received(
            message_id=ctx.message_id,
            author_id=message.author.id,  # real ID for hashing inside log fn
            author_display_name=ctx.author_display_name,
            channel_id=ctx.channel_id,
            guild_id=ctx.guild_id,
            content_length=len(ctx.content),
        )

        social = self._memory.social
        # До записи этой реплики: давно ли человек пропадал, не висит ли её сообщение без ответа.
        if not _followup:  # дописанное уже записано в ленту, пока она отвечала
            absence = await self._absence_note(ctx)
            if absence:
                ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + absence
            await self._memory.note_incoming(ctx)

        # «Аура, забудь обо мне».
        if self._is_addressed(message, ctx) and _FORGET_RE.search(ctx.content):
            removed = await social.forget(ctx.author_id)
            await message.reply(
                random.choice(["ладно. уже не помню, кто ты.", "хм. как скажешь. забыла.", "готово. ты кто вообще?"]),
                mention_author=False,
            )
            _logger.info("Forget requested", extra={"data": {"author_id": ctx.author_id, "removed": removed}})
            return

        relay_sent = False  # в этот раз она реально переслала кому-то сообщение
        # «Аура, напиши Васе, что …» — от того, кому доверяет.
        mentions_other = ctx.is_dm and (
            any(not m.bot and m.id not in (self._bot_user_id, message.author.id) for m in message.mentions)
            or any(k != ctx.author_id for k in social.find_names_in_text(ctx.content or ""))
        )
        if (
            self.relay_hook is not None and config.RELAY_ENABLED
            and (message.author.id in self.relay_pending_ids
                 or mentions_other  # в личке говорит о ком-то — может, просит ему написать
                 # Недавно просил переслать — «ты не отправила», «жду», «ну?» тоже про это.
                 or (ctx.is_dm and time.monotonic() - self.relay_recent.get(message.author.id, -1e9)
                     < config.RELAY_FOLLOWUP_S)
                 or (_RELAY_RE.search(ctx.content or "") and self._is_addressed(message, ctx)))
        ):
            try:
                note = await self.relay_hook(message, ctx)
            except Exception as exc:  # noqa: BLE001
                note = ""
                _logger.warning("relay hook failed", extra={"data": {"error": str(exc)}})
            if note:
                ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + note
                force_reply = True
                relay_sent = "ты написала в личку ТОЛЬКО" in note
        # Книги: подарили файл, «почитай …», «что у тебя на полке?».
        if self.library is not None and (ctx.is_dm or self._is_addressed(message, ctx)):
            try:
                note = await self._books_note(message, ctx)
            except Exception as exc:  # noqa: BLE001
                note = ""
                _logger.warning("books note failed", extra={"data": {"error": str(exc)}})
            if note:
                ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + note
                force_reply = True
        # «Аура, напомни завтра в 6 про экзамен».
        if (
            self.reminders is not None and config.REMINDERS_ENABLED
            and reminders_mod.REMIND_RE.search(ctx.content or "")
            and (ctx.is_dm or self._is_addressed(message, ctx))
        ):
            note = await self._set_reminder(message, ctx)
            if note:
                ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + note
                force_reply = True
        # Чтобы не врала «да, отправила»: что она на самом деле отправляла по его просьбам.
        if ctx.is_dm and self.relay_facts_hook is not None and not ctx.late_note:
            try:
                facts = await self.relay_facts_hook(message)
            except Exception:  # noqa: BLE001
                facts = ""
            if facts:
                ctx.late_note = facts

        # Populate memory fields before the decision engine sees context.
        ctx.memory_snippets = await self._memory.recall(ctx)
        ctx.social = await social.build_context(
            ctx.channel_id,
            (ctx.author_id, ctx.author_display_name, ctx.raw_author_id),
            ctx.participants,
            ctx.content,
        )
        ctx.affinity = ctx.social.speaker_affinity
        if self.library is not None:
            try:
                ctx.book_memories = await self.library.recall(
                    ctx.content or "", context="\n".join(ctx.recent_channel_history[-2:]))
            except Exception as exc:  # noqa: BLE001
                _logger.warning("book recall failed", extra={"data": {"error": str(exc)}})
        familiarity = await self._memory.get_relationship_score(ctx.author_id)
        # Близость = насколько часто общаются + как она к человеку относится.
        ctx.relationship_score = max(0.0, min(1.0, 0.5 * familiarity + 0.5 * (ctx.affinity + 1) / 2))

        # Determine the bot's last-spoke time for this channel.
        last_spoke = self._state.last_spoke.get(ctx.channel_id)

        # Relevance scoring.
        from utils.names import is_bot_mentioned

        mentioned_bot = message.guild is not None and is_bot_mentioned(message, self._bot_user_id)
        result = relevance_mod.score(
            content=ctx.content,
            bot_name=config.BOT_NAME,
            mentioned_bot=mentioned_bot,
            is_reply_to_bot=ctx.replied_to_bot,
            last_bot_spoke=last_spoke,
        )
        ctx.relevance_score = result.score

        log_relevance(
            message_id=ctx.message_id,
            score=result.score,
            breakdown=result.breakdown,
        )

        # Продолжение разговора: решить, ей ли это пишут.
        if (
            not ctx.is_dm
            and not result.breakdown.keys() & {"direct_mention", "reply_to_bot", "name_mention"}
            and self._state.is_followup(ctx.channel_id, ctx.author_id, config.FOLLOWUP_WINDOW_S)
        ):
            ctx.addressed_to_me = await self._is_addressed_to_me(ctx)

        # Behaviour decision.
        decision: ReplyDecision = self._engine.decide(ctx, result, self._state)

        # Рейдж-байт: живой человек не ведётся на каждую провокацию — часто просто игнорит.
        ctx.provocation = is_provocation(ctx.content)
        if ctx.provocation:
            # Оскорбили — отношение падает сразу, а не «симпатичный собеседник» до следующей рефлексии.
            try:
                await self._memory.social.nudge(ctx.author_id, -config.BAIT_AFFINITY_PENALTY, ctx.raw_author_id)
            except Exception:  # noqa: BLE001
                pass
        if ctx.provocation and decision.should_reply and not ctx.is_dm:
            if self._ignore_bait(message.author.id):
                decision.should_reply = False
                if random.random() < 0.5:
                    asyncio.create_task(self._react_to_bait(message))

        if force_reply:
            decision.should_reply = True
            decision.delay_seconds = 0.0

        # Вопрос в общем чате: она сама решает — ответить, посоветоваться в личке или пожаловаться.
        complain = False
        if (
            decision.should_reply and not force_reply and not ctx.is_dm and not ctx.is_voice
            and self.advice_hook is not None and config.CHAT_ASK_ADVICE
            and (result.breakdown.keys() & {"direct_mention", "reply_to_bot", "name_mention"} or ctx.addressed_to_me)
            and _QUESTION_RE.search(ctx.content or "")
            and time.monotonic() - self._last_chat_advice > config.CHAT_ADVICE_COOLDOWN_S
        ):
            verdict = await self._pipeline.chat_doubt(ctx)
            _logger.info("chat_doubt", extra={"data": {"message_id": ctx.message_id, "verdict": verdict}})
            if verdict == "unsure":
                try:
                    asked = await self.advice_hook(message, ctx, "ask", "")
                except Exception as exc:  # noqa: BLE001
                    asked = False
                    _logger.warning("advice hook failed", extra={"data": {"error": str(exc)}})
                if asked:
                    self._last_chat_advice = time.monotonic()
                    await self._memory.log_event(ctx, bot_replied=False)
                    return
            complain = verdict == "complain"

        # Stamp mood onto context so the pipeline can inject it into the prompt.
        ctx.mood_label = decision.mood_label
        # Stamp thinking flag so the pipeline knows whether to use extended reasoning.
        ctx.use_thinking = decision.use_thinking

        log_decision(
            message_id=ctx.message_id,
            should_reply=decision.should_reply,
            reason=decision.reason,
            delay_seconds=decision.delay_seconds,
            typing_duration=decision.typing_duration,
        )

        if not decision.should_reply and not ctx.is_dm:
            await self._maybe_react(message, result.score)
        elif (decision.should_reply and not ctx.is_dm and not ctx.provocation
              and random.random() < config.REACTION_WITH_REPLY_PROBABILITY):
            asyncio.create_task(self._react(message))
        if not decision.should_reply:
            # Молчит, но картинку всё равно рассматривает и запоминает.
            asyncio.create_task(self._pipeline.observe_images(ctx))

        if decision.should_reply:
            # Она уже собирается ответить этому человеку в этом канале, но пока «не видит» (зачиталась) —
            # это сообщение войдёт в тот же ответ, а не получит свой.
            # Тег или ответ на её сообщение — это уведомление: выдёргивает из книги сразу.
            # (в личке relevance тег не считает — проверяем упоминание напрямую)
            pinged = bool(result.breakdown.keys() & {"direct_mention", "reply_to_bot"}) or any(
                m.id == self._bot_user_id for m in message.mentions)
            key = (ctx.channel_id, message.author.id)
            flight = self._inflight.get(key)
            if flight is not None and not force_reply and not _followup and time.monotonic() - flight["since"] < 180:
                # Она уже пишет ему ответ — дописанное войдёт в него (или в следующий, одним сообщением).
                flight["messages"].append(message)
                _logger.info("Message joined reply in progress", extra={"data": {
                    "message_id": message.id, "joined": len(flight["messages"])}})
                await self._memory.log_event(ctx, bot_replied=False)
                return
            pend = self._pending_batch.get(key)
            if pend is not None and not force_reply and not _followup:
                pend["messages"].append(message)
                pend["last_at"] = time.monotonic()
                if ctx.late_note:
                    pend["notes"].append(ctx.late_note)
                if pinged:
                    pend["pinged"] = True
                    pend["wake"].set()  # уведомление — отвечает на всё накопившееся прямо сейчас
                _logger.info("Message batched", extra={"data": {"message_id": message.id,
                                                                "batched": len(pend["messages"])}})
                await self._memory.log_event(ctx, bot_replied=False)
                return

            # Simulate thinking delay.
            if decision.delay_seconds >= 15 and pinged:
                decision.delay_seconds = random.uniform(2.0, 8.0)  # пришло уведомление — глянула сразу
                ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + (
                    "ты читала, но тебя тегнули — уведомление выдернуло тебя из книги")
            batching = not force_reply  # и для «дописанного»: вдруг он пишет дальше
            if decision.delay_seconds > 0 or batching:
                long_wait = decision.delay_seconds >= 15
                if batching:
                    self._pending_batch[key] = {"messages": list(_followup or [message]), "notes": [], "wake": asyncio.Event(),
                                                "pinged": False, "last_at": time.monotonic()}
                try:
                    if long_wait and batching:
                        try:  # ждёт, пока сама не оторвётся от книги или не придёт уведомление (тег)
                            await asyncio.wait_for(self._pending_batch[key]["wake"].wait(),
                                                   timeout=decision.delay_seconds)
                        except asyncio.TimeoutError:
                            pass
                    elif decision.delay_seconds > 0:
                        await asyncio.sleep(decision.delay_seconds)
                    if batching:
                        await self._wait_quiet(key, self._pending_batch[key])  # пусть допишет
                finally:
                    pend = self._pending_batch.pop(key, None) if batching else None
                if long_wait and pend is not None and pend.get("pinged") and len(pend["messages"]) == 1:
                    ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + (
                        "ты читала, но тебя тегнули — уведомление выдернуло тебя из книги")
                if pend is not None and len(pend["messages"]) > 1:
                    # Ответить на всё накопившееся разом — на последнее сообщение.
                    msgs = pend["messages"]
                    ctx.content = "\n".join(m.content for m in msgs if m.content)
                    for m in msgs[1:]:
                        ctx.attachments += [a.url for a in m.attachments]
                    notes = [n for n in pend["notes"] if n]
                    notes.append((f"пока ты читала, {ctx.author_display_name} написал(а)" if long_wait else
                                  f"{ctx.author_display_name} написал(а)")
                                 + " тебе несколько сообщений подряд (они все в «Новом сообщении») — ответь на всё "
                                 "сразу, одним сообщением, как в живой переписке, а не по пунктам"
                                 + ("; от книги тебя оторвало уведомление — тебя тегнули" if pend.get("pinged") else ""))
                    ctx.late_note = "; ".join(([ctx.late_note] if ctx.late_note else []) + notes)
                    message = msgs[-1]
                    _logger.info("Batched reply", extra={"data": {"messages": len(msgs)}})

            self._inflight[key] = {"messages": [], "since": time.monotonic()}
            try:
                # If extended reasoning is active — add a 🤔 reaction so the
                # user sees the bot is "thinking".  Removed once the reply lands.
                if ctx.use_thinking:
                    try:
                        await message.add_reaction("🤔")
                    except (discord.Forbidden, discord.HTTPException):
                        pass

                # Generate reply text.
                gen_started = time.monotonic()
                search_seconds = 0.0
                # Нужно загуглить, чтобы ответить по делу?
                # Вспомнила прочитанное — про это не гуглит (и не спойлерит себе книгу).
                if config.WEB_SEARCH_ENABLED and not ctx.is_voice and not ctx.book_memories:
                    try:
                        query = await self._pipeline.search_query(ctx)
                        if query:
                            ctx.web_query = query
                            ctx.web_results = await self._pipeline.web_lookup(query)
                    except Exception as exc:  # noqa: BLE001
                        _logger.warning("web search failed", extra={"data": {"error": str(exc)}})
                    search_seconds = time.monotonic() - gen_started
                self._pipeline.last_stats = {}
                response = await self._pipeline.generate(ctx)
                if not relay_sent and not ctx.outgoing_media and _false_send_claim(response):
                    # Говорит «отправила» / «сейчас отправлю», хотя в этот раз ничего не ушло, —
                    # отправлять «потом» она не умеет, так что это враньё.  Переписать ответ.
                    _logger.warning("False send claim — regenerating", extra={"data": {"draft": response[:120]}})
                    ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + (
                        "ВАЖНО: в этот раз ты НИКОМУ ничего не отправила и не можешь отправить «потом» или «сейчас» "
                        "сама по себе. Нельзя писать «отправила», «скинула», «сейчас отправлю», «разошлю». Если просят "
                        "переслать — честно скажи, что не отправила, и попроси написать в одном сообщении, кому "
                        "(ники или юзернеймы в Discord) и что именно передать")
                    response = await self._pipeline.generate(ctx)
                    if _false_send_claim(response):
                        response = "блин, ничего не ушло. напиши одним сообщением, кому (ники или юзернеймы в Discord) и что передать — перешлю"
                # Пока она писала ответ, человек дописал ещё — ответить на всё сразу, а не на кусок.
                more = self._inflight[key]["messages"]
                if more and not relay_sent and not ctx.outgoing_media:
                    self._inflight[key]["messages"] = []
                    ctx.content = "\n".join([ctx.content] + [m.content for m in more if m.content])
                    for m in more:
                        ctx.attachments += [a.url for a in m.attachments]
                    ctx.late_note = (ctx.late_note + "; " if ctx.late_note else "") + (
                        f"пока ты писала ответ, {ctx.author_display_name} дописал(а) ещё (всё в «Новом сообщении») "
                        "— ответь на всё сразу, одним сообщением")
                    message = more[-1]
                    _logger.info("Reply redone with new messages", extra={"data": {"added": len(more)}})
                    response = await self._pipeline.generate(ctx)
                gen_seconds = time.monotonic() - gen_started

                # Remove the thinking reaction before sending the reply.
                if ctx.use_thinking:
                    try:
                        await message.remove_reaction("🤔", message.guild.me if message.guild else message.channel.me)
                    except (discord.Forbidden, discord.HTTPException, AttributeError):
                        pass

                # On guild channels — reply to the specific message (shows the
                # reply arrow in Discord).  In DMs — just send normally.
                send_started = time.monotonic()
                await self._send_with_typing(
                    message.channel, response, decision.typing_duration,
                    reply_to=message if not ctx.is_dm else None,
                    reply_if_buried=message if ctx.is_dm else None,
                    already_waited=gen_seconds,
                    media=ctx.outgoing_media,
                )

                if ctx.outgoing_media:
                    # Чтобы в истории/памяти она помнила, что кинула картинку.
                    response = (response + " [кинула картинку из своей коллекции]").strip()
                await self._memory.log_bot_reply(ctx, response)

                if complain and self.advice_hook is not None:
                    self._last_chat_advice = time.monotonic()
                    asyncio.create_task(self.advice_hook(message, ctx, "complain", response))

                log_reply_sent(
                    message_id=ctx.message_id,
                    channel_id=ctx.channel_id,
                    response_length=len(response),
                    search_seconds=round(search_seconds, 1),
                    llm_seconds=round(gen_seconds - search_seconds, 1),
                    typing_seconds=round(time.monotonic() - send_started, 1),
                    **getattr(self._pipeline, "last_stats", {}),
                )
                # Дописал, пока она уже печатала / отправляла, — ответ на это следом, одним сообщением.
                left = self._inflight.pop(key, {}).get("messages", [])
                if left:
                    asyncio.create_task(self.handle(left[0], _followup=left, late_note=(
                        f"{ctx.author_display_name} дописал(а) это, пока ты отвечала на прошлое (твой ответ уже "
                        "в переписке выше) — ответь на новое, не повторяя сказанного")))
            finally:
                self._inflight.pop(key, None)

        # Always log the event — whether or not we replied.  (Дописанное уже записано, когда пришло.)
        if not _followup:
            await self._memory.log_event(ctx, bot_replied=decision.should_reply)

    async def catch_up(self, missed: List[discord.Message], reply: bool) -> None:
        """
        Личка, пришедшая, пока её не было в сети (старые первыми).

        Все сообщения записываются в память; на последнее она отвечает (если *reply*),
        а более ранние модель видит в истории переписки.
        """
        if not missed:
            return
        *earlier, last = missed
        for msg in earlier:
            try:
                ctx = await self._build_context(msg)
                await self._memory.note_incoming(ctx)
                await self._memory.log_event(ctx, bot_replied=False)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("catch-up note failed", extra={"data": {"error": str(exc)}})
        if not reply:
            ctx = await self._build_context(last)
            await self._memory.note_incoming(ctx)
            await self._memory.log_event(ctx, bot_replied=False)
            return
        age = datetime.now(tz=timezone.utc) - last.created_at.replace(tzinfo=timezone.utc)
        minutes = int(age.total_seconds() // 60)
        when = f"{minutes} мин назад" if minutes < 90 else f"{round(minutes / 60)} ч назад"
        n = len(missed)
        note = (
            f"тебя не было в сети, ты только что зашла и увидела "
            f"{'это сообщение' if n == 1 else f'{n} новых сообщения(й)'} — последнее пришло {when}. "
            "Ответь на всё, что тебе написали, одним ответом; можешь коротко упомянуть, что не была в сети, "
            "но не извиняйся долго"
        )
        await self.handle(last, late_note=note)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _build_context(self, message: discord.Message) -> MessageContext:
        """
        Construct a :class:`MessageContext` from a raw Discord message.

        Fetches the last :data:`_HISTORY_LIMIT` messages from the channel
        to populate ``recent_channel_history``.
        """
        # Эмбед гифки (Klipy/Tenor) Discord дорисовывает через секунду-две после сообщения.
        if config.VISION_ENABLED and not message.embeds and _GIF_HOST_RE.search(message.content or ""):
            await asyncio.sleep(2.0)
            try:
                message = await message.channel.fetch_message(message.id)
            except Exception:  # noqa: BLE001
                pass
        # Resolve reply metadata.
        is_reply = message.reference is not None
        replied_to_bot = False
        if is_reply and message.reference is not None:
            ref_author_id = (
                message.reference.resolved.author.id
                if isinstance(message.reference.resolved, discord.Message)
                else None
            )
            replied_to_bot = ref_author_id == self._bot_user_id

        # Fetch recent history (best-effort; ignore failures).
        # Format: "[HH:MM] Name → Other: text" so the LLM knows who said what
        # and to whom; her own messages are included too.
        from pipeline.text import _local_now

        tz = _local_now().tzinfo
        history: List[str] = []
        participants: List[Tuple[str, str, int]] = []
        seen_authors = set()
        others_since_bot = False
        bot_seen = False
        try:
            async for hist_msg in message.channel.history(limit=_HISTORY_LIMIT + 1):
                if hist_msg.id == message.id:
                    continue
                text = hist_msg.content or ""
                if hist_msg.attachments:
                    text = (text + " [картинка/файл]").strip()
                if not text:
                    continue
                is_me = hist_msg.author.id == self._bot_user_id
                if is_me:
                    bot_seen = True
                elif not bot_seen and hist_msg.author.id != message.author.id and not hist_msg.author.bot:
                    others_since_bot = True
                speaker = config.BOT_NAME if is_me else hist_msg.author.display_name
                to = ""
                ref = hist_msg.reference.resolved if hist_msg.reference else None
                if isinstance(ref, discord.Message):
                    to = f" → {config.BOT_NAME if ref.author.id == self._bot_user_id else ref.author.display_name}"
                stamp = hist_msg.created_at.astimezone(tz).strftime("%H:%M")
                history.append(f"[{stamp}] {speaker}{to}: {text[:500]}")
                if not is_me and not hist_msg.author.bot and hist_msg.author.id not in seen_authors \
                        and hist_msg.author.id != message.author.id:
                    seen_authors.add(hist_msg.author.id)
                    participants.append(
                        (anonymise_author_id(hist_msg.author.id), hist_msg.author.display_name, hist_msg.author.id)
                    )
                if len(history) >= _HISTORY_LIMIT:
                    break
            history.reverse()  # oldest first
        except (discord.Forbidden, discord.HTTPException) as exc:
            _logger.warning(
                "Failed to fetch channel history",
                extra={"data": {"channel_id": message.channel.id, "error": str(exc)}},
            )

        reply_to_name: Optional[str] = None
        if message.reference is not None and isinstance(message.reference.resolved, discord.Message):
            ref_author = message.reference.resolved.author
            reply_to_name = config.BOT_NAME if ref_author.id == self._bot_user_id else ref_author.display_name

        return MessageContext(
            message_id=message.id,
            content=message.content or "",
            author_id=anonymise_author_id(message.author.id),
            author_display_name=message.author.display_name,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            timestamp=message.created_at.replace(tzinfo=timezone.utc),
            attachments=[a.url for a in message.attachments] + _embed_media(message),
            is_reply=is_reply,
            replied_to_bot=replied_to_bot,
            recent_channel_history=history,
            participants=participants[:5],
            reply_to_name=reply_to_name,
            others_since_bot=others_since_bot,
            aimed_at_other=(
                (reply_to_name is not None and reply_to_name != config.BOT_NAME)
                or any(m.id not in (self._bot_user_id, message.author.id) and not m.bot
                       for m in message.mentions)
            ),
            is_dm=isinstance(message.channel, discord.DMChannel),
            raw_author_id=message.author.id,
            author_role_ids=(
                [role.id for role in message.author.roles]
                if message.guild and hasattr(message.author, "roles")
                else []
            ),
        )

    def _is_addressed(self, message: discord.Message, ctx: MessageContext) -> bool:
        if ctx.is_dm or ctx.replied_to_bot:
            return True
        from utils.names import is_bot_mentioned

        if is_bot_mentioned(message, self._bot_user_id):
            return True
        from utils.names import mentions_bot_name

        return mentions_bot_name(ctx.content)

    async def _is_addressed_to_me(self, ctx: MessageContext) -> bool:
        """
        Человек, с которым она только что говорила, пишет снова без её имени.
        Ей ли это?  Явные случаи решаем сразу, спорные спрашиваем у модели.
        """
        if ctx.aimed_at_other:
            verdict, how = False, "aimed_at_other"
        elif not ctx.others_since_bot:
            verdict, how = True, "only_them"
        else:
            verdict, how = await self._llm_addressed(ctx), "llm"
        _logger.info(
            "followup_check",
            extra={"data": {"message_id": ctx.message_id, "addressed": verdict, "how": how}},
        )
        return verdict

    async def _llm_addressed(self, ctx: MessageContext) -> bool:
        """Короткий вопрос модели: ДА/НЕТ, обращаются ли к ней.  При сбое — ДА."""
        from pipeline.text import PRIORITY_CHAT

        history = "\n".join(ctx.recent_channel_history[-8:]) or "(пусто)"
        user = (
            f"Чат в Discord (старые сверху):\n{history}\n"
            f"{ctx.author_display_name}: {ctx.content}\n\n"
            f"Последнее сообщение от {ctx.author_display_name} адресовано {config.BOT_NAME} "
            "(продолжает разговор с ней, отвечает ей, спрашивает её) "
            "или кому-то другому / всем людям в чате? Ответь одним словом: ДА (ей) или НЕТ."
        )
        data = await self._pipeline._post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            {
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": user}],
                "stream": False,
                "think": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": 3, "temperature": 0.1},
            },
            timeout=30.0,
            priority=PRIORITY_CHAT,
        )
        if not data:
            return True
        answer = ((data.get("message") or {}).get("content") or "").strip().lower()
        return not (answer.startswith("нет") or answer.startswith("no"))

    def _ignore_bait(self, author_id: int) -> bool:
        """Первую провокацию отбивает, дальше всё чаще игнорит (за последние 10 минут)."""
        now = time.monotonic()
        recent = [t for t in self._bait_log.get(author_id, []) if now - t < 600]
        recent.append(now)
        self._bait_log[author_id] = recent
        n = len(recent)
        chance = config.BAIT_IGNORE_CHANCE if n <= 1 else min(0.9, config.BAIT_IGNORE_CHANCE + 0.25 * (n - 1))
        return random.random() < chance

    async def _react_to_bait(self, message: discord.Message) -> None:
        try:
            await asyncio.sleep(random.uniform(2.0, 6.0))
            await message.add_reaction(random.choice(["🙄", "🤡", "💀", "🥱"]))
        except (discord.Forbidden, discord.HTTPException):
            pass

    async def _maybe_react(self, message: discord.Message, relevance: float) -> None:
        """Иногда вместо ответа — просто реакция, как делают люди."""
        if relevance < 0.2 or random.random() > config.REACTION_PROBABILITY:
            return
        await self._react(message)

    async def _react(self, message: discord.Message) -> None:
        """
        Поставить реакцию: чаще — подходящий по смыслу эмодзи сервера (выбирает модель
        по названиям), иначе или если ничего не подошло — обычный по настроению.
        """
        emoji: Any = None
        server = [e for e in (message.guild.emojis if message.guild else []) if e.available]
        if (server and message.content and random.random() < config.SERVER_EMOJI_SHARE
                and not self._pipeline.busy):
            sample = random.sample(server, min(60, len(server)))
            try:
                name = await self._pipeline.pick_reaction(message.content, [e.name for e in sample])
            except Exception as exc:  # noqa: BLE001
                name = ""
                _logger.warning("pick_reaction failed", extra={"data": {"error": str(exc)}})
            emoji = next((e for e in sample if e.name == name), None)
        if emoji is None:
            emoji = random.choice(_REACTIONS.get(self._state.mood.value, _REACTIONS["default"]))
        try:
            await asyncio.sleep(random.uniform(1.0, 4.0))
            await message.add_reaction(emoji)
            _logger.info("Reaction added", extra={"data": {"message_id": message.id, "emoji": str(emoji)}})
        except (discord.Forbidden, discord.HTTPException):
            pass

    async def _books_note(self, message: discord.Message, ctx: MessageContext) -> str:
        """Подаренная книга / просьба почитать что-то / вопрос про её полку → пометка для ответа."""
        from pathlib import Path

        from memory.library import BOOK_EXTS

        lib = self.library
        notes: List[str] = []
        for a in message.attachments:
            ext = Path(a.filename).suffix.lower()
            if ext not in BOOK_EXTS:
                continue
            if a.size > config.BOOK_MAX_MB * 1024 * 1024:
                notes.append(f"тебе прислали файл «{a.filename}», но он слишком большой — ты его не взяла")
                continue
            dest = lib.dir / re.sub(r'[<>:"/\\|?*]+', "_", a.filename)
            n = 1
            while dest.exists():
                dest = dest.with_name(f"{dest.stem} ({n}){dest.suffix}")
                n += 1
            await a.save(dest)
            book = await lib.add_file(dest, ctx.author_display_name)
            if book is None:
                dest.unlink(missing_ok=True)
                notes.append(f"тебе прислали «{a.filename}», но открыть не получилось (пусто или битый файл) — скажи честно")
                continue
            cur = await lib.current()
            what = "комикс" if book["kind"] == "comic" else "книгу"
            later = ("ничего другого ты сейчас не читала — начнёшь с неё" if cur and cur["id"] == book["id"]
                     else f"сейчас ты читаешь «{cur['title']}», эту возьмёшь следом" if cur else "")
            notes.append(f"{ctx.author_display_name} подарил(а) тебе {what} «{book['title']}»"
                         + (f" ({book['author']})" if book["author"] else "")
                         + f" — ты положила на свою полку{'; ' + later if later else ''}. Ты её ещё НЕ читала — "
                         "не придумывай, о чём она; можешь поблагодарить по-своему")
        text = ctx.content or ""
        url_m = _URL_RE.search(text)
        if (not notes and url_m and self.book_link_hook is not None
                and (_READ_LINK_RE.search(text) or _READER_HOST_RE.search(url_m.group(0)))
                and not _IMAGE_URL_RE.search(url_m.group(0))):
            asyncio.create_task(self.book_link_hook(message, ctx, url_m.group(0)))
            notes.append(f"{ctx.author_display_name} скинул(а) тебе ссылку почитать — ты её открываешь прямо сейчас. "
                         "Скажи коротко, что глянешь; пока НЕ знаешь, что там и откроется ли — не выдумывай")
            return "; ".join(notes)
        if not notes and _READ_CMD_RE.search(text):
            book = await lib.find(_READ_CMD_RE.sub(" ", text))
            req = None
            if book is None and self.book_hook is not None:
                req = await self._pipeline.parse_book_request(text, ctx.recent_channel_history)
            if req is not None and req["kind"] in ("comic", "manga"):
                notes.append(f"тебе советуют {'мангу' if req['kind'] == 'manga' else 'комикс'} «{req['title']}» — "
                             "сама ты комиксы и мангу найти не можешь (бесплатно и легально их почти нигде нет); "
                             "скажи, что с удовольствием почитаешь, если скинут файл (cbz или архив с картинками)")
            elif req is not None:
                asyncio.create_task(self.book_hook(message, ctx, req))
                notes.append(f"{ctx.author_display_name} советует тебе «{req['title']}»"
                             + (f" ({req['author']})" if req["author"] else "")
                             + " — у тебя её нет, ты прямо сейчас пошла искать в бесплатных легальных библиотеках "
                             "(Викитека, Project Gutenberg — там классика). Скажи коротко, что поищешь; "
                             "не обещай, что точно найдёшь, и не рассказывай, о чём книга, если не читала")
            elif book is not None:
                await lib.switch_to(book["id"])
                notes.append(f"тебя попросили почитать «{book['title']}» — ты взялась за неё"
                             + (" (продолжишь с того места, где остановилась)" if book["pos"] else "")
                             + "; сама ещё не читала дальше того, что знаешь")
            elif _BOOKS_RE.search(text):
                titles = await lib.titles()
                notes.append("такой книги у тебя нет; на полке: " + (", ".join(titles) or "пусто")
                             + ". Можно попросить прислать файл (txt, fb2, epub, pdf или комикс cbz)")
        elif not notes and _BOOKS_RE.search(text) and re.search(r"\?|какие|что|есть ли|покажи|список", text, re.I):
            titles = await lib.titles()
            notes.append("твоя полка: " + (", ".join(titles) if titles else "пока пустая — книг тебе ещё не дарили")
                         + ". Книгу или комикс можно прислать тебе файлом в личку (txt, fb2, epub, pdf, cbz)")
        return "; ".join(notes)

    async def _absence_note(self, ctx: MessageContext) -> str:
        """
        Человеческая реакция на тишину (мягко, без обид всерьёз):
        её сообщение в личке висело без ответа — может подколоть; хороший знакомый
        долго не появлялся — рада, что вернулся.
        """
        social = self._memory.social
        now = datetime.now(tz=timezone.utc)
        name = ctx.author_display_name
        try:
            if ctx.is_dm:
                lines = await social.last_lines(ctx.channel_id, 2)
                if lines and lines[0][1]:
                    at, _, text = lines[0]
                    hours = (now - at).total_seconds() / 3600
                    # «Висит без ответа» — только если ответ правда был нужен: вопрос или она сама начала
                    # разговор.  Реакция на его же картинку («какой милый кролик!») ответа не ждёт — иначе
                    # модель додумывала «я всё ещё жду ответа на мой вопрос», которого не было.
                    started = len(lines) < 2 or lines[1][1] or (at - lines[1][0]).total_seconds() > 6 * 3600
                    if hours >= config.IGNORED_AFTER_HOURS and ("?" in text or started):
                        when = f"{int(hours)} ч назад" if hours < 48 else f"{int(hours // 24)} дн назад"
                        return (f"ты писала {name} {when}: «{text[:150]}» — и он(а) так и не ответил(а); можешь "
                                "мимоходом, по-доброму подколоть за это одной фразой («о, вспомнил(а) про меня»), "
                                "но не дуйся, не упрекай всерьёз и не приписывай себе вопросов, которых не было")
            seen = await social.last_seen(ctx.author_id)
            if seen is not None:
                days = (now - seen).total_seconds() / 86400
                if days >= config.MISS_AFTER_DAYS and await social.get_affinity(ctx.author_id) >= 0.3:
                    return (f"{name} не появлялся(лась) {int(days)} дн — ты рада, что он(а) вернулся(лась), "
                            "можешь это показать («о, живой!», «куда пропал(а)?»), без упрёков")
        except Exception as exc:  # noqa: BLE001
            _logger.warning("absence note failed", extra={"data": {"error": str(exc)}})
        return ""

    async def _set_reminder(self, message: discord.Message, ctx: MessageContext) -> str:
        """Разобрать «напомни …» и сохранить.  Возвращает пометку для её ответа ('' — не просьба)."""
        from pipeline.text import _local_now

        now = _local_now()
        system, user = reminders_mod.parse_prompt(ctx.content, ctx.author_display_name, now)
        data = await self._pipeline.call_json(system, user, reminders_mod.PARSE_SCHEMA, num_predict=200)
        _logger.info("Reminder parsed", extra={"data": {"message_id": message.id, "result": data}})
        if not data or not data.get("ok"):
            return ("тебя, кажется, просят что-то напомнить, но непонятно когда — коротко переспроси время"
                    if data is not None and not data.get("ok") and re.search(r"\d|завтра|через|вечер|утр", ctx.content)
                    else "")
        due = reminders_mod.parse_due(str(data.get("when") or ""), now)
        due = reminders_mod.prefer_evening(due, ctx.content, now)
        what = str(data.get("what") or "").strip() or "то, о чём просил(а)"
        if due is None:
            return "тебя просят напомнить, но время получилось непонятным или уже прошло — коротко переспроси, когда"
        is_dm = ctx.is_dm or data.get("where") == "dm"
        await self.reminders.add(message.author.id, ctx.author_display_name,
                                 0 if is_dm else ctx.channel_id, is_dm, due, what, ctx.content)
        when = reminders_mod.human_when(due.astimezone(now.tzinfo), now)
        _logger.info("Reminder set", extra={"data": {"user_id": message.author.id, "due": due.isoformat(),
                                                     "what": what, "dm": is_dm}})
        return (f"ты поставила себе напоминание: {when} напомнить {ctx.author_display_name} — «{what}»"
                + (" (напишешь в личку)" if is_dm and not ctx.is_dm else "")
                + ". Коротко подтверди своими словами и назови время")

    def _buried(self, channel: Any, target: Optional[discord.Message]) -> Optional[discord.Message]:
        """*target*, если после него в канале уже есть другие сообщения (тогда без reply непонятно, на что ответ)."""
        if target is None:
            return None
        return target if self._last_msg_id.get(getattr(channel, "id", 0), 0) > target.id else None

    async def _send_with_typing(
        self,
        channel: discord.abc.Messageable,
        content: str,
        typing_duration: float,
        reply_to: Optional[discord.Message] = None,
        already_waited: float = 0.0,
        media: Optional[List[Any]] = None,
        reply_if_buried: Optional[discord.Message] = None,
    ) -> None:
        """
        Show the typing indicator for *typing_duration* seconds then send.

        If *reply_to* is provided the bot uses ``message.reply()`` so
        Discord renders the reply arrow pointing at the original message.
        Otherwise a plain ``channel.send()`` is used (DMs, proactive messages).
        *reply_if_buried* — reply only if, by the time she sends, the channel already has newer
        messages after it (he wrote more, or she answered something else) — otherwise it's unclear
        what she's answering.

        Handles :exc:`discord.Forbidden` and :exc:`discord.HTTPException`
        without crashing.
        """
        if not config.guild_allowed(getattr(getattr(channel, "guild", None), "id", None)):
            _logger.warning("TEST MODE: send blocked", extra={"data": {"channel_id": getattr(channel, "id", None)}})
            return
        # Несколько строк без кода — несколько сообщений подряд, как пишут люди.
        chunks = [content] if content.strip() else []
        if chunks and "```" not in content:
            lines = [l for l in content.splitlines() if l.strip()]
            if 2 <= len(lines) <= 4:
                chunks = lines
        try:
            # Один ответ за раз: пока она «допечатывает» одному, следующий ждёт — без перемешивания.
            async with self._channel_lock(channel):
                for i, chunk in enumerate(chunks):
                    # Сколько человек печатал бы это сообщение.
                    cps = max(1.0, config.CHAT_TYPING_CPS) * random.uniform(0.8, 1.25)
                    delay = min(config.CHAT_TYPING_MAX, len(chunk) / cps)
                    if i == 0:
                        # Пока модель думала, человек тоже «набирал» — это время не ждём второй раз.
                        floor = typing_duration * 0.5 if already_waited < typing_duration else 0.0
                        delay = max(floor, delay - already_waited, 1.0)
                    else:
                        delay = max(delay, 1.2)
                        await asyncio.sleep(random.uniform(0.3, 1.2))  # пауза перед следующим
                    async with channel.typing():
                        await asyncio.sleep(delay)
                    if i == 0:
                        reply_to = reply_to or self._buried(channel, reply_if_buried)
                    sent = None
                    if i == 0 and reply_to is not None:
                        try:
                            sent = await reply_to.reply(chunk, mention_author=False)
                        except discord.HTTPException:
                            # Сообщение успели удалить — отвечаем просто в канал.
                            sent = await channel.send(chunk)
                    else:
                        sent = await channel.send(chunk)
                    if sent is not None:
                        self._seen(channel.id, sent.id)
                if not chunks:
                    reply_to = reply_to or self._buried(channel, reply_if_buried)
                for i, path in enumerate(media or []):
                    await asyncio.sleep(random.uniform(0.8, 2.0) if chunks or i else 0.5)
                    if not chunks and i == 0 and reply_to is not None:
                        try:
                            await reply_to.reply(file=discord.File(str(path)), mention_author=False)
                            continue
                        except discord.HTTPException:
                            pass
                    await channel.send(file=discord.File(str(path)))
        except discord.Forbidden:
            _logger.warning(
                "Missing permissions to send message",
                extra={"data": {"channel_id": getattr(channel, "id", None)}},
            )
        except discord.HTTPException as exc:
            _logger.error(
                "HTTP error while sending message",
                extra={"data": {"error": str(exc)}},
            )
