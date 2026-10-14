"""
voice/manager.py — голосовой режим Aura (Day 4 + Day 5).

Поток данных::

    Discord voice ──► VoiceListener (VAD, поток) ──► asyncio.Queue
        ──► SpeechToText (Whisper) ──► решение «отвечать?»
        ──► TextPipeline.generate() ──► TextToSpeech (Silero) ──► vc.play()

Когда отвечать:
  * в распознанной фразе есть имя бота (``VOICE_NAME_ALIASES``), или
  * в канале, кроме бота, ровно один человек (разговор наедине).

Каждая распознанная фраза пишется в память (MemoryStore) так же, как
текстовое сообщение, и попадает в историю голосового канала, которую
видит модель.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Tuple

import discord
import numpy as np

import config
from utils.logger import anonymise_author_id, get_logger

if TYPE_CHECKING:
    from behavior.state import BotState
    from memory.store import MemoryStore
    from pipeline.text import TextPipeline

_logger = get_logger("bot.voice")

_HISTORY_LEN = 16

# ---------------------------------------------------------------------------
# Text commands («Аура, зайди в войс» / «Аура, выйди»)
# ---------------------------------------------------------------------------

_JOIN_RE = re.compile(
    r"(зайди|заходи|залетай|подключись|присоединяйся|го\b|иди)\s*(ко мне\s*)?(в|во)?\s*"
    r"(войс|войсе|голосов|голос|канал|вк\b|voice)"
    r"|\bjoin\b|\bзайди\b|\bзаходи\b|\bзалетай\b",
    re.IGNORECASE,
)
_LEAVE_RE = re.compile(
    r"(выйди|выходи|уйди|покинь|отключись|ливни|свали|leave)\b",
    re.IGNORECASE,
)

_JOIN_LINES = ["хм. ладно, захожу.", "ну иду, иду.", "не то чтобы мне хотелось... но ладно."]
_ALREADY_LINES = ["я уже тут, вообще-то.", "я и так здесь. глаза разуй."]
_NOT_IN_VOICE_LINES = ["а ты сам-то где? зайди в войс сначала.", "ты не в голосовом. гений."]
_LEAVE_LINES = ["ну и пожалуйста.", "ухожу. не скучай.", "хм. ладно."]
_FAIL_LINES = ["не могу зайти. права проверь.", "что-то мешает подключиться. хм."]

# Голосовая команда выхода: Whisper пишет «выйти своица», «выйди из войса» и т.п.
_VOICE_LEAVE_RE = re.compile(r"(выйди|выйти|выходи|уйди|покинь|отключись|ливни|ливай)", re.IGNORECASE)


def _pick(lines: list) -> str:
    import random

    return random.choice(lines)


def parse_voice_command(content: str) -> Optional[str]:
    """Вернуть 'join', 'leave' или None.  Вызывать только для сообщений, адресованных боту."""
    if _LEAVE_RE.search(content) and re.search(r"войс|голос|канал|voice|leave|выйди|ливни|отключись", content, re.I):
        return "leave"
    if _JOIN_RE.search(content):
        return "join"
    return None


from utils.names import mentions_bot_name  # noqa: E402  (re-exported for core.bot)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class _GuildSession:
    vc: "discord.VoiceClient"
    channel_id: int
    queue: "asyncio.Queue[Tuple[int, np.ndarray]]"
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    # user_id → последнее известное имя (человек мог уже выйти из канала).
    names: Dict[int, str] = field(default_factory=dict)
    worker: Optional["asyncio.Task"] = None  # type: ignore[type-arg]
    listener: object = None
    last_utterance: float = 0.0
    last_bot_reply: float = 0.0          # когда она последний раз говорила
    last_bot_reply_to: int = 0           # кому
    last_unprompted: float = 0.0         # когда последний раз вмешалась без имени
    warm_task: Optional["asyncio.Task"] = None  # type: ignore[type-arg]
    utterance_seq: int = 0
    pending: Deque["_ReplyJob"] = field(default_factory=deque)
    reply_task: Optional["asyncio.Task"] = None  # type: ignore[type-arg]
    reply_optional: bool = False
    reply_speaking: bool = False


@dataclass
class _ReplyJob:
    ctx: object
    user_id: int
    seq: int
    heard_at: float
    stt_s: float
    optional: bool
    interest: float
    others: list


class _Player:
    """Очередь воспроизведения: куски ответа играются друг за другом, пока остальное ещё генерируется."""

    def __init__(self, manager: "VoiceManager", session: _GuildSession) -> None:
        self._manager, self._session = manager, session
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            pcm = await self._queue.get()
            if pcm is None:
                return
            if self._session.vc.is_connected():
                await self._manager._play(self._session, pcm)

    def put(self, pcm: bytes) -> None:
        if pcm:
            self._queue.put_nowait(pcm)

    async def finish(self) -> None:
        self._queue.put_nowait(None)
        await self._task

    async def abort(self) -> None:
        self._task.cancel()
        try:
            if self._session.vc.is_playing():
                self._session.vc.stop()
        except Exception:  # noqa: BLE001
            pass


_QUESTION_START = re.compile(
    r"^(а\s+)?(как|что|чё|че|почему|зачем|кто|где|когда|куда|откуда|сколько|какой|какая|какое|какие|"
    r"чей|можно|может|правда|серьёзно|серьезно|реально)\b",
    re.IGNORECASE,
)
_GROUP_ADDRESS = re.compile(
    r"\b(ребят|ребята|народ|пацаны|девчонки|кто[- ]нибудь|кто[- ]то знает|как думаете|вы\s+как|все\s+слышали|"
    r"а\s+вы|что\s+скажете|согласны)\b",
    re.IGNORECASE,
)
_ABOUT_HER = re.compile(r"\b(она|её|ее|ей|бот|ботик|нейросет\w*|ии)\b", re.IGNORECASE)
_INTERESTS = re.compile(
    r"(камн|кристалл|минерал|аметист|турмалин|топаз|геншин|genshin|хср|honkai|хонкай|валорант|valorant|"
    r"аниме|кошк|кот\b|магия|магии|таро|геолог)",
    re.IGNORECASE,
)


def voice_interest(text: str, speaker_id: int, session: "_GuildSession", now: float) -> float:
    """
    Насколько уместно ей вмешаться в разговор, если её не звали по имени.
    Быстрая эвристика без модели (0..1); окончательно решает сама модель.
    """
    words = len(text.split())
    score = 0.0
    since_bot = now - session.last_bot_reply if session.last_bot_reply else 1e9
    followup = since_bot < config.VOICE_FOLLOWUP_WINDOW_S
    if followup:
        score += 0.45
        if speaker_id == session.last_bot_reply_to:
            score += 0.15
    if "?" in text or _QUESTION_START.search(text.strip()):
        score += 0.25
    if _GROUP_ADDRESS.search(text):
        score += 0.2
    if since_bot < 120 and _ABOUT_HER.search(text):
        score += 0.25
    if _INTERESTS.search(text):
        score += 0.3 + random.uniform(0.0, 0.2)  # любимые темы — иногда не удержится
    if words < 3:
        score -= 0.3
    # Не встревать слишком часто без приглашения.
    if not followup and now - session.last_unprompted < config.VOICE_PROACTIVE_COOLDOWN_S:
        score -= 0.5
    return max(0.0, min(1.0, score))


class VoiceManager:
    """Подключение к голосовым каналам, распознавание и озвучка."""

    def __init__(
        self,
        client: discord.Client,
        state: "BotState",
        pipeline: "TextPipeline",
        memory: "MemoryStore",
    ) -> None:
        from voice.stt import SpeechToText
        from voice.tts import make_tts

        self._client = client
        self._state = state
        self._pipeline = pipeline
        self._memory = memory
        self._stt = SpeechToText()
        self._tts = make_tts()
        self._sessions: Dict[int, _GuildSession] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._fillers: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def preload(self) -> None:
        """Загрузить Whisper и TTS в фоне, чтобы первый ответ был быстрым."""
        for name, loader in (("whisper", self._stt.load), (f"{config.TTS_ENGINE}_tts", self._tts.load)):
            try:
                await loader()
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "Voice model preload failed",
                    extra={"data": {"model": name, "error": str(exc)}},
                    exc_info=True,
                )
        # Заранее озвученные «хм», «ну...» — играются мгновенно, пока она думает.
        if config.VOICE_FILLERS and not self._fillers and self._tts.ready:
            for word in ("м-м.", "секунду."):
                try:
                    pcm = await self._tts.synthesize(word)
                    if pcm:
                        self._fillers.append(pcm)
                except Exception:  # noqa: BLE001
                    pass

    def recently_active(self, seconds: float) -> bool:
        """Был ли живой разговор в каком-нибудь войсе за последние *seconds* секунд."""
        now = time.monotonic()
        return any(now - s.last_utterance < seconds for s in self._sessions.values())

    async def close(self) -> None:
        for guild_id in list(self._sessions):
            await self.leave(guild_id)
        self._stt.close()
        self._tts.close()

    @property
    def in_voice(self) -> bool:
        """Сидит ли она сейчас хоть в одном войсе."""
        return bool(self._sessions)

    def is_connected(self, guild_id: int) -> bool:
        return guild_id in self._sessions

    # ------------------------------------------------------------------
    # Text command entry point (called from CompanionBot.on_message)
    # ------------------------------------------------------------------

    async def handle_command(self, message: discord.Message, command: str) -> bool:
        """Выполнить команду.  False — команда не наша, пусть сообщение обработает чат."""
        guild = message.guild
        if guild is None:
            return False

        if command == "leave":
            if not self.is_connected(guild.id):
                # «Аура, уйди» вне войса — это просто реплика, а не команда.
                return False
            await self.leave(guild.id)
            await message.reply(_pick(_LEAVE_LINES), mention_author=False)
            return True

        voice = getattr(message.author, "voice", None)
        if voice is None or voice.channel is None:
            await message.reply(_pick(_NOT_IN_VOICE_LINES), mention_author=False)
            return True

        session = self._sessions.get(guild.id)
        if session is not None and session.channel_id == voice.channel.id:
            await message.reply(_pick(_ALREADY_LINES), mention_author=False)
            return True

        ok = await self.join(voice.channel)
        await message.reply(_pick(_JOIN_LINES if ok else _FAIL_LINES), mention_author=False)
        return True

    # ------------------------------------------------------------------
    # Join / leave
    # ------------------------------------------------------------------

    async def join(self, channel: discord.VoiceChannel, _retry: bool = False) -> bool:
        from discord.ext import voice_recv

        from voice import dave_patch
        from voice.listener import VoiceListener

        dave_patch.apply()
        dave_patch.reset_stats()
        self._loop = asyncio.get_running_loop()
        guild_id = channel.guild.id

        if guild_id in self._sessions:
            await self.leave(guild_id)

        try:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, timeout=20)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Failed to connect to voice",
                extra={"data": {"channel_id": channel.id, "error": str(exc)}},
                exc_info=True,
            )
            return False

        session = _GuildSession(vc=vc, channel_id=channel.id, queue=asyncio.Queue(maxsize=20))
        assert self._client.user is not None

        def _on_utterance(user_id: int, audio: np.ndarray, _gid: int = guild_id) -> None:
            # Вызывается из потока VAD — переносим в event loop.
            if self._loop is None:
                return
            self._loop.call_soon_threadsafe(self._enqueue, _gid, user_id, audio)

        listener = VoiceListener(self._client.user.id, _on_utterance)
        session.listener = listener
        vc.listen(listener)  # type: ignore[attr-defined]
        session.worker = asyncio.create_task(self._worker(guild_id))
        self._sessions[guild_id] = session
        if config.VOICE_WARMUP:
            session.warm_task = asyncio.create_task(self._warm_loop(guild_id))

        _logger.info(
            "Joined voice channel",
            extra={"data": {"guild_id": guild_id, "channel_id": channel.id, "channel": channel.name}},
        )

        # Если модели ещё не загружены — грузим в фоне.
        if not (self._stt.ready and self._tts.ready):
            asyncio.create_task(self.preload())
        asyncio.create_task(self._rx_watchdog(channel, vc, _retry))
        return True

    async def _rx_watchdog(self, channel: discord.VoiceChannel, vc: Any, retried: bool) -> None:
        """Через 20 с после входа проверить, что звук реально расшифровывается.

        Бывает, что DAVE-сессия не получила ключи (например, после того как бот
        упал, не выйдя из войса): кадры приходят, но ни один не расшифрован —
        она «глухая».  Тогда один раз переподключаемся.
        """
        from voice import dave_patch

        await asyncio.sleep(20)
        session = self._sessions.get(channel.guild.id)
        if session is None or session.vc is not vc:
            return
        st = dave_patch.stats()
        conn = getattr(vc, "_connection", None)
        dave = getattr(conn, "dave_session", None)
        info = {
            **st,
            "dave_version": getattr(conn, "dave_protocol_version", None),
            "dave_ready": getattr(dave, "ready", None),
            "dave_status": str(getattr(dave, "status", None)),
        }
        _logger.info("voice_rx_check", extra={"data": info})
        deaf = st["dave"] > 100 and st["ok"] == 0
        if deaf and not retried:
            _logger.warning("Voice is deaf (nothing decrypts) — rejoining", extra={"data": info})
            await self.leave(channel.guild.id)
            await asyncio.sleep(2)
            await self.join(channel, _retry=True)
        elif deaf:
            _logger.error("Voice still deaf after rejoin", extra={"data": info})

    async def leave(self, guild_id: int) -> None:
        session = self._sessions.pop(guild_id, None)
        if session is None:
            return
        if session.worker is not None:
            session.worker.cancel()
        if session.reply_task is not None:
            session.reply_task.cancel()
        if session.warm_task is not None:
            session.warm_task.cancel()
        try:
            if session.vc.is_listening():  # type: ignore[attr-defined]
                session.vc.stop_listening()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            await session.vc.disconnect(force=True)
        except Exception:  # noqa: BLE001
            pass
        _logger.info("Left voice channel", extra={"data": {"guild_id": guild_id}})

    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        """Выйти, если в канале не осталось людей; убрать сессию, если бота выкинули."""
        session = self._sessions.get(member.guild.id)
        if session is None:
            return

        if self._client.user is not None and member.id == self._client.user.id:
            if after.channel is None:  # бота отключили/кикнули
                self._sessions.pop(member.guild.id, None)
                if session.worker:
                    session.worker.cancel()
            elif after.channel.id != session.channel_id:
                session.channel_id = after.channel.id  # перетащили в другой канал
            return

        if not config.VOICE_AUTO_LEAVE:
            return
        channel = session.vc.channel
        humans = [m for m in getattr(channel, "members", []) if not m.bot]
        if not humans:
            await self.leave(member.guild.id)

    # ------------------------------------------------------------------
    # Utterance processing
    # ------------------------------------------------------------------

    def _enqueue(self, guild_id: int, user_id: int, audio: np.ndarray) -> None:
        session = self._sessions.get(guild_id)
        if session is None:
            return
        if session.queue.full():
            try:
                session.queue.get_nowait()  # выкидываем самую старую фразу
            except asyncio.QueueEmpty:
                pass
        session.queue.put_nowait((user_id, audio))

    async def _worker(self, guild_id: int) -> None:
        """
        Стадия 1 — слух: распознаём фразы по мере поступления, не дожидаясь,
        пока она договорит или подумает.  Ответы идут отдельной задачей
        (``_respond``), поэтому распознавание никогда не стоит в очереди
        за моделью.
        """
        while True:
            session = self._sessions.get(guild_id)
            if session is None:
                return
            user_id, audio = await session.queue.get()
            try:
                await self._hear(session, user_id, audio)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "Voice utterance processing failed",
                    extra={"data": {"error": str(exc)}},
                    exc_info=True,
                )

    async def _hear(self, session: _GuildSession, user_id: int, audio: np.ndarray) -> None:
        from core.message_handler import MessageContext

        guild = session.vc.guild
        display_name = self._resolve_name(session, user_id)
        member = guild.get_member(user_id)

        t0 = time.monotonic()
        names = [m.display_name for m in getattr(session.vc.channel, "members", []) if not m.bot]
        text = await self._stt.transcribe(audio, names)
        stt_s = time.monotonic() - t0
        if not text:
            return
        heard_at = time.monotonic()
        session.last_utterance = heard_at
        session.utterance_seq += 1
        seq = session.utterance_seq

        channel = session.vc.channel
        humans = [m for m in getattr(channel, "members", []) if not m.bot]
        for m in humans:
            session.names[m.id] = m.display_name
        addressed = mentions_bot_name(text)
        alone = len(humans) == 1
        direct = addressed or (alone and config.VOICE_REPLY_WHEN_ALONE)

        interest = 0.0
        candidate = False
        if not direct and config.VOICE_PROACTIVE:
            interest = voice_interest(text, user_id, session, heard_at)
            candidate = interest >= config.VOICE_PROACTIVE_THRESHOLD

        _logger.info(
            "voice_transcript",
            extra={"data": {
                "author_id_hash": anonymise_author_id(user_id),
                "author_display_name": display_name,
                "channel_id": session.channel_id,
                "chars": len(text),
                "text": text if config.VOICE_LOG_TRANSCRIPTS else None,
                "stt_seconds": round(stt_s, 2),
                "addressed": addressed,
                "alone": alone,
                "interest": round(interest, 2),
                "candidate": candidate,
            }},
        )

        # Голосовая команда «Аура, выйди из войса».
        if addressed and _VOICE_LEAVE_RE.search(text):
            pcm = await self._tts.synthesize(_pick(_LEAVE_LINES))
            if pcm and session.vc.is_connected():
                await self._play(session, pcm)
            await self.leave(guild.id)
            return

        history_before = list(session.history)
        session.history.append(f"{display_name}: {text}")

        others = [m.display_name for m in humans if m.id != user_id]
        ctx = MessageContext(
            message_id=int(time.time() * 1000),
            content=text,
            author_id=anonymise_author_id(user_id),
            author_display_name=display_name,
            channel_id=session.channel_id,
            guild_id=guild.id,
            timestamp=datetime.now(tz=timezone.utc),
            recent_channel_history=history_before,
            raw_author_id=user_id,
            author_role_ids=[r.id for r in getattr(member, "roles", [])] if member else [],
            is_voice=True,
            voice_participants=[display_name] + others,
            mood_label="",
        )
        await self._memory.note_incoming(ctx)

        job = _ReplyJob(ctx=ctx, user_id=user_id, seq=seq, heard_at=heard_at, stt_s=stt_s,
                        optional=not direct, interest=interest,
                        others=[(anonymise_author_id(m.id), m.display_name, m.id) for m in humans if m.id != user_id])

        running = session.reply_task is not None and not session.reply_task.done()
        if direct:
            # К ней обратились: необязательную реплику, которая ещё не зазвучала, отменяем.
            if running and session.reply_optional and not session.reply_speaking:
                session.reply_task.cancel()
                running = False
            session.pending.append(job)
            if not running:
                session.reply_task = asyncio.create_task(self._reply_loop(session))
        elif candidate and not running and not session.pending:
            session.pending.append(job)
            session.reply_task = asyncio.create_task(self._reply_loop(session))
        else:
            await self._memory.log_event(ctx, bot_replied=False)

    async def _reply_loop(self, session: _GuildSession) -> None:
        """Стадия 2 — ответы по очереди (обращения к ней всегда по порядку)."""
        while session.pending:
            job = session.pending.popleft()
            session.reply_optional = job.optional
            session.reply_speaking = False
            try:
                await self._respond(session, job)
            except asyncio.CancelledError:
                _logger.info("voice_reply_cancelled", extra={"data": {"optional": job.optional}})
                await self._memory.log_event(job.ctx, bot_replied=False)
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.error("Voice reply failed", extra={"data": {"error": str(exc)}}, exc_info=True)
            finally:
                session.reply_speaking = False

    async def _respond(self, session: _GuildSession, job: "_ReplyJob") -> None:
        from behavior.engine import mood_label

        ctx, user_id = job.ctx, job.user_id
        display_name = ctx.author_display_name
        t_start = time.monotonic()

        ctx.mood_label = mood_label(self._state)
        ctx.memory_snippets = (await self._memory.recall(ctx))[: config.VOICE_MEMORY_SNIPPETS]
        ctx.social = await self._memory.social.build_context(
            session.channel_id, (ctx.author_id, display_name, user_id), job.others, ctx.content,
            max_people=config.VOICE_SOCIAL_PEOPLE, max_statements=3, max_facts=config.VOICE_SOCIAL_FACTS,
        )
        ctx.affinity = ctx.social.speaker_affinity
        ctx.relationship_score = await self._memory.get_relationship_score(ctx.author_id)

        if job.optional:
            # Пока решали — разговор ушёл дальше или кто-то говорит? Тогда не встреваем.
            if session.utterance_seq > job.seq + 1 or self._someone_speaking(session):
                _logger.info("voice_stayed_silent", extra={"data": {
                    "interest": round(job.interest, 2), "reason": "conversation moved on"}})
                await self._memory.log_event(ctx, bot_replied=False)
                return
            # Встревать или нет, модель решает в том же запросе, что и сам ответ ([молчу]),
            # а не отдельным вопросом «ДА/НЕТ» — это экономило бы лишние ~5 с на каждую реплику.
            ctx.voice_optional = True

        player = _Player(self, session)
        if not job.optional and self._fillers and random.random() < config.VOICE_FILLER_PROBABILITY:
            player.put(random.choice(self._fillers))
            session.reply_speaking = True

        t1 = time.monotonic()
        first_audio_s = -1.0
        tts_s = 0.0
        pieces = []
        try:
            async for piece in self._pipeline.generate_stream(ctx):
                pieces.append(piece)
                t2 = time.monotonic()
                pcm = await self._tts.synthesize(piece, self._state.mood.value)
                tts_s += time.monotonic() - t2
                if pcm:
                    if first_audio_s < 0:
                        first_audio_s = time.monotonic() - job.heard_at
                    session.reply_speaking = True
                    player.put(pcm)
        except asyncio.CancelledError:
            await player.abort()
            raise
        response = " ".join(pieces).strip()
        llm_s = time.monotonic() - t1
        if not response:
            # Не звали по имени, и модель решила промолчать ([молчу]).
            await player.finish()
            _logger.info("voice_stayed_silent", extra={"data": {
                "interest": round(job.interest, 2), "reason": "model chose silence", "llm_seconds": round(llm_s, 2)}})
            await self._memory.log_event(ctx, bot_replied=False)
            return

        session.history.append(f"{config.BOT_NAME} (ответила {display_name}): {response}")
        await self._memory.log_bot_reply(ctx, response)
        self._state.record_reply(session.channel_id)
        session.last_bot_reply = time.monotonic()
        session.last_bot_reply_to = user_id
        if job.optional:
            session.last_unprompted = time.monotonic()

        await player.finish()
        _logger.info(
            "voice_reply_sent",
            extra={"data": {
                "channel_id": session.channel_id,
                "response_length": len(response),
                "replied_to": display_name,
                "unprompted": job.optional,
                "text": response if config.VOICE_LOG_TRANSCRIPTS else None,
                "stt_seconds": round(job.stt_s, 2),
                "prepare_seconds": round(t1 - t_start, 2),
                "first_audio_seconds": round(first_audio_s, 2),
                "llm_seconds": round(llm_s, 2),
                "tts_seconds": round(tts_s, 2),
            }},
        )
        await self._memory.log_event(ctx, bot_replied=True)

    @staticmethod
    def _someone_speaking(session: _GuildSession) -> bool:
        listener = session.listener
        return bool(listener is not None and getattr(listener, "someone_speaking", lambda: False)())

    async def _warm_loop(self, guild_id: int) -> None:
        """
        Держим модель «прогретой» под голосовой режим: сразу после входа и
        потом раз в минуту в простое — крошечный запрос с тем же началом
        промпта.  Ollama кэширует посчитанное начало, и первый ответ в войсе
        не тратит 1–3 секунды на перечитывание характера.
        """
        first = True
        while guild_id in self._sessions:
            try:
                if first or (
                    not self._pipeline.busy and self._pipeline.seconds_since_last_request > 50
                ):
                    await self._pipeline.warmup(voice=True)
                    first = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _logger.debug("warmup failed", extra={"data": {"error": str(exc)}})
            await asyncio.sleep(60)

    def _resolve_name(self, session: _GuildSession, user_id: int) -> str:
        """Имя говорящего: ник на сервере → глобальное имя → запомненное ранее → ID."""
        member = session.vc.guild.get_member(user_id)
        if member is not None:
            session.names[user_id] = member.display_name
            return member.display_name
        if user_id in session.names:
            return session.names[user_id]
        user = self._client.get_user(user_id) if self._client else None
        if user is not None:
            name = user.global_name or user.name
            session.names[user_id] = name
            return name
        return f"кто-то ({str(user_id)[-4:]})"

    async def _play(self, session: _GuildSession, pcm: bytes) -> None:
        vc = session.vc
        # Ждём, пока закончится предыдущая реплика.
        for _ in range(300):
            if not vc.is_playing():
                break
            await asyncio.sleep(0.1)
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _after(err: Optional[Exception]) -> None:
            if err:
                _logger.warning("Playback error", extra={"data": {"error": str(err)}})
            loop.call_soon_threadsafe(done.set)

        vc.play(self._tts.to_audio_source(pcm), after=_after)
        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            vc.stop()
