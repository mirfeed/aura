"""
voice/listener.py — Day 4: приём голоса и нарезка речи на фразы (VAD).

discord-ext-voice-recv вызывает ``AudioSink.write()`` из своего потока для
каждого 20-мс кадра каждого говорящего (48 кГц, стерео, int16).  Мы
не делаем там ничего тяжёлого: кадр кладётся в очередь, а отдельный
рабочий поток:

1. переводит звук в моно 16 кГц float32 (формат Whisper и Silero VAD);
2. прогоняет его через Silero VAD (отдельная копия модели на каждого
   пользователя, потому что у модели есть внутреннее состояние);
3. собирает фразу от начала речи до паузы ``VAD_SILENCE_MS``;
4. отдаёт готовую фразу в ``on_utterance(user_id, audio)``.

Если пакет silero-vad не установлен, используется простой детектор по
громкости (RMS) — хуже, но работает.

Discord сам перестаёт слать пакеты, когда человек замолкает, поэтому
есть ещё «таймер тишины»: если от пользователя ничего не приходило
дольше паузы — фраза закрывается принудительно.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import config
from utils.logger import get_logger

try:  # voice_recv — опциональная зависимость, импортируем мягко
    from discord.ext import voice_recv
    _SinkBase: Any = voice_recv.AudioSink
except Exception:  # noqa: BLE001
    voice_recv = None  # type: ignore[assignment]
    _SinkBase = object

_logger = get_logger("bot.voice.listener")

_SR = 16000           # частота для VAD/Whisper
_CHUNK = 512          # размер окна Silero VAD при 16 кГц (32 мс)
_CHUNK_MS = _CHUNK * 1000 / _SR
_PREROLL_CHUNKS = 10  # ~320 мс звука до начала речи, чтобы не съедать первый слог

UtteranceCallback = Callable[[int, np.ndarray], None]


def _lowpass_taps(n: int = 97, cutoff: float = 7600.0, sr: int = 48000) -> np.ndarray:
    """Windowed-sinc ФНЧ для честного 48→16 кГц (без «звона» и алиасинга)."""
    t = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * cutoff / sr * t) * np.blackman(n)
    return (h / h.sum()).astype(np.float32)


_TAPS = _lowpass_taps()


def resample_48k_to_16k(mono48: np.ndarray) -> np.ndarray:
    """Качественный ресемплинг целой фразы для Whisper."""
    if mono48.size == 0:
        return mono48.astype(np.float32)
    filtered = np.convolve(mono48, _TAPS, mode="same")
    return filtered[::3].astype(np.float32)


def pcm48k_stereo_to_48k_mono(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16)
    usable = samples.size - (samples.size % 6)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    return (samples[:usable].reshape(-1, 2).astype(np.float32).mean(axis=1) / 32768.0).astype(np.float32)


def pcm48k_stereo_to_16k_mono(pcm: bytes) -> np.ndarray:
    """Discord PCM (48 кГц, стерео, int16) → float32 моно 16 кГц в [-1, 1]."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size < 6:
        return np.zeros(0, dtype=np.float32)
    usable = samples.size - (samples.size % 6)          # 2 канала × децимация 3
    stereo = samples[:usable].reshape(-1, 2).astype(np.float32)
    mono = stereo.mean(axis=1)
    # Децимация 48k → 16k с усреднением по 3 отсчёта (простой ФНЧ).
    mono16 = mono.reshape(-1, 3).mean(axis=1)
    return (mono16 / 32768.0).astype(np.float32)


# ---------------------------------------------------------------------------
# VAD backends
# ---------------------------------------------------------------------------

class _SileroVAD:
    """Обёртка над Silero VAD: одна копия модели на одного говорящего."""

    def __init__(self) -> None:
        from silero_vad import load_silero_vad  # type: ignore
        import torch  # noqa: F401

        self._model = load_silero_vad()

    def prob(self, chunk: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            return float(self._model(torch.from_numpy(chunk), _SR).item())

    def reset(self) -> None:
        try:
            self._model.reset_states()
        except Exception:  # noqa: BLE001
            pass


class _EnergyVAD:
    """Запасной детектор речи по громкости."""

    def __init__(self) -> None:
        self._noise = 0.003

    def prob(self, chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-9)
        # Медленно подстраиваемся под фоновый шум.
        if rms < self._noise * 2:
            self._noise = 0.95 * self._noise + 0.05 * rms
        return 1.0 if rms > max(0.01, self._noise * 4) else 0.0

    def reset(self) -> None:
        pass


_silero_ok: Optional[bool] = None


def _make_vad() -> Any:
    global _silero_ok
    if _silero_ok is not False:
        try:
            vad = _SileroVAD()
            if _silero_ok is None:
                _logger.info("Silero VAD loaded", extra={"data": {}})
            _silero_ok = True
            return vad
        except Exception as exc:  # noqa: BLE001
            _silero_ok = False
            _logger.warning(
                "silero-vad unavailable — falling back to energy VAD",
                extra={"data": {"error": str(exc)}},
            )
    return _EnergyVAD()


# ---------------------------------------------------------------------------
# Per-user segmentation state
# ---------------------------------------------------------------------------

@dataclass
class _Speaker:
    vad: Any
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pending48: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    preroll: List[np.ndarray] = field(default_factory=list)
    speech: List[np.ndarray] = field(default_factory=list)
    # Тот же звук в исходных 48 кГц — для качественного ресемплинга под Whisper.
    preroll48: List[np.ndarray] = field(default_factory=list)
    speech48: List[np.ndarray] = field(default_factory=list)
    in_speech: bool = False
    speech_chunks: int = 0
    silence_chunks: int = 0
    last_packet: float = 0.0


class VoiceListener(_SinkBase):  # type: ignore[misc]
    """
    AudioSink для discord-ext-voice-recv, который режет речь на фразы.

    Parameters
    ----------
    bot_user_id:
        ID самого бота — его звук игнорируется.
    on_utterance:
        Вызывается из рабочего потока: ``on_utterance(user_id, audio16k)``.
        Должна быть потокобезопасной (например, через
        ``loop.call_soon_threadsafe``).
    """

    def __init__(self, bot_user_id: int, on_utterance: UtteranceCallback) -> None:
        if _SinkBase is not object:
            super().__init__()
        self._bot_user_id = bot_user_id
        self._on_utterance = on_utterance
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=2000)
        self._speakers: Dict[int, _Speaker] = {}
        self._threshold = config.VAD_THRESHOLD
        self._silence_limit = max(1, int(config.VAD_SILENCE_MS / _CHUNK_MS))
        self._min_speech = max(1, int(config.VAD_MIN_SPEECH_MS / _CHUNK_MS))
        self._max_speech = max(10, int(config.VAD_MAX_SPEECH_S * 1000 / _CHUNK_MS))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="voice-vad", daemon=True)
        self._thread.start()

    # ----- AudioSink API ---------------------------------------------------

    def wants_opus(self) -> bool:
        return False  # хотим уже декодированный PCM

    def write(self, user: Any, data: Any) -> None:
        if user is None or getattr(user, "bot", False):
            return
        if user.id == self._bot_user_id:
            return
        pcm = getattr(data, "pcm", b"")
        if not pcm:
            return
        try:
            self._queue.put_nowait((user.id, pcm, time.monotonic()))
        except queue.Full:
            pass  # перегрузка — лучше потерять кадр, чем зависнуть

    def cleanup(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def someone_speaking(self) -> bool:
        """Говорит ли кто-то прямо сейчас (по VAD)."""
        now = time.monotonic()
        return any(sp.in_speech and now - sp.last_packet < 0.6 for sp in list(self._speakers.values()))

    # ----- worker thread ----------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                item = ()
            if item is None:
                break
            try:
                if item:
                    user_id, pcm, ts = item
                    self._feed(user_id, pcm, ts)
                if self._queue.empty():
                    # Проверяем «тишину» только когда разобрали всё накопленное,
                    # иначе при отставании потока фразы рвались бы посередине.
                    self._check_timeouts()
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "VAD worker error",
                    extra={"data": {"error": str(exc)}},
                    exc_info=True,
                )

    def _speaker(self, user_id: int) -> _Speaker:
        sp = self._speakers.get(user_id)
        if sp is None:
            sp = _Speaker(vad=_make_vad())
            self._speakers[user_id] = sp
        return sp

    def _feed(self, user_id: int, pcm: bytes, ts: float) -> None:
        sp = self._speaker(user_id)
        sp.last_packet = ts
        audio = pcm48k_stereo_to_16k_mono(pcm)        # грубо, но быстро — для VAD
        audio48 = pcm48k_stereo_to_48k_mono(pcm)       # оригинал — для распознавания
        sp.pending = np.concatenate([sp.pending, audio]) if sp.pending.size else audio
        sp.pending48 = np.concatenate([sp.pending48, audio48]) if sp.pending48.size else audio48

        while sp.pending.size >= _CHUNK and sp.pending48.size >= _CHUNK * 3:
            chunk = sp.pending[:_CHUNK].copy()
            chunk48 = sp.pending48[: _CHUNK * 3].copy()
            sp.pending = sp.pending[_CHUNK:]
            sp.pending48 = sp.pending48[_CHUNK * 3:]
            self._process_chunk(user_id, sp, chunk, chunk48)

    def _process_chunk(self, user_id: int, sp: _Speaker, chunk: np.ndarray, chunk48: np.ndarray) -> None:
        is_speech = sp.vad.prob(chunk) >= self._threshold

        if not sp.in_speech:
            sp.preroll.append(chunk)
            sp.preroll48.append(chunk48)
            if len(sp.preroll) > _PREROLL_CHUNKS:
                sp.preroll.pop(0)
                sp.preroll48.pop(0)
            if is_speech:
                sp.in_speech = True
                sp.speech = list(sp.preroll)
                sp.speech48 = list(sp.preroll48)
                sp.preroll = []
                sp.preroll48 = []
                sp.speech_chunks = 1
                sp.silence_chunks = 0
            return

        sp.speech.append(chunk)
        sp.speech48.append(chunk48)
        if is_speech:
            sp.speech_chunks += 1
            sp.silence_chunks = 0
        else:
            sp.silence_chunks += 1

        if sp.silence_chunks >= self._silence_limit or len(sp.speech) >= self._max_speech:
            self._finish(user_id, sp)

    def _check_timeouts(self) -> None:
        now = time.monotonic()
        limit_s = config.VAD_SILENCE_MS / 1000.0
        for user_id, sp in list(self._speakers.items()):
            if sp.in_speech and now - sp.last_packet > limit_s:
                self._finish(user_id, sp)

    def _finish(self, user_id: int, sp: _Speaker) -> None:
        speech_chunks = sp.speech_chunks
        if sp.speech48:
            audio = resample_48k_to_16k(np.concatenate(sp.speech48))
        else:
            audio = np.concatenate(sp.speech) if sp.speech else np.zeros(0, dtype=np.float32)
        # Тихий микрофон — нормализуем громкость (Whisper хуже слышит тихую речь).
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if 0.0 < peak < 0.5:
            audio = (audio * min(4.0, 0.7 / peak)).astype(np.float32)
        sp.in_speech = False
        sp.speech = []
        sp.speech48 = []
        sp.preroll = []
        sp.preroll48 = []
        sp.pending48 = np.zeros(0, dtype=np.float32)
        sp.speech_chunks = 0
        sp.silence_chunks = 0
        sp.pending = np.zeros(0, dtype=np.float32)
        sp.vad.reset()

        if speech_chunks < self._min_speech:
            return  # кашель, щелчок, «м»

        _logger.debug(
            "Utterance captured",
            extra={"data": {"user_id": user_id, "seconds": round(audio.size / _SR, 2)}},
        )
        try:
            self._on_utterance(user_id, audio)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "on_utterance callback failed",
                extra={"data": {"error": str(exc)}},
                exc_info=True,
            )
