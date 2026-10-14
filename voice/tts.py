"""
voice/tts.py — Day 5: озвучка ответов через Silero TTS (или F5-TTS).

* ``TTS_ENGINE=f5``: сервер F5-TTS (русская модель ESpeech).  Текст готовится
  как для Silero (кириллица, «+» ударения), эмоция — образец голоса под настроение.
  Если сервер не отвечает — фразу озвучивает Silero (см. ``FallbackTTS``).

* Модель скачивается один раз в ``data/models/silero_<TTS_MODEL>.pt``
  (standalone-вариант через torch.package — без torch.hub и git).
* Синтез сразу в 48 кГц — это родная частота Discord, поэтому FFmpeg
  не нужен: результат отдаётся как ``discord.PCMAudio``.
* Перед синтезом текст чистится: эмодзи и markdown убираются, числа
  переводятся в слова (num2words), латиница транслитерируется — Silero
  для русского читает только кириллицу.
"""

from __future__ import annotations

import asyncio
import io
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import numpy as np

import config
from utils.logger import get_logger

_logger = get_logger("bot.voice.tts")

_SAMPLE_RATE = 48000
_MAX_CHUNK_CHARS = 800  # у Silero есть ограничение на длину входа

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_TRANSLIT = {
    "sh": "ш", "ch": "ч", "zh": "ж", "ya": "я", "yu": "ю", "yo": "ё", "ts": "ц",
    "th": "т", "ph": "ф", "oo": "у", "ee": "и", "kh": "х",
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}
_WORD_REPLACEMENTS = {
    "aura": "аура", "ok": "окей", "okay": "окей", "lol": "лол", "discord": "дискорд",
    "ai": "эй ай", "wtf": "вэ тэ эф", "idk": "не знаю", "hm": "хм", "ngl": "честно",
    "valorant": "валор+ант", "valo": "в+ало", "genshin": "г+еншин", "impact": "импакт",
    "honkai": "хонк+ай", "star": "стар", "rail": "рейл", "hsr": "хэ эс эр", "gg": "гэ гэ",
    "wp": "вэ пэ", "gl": "гуд лак", "hf": "хэв фан", "afk": "афк", "pvp": "пэ вэ пэ", "pve": "пэ вэ е",
    "fps": "эф пэ эс", "cs": "кс", "dota": "д+ота", "minecraft": "майнкр+афт", "steam": "стим",
    "youtube": "ютуб", "twitch": "твич", "tiktok": "тикток", "telegram": "телеграм", "google": "гугл",
    "anime": "аниме", "nice": "найс", "cool": "кул", "bro": "бро", "yes": "ес", "no": "ноу",
    "sorry": "сорри", "please": "плиз", "thanks": "сенкс", "hi": "хай", "hello": "хэлло", "bye": "бай",
    "rofl": "рофл", "kek": "кек", "omg": "о май гад", "pog": "пог", "noob": "нуб", "pro": "про",
    "mirfeed": "мирф+ид", "spongebob": "спанч боб",
}

# Аббревиатуры кириллицей — по буквам.
_RU_LETTERS = {
    "А": "а", "Б": "бэ", "В": "вэ", "Г": "гэ", "Д": "дэ", "Е": "е", "Ж": "жэ", "З": "зэ", "И": "и",
    "К": "ка", "Л": "эл", "М": "эм", "Н": "эн", "О": "о", "П": "пэ", "Р": "эр", "С": "эс", "Т": "тэ",
    "У": "у", "Ф": "эф", "Х": "ха", "Ц": "цэ", "Ч": "чэ", "Ш": "ша", "Щ": "ща", "Э": "э", "Ю": "ю", "Я": "я",
}
_EN_LETTERS = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
    "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "икс", "y": "уай", "z": "зед",
}
_RU_ABBR_RE = re.compile(r"\b[А-ЯЁ]{2,5}\b")
_KNOWN_RU_ABBR = {"ИИ", "ХСР", "РФ", "США", "ВК", "ТГ", "ДС", "ЛС", "ЧСВ", "МГУ", "ЕГЭ", "ОГЭ", "ПК", "ГГ", "НПС", "ФПС"}
_RU_VOWELS = set("АЕЁИОУЫЭЮЯ")


def _ru_caps(m: "re.Match[str]") -> str:
    """Аббревиатуру — по буквам; слово капсом («НЕТ», «ДА») — просто обычным словом."""
    w = m.group(0)
    if w in _KNOWN_RU_ABBR or not (set(w) & _RU_VOWELS):
        return " ".join(_RU_LETTERS.get(c, c.lower()) for c in w)
    return w.lower()
_EN_ABBR_RE = re.compile(r"\b[A-Z]{2,5}\b")


def _load_pronunciations() -> List[Tuple[re.Pattern, str]]:
    """pronunciations.txt: «слово = как произносить» (+ перед ударной гласной)."""
    rules: List[Tuple[re.Pattern, str]] = []
    try:
        lines = config.PRONUNCIATIONS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rules
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        src, dst = (x.strip() for x in line.split("=", 1))
        if src and dst:
            rules.append((re.compile(rf"(?<![\w+]){re.escape(src)}(?!\w)", re.IGNORECASE), dst))
    return rules


_pron_cache: Tuple[float, List[Tuple[re.Pattern, str]]] = (-1.0, [])


def _pronunciations() -> List[Tuple[re.Pattern, str]]:
    """Словарь перечитывается при изменении файла — правки без перезапуска."""
    global _pron_cache
    try:
        mtime = config.PRONUNCIATIONS_FILE.stat().st_mtime
    except OSError:
        return []
    if mtime != _pron_cache[0]:
        _pron_cache = (mtime, _load_pronunciations())
    return _pron_cache[1]

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]+"
)
_CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>")
_MENTION_RE = re.compile(r"<[@#][!&]?\d+>")
_URL_RE = re.compile(r"https?://\S+")
_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _translit_word(word: str) -> str:
    low = word.lower()
    if low in _WORD_REPLACEMENTS:
        return _WORD_REPLACEMENTS[low]
    out: List[str] = []
    i = 0
    while i < len(low):
        pair = low[i:i + 2]
        if len(pair) == 2 and pair in _TRANSLIT:
            out.append(_TRANSLIT[pair])
            i += 2
        else:
            out.append(_TRANSLIT.get(low[i], ""))
            i += 1
    return "".join(out)


def _number_to_words(match: "re.Match[str]") -> str:
    raw = match.group(0).replace(",", ".")
    try:
        from num2words import num2words  # type: ignore

        value: Any = float(raw) if "." in raw else int(raw)
        return num2words(value, lang="ru")
    except Exception:  # noqa: BLE001
        return raw


def _clean_common(text: str) -> str:
    """Общая чистка для любого движка: код, ссылки, эмодзи, markdown, словарь произношений."""
    text = _CODE_RE.sub(" код ", text)
    text = _URL_RE.sub(" ссылка ", text)
    text = _CUSTOM_EMOJI_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"[*_~`|>#\[\]{}<\"]", " ", text)   # markdown
    for pattern, repl in _pronunciations():               # пользовательский словарь — первым
        text = pattern.sub(repl, text)
    return text


def normalize_for_speech(text: str, keep_ellipsis: bool = False) -> str:
    """Превратить ответ бота в текст, который Silero прочитает нормально.

    keep_ellipsis: оставить «...» (F5 делает на нём паузу; Silero — нет, ему точка).
    """
    text = _clean_common(text)
    text = _EN_ABBR_RE.sub(
        lambda m: m.group(0) if m.group(0).lower() in _WORD_REPLACEMENTS
        else " ".join(_EN_LETTERS.get(c, c) for c in m.group(0).lower()), text)
    text = _RU_ABBR_RE.sub(_ru_caps, text)
    text = _NUMBER_RE.sub(_number_to_words, text)
    text = _LATIN_WORD_RE.sub(lambda m: _translit_word(m.group(0)), text)
    text = text.replace("…", "...")
    text = re.sub(r"\.{2,}", "..." if keep_ellipsis else ".", text)
    # Оставляем только то, что Silero умеет: кириллица, пробелы, пунктуация.
    text = re.sub(r"[^А-Яа-яЁё0-9\s.,!?:;\-—+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\s.,!?:;\-—]+", "", text)
    return text


# --- эмоции: настроение + тип фразы → темп и высота голоса (SSML Silero) ---
_RATES = ["x-slow", "slow", "medium", "fast", "x-fast"]
_PITCHES = ["x-low", "low", "medium", "high", "x-high"]
_MOOD_PROSODY = {          # (сдвиг темпа, сдвиг высоты)
    "cheerful": (0, 1),
    "amused": (0, 1),
    "curious": (0, 0),
    "irritated": (1, -1),
    "tired": (-1, -1),
    "withdrawn": (-1, 0),
    "neutral": (0, 0),
}


def _prosody_for(sentence: str, mood: str) -> Tuple[str, str]:
    r, p = _MOOD_PROSODY.get(mood or "neutral", (0, 0))
    s = sentence.strip()
    if s.endswith("!") or s.count("!") >= 1:
        r, p = r + 1, p + 1          # восклицание — живее и выше
    elif s.endswith("?"):
        p += 1                        # вопрос — чуть выше
    elif s.endswith("...") or s.endswith("…"):
        r -= 1                        # многоточие — медленнее, задумчиво
    # Без крайностей: x-fast / x-high звучат как мультик.
    r = max(-1, min(1, r))
    p = max(-1, min(1, p))
    return _RATES[2 + r], _PITCHES[2 + p]


def build_ssml(text: str, mood: str) -> Tuple[str, str]:
    """(ssml, plain): разбивает на предложения, каждому — свой темп/высота."""
    sentences = [x for x in re.split(r"(?<=[.!?…])\s+", text.strip()) if x.strip()]
    parts: List[str] = []
    plain: List[str] = []
    for sent in sentences:
        clean = normalize_for_speech(sent)
        if not re.search(r"[А-Яа-яЁё]", clean):
            continue
        rate, pitch = _prosody_for(sent, mood)
        safe = clean.replace("&", " и ").replace("<", " ").replace(">", " ")
        if rate == "medium" and pitch == "medium":
            parts.append(f"<s>{safe}</s>")
        else:
            parts.append(f'<s><prosody rate="{rate}" pitch="{pitch}">{safe}</prosody></s>')
        plain.append(clean)
    return "<speak><p>" + "".join(parts) + "</p></speak>", " ".join(plain)


def _split_chunks(text: str) -> List[str]:
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        while len(s) > _MAX_CHUNK_CHARS:  # очень длинное «предложение»
            chunks.append(s[:_MAX_CHUNK_CHARS])
            s = s[_MAX_CHUNK_CHARS:]
        if len(cur) + len(s) + 1 > _MAX_CHUNK_CHARS and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# TTS engine
# ---------------------------------------------------------------------------

class TextToSpeech:
    """Асинхронная обёртка над Silero TTS."""

    def __init__(self) -> None:
        self._model: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="silero-tts")
        self._load_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def load(self) -> None:
        async with self._load_lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_sync)

    def _load_sync(self) -> None:
        import torch

        t0 = time.monotonic()
        models_dir = config.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        local_file = models_dir / f"silero_{config.TTS_MODEL}.pt"
        if not local_file.is_file():
            url = config.TTS_MODEL_URL or f"https://models.silero.ai/models/tts/ru/{config.TTS_MODEL}.pt"
            _logger.info("Downloading Silero TTS model", extra={"data": {"url": url}})
            tmp = local_file.with_suffix(".part")
            torch.hub.download_url_to_file(url, str(tmp))
            tmp.replace(local_file)

        wanted = config.TTS_DEVICE.lower()
        use_cuda = wanted in ("cuda", "gpu") or (wanted == "auto" and torch.cuda.is_available())
        device = torch.device("cuda" if use_cuda else "cpu")
        if device.type == "cpu":
            torch.set_num_threads(max(1, config.TTS_CPU_THREADS))

        model = torch.package.PackageImporter(str(local_file)).load_pickle("tts_models", "model")
        model.to(device)
        self._model = model
        _logger.info(
            "Silero TTS loaded",
            extra={"data": {
                "model": config.TTS_MODEL,
                "speaker": config.TTS_SPEAKER,
                "device": device.type,
                "seconds": round(time.monotonic() - t0, 1),
            }},
        )

    async def synthesize(self, text: str, mood: str = "neutral") -> bytes:
        """Текст → PCM 48 кГц стерео int16 (формат discord.PCMAudio).  b'' если нечего читать."""
        if self._model is None:
            await self.load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._synthesize_sync, text, mood)

    # Параметры автоударений Silero (у разных версий набор разный — пробуем по убыванию).
    _ACCENT_SETS = (
        dict(put_accent=True, put_yo=True, put_stress_homo=True, put_yo_homo=True),
        dict(put_accent=True, put_yo=True),
        {},
    )

    def _call(self, **kwargs: Any) -> np.ndarray:
        last: Optional[Exception] = None
        for extra in self._ACCENT_SETS:
            try:
                audio = self._model.apply_tts(
                    speaker=config.TTS_SPEAKER, sample_rate=_SAMPLE_RATE, **extra, **kwargs
                )
                return audio.detach().cpu().numpy().astype(np.float32)
            except TypeError as exc:
                last = exc
        raise last or RuntimeError("apply_tts failed")

    def _apply(self, chunk: str) -> np.ndarray:
        return self._call(text=chunk)

    def _synthesize_sync(self, text: str, mood: str = "neutral") -> bytes:
        t0 = time.monotonic()
        pieces: List[np.ndarray] = []
        pause = np.zeros(int(_SAMPLE_RATE * 0.15), dtype=np.float32)

        ssml, clean = build_ssml(text, mood) if config.TTS_EMOTIONAL else ("", normalize_for_speech(text))
        if not re.search(r"[А-Яа-яЁё]", clean):
            return b""
        if ssml and len(clean) <= _MAX_CHUNK_CHARS:
            try:
                pieces.append(self._call(ssml_text=ssml))
                pieces.append(pause)
            except Exception as exc:  # noqa: BLE001
                _logger.debug("SSML synthesis failed — plain text", extra={"data": {"error": str(exc)}})
                pieces = []
        if not pieces:
            for chunk in _split_chunks(clean):
                try:
                    pieces.append(self._apply(chunk))
                    pieces.append(pause)
                except Exception as exc:  # noqa: BLE001
                    _logger.warning(
                        "Silero failed on chunk — skipped",
                        extra={"data": {"error": str(exc), "chunk": chunk[:80]}},
                    )
        if not pieces:
            return b""

        mono = np.concatenate(pieces)
        mono = np.clip(mono * config.TTS_VOLUME, -1.0, 1.0)
        pcm16 = (mono * 32767).astype(np.int16)
        stereo = np.repeat(pcm16[:, None], 2, axis=1)  # дублируем в оба канала
        _logger.debug(
            "Synthesized",
            extra={"data": {
                "chars": len(clean),
                "audio_seconds": round(mono.size / _SAMPLE_RATE, 2),
                "latency_seconds": round(time.monotonic() - t0, 2),
            }},
        )
        return stereo.tobytes()

    @staticmethod
    def to_audio_source(pcm: bytes) -> Any:
        import discord

        return discord.PCMAudio(io.BytesIO(pcm))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# F5-TTS (HTTP-сервер f5/f5_server.py)
# ---------------------------------------------------------------------------

def _wav_to_pcm48(data: bytes) -> np.ndarray:
    """WAV от сервера (любая частота, моно/стерео, int16) → моно float32 48 кГц."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != _SAMPLE_RATE and audio.size:
        n_out = int(round(audio.size * _SAMPLE_RATE / rate))
        x_old = np.arange(audio.size, dtype=np.float64)
        x_new = np.linspace(0, audio.size - 1, n_out)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


def _mono_to_discord(mono: np.ndarray) -> bytes:
    mono = np.clip(mono * config.TTS_VOLUME, -1.0, 1.0)
    pcm16 = (mono * 32767).astype(np.int16)
    return np.repeat(pcm16[:, None], 2, axis=1).tobytes()


class F5TTS:
    """
    Клиент к серверу F5-TTS (f5/f5_server.py): POST /v1/tts → WAV → PCM 48 кГц стерео.
    Модель ESpeech читает только кириллицу и понимает «+» ударения — поэтому текст готовится
    как для Silero (словарь произношений работает).  Эмоция — образец голоса под настроение (mood).
    """

    name = "F5"

    def __init__(self) -> None:
        self._session: Any = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def _http(self) -> Any:
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def load(self) -> None:
        """Проверка сервера одной короткой фразой (заодно прогрев)."""
        t0 = time.monotonic()
        await self._request("Привет.", timeout=max(60.0, config.F5_TIMEOUT))
        self._ready = True
        _logger.info(
            "F5 TTS ready",
            extra={"data": {
                "url": config.F5_API_URL,
                "reference_id": config.F5_REFERENCE_ID,
                "seconds": round(time.monotonic() - t0, 1),
            }},
        )

    @staticmethod
    def _payload(text: str, mood: str) -> dict:
        return {
            "text": text,
            "reference_id": config.F5_REFERENCE_ID,
            "mood": (mood or "neutral") if config.TTS_EMOTIONAL else "neutral",
            "speed": config.F5_SPEED,
        }

    async def _request(self, text: str, timeout: Optional[float] = None, mood: str = "neutral") -> bytes:
        import aiohttp

        session = await self._http()
        tmo = aiohttp.ClientTimeout(total=timeout or config.F5_TIMEOUT)
        async with session.post(f"{config.F5_API_URL}/v1/tts", json=self._payload(text, mood), timeout=tmo) as resp:
            body = await resp.read()
            if resp.status != 200:
                raise RuntimeError(f"F5 HTTP {resp.status}: {body[:200]!r}")
            return body

    async def synthesize(self, text: str, mood: str = "neutral") -> bytes:
        """Текст → PCM 48 кГц стерео int16.  b'' если нечего читать; при сбое — исключение."""
        clean = self._prepare(text)
        if not clean:
            return b""
        t0 = time.monotonic()
        wav = await self._request(clean, mood=mood)
        mono = _wav_to_pcm48(wav)
        if not mono.size:
            return b""
        mono = np.concatenate([mono, np.zeros(int(_SAMPLE_RATE * 0.15), dtype=np.float32)])
        _logger.debug(
            "Synthesized (F5)",
            extra={"data": {
                "text": clean[:120],
                "audio_seconds": round(mono.size / _SAMPLE_RATE, 2),
                "latency_seconds": round(time.monotonic() - t0, 2),
            }},
        )
        return _mono_to_discord(mono)

    @staticmethod
    def _prepare(text: str) -> str:
        clean = normalize_for_speech(text, keep_ellipsis=True)
        # Многоточие посреди фразы F5 читает рваной паузой: внутри — запятая, в конце — точка.
        clean = re.sub(r"\s*\.\.\.\s*(?=[А-Яа-яЁё0-9+])", ", ", clean)
        clean = re.sub(r"\.\.\.(?=\s*[!?])", "", clean)
        clean = clean.replace("...", ".")
        clean = re.sub(r",\s*,", ",", clean)
        return clean if re.search(r"[А-Яа-яЁё]", clean) else ""

    @staticmethod
    def to_audio_source(pcm: bytes) -> Any:
        return TextToSpeech.to_audio_source(pcm)

    def close(self) -> None:
        if self._session is not None and not self._session.closed:
            try:
                asyncio.get_running_loop().create_task(self._session.close())
            except RuntimeError:
                pass


class FallbackTTS:
    """F5-TTS, а если сервер не отвечает — та же фраза через Silero."""

    def __init__(self, primary: Optional[F5TTS] = None) -> None:
        self._primary = primary or F5TTS()
        self._silero = TextToSpeech()
        self._primary_down_until = 0.0

    @property
    def ready(self) -> bool:
        return self._primary.ready or self._silero.ready

    async def load(self) -> None:
        # Silero грузим всегда: без него нечем говорить, пока F5 недоступен.
        await self._silero.load()
        try:
            await self._primary.load()
        except Exception as exc:  # noqa: BLE001
            self._mark_down(exc)

    def _mark_down(self, exc: Exception) -> None:
        self._primary_down_until = time.monotonic() + config.F5_RETRY_AFTER_S
        _logger.warning(
            "f5_tts_failed — Silero speaks instead",
            extra={"data": {"engine": self._primary.name, "error": str(exc) or type(exc).__name__,
                            "retry_in_s": config.F5_RETRY_AFTER_S}},
        )

    async def synthesize(self, text: str, mood: str = "neutral") -> bytes:
        if time.monotonic() >= self._primary_down_until:
            try:
                return await self._primary.synthesize(text, mood)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._mark_down(exc)
        return await self._silero.synthesize(text, mood)

    @staticmethod
    def to_audio_source(pcm: bytes) -> Any:
        return TextToSpeech.to_audio_source(pcm)

    def close(self) -> None:
        self._primary.close()
        self._silero.close()


def make_tts() -> Any:
    """Движок озвучки по TTS_ENGINE: silero (по умолчанию) или f5 (с запасным Silero)."""
    if config.TTS_ENGINE == "f5":
        return FallbackTTS(F5TTS())
    return TextToSpeech()
