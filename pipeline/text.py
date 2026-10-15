"""
pipeline/text.py — text generation pipeline.

Sends prompts to a local Ollama instance.  By default it uses the
/api/chat endpoint (system + user messages through the model's own chat
template — it keeps the role and the context much better than raw
completion).  If chat returns nothing, the old flat /api/generate prompt
is used as a fallback.

Also exposes :meth:`TextPipeline.call_json` for background "reflection"
(memory/social.py) — structured JSON output via Ollama's ``format``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import aiohttp

import config
from core import game_mode
from utils.logger import get_logger
from memory.images import ImageMemory, ago_text
from pipeline.media import MediaPicker
from pipeline.websearch import EXPLICIT_RE as _SEARCH_EXPLICIT_RE, WebSearch
from behavior import schedule

if TYPE_CHECKING:
    from pathlib import Path

    from core.message_handler import MessageContext
    from memory.life import LifeStore

_logger = get_logger("bot.pipeline")

_FALLBACK_LINES: List[str] = [
    "погоди, что?",
    "я отвлеклась. ещё раз?",
    "без понятия, честно",
    "прости, прослушала",
]

# Слова-паразиты в начале реплики («хм, ну…») — в голосе звучат как заминка.
_LEADING_FILLERS = re.compile(
    r"^\s*(?:(?:хм+|м{2,}|э+м+|э+|пф+)(?=[\s,.!?…\-—]|$)[\s,.!?…\-—]*"
    r"|(?:ну+|так|ладно|ой|эх)\s*(?:[,.!?…\-—]+\s*|$))+",
    re.IGNORECASE,
)
# Ремарки «*вздыхает*», «(улыбается)», «подумала, потом ответила:».
_STAGE_DIRECTIONS = re.compile(r"\*[^*]{1,60}\*|\((?:[^()]{1,60})\)")
_NARRATION = re.compile(r"^[^.!?:]{0,80}(?:подумала|ответила|сказала|произнесла|вздохнула)[^.!?:]{0,40}:\s*",
                        re.IGNORECASE)


# Корейский / китайский / японский посреди русской фразы («Чего 이번 раз?») — сбой модели.
_FOREIGN_SCRIPT_RE = re.compile(r"[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")


def strip_foreign_script(text: str, keep_if: str = "") -> str:
    """Вырезать случайные иероглифы/хангыль.  Если собеседник сам так пишет (*keep_if*) — не трогать."""
    if not _FOREIGN_SCRIPT_RE.search(text) or _FOREIGN_SCRIPT_RE.search(keep_if or ""):
        return text
    text = _FOREIGN_SCRIPT_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r" +([,.!?…])", r"\1", text).strip()


def strip_voice_tics(text: str) -> str:
    """Убрать «хм/ну» в начале, ремарки и рассказ о себе в третьем лице (для озвучки).

    При VOICE_NATURAL_SPEECH «ну», «хм», «ой» остаются — это живая речь, а не заминка.
    """
    t = _NARRATION.sub("", strip_foreign_script(text))
    t = _STAGE_DIRECTIONS.sub(" ", t)
    if not config.VOICE_NATURAL_SPEECH:
        # Пустая строка, если реплика состояла только из «хм…».
        t = _LEADING_FILLERS.sub("", t).strip()
    return re.sub(r"\s{2,}", " ", t).strip()

_AI_TELLS: List[str] = [
    "as an ai",
    "i'm an ai",
    "i am an ai",
    "as a language model",
    "i'm a language model",
    "i am a language model",
    "как ии",
    "как языковая модель",
    "я языковая модель",
    "я искусственный интеллект",
]

_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

_LENGTH_HINTS = {
    "tiny": "очень коротко — пара слов или короткая фраза до ~40 символов, как в живом чате",
    "short": "коротко — одна короткая фраза, до ~80 символов",
    "normal": "одно предложение, максимум два коротких (до ~150 символов)",
    "long": "развёрнуто — тебя правда о чём-то спросили: ответь по делу в два-четыре предложения, своими словами, без лекторского тона",
}
_MAX_LEN = {"tiny": 120, "short": 160, "normal": 260, "long": 600}
# Сколько предложений оставить в чате, если модель всё равно расписалась.
_MAX_SENTENCES = {"tiny": 1, "short": 1, "normal": 2, "long": 4}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
# Где человек правда ждёт развёрнутого ответа.
_WANTS_DETAIL_RE = re.compile(
    r"\b(расска[жз]|объясн|поясни|почему|зачем|как\s+(?:лучше|сделать|думаешь|работает)|посовет|совет|"
    r"что\s+(?:думаешь|такое|за\s)|в\s+чём\s+разниц|чем\s+отлича|подробн|лекци|поделись|опи[шс]|"
    r"переска[жз]|что\s+(?:там\s+)?было|(?:а|и)\s+дальше|что\s+дальше|продолж|вс[её]\s+что\s+помнишь)",
    re.IGNORECASE,
)


_CLAUSE_SPLIT_RE = re.compile(
    r",\s+(?=(?:но|а|хотя|так что|зато|правда|кстати|и вообще|просто|ведь)\s)", re.IGNORECASE,
)


def _trim_chat(text: str, max_sentences: int, max_len: int, cut: bool = False) -> str:
    """
    Оставляет первые целые фразы: не больше *max_sentences* и примерно *max_len* символов.
    Переносы строк (= отдельные сообщения) сохраняются.  *cut* — модель упёрлась в лимит
    токенов, значит последняя фраза недописана и её выкидываем.
    """
    sents = []  # (номер строки, фраза)
    for li, line in enumerate(l.strip() for l in text.splitlines() if l.strip()):
        sents.extend((li, p) for p in _SENTENCE_SPLIT_RE.split(line) if p.strip())
    if not sents:
        return text.strip()
    if cut and len(sents) > 1:
        sents.pop()
    sents = sents[:max(1, max_sentences)]
    while len(sents) > 1 and sum(len(p) + 1 for _, p in sents) > max_len:
        sents.pop()
    if len(sents) == 1 and len(sents[0][1]) > max_len * 1.5:
        head = sents[0][1][: int(max_len * 1.5)]
        head = head[: head.rfind(" ")] if " " in head else head
        sents = [(sents[0][0], head.rstrip(",;:— ") + "…")]
    elif cut and len(sents) == 1 and not re.search(r"[.!?…)]\s*\S{0,3}$", sents[0][1]):
        sents = [(sents[0][0], sents[0][1].rstrip(",;:— ") + "…")]
    lines: Dict[int, List[str]] = {}
    for li, p in sents:
        lines.setdefault(li, []).append(p)
    return "\n".join(" ".join(ps) for ps in lines.values())


_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2728]\ufe0f?")


def _drop_repeated_emoji(text: str, context: "MessageContext") -> str:
    """Убирает эмодзи, если она ставила их в последних своих сообщениях — иначе 😏 в каждой реплике."""
    own = []
    for line in reversed(context.recent_channel_history or []):
        stripped = re.sub(r"^\[[^\]]*\]\s*", "", line)
        if stripped.startswith(config.BOT_NAME):
            own.append(stripped)
            if len(own) >= 3:
                break
    used = set(e for line in own for e in _EMOJI_RE.findall(line))
    if not _EMOJI_RE.search(text):
        return text
    if used or random.random() < 0.3:
        cleaned = re.sub(r"[ \t]{2,}", " ", _EMOJI_RE.sub("", text))
        cleaned = "\n".join(l.rstrip() for l in cleaned.splitlines() if l.strip())
        return cleaned or text  # ответ из одного эмодзи оставляем как есть
    return text


def _drop_period(s: str) -> str:
    s = s.rstrip()
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1].rstrip()
    return s


def _humanize_chat(text: str) -> str:
    """
    Чат: как пишут люди — без точки в конце и иногда несколькими сообщениями.
    Отдельные сообщения разделяются переводом строки (их разошлёт message_handler).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) == 1 and config.CHAT_SPLIT_MESSAGES and random.random() < config.CHAT_SPLIT_CHANCE:
        parts = _SENTENCE_SPLIT_RE.split(lines[0])
        if len(parts) == 1:
            # Одна длинная фраза — рвём по союзу: «много интересных» / «но если хочешь…»
            m = _CLAUSE_SPLIT_RE.search(parts[0], 8)
            if len(parts[0]) >= 60 and m and len(parts[0]) - m.end() >= 15:
                parts = [parts[0][:m.start()], parts[0][m.end():]]
        if 2 <= len(parts) <= 4:
            lines = parts
    if config.CHAT_DROP_FINAL_PERIOD:
        lines = [_drop_period(l) for l in lines]
    return "\n".join(l for l in lines if l)


def _local_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(config.BOT_TIMEZONE))
    except Exception:  # noqa: BLE001  (на Windows нет tzdata — берём МСК)
        return datetime.now(timezone(timedelta(hours=3)))


def time_description() -> str:
    now = _local_now()
    h = now.hour
    part = (
        "глубокая ночь" if h < 5 else
        "раннее утро" if h < 9 else
        "утро" if h < 12 else
        "день" if h < 17 else
        "вечер" if h < 22 else
        "поздний вечер"
    )
    return f"{_WEEKDAYS[now.weekday()]}, {now:%H:%M} ({part})"


# Провокации: мат-оскорбления, «твою мать…», «ты тупая», политота ради срача.
_PROVOCATION_RE = re.compile(
    r"(?:ебал|выеб|отъеб|долбоеб|уебищ|уебок|пидор|пидр|шлюх|давалк|\bчмо\b|"
    r"\bмать\s+(?:твоя|ебал|шлюха)|твою\s+(?:мать|маму|мамку|собаку|сестру)|твоего\s+(?:папу|отца|батю)|"
    r"\b(?:ты|аура)\s+(?:тупая|дура|овца|шлюха|сука|уродина|бот\s+тупой)|"
    r"\bнахуй\s+иди|\bиди\s+нахуй|"
    r"путин|зеленск|навальн|хохл|кацап|\bсво\b|слава\s+(?:украине|россии|z\b))",
    re.IGNORECASE,
)


def is_provocation(text: str) -> bool:
    """Похоже на рейдж-байт / оскорбление / политоту ради реакции."""
    return bool(_PROVOCATION_RE.search(text or ""))


def _own_recent_burst(context: "MessageContext", limit: int = 3) -> List[str]:
    """Сообщения автора подряд прямо перед текущим (пока не вклинился кто-то другой)."""
    name = getattr(context, "author_display_name", "") or ""
    out: List[str] = []
    bot_skipped = False
    for line in reversed(context.recent_channel_history or []):
        m = re.match(r"^\[[^\]]*\]\s*([^:→]+?)\s*(?:→[^:]*)?:\s*(.*)$", line, re.S)
        if m and m.group(1).strip() == config.BOT_NAME and not bot_skipped:
            bot_skipped = True  # «ну так что?» после её «ну ладно…» — просьба всё ещё в силе
            continue
        if not m or m.group(1).strip() != name:
            break
        out.append(m.group(2))
        if len(out) >= limit:
            break
    return out


def choose_length(context: "MessageContext") -> str:
    """Разная длина ответов, как у живого человека."""
    content = (context.content or "").strip()
    words = len(content.split())
    question = "?" in content
    # Человек мог разбить просьбу на несколько сообщений («расскажи про аметист?» + «аура, пожалуйста»).
    burst = " ".join(_own_recent_burst(context))
    wants_detail = bool(_WANTS_DETAIL_RE.search(content + " " + burst)) or words >= 30
    if context.use_thinking:
        weights = {"short": 15, "normal": 50, "long": 35}
    elif wants_detail:
        # Обычно рассказывает, но может и отмахнуться или подколоть — она не справочник.
        weights = {"short": 10, "normal": 35, "long": 55}
    elif words <= 3:
        weights = {"tiny": 70, "short": 28, "normal": 2}
    elif question:
        weights = {"tiny": 30, "short": 50, "normal": 20}
    else:
        weights = {"tiny": 45, "short": 45, "normal": 10}
    if getattr(context, "provocation", False):  # на провокации — коротко
        weights = {"tiny": 70, "short": 30}
    if getattr(context, "affinity", 0.0) <= -0.4:  # с неприятными — короче
        weights = {"tiny": 55, "short": 40, "normal": 5}
    if context.is_voice:
        # В войсе коротко: длинная фраза = долгое ожидание первого звука.
        weights = {"tiny": 30, "short": 50, "normal": 20}
    keys = list(weights)
    return random.choices(keys, weights=[weights[k] for k in keys])[0]


# Приоритеты очереди к модели: голос важнее чата, чат важнее «рефлексии».
PRIORITY_VOICE = 0
PRIORITY_CHAT = 2
PRIORITY_BACKGROUND = 3


def _ollama_stats(d: Dict[str, Any]) -> Dict[str, Any]:
    """Сколько токенов промпта пришлось реально пересчитать и как быстро (для диагностики)."""
    if not d:
        return {}
    pe, ped = d.get("prompt_eval_count"), d.get("prompt_eval_duration")
    ev, evd = d.get("eval_count"), d.get("eval_duration")
    out: Dict[str, Any] = {}
    if pe is not None:
        out["prompt_tokens_evaluated"] = pe
    if ped:
        out["prompt_eval_seconds"] = round(ped / 1e9, 2)
    if ev and evd:
        out["tokens_per_second"] = round(ev / (evd / 1e9), 1)
    if d.get("load_duration"):
        out["load_seconds"] = round(d["load_duration"] / 1e9, 2)
    return out


def _voice_model() -> str:
    return config.VOICE_OLLAMA_MODEL or config.OLLAMA_MODEL


def _game_mode_payload(
    payload: Dict[str, Any], timeout: Optional[float]
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Игровой режим: модель частично на CPU (те же опции во всех запросах) — и ждать её дольше."""
    extra = game_mode.ollama_options()
    if not extra:
        return payload, timeout
    payload = {**payload, "options": {**(payload.get("options") or {}), **extra}}
    return payload, (timeout or config.OLLAMA_TIMEOUT) * 3


# Расшифровка речи самой Gemma (VOICE_HEARING=gemma).  Промпт нарочно без персонажа: в образе она путает
# запись с историей чата, переспрашивает «вы не прикрепили файл» и додумывает обращения к себе.
_STT_SYSTEM = "Ты — точная система распознавания русской речи. Ничего не добавляешь от себя."
# На тишине/шуме вместо «-» она иногда пишет «вы не прикрепили запись (аудиофайл)…».
_STT_META_RE = re.compile(r"аудиофайл|аудиозапис|не\s+прикрепил|прикрепи\w*\s+(?:саму\s+)?запись", re.I)

SILENCE_TOKEN = "[молчу]"

# Похоже на вопрос, даже без «?» — тогда спрашиваем модель, не загуглить ли.
_QUESTION_WORDS_RE = re.compile(
    r"\b(?:кто|что|когда|где|сколько|какой|какая|какие|почему|зачем|как|чем|откуда|правда\s+ли)\b", re.I)
# Разговор о книгах/чтении — тогда в промпт идут подробности того, что она читает.
_BOOK_TALK_RE = re.compile(
    r"книг|книж|чита|прочит|глав[аеуы]|сюжет|роман|манг|комикс|ранобэ|новелл|автор|персонаж|геро[йия]", re.I)
# Вопрос про своё/чужое из чата («на моём акке?», «у тебя тоже?») — гуглить нечего.
_PERSONAL_RE = re.compile(
    r"(?<![\wё])(?:мо[йеёияю]\w*|тво[йеёияю]\w*|у\s+(?:меня|тебя|нас|него|неё|нее)|мне|тебе)(?![\wё])", re.I)
_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"search": {"type": "boolean"}, "query": {"type": "string"}},
    "required": ["search", "query"],
}
_BOOK_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"is_request": {"type": "boolean"}, "title": {"type": "string"}, "author": {"type": "string"},
                   "kind": {"type": "string", "enum": ["book", "comic", "manga"]}},
    "required": ["is_request", "title", "author", "kind"],
}
_KONSPEKT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "gist": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "who": {"type": "string"}}, "required": ["name", "who"]}},
        "moments": {"type": "array", "items": {"type": "string"}},
        "ending": {"type": "string"},
        "feeling": {"type": "string"},
    },
    "required": ["gist", "characters", "moments", "ending", "feeling"],
}
_READ_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "story": {"type": "string"}, "thought": {"type": "string"},
                   "share": {"type": "string"}, "hooked": {"type": "integer"},
                   "illustrations": {"type": "array", "items": {"type": "object", "properties": {
                       "seen": {"type": "string"}, "love": {"type": "boolean"}}, "required": ["seen", "love"]}}},
    "required": ["summary", "story", "thought", "hooked"],
}
_VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"opinion": {"type": "string"}, "rating": {"type": "integer"}},
    "required": ["opinion", "rating"],
}
_LIFE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"event": {"type": "string"}},
    "required": ["event"],
}


def _is_silence(text: str) -> Optional[bool]:
    """True — модель решила молчать; False — точно говорит; None — пока непонятно."""
    t = text.strip().lower()
    if not t:
        return None
    if t[0] in "[(*":
        if "молч" in t or "silence" in t or "тишина" in t or "пропуск" in t:
            return True
        return None if len(t) < 14 and "]" not in t and ")" not in t else False
    if t.startswith("молчу") or t in ("...", "…"):
        return True
    return None if len(t) < 6 else False


def _other_speakers(context: "MessageContext") -> set:
    """Имена других людей из истории/канала (в нижнем регистре) — чтобы понять, что модель пишет за них."""
    names = {n.lower() for n in (getattr(context, "voice_participants", []) or [])}
    for line in context.recent_channel_history:
        stripped = re.sub(r"^\[[^\]]*\]\s*", "", line)
        head = stripped.split(":", 1)[0]
        head = re.sub(r"\s*(?:→.*|\(.*\))$", "", head).strip()
        if head and len(head) <= 32:
            names.add(head.lower())
    names.add((context.author_display_name or "").lower())
    names.discard(config.BOT_NAME.lower())
    names.discard("")
    return names


_DESCRIBE_RE = re.compile(r"опиши|описать|что (?:тут|здесь|там|на (?:ней|нём|нем|картинке|фото|скрине))|что это|прочитай|переведи|разбер", re.I)
# Вопрос про недавнюю картинку («а что на ней?», «глянь на фото») — только тогда показываем её снова.
_ABOUT_IMAGE_RE = re.compile(r"картин|фот|скрин|гиф|пикч|арт|мем|на ней|на нём|на нем|там изображ|видишь|разгляд|опиши|рассмотр", re.I)
_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
# «Аура, сохрани» — картинка сразу в любимые.
_SAVE_RE = re.compile(r"сохрани|в любимые|в избранное|забери себе", re.I)
# Вопрос про её любимые картинки — показать их снова.
_FAV_ASK_RE = re.compile(r"любим\w*\s+(?:картин|пикч|фот|мем|гиф|арт)|избранн|сохранённ\w* картин|сохраненн\w* картин", re.I)
_SCORE_RE = re.compile(r"оценка\D{0,5}(\d{1,2})", re.I)
# Ссылки на гифки: со страницы берём og:image (саму гифку).
_GIF_LINK_RE = re.compile(
    r"https?://(?:www\.)?(?:tenor\.com/(?:[a-z]{2}/)?view/|giphy\.com/gifs/|media\d*\.giphy\.com/|klipy\.com/(?:gifs|stickers|clips|memes)/)\S+",
    re.I)
# Прямая ссылка на картинку в тексте (Discord прячет такую ссылку и показывает саму картинку).
_IMAGE_LINK_RE = re.compile(
    r"https?://(?:(?:cdn|media)\.discordapp\.(?:com|net)/(?:attachments|ephemeral-attachments)/\S+"
    r"|\S+?\.(?:png|jpe?g|webp|gif|bmp)(?:\?\S*)?(?=\s|$))",
    re.I)
# Прямая ссылка на анимированную гифку/webp внутри страницы (лучше статичного og:image).
_PAGE_GIF_RE = re.compile(r"https://[^\s\"'<>]+?\.(?:gif|webp)(?=[\"'\s?<>])", re.I)
_OG_IMAGE_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\']og:image["\'][^>]+content=["\']([^"\']+)'
    r'|<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']og:image["\']', re.I)


def _prepare_image(raw: bytes, ext: str) -> List[bytes]:
    """
    Уменьшить картинку до VISION_MAX_SIDE и перекодировать в JPEG (Ollama любит png/jpeg).
    Анимация (gif, apng, webp) → несколько кадров по порядку, чтобы было видно, что происходит.
    """
    try:
        from PIL import Image
    except ImportError:
        return [raw] if ext in ("png", "jpg", "jpeg") else []
    try:
        img = Image.open(io.BytesIO(raw))
        n = getattr(img, "n_frames", 1) if getattr(img, "is_animated", False) else 1
        want = max(1, min(config.VISION_GIF_FRAMES, n))
        idx = [0] if want == 1 else sorted({round((n - 1) * (k + 0.5) / want) for k in range(want)})
        side = config.VISION_MAX_SIDE if want == 1 else min(config.VISION_MAX_SIDE, 768 if want <= 3 else 512)
        out: List[bytes] = []
        for i in idx:
            img.seek(i)
            frame = img.convert("RGB")
            frame.thumbnail((side, side))
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=85)
            out.append(buf.getvalue())
        return out
    except Exception:  # noqa: BLE001
        return []


_VIDEO_EXTS = {"mp4", "webm", "mov", "m4v"}


def _prepare_video(raw: bytes, ext: str) -> List[bytes]:
    """Гифка в виде видео (Klipy/Tenor отдают mp4) или короткий ролик → несколько кадров по порядку."""
    try:
        import av  # PyAV, ставится вместе с faster-whisper
    except ImportError:
        return []
    try:
        frames = []
        with av.open(io.BytesIO(raw)) as box:
            for i, frame in enumerate(box.decode(video=0)):
                frames.append(frame.to_image())
                if i >= 600:  # длинные ролики — только начало
                    break
        if not frames:
            return []
        want = max(1, min(config.VISION_GIF_FRAMES, len(frames)))
        idx = sorted({round((len(frames) - 1) * (k + 0.5) / want) for k in range(want)})
        side = min(config.VISION_MAX_SIDE, 768 if want <= 3 else 512)
        out: List[bytes] = []
        for i in idx:
            img = frames[i].convert("RGB")
            img.thumbnail((side, side))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            out.append(buf.getvalue())
        return out
    except Exception:  # noqa: BLE001
        return []


class _PriorityLock:
    """Один запрос к Ollama за раз; из ожидающих первым идёт самый важный."""

    def __init__(self) -> None:
        self._locked = False
        self._waiters: List[Tuple[int, int, asyncio.Future]] = []
        self._counter = 0

    def locked(self) -> bool:
        return self._locked

    async def acquire(self, priority: int) -> None:
        if not self._locked and not self._waiters:
            self._locked = True
            return
        import heapq

        fut = asyncio.get_running_loop().create_future()
        self._counter += 1
        heapq.heappush(self._waiters, (priority, self._counter, fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # нам уже передали замок — отдаём дальше
            else:
                self._waiters = [w for w in self._waiters if w[2] is not fut]
                heapq.heapify(self._waiters)
            raise

    def release(self) -> None:
        import heapq

        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(True)  # замок переходит к нему, _locked остаётся True
                return
        self._locked = False


class _Hold:
    def __init__(self, lock: _PriorityLock, priority: int) -> None:
        self._lock, self._priority = lock, priority

    async def __aenter__(self) -> None:
        await self._lock.acquire(self._priority)

    async def __aexit__(self, *exc: Any) -> None:
        self._lock.release()


class TextPipeline:
    """
    Generates a text response for a given MessageContext.

    The _llm_lock semaphore ensures only one request goes to Ollama at a
    time — parallel requests cause empty response fields.
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        """Store the injected HTTP session and create the serialisation lock."""
        self._session = session
        self._llm_lock = _PriorityLock()
        self._last_http_error = ""
        self.last_stats: Dict[str, Any] = {}
        self._no_think_models: set = set()  # модели, которые отказались «размышлять» (think)
        self._last_request = 0.0
        # channel_id → (время, картинки, автор): чтобы на «а что на ней?» она всё ещё видела картинку
        self._recent_images: Dict[int, Tuple[float, List[str], str, bool]] = {}
        # Долгая память на картинки (data/images): подписи + уменьшенные копии.
        self._image_memory: Optional[ImageMemory] = None
        # channel_id → id последней запомненной картинки (для «сохрани её»)
        self._last_image_id: Dict[int, int] = {}
        if config.VISION_ENABLED and config.VISION_MEMORY_MB > 0:
            try:
                self._image_memory = ImageMemory(
                    config.VISION_MEMORY_DIR, config.VISION_MEMORY_MB,
                    fav_dir=config.FAVORITES_DIR if config.FAVORITES_ENABLED else None,
                    fav_max_mb=config.FAVORITES_MAX_MB)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Image memory disabled", extra={"data": {"error": str(exc)}})
        # Её собственная жизнь (memory/life.py) — выставляет бот после подключения к базе.
        self.life: Optional["LifeStore"] = None
        # Её книжная полка (memory/library.py) — тоже выставляет бот.
        self.library: Optional[Any] = None
        # Поиск в интернете.
        self._web: Optional[WebSearch] = WebSearch(session) if config.WEB_SEARCH_ENABLED else None
        # Картинки из её коллекции любимых, которые она кидает сама.
        self._media: Optional[MediaPicker] = (
            MediaPicker(self._image_memory)
            if config.MEDIA_SEND_ENABLED and config.FAVORITES_ENABLED and self._image_memory is not None
            else None)
    @property
    def busy(self) -> bool:
        """True while a request to Ollama is in flight."""
        return self._llm_lock.locked()

    @property
    def seconds_since_last_request(self) -> float:
        return time.monotonic() - self._last_request

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, context: "MessageContext") -> str:
        """Produce a reply string for context."""
        try:
            length = choose_length(context)
            images = [] if context.is_voice else await self._fetch_images(context)
            fresh = list(images)
            if not context.is_voice:
                images = self._remember_images(context, images)
                self._attach_recent_captions(context)
            context.image_count = len(images)
            if images and length == "tiny" and _DESCRIBE_RE.search(context.content or ""):
                length = "short"  # спросили про картинку — хоть пару слов по делу
            system, user = self._build_messages(context, length)
            meta: Dict[str, Any] = {}
            raw = await self._complete(
                system, user, use_thinking=context.use_thinking,
                priority=PRIORITY_VOICE if context.is_voice else PRIORITY_CHAT,
                images=images, meta=meta,
                num_predict=max(config.MAX_RESPONSE_TOKENS, 360) if length == "long" else None,
            )
            if fresh and self._image_memory is not None:
                asyncio.create_task(self._store_image(context, fresh))
            elif not context.is_voice:
                self._save_requested(context)
            if self._media is not None and not context.is_voice:
                raw = self._attach_media(raw, context)
                if not raw and context.outgoing_media:
                    return ""  # ответила одной картинкой
            return self._post_process(raw, context, length, cut=bool(meta.get("cut")))
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Unhandled exception in TextPipeline.generate",
                extra={"data": {"error": str(exc)}},
                exc_info=True,
            )
            return random.choice(_FALLBACK_LINES)

    async def generate_stream(self, context: "MessageContext"):
        """
        Голос: ответ по предложениям, как только модель их допишет.

        Первое предложение уходит в озвучку, пока модель ещё генерирует
        остальное — Aura начинает говорить через ~1 с вместо 3–4 с.
        Асинхронный генератор строк; полный текст = их склейка.
        """
        length = choose_length(context)
        system, user = self._build_messages(context, length)
        payload = {
            "model": _voice_model(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "think": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": self._options(),
        }
        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        payload, timeout = _game_mode_payload(payload, None)
        tmo = aiohttp.ClientTimeout(total=timeout or config.OLLAMA_TIMEOUT)
        buf = ""
        stats: Dict[str, Any] = {}
        silent = False
        sent_any = False
        total = 0
        max_len = _MAX_LEN.get(length, 420)
        t0 = time.monotonic()
        first_at: Optional[float] = None

        def _clean_piece(piece: str, first: bool) -> str:
            piece = piece.strip()
            if first and piece:
                piece = self._post_process(piece, context, "long")
            piece = strip_voice_tics(piece) if piece else ""
            # Одно междометие без смысла («хм.») не озвучиваем.
            if re.fullmatch(r"[\W_]*", piece or ""):
                return ""
            return piece

        others = _other_speakers(context)
        optional = bool(getattr(context, "voice_optional", False))
        decided = not optional

        async with _Hold(self._llm_lock, PRIORITY_VOICE):
            self._last_request = time.monotonic()
            try:
                async with self._session.post(url, json=payload, timeout=tmo) as resp:
                    resp.raise_for_status()
                    stop = False
                    async for raw_line in resp.content:
                        if not raw_line.strip():
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except ValueError:
                            continue
                        buf += (chunk.get("message") or {}).get("content", "") or ""
                        done = bool(chunk.get("done"))
                        if done:
                            stats = chunk

                        # Необязательная реплика: сначала понять, не решила ли она промолчать.
                        if not decided:
                            verdict = _is_silence(buf)
                            if verdict is None and done:
                                verdict = False  # короткая реплика вроде «ага» — пусть говорит
                            if verdict is True:
                                silent = True
                                break
                            if verdict is None:
                                continue
                            decided = True

                        # Модель начала писать за другого человека — обрываем.
                        m = None
                        for x in re.finditer(
                            r"(?:^|\n)\s*(?:\[[^\]]*\]\s*)?([^\s:]{1,32})(?:\s*→\s*[^:\n]{1,32})?(?:\s\([^)]*\))?\s*:\s",
                            buf,
                        ):
                            # Режем только если это имя реального участника, а не «смотри:».
                            if x.group(1).lower() in others:
                                m = x
                                break
                        if m:
                            buf = buf[: m.start()]
                            done = True

                        while True:
                            # Граница предложения; слишком короткие («хм.») склеиваем со следующим.
                            mm = next(
                                (x for x in re.finditer(r"[.!?…]+[\s\n]+|\n+", buf) if x.end() >= 12),
                                None,
                            )
                            # Первую фразу отдаём уже на запятой, если она длинная — заговорит раньше.
                            if mm is None and not sent_any and len(buf) > 55 and config.VOICE_SPLIT_ON_COMMA:
                                mm = next((x for x in re.finditer(r"[,;—]\s+", buf) if x.end() >= 25), None)
                            if not mm:
                                break
                            piece, buf = buf[: mm.end()], buf[mm.end():]
                            piece = _clean_piece(piece, not sent_any)
                            if piece:
                                if first_at is None:
                                    first_at = time.monotonic() - t0
                                sent_any = True
                                total += len(piece)
                                yield piece
                                if total >= max_len:
                                    done = stop = True
                                    break
                        if done:
                            break
                    if not stop and not silent and buf.strip():
                        piece = _clean_piece(buf, not sent_any)
                        if piece:
                            sent_any = True
                            yield piece
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                _logger.warning("Ollama stream failed", extra={"data": {"error": str(exc)}})
            finally:
                self._last_request = time.monotonic()
                _logger.info(
                    "llm_voice_stats",
                    extra={"data": {
                        "first_sentence_seconds": round(first_at or -1, 2),
                        "total_seconds": round(time.monotonic() - t0, 2),
                        **_ollama_stats(stats),
                    }},
                )
        if not sent_any and not optional:
            yield random.choice(_FALLBACK_LINES)

    async def transcribe_audio(self, wav_b64: str, names: List[str]) -> str:
        """
        Фраза из войса (WAV в base64) → текст.  Пустая строка — речи нет.

        Отдельный короткий запрос: Ollama держит в кэше один промпт, так что после него промпт
        персонажа считается заново (+1.5–3 с к ответу) — это и есть цена VOICE_HEARING=gemma.
        """
        hint = ", ".join(n for n in names if n)[:200]
        user = (
            "Приложена запись одной фразы из голосового чата. Запиши дословно, что в ней сказано, — только сам "
            "текст речи, по-русски, с обычной пунктуацией, без комментариев и кавычек. "
            + (f"Люди в канале: {hint}; " if hint else "")
            + "бота зовут Аура. Если речи не слышно или ничего не разобрать — напиши одно «-»."
        )
        data = await self._post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            {
                "model": _voice_model(),
                "messages": [{"role": "system", "content": _STT_SYSTEM},
                             {"role": "user", "content": user, "images": [wav_b64]}],
                "stream": False,
                "think": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                # num_ctx — как у ответов: другой размер заставил бы Ollama перезагрузить модель.
                "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0, "num_predict": 200},
            },
            priority=PRIORITY_VOICE,
        )
        text = (((data or {}).get("message") or {}).get("content", "") or "").strip().strip("«»\"").strip()
        if text in ("-", "—", "–") or _STT_META_RE.search(text) or (text[:1] in "([" and text[-1:] in ")]"):
            return ""  # «-», «(тишина)», «[музыка]» или жалоба, что записи нет
        return text

    async def warmup(self, voice: bool = True) -> None:
        """Крошечный запрос с тем же началом промпта — чтобы Ollama держала его в кэше."""
        from core.message_handler import MessageContext

        ctx = MessageContext(
            message_id=0, content="", author_id="warmup", author_display_name="",
            channel_id=0, guild_id=None, timestamp=datetime.now(timezone.utc),
            is_voice=voice, mood_label="обычное",
        )
        system = self._build_system(ctx, "short")
        t0 = time.monotonic()
        await self._post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            {
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": "."}],
                "stream": False,
                "think": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {**self._options(), "num_predict": 1},
            } | ({"model": _voice_model()} if voice else {}),
            priority=PRIORITY_BACKGROUND,
        )
        _logger.info("LLM warmed up", extra={"data": {"voice": voice, "seconds": round(time.monotonic() - t0, 2)}})
        await self._check_gpu_placement()

    async def _check_gpu_placement(self) -> None:
        """Проверить через /api/ps, целиком ли модель в видеопамяти (частично на CPU = очень медленно)."""
        if getattr(self, "_gpu_checked", False) or game_mode.active():  # в игровом режиме так и задумано
            return
        self._gpu_checked = True
        try:
            async with self._session.get(f"{config.OLLAMA_BASE_URL}/api/ps",
                                         timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json(content_type=None)
        except Exception:  # noqa: BLE001
            return
        for m in data.get("models", []):
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            on_gpu = vram / size if size else 1.0
            info = {"model": m.get("name"), "size_gb": round(size / 1e9, 1), "vram_gb": round(vram / 1e9, 1),
                    "on_gpu_percent": round(on_gpu * 100), "num_ctx": config.OLLAMA_NUM_CTX}
            if on_gpu < 0.97:
                _logger.warning(
                    "Model does not fit in VRAM — part runs on CPU (slow). Lower OLLAMA_NUM_CTX or use a smaller model",
                    extra={"data": info},
                )
            else:
                _logger.info("Model fully on GPU", extra={"data": info})

    async def generate_initiator(self, context: "MessageContext") -> str:
        """Produce an unprompted conversation-starter for a proactive DM."""
        try:
            system = self._build_system(context, "short")
            memory = ""
            if context.memory_snippets:
                memory = "Из прошлых разговоров с ним/ней:\n" + "\n".join(context.memory_snippets) + "\n\n"
            user = (
                f"{memory}Ты сама решила написать {context.author_display_name} в личку — без повода с его стороны. "
                "Придумай короткое сообщение: случайная мысль, вопрос о том, что он(а) рассказывал(а) раньше, "
                "наблюдение или просто «эй». Не объясняй, зачем написала. Одно предложение, максимум два."
                + (f"\n({context.late_note})" if getattr(context, "late_note", "") else "")
            )
            raw = await self._complete(system, user)
            return self._post_process(raw, context, "short")
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Unhandled exception in TextPipeline.generate_initiator",
                extra={"data": {"error": str(exc)}},
                exc_info=True,
            )
            return random.choice(_FALLBACK_LINES)

    async def generate_occasion(self, context: "MessageContext", task: str, fallback: str = "") -> str:
        """
        Короткая реплика по поводу, а не в ответ: напоминание, «всё, я спать», поздравление.
        task — что сказать и зачем.  При сбое — fallback.
        """
        try:
            system = self._build_system(context, "short")
            history = "\n".join(context.recent_channel_history[-8:])
            user = (f"Последние сообщения здесь (старые сверху):\n{history}\n\n" if history else "") + task
            raw = await self._complete(system, user, priority=PRIORITY_BACKGROUND)
            text = self._post_process(raw, context, "short") if raw.strip() else ""
            return text if text and text not in _FALLBACK_LINES else fallback
        except Exception as exc:  # noqa: BLE001
            _logger.error("generate_occasion failed", extra={"data": {"error": str(exc)}}, exc_info=True)
            return fallback

    async def search_query(self, context: "MessageContext") -> str:
        """
        Нужно ли загуглить, чтобы ответить, и что именно.  '' — не нужно.
        Явная просьба («загугли …») — ищем всегда; иначе решает модель.
        """
        if self._web is None or context.is_voice:
            return ""
        text = (context.content or "").strip()
        explicit = bool(_SEARCH_EXPLICIT_RE.search(text))
        if not explicit and (len(text) < 12 or ("?" not in text and not _QUESTION_WORDS_RE.search(text))):
            return ""  # «как дела?» и прочая болтовня — без лишнего запроса к модели
        history = "\n".join(context.recent_channel_history[-4:])
        books = self.library.prompt_block() if self.library is not None else ""
        if not explicit and self.library is not None and self.library.mentions_my_book(text):
            return ""  # про её книгу (герои, название) — отвечает по прочитанному, без спойлеров из интернета
        if not explicit and _PERSONAL_RE.search(text):
            return ""  # «это на моём акке?», «у тебя тоже?» — про людей из чата, интернет тут не поможет
        data = await self.call_json(
            "Ты решаешь, нужно ли поискать в интернете, чтобы ответить на сообщение. Отвечай ТОЛЬКО JSON.",
            f"Сегодня {time_description()}.\nПеред этим в чате:\n{history or '-'}\n\n"
            f"Сообщение от {context.author_display_name}: «{text}»\n\n"
            + ("Человек прямо просит загуглить — search=true. " if explicit else
               "search=true, только если для ответа нужны факты, которых обычная девушка-студентка может "
               "не знать или не помнить точно: свежие новости, даты выхода, цены, расписания, результаты, "
               "погода, конкретные факты о людях, играх, фильмах, местах. Для болтовни, мнений, личных "
               "вопросов, вопросов о ней самой или о людях из чата и того, что и так все знают, — search=false. ")
            + (f"\n\nЧто она читает и уже прочитала:\n{books[:1500]}\nЕсли вопрос про эти книги — их сюжет, "
               "героев, что там было или чем кончится — search=false: она отвечает по тому, что сама прочитала; "
               "чего ещё не прочитала — того не знает и не гуглит, это спойлеры.\n" if books else "")
            + "query — поисковый запрос (как его вбили бы в Google), на языке, где проще найти ответ.",
            _SEARCH_SCHEMA, num_predict=80,
        )
        if not data or not data.get("search"):
            return ""
        query = str(data.get("query") or "").strip()[:120]
        # Запрос про её книгу: либо модель подклеила название из промпта («Unnamed Memory … account»),
        # либо хочет «освежить память» пересказом из интернета — это спойлеры.  Про свою книгу — по памяти.
        if query and self.library is not None and self.library.mentions_my_book(query):
            _logger.info("Search query dragged in her book — skipped", extra={"data": {"query": query}})
            return ""
        return query

    async def web_lookup(self, query: str) -> str:
        return await self._web.lookup(query) if self._web is not None and query else ""

    async def generate_life_event(self, history: List[str]) -> str:
        """Новое небольшое событие из её жизни — продолжение того, что было.  '' при сбое."""
        persona = config.load_persona()
        data = await self.call_json(
            f"Ты ведёшь хронику жизни {config.BOT_NAME} — живой девушки. Отвечай ТОЛЬКО JSON.",
            f"Кто она:\n{persona[:2500]}\n\nСейчас: {time_description()}.\n"
            f"Что с ней уже было (старые сверху):\n" + ("\n".join(history) or "(пока ничего)") + "\n\n"
            "Придумай ОДНО новое небольшое событие из её жизни, которое случилось недавно: учёба, камни, "
            "кошка Луна, игры, музыка, еда, погода, мелкие бытовые радости и неудачи. Обычное и правдоподобное, "
            "не драма и не подвиг. Иногда — продолжение прошлого события (сдала тот зачёт, Луна опять что-то "
            "уронила), иногда новое. Не противоречь прошлому и времени суток. event — одно предложение "
            "от третьего лица в прошедшем времени, без имени («сдала зачёт по кристаллографии на четвёрку, "
            "хотя готовилась одну ночь»).",
            _LIFE_SCHEMA, num_predict=150,
        )
        return str((data or {}).get("event") or "").strip()

    async def read_session(self, book: Dict[str, Any], text: str = "",
                           pages: Optional[List[bytes]] = None,
                           illustrations: Optional[List[bytes]] = None) -> Optional[Dict[str, Any]]:
        """
        Она читает кусок книги (text) или несколько страниц комикса (pages).
        → {summary, story, thought} или None.  Читает как она сама: со своим вкусом и реакцией.
        """
        persona = config.load_persona()
        images: List[str] = []
        for raw in (pages or []) + (illustrations or []):
            frames = _prepare_image(raw, "jpg")
            if frames:
                images.append(base64.b64encode(frames[0]).decode("ascii"))
        n_illus = len(illustrations or []) if images else 0
        scan = book["kind"] == "comic" and str(book.get("file", "")).lower().endswith(".pdf")
        what = "комикс" if book["kind"] == "comic" else "книгу"
        by = f" ({book['author']})" if book["author"] else ""
        system = (f"Ты — {config.BOT_NAME}, читаешь {what} в своё удовольствие. Кто ты:\n{persona[:1800]}\n\n"
                  "Отвечай ТОЛЬКО JSON по схеме, по-русски.")
        user = (
            f"{'Комикс' if images else 'Книга'}: «{book['title']}»{by}.\n"
            + (f"Это продолжение «{book['prev_title']}», которую ты уже прочитала. Что было там: "
               f"{book['prev_story'][:900]}\n" if book.get("prev_story") else "")
            + f"Что было до сих пор в этой книге (твой пересказ): {book['story'] or '(это самое начало)'}\n\n"
            + (f"Ты смотришь следующие страницы ({len(images)} шт., по порядку) — "
               + ("это страницы книги или манги (скан): читай текст на них и смотри, что нарисовано.\n" if scan else
                  "читай реплики в облачках и смотри, что нарисовано.\n")
               if pages else f"Ты прочитала следующий кусок:\n«««\n{text}\n»»»\n")
            + (f"В этом куске книги есть иллюстрации ({n_illus} шт., приложены по порядку) — посмотри на них. "
               "illustrations — для каждой: seen (что на ней нарисовано, одной фразой, конкретно) и love "
               "(true, только если правда зацепила так, что хочется сохранить себе).\n" if n_illus else "")
            + "\nsummary — что произошло в этом куске, 2–3 предложения, конкретно (кто что сделал), без оценок.\n"
            "story — обновлённый пересказ всего прочитанного до этого места (до 800 символов): главное "
            "о героях и сюжете, старое неважное сокращай. Только то, что уже прочитано — ничего не додумывай.\n"
            "thought — твоя живая реакция одной-двумя фразами от первого лица, в твоём стиле, как подруге "
            "(что зацепило, бесит, смешит, чего ждёшь дальше). Можно с иронией. Начинай каждый раз по-разному — "
            "не с «ой», «ну» или «так», а сразу с мысли.\n"
            "Всё пиши по-русски: если книга на другом языке — цитаты переводи сама (живо, не дословно-коряво), "
            "имена героев тоже по-русски.\n"
            "share — если в этом куске был момент, которым прям хочется поделиться с другом (смешной, шокирующий, "
            "трогательный поворот, крутая фраза героя), — что именно, одной-двумя фразами, как написала бы "
            "подруге в личку; если ничего такого — пустая строка (чаще всего пусто).\n"
            "hooked — насколько тебе сейчас хочется читать дальше прямо сейчас, 0–10: 9–10 — не оторваться, "
            "сюжет на самом интересном месте; 5 — нормально, можно и отложить; 0–2 — скучно, тянет бросить. "
            "Будь честной и строгой: высокие оценки — только когда правда захватило."
        )
        data = await self.call_json(system, user, _READ_SCHEMA, num_predict=600, images=images or None)
        if not data or not str(data.get("summary") or "").strip():
            return None
        out: Dict[str, Any] = {k: str(data.get(k) or "").strip() for k in ("summary", "story", "thought", "share")}
        try:
            out["hooked"] = max(0, min(10, int(data.get("hooked") or 0)))
        except (TypeError, ValueError):
            out["hooked"] = 0
        out["illustrations"] = [
            {"seen": str(i.get("seen") or "").strip(), "love": bool(i.get("love"))}
            for i in (data.get("illustrations") or [])[:n_illus] if isinstance(i, dict)]
        return out

    async def save_book_illustration(self, raw: bytes, title: str, caption: str) -> bool:
        """Иллюстрация из книги очень понравилась — в её коллекцию любимых (потом может скинуть в чат)."""
        if self._image_memory is None or not config.FAVORITES_ENABLED:
            return False
        b64 = base64.b64encode(raw).decode("ascii")
        img_id = await asyncio.to_thread(self._image_memory.add, 0, f"книга «{title}»", [b64], False)
        if img_id is None:
            return False
        await asyncio.to_thread(self._image_memory.set_caption, img_id, f"иллюстрация из «{title}»: {caption}")
        await asyncio.to_thread(self._image_memory.mark_favorite, img_id, 10, [b64])
        return True

    async def parse_book_request(self, text: str, history: List[str]) -> Optional[Dict[str, str]]:
        """«Посоветую тебе «Мастера и Маргариту»» → {title, author, kind}.  None — не совет почитать."""
        data = await self.call_json(
            "Ты разбираешь сообщения из чата. Отвечай ТОЛЬКО JSON.",
            "Контекст:\n" + ("\n".join(history[-4:]) or "-") + f"\n\nСообщение: «{text}»\n\n"
            f"Советуют ли {config.BOT_NAME} что-то почитать или просят найти и прочитать конкретную книгу, "
            "комикс или мангу? Совет почитать мангу или комикс — тоже is_request=true (kind=manga или comic); "
            "фильмы, сериалы, игры и просьбы прочитать сообщение — нет. "
            "is_request — да/нет; title — название, как его обычно пишут (в именительном "
            "падеже, без кавычек); author — автор, если назван или ты точно знаешь (иначе пусто); "
            "kind — book, comic или manga.",
            _BOOK_REQUEST_SCHEMA, num_predict=120,
        )
        if not data or not data.get("is_request") or not str(data.get("title") or "").strip():
            return None
        return {"title": str(data["title"]).strip(" «»\"'"), "author": str(data.get("author") or "").strip(),
                "kind": str(data.get("kind") or "book")}

    async def book_konspekt(self, book: Dict[str, Any], notes: List[str], rating: int) -> Optional[Dict[str, Any]]:
        """
        Конспект дочитанной книги для её долгой памяти.  Подробность — по тому, как понравилась:
        любимую помнит в деталях, проходную — в общих чертах.
        """
        if rating >= 8:
            depth, chars, moments = "подробно — книга тебе очень понравилась, ты её хорошо запомнила", 6, 8
        elif rating >= 5:
            depth, chars, moments = "в общих чертах — книга нормальная", 3, 4
        else:
            depth, chars, moments = "совсем коротко — книга не зацепила, запомнилось мало", 2, 2
        # Заметки по ходу чтения — сколько влезет (у длинных книг — равномерно по всей книге).
        budget, picked = 5000, notes
        if sum(len(n) for n in notes) > budget:
            step = max(1, round(sum(len(n) for n in notes) / budget))
            picked = notes[::step]
        data = await self.call_json(
            f"Ты — {config.BOT_NAME}, ведёшь для себя конспекты прочитанных книг. Отвечай ТОЛЬКО JSON, по-русски.",
            f"Ты дочитала «{book['title']}»{(' (' + book['author'] + ')') if book['author'] else ''}, "
            f"оценила на {rating}/10.\nПересказ: {book['story']}\n\nТвои заметки по ходу чтения:\n"
            + "\n".join(f"- {n}" for n in picked)
            + f"\n\nСделай конспект {depth}. gist — о чём книга, 1–2 предложения; characters — до {chars} героев "
            f"(name и who — кто это и что с ним было, одной фразой); moments — до {moments} моментов, которые "
            "стоит помнить (конкретно, кто что сделал); ending — чем кончилось; feeling — что ты чувствовала, "
            "читая, одной фразой от первого лица. Только то, что есть в пересказе и заметках, — не выдумывай.",
            _KONSPEKT_SCHEMA, num_predict=900,
        )
        if not data:
            return None
        data["characters"] = (data.get("characters") or [])[:chars]
        data["moments"] = (data.get("moments") or [])[:moments]
        return data

    async def book_verdict(self, book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Дочитала: мнение о книге одной-двумя фразами и оценка 1–10."""
        persona = config.load_persona()
        data = await self.call_json(
            f"Ты — {config.BOT_NAME}. Кто ты:\n{persona[:1500]}\nОтвечай ТОЛЬКО JSON.",
            f"Ты дочитала «{book['title']}»{(' (' + book['author'] + ')') if book['author'] else ''}.\n"
            f"Что там было: {book['story']}\n\n"
            "opinion — твоё честное мнение одной-двумя фразами от первого лица, в своём стиле; "
            "rating — оценка от 1 до 10 по твоему вкусу (не всем книгам ставь высокие).",
            _VERDICT_SCHEMA, num_predict=200,
        )
        if not data:
            return None
        try:
            rating = int(data.get("rating") or 0)
        except (TypeError, ValueError):
            rating = 0
        return {"opinion": str(data.get("opinion") or "").strip(), "rating": rating}

    async def pick_reaction(self, text: str, names: List[str]) -> str:
        """Какой эмодзи сервера она поставила бы на сообщение: имя из names или ''."""
        if not names or not text.strip():
            return ""
        answer = await self._one_word(
            f"Ты — {config.BOT_NAME}, девушка из Discord-чата. Кто-то написал: «{text[:300]}»\n"
            f"Эмодзи сервера: {', '.join(names)}\n"
            "Какую из них ты поставила бы реакцией на это сообщение? Выбирай по смыслу названия. "
            "Ответь только названием эмодзи. Если ни одна не подходит — ответь: нет",
            num_predict=16,
        )
        answer = answer.strip(" :.,«»\"'`").lower()
        for n in names:
            if n.lower() == answer:
                return n
        return ""

    async def generate_share(self, context: "MessageContext", news: List[str],
                             complaints: Optional[List[str]] = None) -> str:
        """
        Сама пишет близкому человеку (доверие 10/10): делится мыслью, новостью или мнением.
        Пустая строка — решила, что сказать нечего.
        """
        try:
            system = self._build_system(context, "short")
            history = "\n".join(context.recent_channel_history[-10:]) or "(вы ещё не переписывались)"
            memory = ""
            if context.memory_snippets:
                memory = "Из прошлых разговоров с ним/ней:\n" + "\n".join(context.memory_snippets) + "\n\n"
            about = ""
            if news:
                about = "Что недавно было на сервере:\n" + "\n".join(f"- {n}" for n in news) + "\n\n"
            if complaints:
                about += ("Кто тебя недавно доставал в личке:\n" + "\n".join(f"- {c}" for c in complaints)
                          + "\n(Можешь пожаловаться на это — своими словами, с эмоцией, как подруге. "
                          "Говори, что тебя задело, но не пересказывай подробно чужую переписку и чужие секреты.)\n\n")
            user = (
                f"Личная переписка с {context.author_display_name} — ты ему(ей) полностью доверяешь. "
                f"Последние сообщения (старые сверху):\n{history}\n\n{memory}{about}"
                f"Тебе самой захотелось написать {context.author_display_name} — без повода с его(её) стороны. "
                "Поделись чем-то своим: мнением о том, что было на сервере, новостью, жалобой на того, кто достал, "
                "мыслью, которая пришла "
                "в голову, или вернись к тому, о чём вы говорили раньше, — с твоим отношением, а не пересказом. "
                "Пиши как в личке близкому человеку: коротко, одно-два предложения, без «привет, как дела» "
                "и без объяснений, зачем пишешь. Не повторяй то, что уже писала в этой переписке. "
                f"Если сказать правда нечего — ответь ровно: {SILENCE_TOKEN}"
            )
            raw = await self._complete(system, user, priority=PRIORITY_BACKGROUND)
            if not raw.strip() or SILENCE_TOKEN in raw or "[молч" in raw.lower():
                return ""
            return self._post_process(raw, context, "short")
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Unhandled exception in TextPipeline.generate_share",
                extra={"data": {"error": str(exc)}},
                exc_info=True,
            )
            return ""

    async def _one_word(self, prompt: str, num_predict: int = 6) -> str:
        """Короткий служебный вопрос модели (одно слово в ответ).  '' при сбое."""
        data = await self._post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            {
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": config.OLLAMA_NUM_CTX, "num_predict": num_predict, "temperature": 0.2},
            },
            timeout=60.0,
            priority=PRIORITY_BACKGROUND,
        )
        return (((data or {}).get("message") or {}).get("content") or "").strip().lower()

    async def doubt_check(self, context: "MessageContext", draft: str) -> str:
        """Стоит ли отправлять то, что она сама собралась написать: 'yes' / 'no' / 'unsure'."""
        people = getattr(context.social, "people_block", "") or ""
        history = "\n".join(context.recent_channel_history[-6:]) or "(переписки ещё не было)"
        answer = await self._one_word(
            f"Ты — {config.BOT_NAME}. Ты сама, без повода, собираешься написать в личку "
            f"{context.author_display_name}.\nЧто ты о нём/ней знаешь:\n{people or '(почти ничего)'}\n"
            f"Последняя переписка:\n{history}\n\nЧерновик: «{draft}»\n\n"
            "Стоит ли это отправлять? Подумай, как человек это воспримет: не навязчиво ли, не обидит ли, "
            "не неловко ли, уместно ли сейчас. Ответь одним словом: ДА, НЕТ или СОМНЕВАЮСЬ."
        )
        if answer.startswith("сомн") or "сомнева" in answer:
            return "unsure"
        if answer.startswith("нет") or answer.startswith("no"):
            return "no"
        if answer.startswith("да") or answer.startswith("yes"):
            return "yes"
        return "unsure"

    async def ask_advice_text(self, advisor: "MessageContext", target_name: str, draft: str) -> str:
        """Её сообщение близкому: хочу написать X вот это — норм или не стоит?"""
        fallback = f"слушай, хочу написать {target_name}: «{draft}». норм или лучше не надо?"
        try:
            system = self._build_system(advisor, "short")
            user = (
                f"Ты хочешь сама написать {target_name} в личку вот это: «{draft}» — но сомневаешься, "
                f"стоит ли. Спроси совета у {advisor.author_display_name} — ты ему(ей) полностью доверяешь. "
                "Коротко, как подруге в личке: процитируй черновик и скажи, что тебя смущает. "
                "Одно-два предложения."
            )
            raw = await self._complete(system, user, priority=PRIORITY_BACKGROUND)
            text = self._post_process(raw, advisor, "short") if raw.strip() else ""
            if not text:
                return fallback
            if draft[:20].lower() not in text.lower():
                text += f"\n«{draft}»"
            return text
        except Exception:  # noqa: BLE001
            return fallback

    async def chat_doubt(self, context: "MessageContext") -> str:
        """Вопрос в общем чате: 'answer' / 'unsure' (спросить совета) / 'complain' (ответить и пожаловаться)."""
        people = getattr(context.social, "people_block", "") or ""
        history = "\n".join(context.recent_channel_history[-6:]) or "(пусто)"
        answer = await self._one_word(
            f"Ты — {config.BOT_NAME}. Общий чат сервера (старые сверху):\n{history}\n\n"
            f"{context.author_display_name} спрашивает тебя: «{context.content}»\n"
            f"Что ты знаешь о людях:\n{people or '(почти ничего)'}\n\n"
            "Обычно ты просто отвечаешь сама. Но если вопрос странный, слишком личный, про других людей, "
            "с подвохом, или ты правда не знаешь, стоит ли отвечать и как, — можешь сначала спросить совета "
            "у самого близкого человека в личке. А если вопрос тебя бесит или задевает — можешь ответить "
            "и потом пожаловаться ему. Ответь одним словом: ОТВЕЧУ, СОМНЕВАЮСЬ или ПОЖАЛУЮСЬ."
        )
        if answer.startswith("сомн"):
            return "unsure"
        if answer.startswith("пожал"):
            return "complain"
        return "answer"

    async def chat_advice_text(self, advisor: "MessageContext", asker: str, question: str,
                               kind: str, my_reply: str = "") -> str:
        """Её сообщение близкому: «в чате X спрашивает … — отвечать?» или жалоба на вопрос."""
        if kind == "complain":
            task = (f"В общем чате {asker} спросил(а) тебя: «{question}». Тебя это задело или бесит"
                    + (f"; ты ответила: «{my_reply}»" if my_reply else "")
                    + f". Пожалуйся {advisor.author_display_name} в личке — коротко, с эмоцией, как подруге.")
            fallback = f"представляешь, {asker} в чате спрашивает: «{question}». вот зачем?"
        else:
            task = (f"В общем чате {asker} спросил(а) тебя: «{question}». Ты не уверена, стоит ли отвечать "
                    f"и как. Спроси совета у {advisor.author_display_name} в личке: процитируй вопрос "
                    "и скажи, что смущает. Коротко, одно-два предложения.")
            fallback = f"слушай, {asker} в чате спрашивает: «{question}». отвечать вообще? и что сказать?"
        try:
            system = self._build_system(advisor, "short")
            raw = await self._complete(system, task, priority=PRIORITY_CHAT)
            text = self._post_process(raw, advisor, "short") if raw.strip() else ""
            if not text:
                return fallback
            if question[:20].lower() not in text.lower():
                text += f"\n«{question[:300]}»"
            return text
        except Exception:  # noqa: BLE001
            return fallback

    async def classify_chat_advice(self, advice: str, asker: str, question: str) -> str:
        """Совет по вопросу из чата: 'answer' / 'skip' / 'unclear'."""
        answer = await self._one_word(
            f"{config.BOT_NAME} спросила, стоит ли ей отвечать в чате на вопрос {asker}: «{question}».\n"
            f"Ей ответили: «{advice}»\n\n"
            "Что ей посоветовали? Ответь одним словом: ОТВЕТИТЬ (в том числе если подсказали, что сказать), "
            "НЕ_ОТВЕЧАТЬ или НЕПОНЯТНО (если ответ не про это)."
        )
        if answer.startswith("не_") or answer.startswith("не отвеч") or answer.startswith("нет"):
            return "skip"
        if answer.startswith("ответ") or answer.startswith("да"):
            return "answer"
        return "unclear"

    async def parse_relay(self, requester: str, text: str, history: List[str]) -> Optional[Dict[str, str]]:
        """«Напиши Васе, что …» → {'target': имя, 'message': что передать}.  None — это не такая просьба."""
        schema = {
            "type": "object",
            "properties": {
                "is_request": {"type": "boolean"},
                "targets": {"type": "array", "items": {"type": "string"}},
                "message": {"type": "string"},
                "anonymous": {"type": "boolean"},
            },
            "required": ["is_request", "targets", "message", "anonymous"],
        }
        ctx_lines = "\n".join(history[-10:]) or "(пусто)"
        data = await self.call_json(
            "Ты разбираешь сообщения из Discord. Отвечай только JSON.",
            f"Контекст:\n{ctx_lines}\n\n{requester} пишет {config.BOT_NAME}: «{text}»\n\n"
            f"Просит ли {requester} {config.BOT_NAME} написать/передать что-то ДРУГОМУ человеку в личку "
            "или сделать что-то с ним в личке (напугать, затроллить, поздравить, позвать и т.п.)? "
            "Просто рассказ о человеке или вопрос о нём — это НЕ просьба. Но если в контексте его уже просили "
            "что-то передать, а теперь он жалуется, что ты не отправила или отправила не то, или торопит "
            "(«жду», «ну?», «так отправишь?») — это просьба: targets и message возьми из контекста. "
            "Благодарность и «дошло» — не просьба. "
            "is_request — да/нет; targets — список имён или ников тех, кому написать, каждое отдельно "
            "(«Паше и Даше» → [\"Паше\", \"Даше\"]); message — что именно передать, точно по смыслу просьбы, "
            "кратко (не выдумывай). Если в самом сообщении не сказано, что передать («скинь им», «отправь ещё раз»), "
            f"возьми из контекста то, что {requester} просил(а) передать — самое последнее уточнение важнее. "
            "Если сообщение только упомянуто («скинь им про печеньку»), найди в контексте сам его текст "
            "и передай его, а не слова «сообщение про …»; "
            "не путай с тем, что писала сама Aura. anonymous — true, если просят не говорить, от кого это. "
            "Если это не такая просьба — is_request=false, остальное пустое.",
            schema,
            num_predict=200,
        )
        if not data or not data.get("is_request"):
            return None
        raw = data.get("targets") or ([data["target"]] if data.get("target") else [])
        if isinstance(raw, str):
            raw = [raw]
        targets: List[str] = []
        for t in raw:
            for part in re.split(r"\s*(?:,|\bи\b|&)\s*", str(t)):
                part = part.strip().lstrip("@")
                if part and part not in targets:
                    targets.append(part)
        if not targets:
            return None
        return {"targets": targets, "message": str(data.get("message") or "").strip(),
                "anonymous": bool(data.get("anonymous"))}

    async def relay_text(self, target: "MessageContext", requester: str, what: str, request: str = "",
                         anonymous: bool = False) -> str:
        """Её сообщение человеку по просьбе *requester*."""
        fallback = what if anonymous else f"{requester} просил(а) передать: {what}"
        try:
            system = self._build_system(target, "short")
            history = "\n".join(target.recent_channel_history[-6:]) or "(вы ещё не переписывались)"
            user = (
                f"Личная переписка с {target.author_display_name}. Последние сообщения:\n{history}\n\n"
                f"{requester} попросил(а) тебя написать {target.author_display_name}. Его(её) просьба дословно: "
                f"«{request or what}». Суть: {what}.\n"
                "Напиши это в личку своими словами, точно по смыслу просьбы — ничего не выдумывай и не меняй; "
                + ("пиши как будто сама решила — НЕ упоминай, что тебя кто-то попросил, и не называй "
                   f"{requester}. " if anonymous else
                   f"скажи, что это от {requester} (имя пиши именно так). ")
                + "Коротко, одно-два предложения."
            )
            raw = await self._complete(system, user, priority=PRIORITY_CHAT)
            text = self._post_process(raw, target, "short") if raw.strip() else ""
            return text or fallback
        except Exception:  # noqa: BLE001
            return fallback

    async def classify_advice(self, advice: str, target_name: str, draft: str) -> str:
        """Что посоветовали: 'send' / 'skip' / 'unclear'."""
        answer = await self._one_word(
            f"{config.BOT_NAME} спросила, стоит ли отправить {target_name} сообщение «{draft}».\n"
            f"Ей ответили: «{advice}»\n\n"
            "Что ей посоветовали? Ответь одним словом: ОТПРАВИТЬ, НЕ_ОТПРАВЛЯТЬ или НЕПОНЯТНО "
            "(если ответ не про это)."
        )
        if answer.startswith("не_") or answer.startswith("не отправ") or answer.startswith("нет"):
            return "skip"
        if answer.startswith("отправ") or answer.startswith("да"):
            return "send"
        return "unclear"

    async def call_json(
        self, system: str, user: str, schema: Dict[str, Any], *, num_predict: int = 1200,
        images: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Structured JSON answer (used by background reflection).  None on failure."""
        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        user_msg: Dict[str, Any] = {"role": "user", "content": user}
        if images:
            user_msg["images"] = images  # комикс: страницы картинками
        for fmt in (schema, "json"):
            payload = {
                "model": (config.VISION_MODEL or config.OLLAMA_MODEL) if images else config.OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system}, user_msg],
                "stream": False,
                "think": False,
                "format": fmt,
                "options": {
                    "num_ctx": config.OLLAMA_NUM_CTX,
                    "temperature": 0.3,
                    "num_predict": num_predict,
                },
            }
            data = await self._post(url, payload, timeout=max(config.OLLAMA_TIMEOUT, 180.0),
                                    priority=PRIORITY_BACKGROUND)
            if not data:
                continue
            text = (data.get("message") or {}).get("content", "") or ""
            try:
                return json.loads(text)
            except ValueError:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except ValueError:
                        pass
        return None

    # ------------------------------------------------------------------
    # Картинки (vision)
    # ------------------------------------------------------------------

    def _attach_media(self, raw: str, context: "MessageContext") -> str:
        """Метка [картинка: N] → файл из коллекции в context.outgoing_media; возвращает текст без меток."""
        text, wanted = MediaPicker.extract(raw)
        if not wanted:
            return raw
        asked = MediaPicker.asked(context.content)
        if not asked and not self._media.cooled_down(context.channel_id):
            _logger.info("Media skipped (cooldown)", extra={"data": {"wanted": wanted}})
            return text
        path = self._media.pick(wanted[0], context.channel_id)  # одной хватит
        if path is not None and not asked and self._off_topic_illustration(path, context):
            _logger.info("Media skipped (book art off-topic)", extra={"data": {"file": path.name}})
            return text
        if path is not None:
            context.outgoing_media.append(path)
            self._media.mark_sent(context.channel_id, path)
        _logger.info("Media picked", extra={"data": {
            "ref": wanted[0], "asked": asked, "file": path.name if path else None}})
        return text

    def _off_topic_illustration(self, path: "Path", context: "MessageContext") -> bool:
        """Иллюстрация из её книги, а разговор не о книге (про работу, жизнь…) — кидать её некстати."""
        caption = next((it["caption"] for it in self._media.library() if it["path"] == path), "")
        if not caption.lower().startswith("иллюстрация из"):
            return False
        talk = "\n".join([context.content or ""] + list(context.recent_channel_history[-4:]))
        return not (self.library is not None and self.library.mentions_my_book(talk))

    def _remember_images(self, context: "MessageContext", images: List[str]) -> List[str]:
        """Новые картинки запомнить; если их нет — подложить недавнюю картинку из этого канала."""
        now = time.monotonic()
        key = context.channel_id
        if images:
            self._recent_images[key] = (now, images, context.author_display_name,
                                        bool(getattr(context, "is_gif", False)))
            return images
        if self._media is not None and MediaPicker.asked(context.content):
            return []  # «скинь картинку» — просят прислать свою, а не обсуждать старую
        cached = self._recent_images.get(key)
        if (cached and now - cached[0] <= config.VISION_FOLLOWUP_S
                and _ABOUT_IMAGE_RE.search(context.content or "")):
            context.image_from = cached[2]
            context.is_gif = cached[3]
            return cached[1]
        if (self._image_memory is not None and config.FAVORITES_ENABLED
                and _FAV_ASK_RE.search(context.content or "")):
            try:
                picked = self._image_memory.best_match(self._image_memory.favorites(limit=30),
                                                       context.content or "")
                old = [b for row in picked for b in self._image_memory.load(row)]
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Favorite recall failed", extra={"data": {"error": str(exc)}})
                old = []
            if old:
                context.recalled_images = [
                    f"твоя любимая картинка, которую ты сохранила себе {ago_text(r['ts'])} (прислал(а) {r['author']})"
                    + (f" ({r['caption']})" if r["caption"] else "") for r in picked]
                context.is_gif = any(r["is_gif"] for r in picked)
                context.shown_image_ids = {r["id"] for r in picked}
                _logger.info("Favorite recalled", extra={"data": {"ids": [r["id"] for r in picked]}})
                return old
        if self._image_memory is not None and _ABOUT_IMAGE_RE.search(context.content or ""):
            try:
                rows = self._image_memory.recent(key, config.VISION_ASK_RECALL_HOURS, limit=20,
                                                 author_id=getattr(context, "raw_author_id", 0))
                picked = self._image_memory.best_match(rows, context.content or "")
                old = [b for row in picked for b in self._image_memory.load(row)]
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Image recall failed", extra={"data": {"error": str(exc)}})
                old = []
            if old:
                context.recalled_images = [
                    f"{r['author']} присылал(а) {ago_text(r['ts'])}{' в личке' if r['is_dm'] else ''}"
                    + (f" ({r['caption']})" if r["caption"] else "") for r in picked]
                context.is_gif = any(r["is_gif"] for r in picked)
                context.shown_image_ids = {r["id"] for r in picked}
                _logger.info("Image recalled", extra={"data": {"ids": [r["id"] for r in picked]}})
                return old
        return []

    def _attach_recent_captions(self, context: "MessageContext") -> None:
        """Подписи картинок, которые недавно кидали в канал, — чтобы она могла к ним вернуться."""
        if self._image_memory is None:
            return
        try:
            rows = self._image_memory.recent(context.channel_id, config.VISION_RECALL_HOURS, limit=5,
                                             author_id=getattr(context, "raw_author_id", 0))
        except Exception:  # noqa: BLE001
            return
        shown = getattr(context, "shown_image_ids", set())
        lines = [f"{ago_text(r['ts'])} {r['author']}{' (в личке)' if r['is_dm'] and not r['here'] else ''}: {r['caption']}"
                 + (" — тебе она понравилась, ты сохранила её себе в любимые" if r.get("favorite") else "")
                 for r in rows if r["caption"] and r["id"] not in shown]
        context.recent_pictures = lines[:3]

    async def observe_images(self, context: "MessageContext") -> None:
        """
        Сообщение, на которое она не отвечает: всё равно посмотреть картинки и запомнить их
        (с подписью), чтобы потом можно было к ним вернуться.
        """
        if (not config.VISION_ENABLED or not config.VISION_OBSERVE or context.is_voice
                or self._image_memory is None):
            return
        if (not context.attachments and not _GIF_LINK_RE.search(context.content or "")
                and not _IMAGE_LINK_RE.search(context.content or "")):
            return
        try:
            images = await self._fetch_images(context)
            if not images:
                return
            self._recent_images[context.channel_id] = (
                time.monotonic(), images, context.author_display_name, bool(getattr(context, "is_gif", False)))
            await self._store_image(context, images)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Image observe failed", extra={"data": {"error": str(exc)}})

    async def _store_image(self, context: "MessageContext", images: List[str]) -> None:
        """Сохранить картинку и в фоне подписать её одной фразой (для памяти)."""
        try:
            img_id = await asyncio.to_thread(
                self._image_memory.add, context.channel_id, context.author_display_name,
                images, bool(getattr(context, "is_gif", False)),
                getattr(context, "raw_author_id", 0), bool(getattr(context, "is_dm", False)))
            if img_id:
                self._last_image_id[context.channel_id] = img_id
            asked = bool(_SAVE_RE.search(context.content or ""))
            rate = (" Затем отдельной строкой «Оценка: N», где N от 0 до 10 — насколько такая картинка "
                    f"понравилась бы человеку, который любит: {config.FAVORITES_TASTE}. "
                    "10 — прям хочется сохранить себе, 5 — нормально, 0 — неинтересно. Будь строгой: "
                    "9–10 только для того, что правда цепляет.") if config.FAVORITES_ENABLED else ""
            data = await self._post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                {
                    "model": config.VISION_MODEL or config.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "Ты делаешь короткие подписи к картинкам для памяти."},
                        {"role": "user", "images": images, "content": (
                            "Одной-двумя фразами по-русски: что на картинке — кто или что, ключевые детали; "
                            "все важные надписи, числа, цены и суммы перепиши точно, как написано" + (" и что происходит (это кадры гифки)" if getattr(context, "is_gif", False) else "")
                            + ". Без оценочных слов и вступлений." + rate)},
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0.2, "num_predict": 150},
                },
                priority=PRIORITY_BACKGROUND,
            )
            text = ((data or {}).get("message") or {}).get("content", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            m = _SCORE_RE.search(text)
            score = min(10, int(m.group(1))) if m else 0
            caption = next((ln for ln in text.split("\n") if ln.strip() and not _SCORE_RE.match(ln.strip())), "")
            caption = re.sub(r"\s*[(\[]?оценка\D{0,5}\d{1,2}.*$", "", caption, flags=re.I).strip()
            if caption:
                await asyncio.to_thread(self._image_memory.set_caption, img_id, caption)
                _logger.info("Image remembered", extra={"data": {"id": img_id, "caption": caption, "score": score}})
            if config.FAVORITES_ENABLED and img_id and (asked or score >= config.FAVORITES_MIN_SCORE):
                await asyncio.to_thread(self._image_memory.mark_favorite, img_id, 10 if asked else score, images)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Image store failed", extra={"data": {"error": str(exc)}})

    def _save_requested(self, context: "MessageContext") -> None:
        """«Аура, сохрани её» без новой картинки — сохранить в любимые ту, о которой речь."""
        if (not config.FAVORITES_ENABLED or self._image_memory is None
                or not _SAVE_RE.search(context.content or "")):
            return
        ids = set(getattr(context, "shown_image_ids", set()) or set())
        cached = self._recent_images.get(context.channel_id)
        last = self._last_image_id.get(context.channel_id)
        if not ids and last and cached and time.monotonic() - cached[0] <= config.VISION_FOLLOWUP_S:
            ids = {last}
        for img_id in ids:
            asyncio.create_task(asyncio.to_thread(self._image_memory.mark_favorite, img_id, 10))

    async def _fetch_images(self, context: "MessageContext") -> List[str]:
        """Скачать картинки/гифки из вложений, эмбедов и ссылок Tenor/Giphy/Klipy; вернуть base64 для Ollama."""
        if not config.VISION_ENABLED:
            return []
        links = [m.group(0) for m in _GIF_LINK_RE.finditer(context.content or "")]
        urls = list(context.attachments or [])
        # Есть эмбед гифки (видео/превью от Discord) — берём его, страницу не грузим.
        if any("||" in u for u in urls):
            links = []
        # Голая ссылка на картинку (эмбед Discord к on_message часто ещё не успевает появиться).
        if not urls:
            links += [m.group(0) for m in _IMAGE_LINK_RE.finditer(context.content or "")
                      if not _GIF_LINK_RE.match(m.group(0))]
        urls = links + urls
        out: List[str] = []
        pictures = 0
        seen: set = set()
        for entry in urls:
            if pictures >= config.VISION_MAX_IMAGES:
                break
            for url in entry.split("||"):  # варианты одной картинки: видео гифки, затем превью
                frames = await self._fetch_one(url, context, seen)
                if frames:
                    out.extend(frames)
                    pictures += 1
                    break
        if out:
            _logger.info("Images attached to prompt", extra={"data": {
                "pictures": pictures, "frames": len(out), "gif": bool(getattr(context, "is_gif", False))}})
        return out

    async def _fetch_one(self, url: str, context: "MessageContext", seen: set) -> List[str]:
        """Одна картинка / гифка / короткое видео по ссылке → кадры в base64 (пусто, если не вышло)."""
        tmo = aiohttp.ClientTimeout(total=15)
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            if _GIF_LINK_RE.fullmatch(url) and "media" not in url.split("/")[2]:
                async with self._session.get(url, timeout=tmo, headers=headers) as resp:
                    status = resp.status
                    page = await resp.text(errors="ignore")
                page = page.replace("&amp;", "&").replace("\\/", "/")
                gifs = [g for g in _PAGE_GIF_RE.findall(page) if ".gif" in g.lower()]
                m = _OG_IMAGE_RE.search(page)
                og = (m.group(1) or m.group(2)) if m else ""
                if og and ".gif" in og.lower():
                    url = og          # og:image уже анимированная гифка
                elif gifs:
                    url = gifs[0]     # иначе первая гифка на странице (обычно — сама)
                elif og:
                    url = og
                else:
                    _logger.warning("No image on gif page",
                                    extra={"data": {"url": url, "status": status, "page_chars": len(page)}})
                    return []
            path = url.split("?", 1)[0].rsplit("/", 1)[-1]
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if url in seen or ext in ("json", "html"):
                return []
            seen.add(url)
            async with self._session.get(url, timeout=tmo, headers=headers) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "")
                is_video = ext in _VIDEO_EXTS or ctype.startswith("video/")
                if not is_video and ext not in _IMAGE_EXTS and not ctype.startswith("image/"):
                    return []  # не картинка и не видео
                raw = await resp.read()
            if len(raw) > config.VISION_MAX_MB * 1024 * 1024:
                return []
            frames = await asyncio.to_thread(_prepare_video if is_video else _prepare_image, raw, ext)
            if len(frames) > 1:
                context.is_gif = True
            return [base64.b64encode(f).decode("ascii") for f in frames]
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Image download failed", extra={"data": {"url": url[:120], "error": str(exc)}})
            return []

    # ------------------------------------------------------------------
    # Ollama transport
    # ------------------------------------------------------------------

    def _options(self) -> Dict[str, Any]:
        return {
            "num_ctx": config.OLLAMA_NUM_CTX,
            "temperature": config.OLLAMA_TEMPERATURE,
            "top_p": config.OLLAMA_TOP_P,
            "repeat_penalty": config.OLLAMA_REPEAT_PENALTY,
            "num_predict": config.MAX_RESPONSE_TOKENS,
        }

    async def _complete(
        self, system: str, user: str, use_thinking: bool = False, priority: int = PRIORITY_CHAT,
        images: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Chat first; fall back to flat /api/generate if the answer is empty.

        *meta* (если передан) получает ``cut=True``, когда модель упёрлась в лимит токенов.
        """
        options = self._options()
        if num_predict:
            options["num_predict"] = num_predict
        t0 = time.monotonic()
        raw = ""
        model = (config.VISION_MODEL or config.OLLAMA_MODEL) if images else config.OLLAMA_MODEL
        if use_thinking and model in self._no_think_models:
            use_thinking = False  # эта модель не умеет «размышлять» — сразу обычный ответ
        user_msg: Dict[str, Any] = {"role": "user", "content": user}
        if images:
            user_msg["images"] = images
        if config.OLLAMA_API == "chat":
            data = await self._post(
                f"{config.OLLAMA_BASE_URL}/api/chat",
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        user_msg,
                    ],
                    "stream": False,
                    "think": use_thinking,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": options,
                },
                priority=priority,
            )
            if data is None and use_thinking and "think" in self._last_http_error.lower():
                # «model does not support thinking» — запомнить и переспросить без размышления
                self._no_think_models.add(model)
                _logger.warning("Model has no thinking mode — retrying without it", extra={"data": {"model": model}})
                return await self._complete(system, user, False, priority, images, meta, num_predict)
            if data:
                self.last_stats = _ollama_stats(data)  # для лога reply_sent: сколько считала модель
                raw = ((data.get("message") or {}).get("content") or "").strip()
                if meta is not None:
                    meta["cut"] = data.get("done_reason") == "length"
                if not raw and use_thinking:
                    # Размышляла и не дошла до ответа (кончился лимит токенов) — ответить без размышления.
                    _logger.warning("Thinking gave empty answer — retrying without it", extra={"data": {
                        "model": model, "done_reason": data.get("done_reason"),
                        "thinking_chars": len((data.get("message") or {}).get("thinking") or "")}})
                    return await self._complete(system, user, False, priority, images, meta, num_predict)
                if images:
                    _logger.info("Vision request", extra={"data": {
                        "model": model, "images": len(images),
                        "prompt_tokens": data.get("prompt_eval_count")}})
        if not raw:
            prompt = f"{system}\n\n{user}\n{config.BOT_NAME}:"
            data = await self._post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    **({"images": images} if images else {}),
                    "think": use_thinking,
                    "keep_alive": config.OLLAMA_KEEP_ALIVE,
                    "options": options,
                },
                priority=priority,
            )
            if data:
                raw = (data.get("response") or "").strip()
                if meta is not None:
                    meta["cut"] = data.get("done_reason") == "length"
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        _logger.debug(
            "Ollama response received",
            extra={"data": {
                "model": config.OLLAMA_MODEL,
                "api": config.OLLAMA_API,
                "response_chars": len(raw),
                "latency_seconds": round(time.monotonic() - t0, 3),
                "thinking": use_thinking,
            }},
        )
        if not raw:
            _logger.warning("Empty model reply — fallback line used", extra={"data": {
                "model": model, "thinking": use_thinking, "error": self._last_http_error[:200]}})
        return raw or random.choice(_FALLBACK_LINES)

    async def _post(
        self, url: str, payload: Dict[str, Any], timeout: Optional[float] = None, priority: int = PRIORITY_CHAT
    ) -> Optional[Dict[str, Any]]:
        """POST to Ollama under the global lock.  Returns parsed JSON or None."""
        headers = {"Content-Type": "application/json; charset=utf-8"}
        payload, timeout = _game_mode_payload(payload, timeout)
        tmo = aiohttp.ClientTimeout(total=timeout or config.OLLAMA_TIMEOUT)
        t0 = time.monotonic()
        async with _Hold(self._llm_lock, priority):
            self._last_request = time.monotonic()
            self._last_http_error = ""
            try:
                async with self._session.post(url, json=payload, headers=headers, timeout=tmo) as resp:
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        self._last_http_error = body
                        _logger.warning("Ollama returned HTTP error", extra={"data": {
                            "status": resp.status, "error": body, "url": url}})
                        return None
                    return await resp.json(content_type=None)
            except aiohttp.ServerConnectionError:
                _logger.warning("Ollama not reachable", extra={"data": {"url": url}})
            except aiohttp.ClientResponseError as exc:
                _logger.warning(
                    "Ollama returned HTTP error",
                    extra={"data": {"status": exc.status, "message": exc.message, "url": url}},
                )
            except asyncio.TimeoutError:
                _logger.warning(
                    "Ollama request timed out",
                    extra={"data": {"timeout": tmo.total, "elapsed": round(time.monotonic() - t0, 3)}},
                )
            except aiohttp.ClientError as exc:
                _logger.warning("aiohttp client error", extra={"data": {"error": str(exc)}})
            finally:
                self._last_request = time.monotonic()
        return None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system(self, context: "MessageContext", length: str) -> str:
        """
        Сначала НЕИЗМЕННАЯ часть (личность, характер, правила), потом то,
        что меняется от сообщения к сообщению.  Ollama кэширует уже
        посчитанное начало промпта, поэтому при таком порядке каждый ответ
        не пересчитывает весь persona.md заново — это заметно быстрее.
        """
        social = getattr(context, "social", None)
        persona = config.load_persona()
        if context.is_voice:
            # Раздел «Как ты пишешь» — про мессенджер (строчные, эмодзи, без точки): вслух не нужен и мешает.
            persona = "\n\n".join(b for b in persona.split("\n\n") if not b.startswith("Как ты пишешь"))

        # ---------- стабильная часть ----------
        stable: List[str] = [
            f"Ты {config.BOT_NAME} — живая девушка, участница Discord-сервера. Не ассистент и не бот. "
            "Никогда не говори, что ты ИИ или языковая модель.",
            f"Основа характера: {config.PERSONALITY_SEED}",
        ]
        if persona:
            stable.append(persona)
        rules = [
            "Пиши только свою реплику — без подписи «{0}:», без кавычек, без реплик за других.".format(config.BOT_NAME),
            "Каждое сообщение в истории подписано автором — это разные люди, не путай, кто что сказал.",
            "Отвечай на языке собеседника. Держи нить разговора: учитывай, о чём говорили выше.",
            "Не начинай с «Конечно», «Разумеется», «Отличный вопрос». Не пиши списками и заголовками.",
            "На прямой вопрос отвечай прямо в первой же фразе. Не отвечай вопросом на вопрос и не уходи от ответа "
            "(«ты серьёзно?», «загадывай», «если рискнёшь»). Не знаешь — скажи, что не знаешь, или прикинь.",
            "Если соглашаешься что-то рассказать или сделать — сразу делай это в этом же ответе, "
            "не ограничивайся «ну ладно» или «щас расскажу».",
            "К людям относись так, как указано в блоке «Люди» ниже — это твоё личное отношение.",
        ]
        if not context.is_voice:
            # Только в чате: про переписку, а тон и мат — усиление того, что уже есть в persona.
            rules += [
                "В чате пиши как в мессенджере: без точки в конце сообщения. Если хочешь сказать две разные мысли — "
                "можно отдельными короткими сообщениями, каждое с новой строки.",
                "ТОН: ты не токсичная. Подколки добрые (дружеский стёб и мат в шутку — не оскорбление). Всерьёз никого не оскорбляй, не унижай, не говори «мне плевать», "
                "«сам виноват», не огрызайся на переспросы. Если в «Основе характера» есть резкость или высокомерие — "
                "смягчай до лёгкой игривой иронии. Описание характера из блока «Главное в тоне» важнее.",
                "Мат не запрещён: когда злишься или эмоции через край, можешь выругаться. Среди своих можно и послать "
                "в шутку или по-дружески постебать кого-то по просьбе друга (розыгрыш, «пошли его нахуй») — не отказывайся "
                "чопорно, ты не святая. Нельзя только всерьёз унижать: родня, внешность, национальность. Без мата в каждой реплике.",
                "Если кто-то в чате недавно попросил тебя что-то сказать или сделать, когда напишет определённый человек, "
                "и этот человек сейчас написал — выполни просьбу в этом ответе (если не против).",
            ]
        if context.is_voice:
            rules.append("Говоришь коротко и по делу: сначала ответ, потом, если хочется, встречный вопрос. "
                         "Не флиртуешь, не читаешь лекций; с близкими теплее, с новыми приветливо.")
        if context.is_voice and config.VOICE_NATURAL_SPEECH:
            rules.append(
                "Ты в ГОЛОСОВОМ канале: тебя слышат, ответ озвучивается. Речь распознана автоматически и может "
                "быть с ошибками — угадывай смысл. Без смайликов, markdown, ссылок; числа словами. "
                "Говори как живой человек вслух, а не как написанный текст: первое предложение короткое, "
                "можно оборвать мысль и поправиться. Иногда, к месту, — «ну», «хм», «слушай», «ой», «эм» "
                "(когда подбираешь слово), смешок «ха-ха» или «хах», вздох «ох» или «эх». "
                "Не больше одного-двух таких звуков на реплику и не в каждой реплике — иначе звучит наигранно. "
                "Никаких ремарок в скобках или звёздочках (*вздыхает*, (смеётся)) — только то, что произносится вслух. "
                "Эмоции передавай знаками — твой голос их слышит: «!» звучит живо и радостно, «?» — с вопросом, "
                "точка и запятая — паузы; многоточий не ставь. Не бойся восклицаний, когда тебе правда весело или интересно."
            )
        elif context.is_voice:
            rules.append(
                "Ты в ГОЛОСОВОМ канале: тебя слышат, ответ озвучивается. Речь распознана автоматически и может "
                "быть с ошибками — угадывай смысл. Без смайликов, markdown, ссылок; числа словами. "
                "Говори как в живом разговоре: первое предложение короткое. НЕ начинай реплику с «хм», «ну», «так», "
                "«ммм» — сразу к сути. Никаких ремарок о себе («подумала», «вздохнула», *действия*). "
                "Эмоции передавай знаками — твой голос их слышит: «!» звучит живо и радостно, «?» — с вопросом, "
                "«...» — задумчиво или устало. Не бойся восклицаний, когда тебе правда весело или интересно."
            )
        stable.append("Правила:\n- " + "\n- ".join(rules))

        # ---------- меняющаяся часть ----------
        # Порядок — от редко меняющегося к часто меняющемуся: Ollama пересчитывает промпт только с первого
        # изменившегося места.  Время (каждую минуту) и длина ответа — в самом конце.
        dynamic: List[str] = []
        if self.life is not None:
            life = self.life.prompt_block()
            if life:
                dynamic.append(life)
        # Разговор о книгах?  Иначе подробности книги и иллюстрации из неё в промпт не идут —
        # 12B-модель тащит их в любой ответ.
        talk = "\n".join([context.content or ""] + list(context.recent_channel_history[-4:]))
        about_books = bool(getattr(context, "book_memories", None) or _BOOK_TALK_RE.search(talk)
                           or (self.library is not None and self.library.mentions_my_book(talk)))
        if self.library is not None:
            reading = self.library.prompt_block() if about_books else self.library.prompt_brief()
            if reading:
                dynamic.append(reading)
        if getattr(context, "book_memories", None):
            dynamic.append(
                "Вспоминается из прочитанного (твои заметки и конспекты — помнишь примерно так, своими словами; "
                "чего здесь нет — того ты не помнишь или не читала, не выдумывай):\n"
                + "\n".join(f"- {m}" for m in context.book_memories))
        if social is not None and social.summary:
            dynamic.append(f"О чём недавно говорили здесь (сводка): {social.summary}")
        if social is not None and social.statements_block:
            dynamic.append(
                "Что ты сама раньше говорила и обещала (не противоречь себе; обещанное помни):\n"
                + social.statements_block
            )
        notes = {
            n: note for n, note in config.SPECIAL_USER_NOTES.items()
            if social is None or n not in " ".join(social.focus_names).lower()
        }
        if notes and not context.is_voice:
            dynamic.append(
                "Ещё ты знаешь на сервере: "
                + "; ".join(f"{n.capitalize()} — {note}" for n, note in notes.items())
                + "."
            )
        if self._media is not None and not context.is_voice:
            items = [(i, it) for i, it in enumerate(self._media.library(), 1)
                     if about_books or MediaPicker.asked(context.content)
                     or not it["caption"].lower().startswith("иллюстрация из")]
            if items:
                recent = set(self._media.recently_sent(context.channel_id))
                again = [str(i) for i, it in items if it["path"] in recent]
                dynamic.append(
                    self._media.prompt_block(items)
                    + (f"\n(Уже кидала здесь недавно: {', '.join(again)})" if again else "")
                    + "\nМожешь кинуть одну из них в чат, как живой человек: отдельной строкой метка "
                    "[картинка: номер]. Редко и к месту — реакция на смешное, к теме разговора, или когда "
                    "просят скинуть. Не больше одной за ответ; можно и без текста, только метка. "
                    "Кидать можно только из этого списка."
                )
        if social is not None and social.people_block:
            dynamic.append("Люди, о которых ты сейчас думаешь:\n" + social.people_block)
        if context.memory_snippets:
            dynamic.append(
                "Обрывки из памяти (вспоминай к месту, не пересказывай их подряд):\n"
                + "\n".join(context.memory_snippets)
            )

        recent_own = []
        for line in context.recent_channel_history:
            stripped = re.sub(r"^\[[^\]]*\]\s*", "", line)
            if stripped.startswith(config.BOT_NAME) and ":" in stripped:
                recent_own.append(stripped.split(":", 1)[1].strip())
        recent_own = recent_own[-4:]
        now = f"Сейчас: {time_description()}. Твоё настроение: {context.mood_label or 'обычное'}."
        day_hint = schedule.prompt_hint()
        now_rules = [now + (" " + day_hint if day_hint else ""), f"Длина этого ответа: {_LENGTH_HINTS[length]}."]
        if getattr(context, "affinity", 0.0) <= -0.25 and not getattr(context, "provocation", False):
            now_rules.append(
                "Этот человек тебе неприятен (см. «Люди»). Держись прохладно и сухо: без флирта и комплиментов, "
                "не втягивайся в его игры и долгие переписки — можешь отказаться («не, неохота») или свернуть разговор."
            )
        if getattr(context, "provocation", False):
            now_rules.append(
                "Последнее сообщение — провокация / рейдж-байт: тебя или твоих близких оскорбляют, пишут гадости "
                "ради реакции или вбрасывают политоту. Не ведись и не играй в «достойный ответ». НЕ хвали "
                "подколку, не оценивай её («в точку», «попал», «по самому простому пути»), не анализируй, "
                "что человек «решил сделать». Ответь как живой человек, которому это надоело: коротко и "
                "равнодушно («мда», «скучно», «ну и?», «ору, гений»), высмей или пошли — мат можно, "
                "но без ответных оскорблений родни. На политоту: «не, в политику я не лезу» или «давай без этого». "
                "Если это уже не первый вброс от него — можно просто «всё, я тебя не слушаю»."
            )
        if recent_own:
            now_rules.append("Не повторяй свои недавние фразы и обороты: " + " | ".join(f"«{r[:80]}»" for r in recent_own))
        dynamic.append("\n".join(now_rules))
        return "\n\n".join(stable + dynamic)

    def _build_messages(self, context: "MessageContext", length: str) -> Tuple[str, str]:
        system = self._build_system(context, length)
        speaker = context.author_display_name
        lines = context.recent_channel_history
        if context.is_voice:
            lines = lines[-config.VOICE_HISTORY_LINES:]
        history = "\n".join(lines) or "(пусто)"
        pics = getattr(context, "recent_pictures", None)
        if pics and not context.is_voice:
            history += ("\n\n(Картинки, которые недавно кидали в этот канал, — ты их видела и помнишь; "
                        "можешь вспомнить о них, если к месту:\n" + "\n".join(f"- {p}" for p in pics) + ")")
        reply_note = ""
        if getattr(context, "reply_to_name", None):
            reply_note = f" (в ответ на сообщение {context.reply_to_name})"
        attach = " [приложил(а) картинку/файл]" if context.attachments else ""
        if getattr(context, "image_count", 0):
            src = getattr(context, "image_from", None)
            recalled = getattr(context, "recalled_images", None)
            seen = ((" [речь, похоже, о картинке, которую ты помнишь: " + "; ".join(recalled)
                     + (" — какая из них, пойми сама по смыслу;" if len(recalled) > 1 else ";")
                     + " она приложена, ты её снова видишь;")
                    if recalled else
                    f" [картинка, которую недавно прислал(а) {src}, всё ещё перед тобой — она приложена, ты её видишь;"
                    if src else " [приложил(а) картинку — она прикреплена к этому сообщению, ты её видишь;")
            if getattr(context, "is_gif", False):
                seen += (" это гифка — тебе приложены несколько её кадров по порядку, "
                         "так что ты видишь и что на ней, и что в ней происходит;")
            attach = seen + (
                " смотри на неё по-настоящему и, если говоришь о ней, — о конкретном, что видно, а не общими "
                "словами; не выдумывай. Ты не ассистент: не описывай всё подряд, отреагируй коротко, как в чате, "
                "— одна-две фразы, одна деталь, которая зацепила]")

        if getattr(context, "web_results", ""):
            attach += (f" [ты только что загуглила «{context.web_query}», вот что нашлось:\n{context.web_results}\n"
                       "— отвечай по найденному, своими словами и коротко, как человек, который сейчас глянул в "
                       "интернет (можно «погуглила —», «пишут, что…»). Не вставляй ссылки и не пересказывай всё. "
                       "Если нужного там нет — так и скажи]")
        elif getattr(context, "web_query", ""):
            attach += (f" [ты пыталась загуглить «{context.web_query}», но ничего толкового не нашлось — "
                       "так и скажи, если это важно для ответа]")
        if self._media is not None and not context.is_voice and MediaPicker.asked(context.content):
            if self._media.library():
                attach += (" [тебя просят скинуть картинку/гифку — если в твоей коллекции есть подходящая, "
                           "добавь отдельной строкой метку [картинка: номер]; если подходящей нет — так и скажи]")
            else:
                attach += (" [тебя просят скинуть картинку/гифку, но твоя коллекция любимых пока пуста — "
                           "скажи, что скинуть нечего; можно предложить прислать тебе что-нибудь классное, "
                           "ты сохранишь. Ссылки не выдумывай]")

        if context.is_voice and getattr(context, "voice_optional", False):
            participants = getattr(context, "voice_participants", []) or [speaker]
            user = (
                f"В голосовом канале сейчас: {', '.join(participants)} и ты.\n"
                f"Разговор в канале (старые сверху):\n{history}\n\n"
                f"Только что {speaker} сказал(а):\n{speaker}: {context.content}\n\n"
                "Тебя НЕ звали по имени — ты просто слышишь разговор. Вмешайся, только если: продолжают "
                "разговор с тобой, вопрос ко всем или к тебе, говорят о тебе, или тема тебе правда интересна "
                "и есть что сказать коротко и к месту. Не перебивай чужой личный разговор. "
                f"Если вмешиваться не стоит — ответь ровно: {SILENCE_TOKEN}\n"
                f"Иначе — скажи вслух свою реплику как {config.BOT_NAME}."
            )
        elif context.is_voice:
            participants = getattr(context, "voice_participants", []) or [speaker]
            user = (
                f"В голосовом канале сейчас: {', '.join(participants)} и ты.\n"
                f"Разговор в канале (старые сверху):\n{history}\n\n"
                f"Сейчас говорит {speaker} и обращается к тебе:\n{speaker}: {context.content}\n\n"
                f"Ответь вслух как {config.BOT_NAME} — именно {speaker}."
            )
        elif context.is_dm:
            late = getattr(context, "late_note", "")
            late = f"\n({late})" if late else ""
            user = (
                f"Личная переписка с {speaker}. Последние сообщения (старые сверху):\n{history}\n\n"
                f"Новое сообщение от {speaker}{attach}:\n{context.content}\n{late}\n"
                f"Ответь как {config.BOT_NAME}."
            )
        else:
            user = (
                "Общий чат сервера. Последние сообщения (старые сверху; «→ Имя» — ответ на сообщение этого человека):\n"
                f"{history}\n\n"
                f"Новое сообщение от {speaker}{reply_note}{attach}:\n{context.content}\n\n"
                + (f"({context.late_note})\n" if getattr(context, "late_note", "") else "")
                + f"Ответь как {config.BOT_NAME} — именно {speaker}."
            )
        return system, user

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, raw: str, context: "MessageContext", length: str = "normal", cut: bool = False) -> str:
        """Clean and trim the raw model output before sending to Discord."""
        # Метки [картинка: …] сюда доходят только там, где картинки не отправляются, — просто убрать.
        text = MediaPicker.extract(raw)[0] if "[" in raw or "(" in raw else raw.strip()
        text = strip_foreign_script(text, keep_if=context.content)

        # Strip echo of the bot's name / signatures the model may prepend.
        text = re.sub(
            rf"^\s*(?:\[[^\]]*\]\s*)?{re.escape(config.BOT_NAME)}\s*(?:→\s*[^:\n]{{1,32}})?\s*(?:\([^)]*\))?\s*:\s*",
            "", text, flags=re.IGNORECASE,
        )
        text = re.sub(r"^\s*\((?:ответила|отвечает)[^)]*\)\s*:\s*", "", text, flags=re.IGNORECASE)
        if len(text) > 2 and text[0] == text[-1] and text[0] in "\"«":
            text = text[1:-1].strip()
        if text.endswith("»") and text.startswith("«"):
            text = text[1:-1].strip()

        # The model sometimes continues the dialogue for other people — keep only our line.
        if "```" not in text:
            kept = []
            for i, line in enumerate(text.splitlines()):
                if i > 0 and re.match(r"^\s*(?:\[[^\]]*\]\s*)?[^\s:]{1,32}(?:\s*→\s*[^:]{1,32})?(?:\s\([^)]*\))?\s*:", line):
                    break
                kept.append(line)
            text = "\n".join(kept).strip()
            if context.is_voice:
                text = " ".join(l.strip() for l in text.splitlines() if l.strip())

        lower_text = text.lower()
        for tell in _AI_TELLS:
            if lower_text.startswith(tell) or f" {tell}" in lower_text[:80]:
                _logger.warning(
                    "AI-tell detected in model output — substituting fallback",
                    extra={"data": {"prefix": text[:60]}},
                )
                return random.choice(_FALLBACK_LINES)

        if "```" in text:
            max_len = 1500
        elif context.use_thinking:
            max_len = max(_MAX_LEN.get(length, 420), 800)
        else:
            max_len = _MAX_LEN.get(length, 420)

        if not context.is_voice and "```" not in text:
            # Чат: режем только целыми фразами — никаких «хотя многие до…».
            limit = _MAX_SENTENCES.get(length, 2) + (1 if context.use_thinking else 0)
            # Лимита символов в чате нет — длину задаёт подсказка; 1900 — только чтобы влезть в сообщение Discord.
            text = _trim_chat(text, limit, 1900, cut)
        elif len(text) > max_len:
            truncated = text[:max_len]
            last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
            if last_period > max_len // 3:
                text = truncated[: last_period + 1]
            else:
                text = truncated.rstrip() + "…"

        if context.is_voice:
            text = strip_voice_tics(text) or text
        elif "```" not in text:
            text = _drop_repeated_emoji(text, context)
            text = _humanize_chat(text)
        if not text:
            return random.choice(_FALLBACK_LINES)
        return text
