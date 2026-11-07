"""
config.py — centralised settings loaded from .env via python-dotenv.

All file/folder references use pathlib.Path.  Every other module imports
settings from this module rather than reading environment variables
directly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Locate the project root (the directory that contains this file) and load
# the .env file that lives beside it.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_str(key: str, default: str = "") -> str:
    """Return a string env-var, falling back to *default*."""
    return os.getenv(key, default)


def _get_float(key: str, default: float) -> float:
    """Return a float env-var, falling back to *default*."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    """Return an int env-var, falling back to *default*."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_id_list(key: str) -> List[int]:
    """Parse a comma-separated list of integer IDs from an env-var."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    result: List[int] = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

DISCORD_TOKEN: str = _get_str("DISCORD_TOKEN")
GUILD_IDS: List[int] = _get_id_list("GUILD_IDS")

# Тестовый режим: читает и пишет только на серверах TEST_GUILD_IDS и в личке.
# Остальные серверы для неё как будто не существуют (не читает, не запоминает, не пишет, не заходит в войс).
TEST_MODE: bool = os.getenv("TEST_MODE", "false").strip().lower() in ("1", "true", "yes", "on")
TEST_GUILD_IDS: List[int] = _get_id_list("TEST_GUILD_IDS") or [1403148326769135748]  # сервер «11»


def guild_allowed(guild_id: Optional[int]) -> bool:
    """Можно ли ей здесь читать и писать: личка — всегда, сервер — если не тестовый режим или он в списке."""
    return guild_id is None or not TEST_MODE or guild_id in TEST_GUILD_IDS


# Optional whitelist — empty list means "all channels allowed"
_raw_whitelist = os.getenv("CHANNEL_WHITELIST", "").strip()
CHANNEL_WHITELIST: List[int] = (
    [int(x.strip()) for x in _raw_whitelist.split(",") if x.strip().isdigit()]
    if _raw_whitelist
    else []
)

# ---------------------------------------------------------------------------
# Bot personality
# ---------------------------------------------------------------------------

BOT_NAME: str = _get_str("BOT_NAME", "Aria")
PERSONALITY_SEED: str = _get_str(
    "PERSONALITY_SEED",
    "introverted, slightly chaotic, dry humor, warm when comfortable",
)

# ---------------------------------------------------------------------------
# Response behaviour
# ---------------------------------------------------------------------------

BASE_RESPONSE_PROBABILITY: float = _get_float("BASE_RESPONSE_PROBABILITY", 0.3)
# Сколько секунд после её ответа реплики того же человека считаются продолжением разговора.
FOLLOWUP_WINDOW_S: int = _get_int("FOLLOWUP_WINDOW_S", 180)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = _get_str("LOG_LEVEL", "INFO").upper()
LOG_DIR: Path = _PROJECT_ROOT / _get_str("LOG_DIR", "logs")

# ---------------------------------------------------------------------------
# Ollama / LLM  (active from Day 2)
# ---------------------------------------------------------------------------

OLLAMA_MODEL: str = _get_str("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_BASE_URL: str = _get_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX: int = _get_int("OLLAMA_NUM_CTX", 8192)
OLLAMA_TIMEOUT: float = _get_float("OLLAMA_TIMEOUT", 30.0)
OLLAMA_TEMPERATURE: float = _get_float("OLLAMA_TEMPERATURE", 0.85)

# Maximum tokens the model may generate per reply.  Keeps responses concise.
MAX_RESPONSE_TOKENS: int = _get_int("MAX_RESPONSE_TOKENS", 180)

# Игровой режим (aura.bat → G): только чат — без войса, раздумий и сообщений по своей инициативе, читает реже;
# модель частично на CPU, чтобы освободить видеопамять под игру.  Включён, пока лежит data/game_mode.flag.
# Замеры 12B на RTX 3060 + 7800X3D: 8 слоёв — 2.2 ГБ видеопамяти и ~7 ток/с; 14 — 3.0 ГБ, ~8 ток/с
# (всё на видеокарте — 8.2 ГБ, 30 ток/с).  Совсем без видеокарты (0) промпт считается 1.5–2 минуты.
GAME_MODE_FLAG: Path = _PROJECT_ROOT / "data" / "game_mode.flag"
GAME_MODE_GPU_LAYERS: int = _get_int("GAME_MODE_GPU_LAYERS", 8)
GAME_MODE_THREADS: int = _get_int("GAME_MODE_THREADS", 4)  # ядер CPU под модель — остальные игре

# Пульт (tools/panel.py, http://127.0.0.1:8770) говорит с ботом через этот локальный порт (0 — выключить).
# Ключ доступа бот сам кладёт в data/control.token при каждом запуске.
CONTROL_PORT: int = _get_int("CONTROL_PORT", 8771)
CONTROL_TOKEN_FILE: Path = _PROJECT_ROOT / "data" / "control.token"
BOT_PID_FILE: Path = _PROJECT_ROOT / "data" / "bot.pid"

# ---------------------------------------------------------------------------
# Memory / vector store  (active from Day 3)
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / _get_str("CHROMA_PERSIST_DIR", "data/chroma")
SQLITE_DB_PATH: Path = _PROJECT_ROOT / _get_str("SQLITE_DB_PATH", "data/events.db")
# Многоязычная модель: старая all-MiniLM-L6-v2 плохо понимает русский.
# При смене модели воспоминания автоматически переносятся (memory/store.py).
EMBEDDING_MODEL: str = _get_str("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
MEMORY_TOP_K: int = _get_int("MEMORY_TOP_K", 4)
# Насколько похожим должно быть воспоминание (косинусное расстояние, меньше = строже).
MEMORY_MAX_DISTANCE: float = _get_float("MEMORY_MAX_DISTANCE", 0.55)
# Не вспоминать то, что было последние N минут — это и так видно в истории чата.
MEMORY_SKIP_RECENT_MINUTES: int = _get_int("MEMORY_SKIP_RECENT_MINUTES", 30)
RELATIONSHIP_DECAY_DAYS: int = _get_int("RELATIONSHIP_DECAY_DAYS", 14)

# ---------------------------------------------------------------------------
# Special users — per-ID relationship overrides
# ---------------------------------------------------------------------------
# Format in .env:
#   SPECIAL_USER_IDS=123456789,987654321
#   SPECIAL_USER_NOTES=MirFeeD:особый человек,Kira:старый друг
#
# Each ID in SPECIAL_USER_IDS gets the "special" treatment in the pipeline.
# SPECIAL_USER_NOTES is an optional per-name note injected into the prompt.
SPECIAL_USER_IDS: List[int] = _get_id_list("SPECIAL_USER_IDS")


def _get_special_user_notes() -> dict:
    """Parse SPECIAL_USER_NOTES=Name1:note1,Name2:note2 into a dict."""
    raw = os.getenv("SPECIAL_USER_NOTES", "").strip()
    result: dict = {}
    if not raw:
        return result
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            name, note = part.split(":", 1)
            result[name.strip().lower()] = note.strip()
    return result


SPECIAL_USER_NOTES: dict = _get_special_user_notes()

# ---------------------------------------------------------------------------
# Role-based relationship — hidden gratitude / warmth
# ---------------------------------------------------------------------------
# Proactive DM / personal conversations
# ---------------------------------------------------------------------------

# Comma-separated list of Discord user IDs the bot may initiate DMs with.
PROACTIVE_USER_IDS: List[int] = _get_id_list("PROACTIVE_USER_IDS")

# Probability (0.0–1.0) that the bot initiates a DM on each background tick.
# Default: 0.05 → roughly 3 times per hour at the 60-second tick rate.
PROACTIVE_DM_PROBABILITY: float = _get_float("PROACTIVE_DM_PROBABILITY", 0.05)

# Minimum seconds between two bot-initiated DMs to the same user.
# Default: 3600 (1 hour).
PROACTIVE_MIN_INTERVAL: int = _get_int("PROACTIVE_MIN_INTERVAL", 3600)

# --- Сама пишет тем, кому полностью доверяет (доверие 10/10) ---
# Включить: делится мыслями/новостями/мнением в личке по своей инициативе.
PROACTIVE_TRUSTED: bool = os.getenv("PROACTIVE_TRUSTED", "true").strip().lower() in ("1", "true", "yes", "on")
# Порог доверия по шкале -10..10 (как в people_ui / people_set).
PROACTIVE_TRUST_MIN: float = _get_float("PROACTIVE_TRUST_MIN", 10)
# Шанс на каждом тике (раз в минуту) для одного человека, когда все условия выполнены.
# 0.01 → в среднем пишет примерно через 1.5 часа после того, как стало «можно».
PROACTIVE_TRUSTED_PROBABILITY: float = _get_float("PROACTIVE_TRUSTED_PROBABILITY", 0.01)
# Не чаще, чем раз в столько секунд одному человеку (считается от её последнего сообщения в личке).
PROACTIVE_TRUSTED_MIN_INTERVAL: int = _get_int("PROACTIVE_TRUSTED_MIN_INTERVAL", 21600)
# Если её прошлое сообщение осталось без ответа — не пишет снова столько часов.
PROACTIVE_TRUSTED_UNANSWERED_H: int = _get_int("PROACTIVE_TRUSTED_UNANSWERED_H", 48)
# Тихие часы (по BOT_TIMEZONE): с какого по какой час сама не пишет. Пусто — без тихих часов.
PROACTIVE_QUIET_HOURS: str = _get_str("PROACTIVE_QUIET_HOURS", "23-10")
# Может пожаловаться близкому на того, кто достаёт её в личке: отношение к нему не выше этого (-10..10).
PROACTIVE_COMPLAIN_AFFINITY: float = _get_float("PROACTIVE_COMPLAIN_AFFINITY", -3)

# --- Совет у близкого: сомневается, отправлять ли сообщение ---
# Перед тем как самой написать кому-то (не близкому), прикидывает, стоит ли. Если сомневается —
# спрашивает совета в личке у человека с доверием 10 и отправляет только после его «да».
PROACTIVE_ASK_ADVICE: bool = os.getenv("PROACTIVE_ASK_ADVICE", "true").strip().lower() in ("1", "true", "yes", "on")
# Сколько часов ждёт ответа на вопрос; не ответили — не отправляет.
PROACTIVE_ADVICE_TIMEOUT_H: float = _get_float("PROACTIVE_ADVICE_TIMEOUT_H", 6)
# Иногда сама пишет и другим знакомым, кто ей симпатичен и уже писал ей в личку (с проверкой выше).
PROACTIVE_FRIENDS: bool = os.getenv("PROACTIVE_FRIENDS", "true").strip().lower() in ("1", "true", "yes", "on")
# Порог симпатии для этого (-10..10).
PROACTIVE_FRIEND_AFFINITY: float = _get_float("PROACTIVE_FRIEND_AFFINITY", 5)
# Шанс на тик (раз в минуту) на человека и минимальный интервал (сек) между её сообщениями ему.
PROACTIVE_FRIEND_PROBABILITY: float = _get_float("PROACTIVE_FRIEND_PROBABILITY", 0.003)
PROACTIVE_FRIEND_MIN_INTERVAL: int = _get_int("PROACTIVE_FRIEND_MIN_INTERVAL", 86400)

# --- Вопрос в общем чате: не уверена — советуется или жалуется близкому в личке ---
# Когда к ней в чате обращаются с вопросом, она сама решает: ответить, спросить совета
# у человека с доверием 10 (стоит ли отвечать / как ответить) или ответить и пожаловаться ему.
CHAT_ASK_ADVICE: bool = os.getenv("CHAT_ASK_ADVICE", "true").strip().lower() in ("1", "true", "yes", "on")
# Не чаще одного вопроса/жалобы в личку за столько секунд.
CHAT_ADVICE_COOLDOWN_S: int = _get_int("CHAT_ADVICE_COOLDOWN_S", 900)
# Сколько минут ждёт совета; не дождалась — решает сама и отвечает.
CHAT_ADVICE_TIMEOUT_MIN: float = _get_float("CHAT_ADVICE_TIMEOUT_MIN", 10)

# --- Пишет другим по просьбе тех, кому доверяет («Аура, напиши Васе, что …») ---
RELAY_ENABLED: bool = os.getenv("RELAY_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
# С какого доверия (-10..10) выполняет такие просьбы.
RELAY_TRUST_MIN: float = _get_float("RELAY_TRUST_MIN", 6)
# Не больше стольких сообщений по просьбам одного человека в час.
RELAY_PER_HOUR: int = _get_int("RELAY_PER_HOUR", 5)
# После просьбы переслать столько секунд «ты не отправила», «жду» и т.п. тоже разбираются как просьба.
RELAY_FOLLOWUP_S: int = _get_int("RELAY_FOLLOWUP_S", 1800)

# --- Личка, пришедшая, пока её не было в сети ---
# При запуске прочитать новые сообщения в личке и ответить на них.
DM_CATCHUP: bool = os.getenv("DM_CATCHUP", "true").strip().lower() in ("1", "true", "yes", "on")
# Насколько далеко назад смотреть (часы).
DM_CATCHUP_LOOKBACK_H: int = _get_int("DM_CATCHUP_LOOKBACK_H", 72)
# Отвечать, только если последнее сообщение не старше стольких часов (старше — просто запомнит).
DM_CATCHUP_REPLY_MAX_AGE_H: int = _get_int("DM_CATCHUP_REPLY_MAX_AGE_H", 24)
# Пауза после запуска перед чтением лички (секунды) — чтобы модель успела проснуться.
DM_CATCHUP_DELAY_S: int = _get_int("DM_CATCHUP_DELAY_S", 20)

# ---------------------------------------------------------------------------
# Voice — Day 4 (STT) + Day 5 (TTS)
# ---------------------------------------------------------------------------

def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "да")


def _get_str_list(key: str, default: str) -> List[str]:
    raw = os.getenv(key, default)
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


VOICE_ENABLED: bool = _get_bool("VOICE_ENABLED", True)
# Загружать Whisper и Silero сразу при старте (иначе — при первом входе в войс).
VOICE_PRELOAD: bool = _get_bool("VOICE_PRELOAD", True)
# Выходить из канала, когда в нём не осталось людей.
VOICE_AUTO_LEAVE: bool = _get_bool("VOICE_AUTO_LEAVE", True)
# Отвечать на любую фразу, если в канале с ботом только один человек.
VOICE_REPLY_WHEN_ALONE: bool = _get_bool("VOICE_REPLY_WHEN_ALONE", True)
# Писать распознанный текст в logs/bot.log (удобно при настройке).
VOICE_LOG_TRANSCRIPTS: bool = _get_bool("VOICE_LOG_TRANSCRIPTS", True)
# Как Whisper может записать имя бота — по любому из них бот поймёт, что зовут её.
VOICE_NAME_ALIASES: List[str] = _get_str_list(
    "VOICE_NAME_ALIASES", "аура,ауры,ауре,ауру,аурой,аур,aura"
)

MODELS_DIR: Path = _PROJECT_ROOT / _get_str("MODELS_DIR", "data/models")

# --- Day 4: VAD + Whisper ---
VAD_THRESHOLD: float = _get_float("VAD_THRESHOLD", 0.5)
VAD_SILENCE_MS: int = _get_int("VAD_SILENCE_MS", 600)     # пауза, после которой фраза закончена
VAD_MIN_SPEECH_MS: int = _get_int("VAD_MIN_SPEECH_MS", 400)
VAD_MAX_SPEECH_S: float = _get_float("VAD_MAX_SPEECH_S", 15.0)

# Кто расшифровывает речь в войсе:
#   gemma   — та же Gemma, что отвечает (слышит звук сама): точнее на русском, с именами и сленгом, Whisper
#             не грузится (минус ~1 ГБ видеопамяти).  ~1–1.5 с на фразу, и запрос расшифровки выбивает из
#             кэша Ollama промпт персонажа — вживую ответ выходит на ~1.5 с позже.  Модель войса должна уметь
#             audio (gemma4);
#   whisper — отдельная модель (настройки ниже), ~0.3–0.6 с на фразу.
VOICE_HEARING: str = _get_str("VOICE_HEARING", "gemma").lower()

# large-v3-turbo — заметно лучше small на русском, на RTX ~0.3–0.6 с на фразу.
WHISPER_MODEL: str = _get_str("WHISPER_MODEL", "large-v3-turbo")  # small/medium/large-v3/large-v3-turbo
WHISPER_DEVICE: str = _get_str("WHISPER_DEVICE", "auto")  # auto / cuda / cpu
WHISPER_COMPUTE_TYPE: str = _get_str("WHISPER_COMPUTE_TYPE", "auto")
WHISPER_LANGUAGE: str = _get_str("WHISPER_LANGUAGE", "ru")  # пусто = автоопределение
WHISPER_BEAM_SIZE: int = _get_int("WHISPER_BEAM_SIZE", 5)
WHISPER_INITIAL_PROMPT: str = _get_str(
    "WHISPER_INITIAL_PROMPT", "Разговор друзей в голосовом чате Discord, обращаются к Ауре."
)
# Слова, которые Whisper должен знать (игры, сленг сервера), через запятую.
WHISPER_HOTWORDS: List[str] = [
    w.strip() for w in _get_str(
        "WHISPER_HOTWORDS", "Valorant, Genshin, Honkai, Star Rail, катка, агент, скин"
    ).split(",") if w.strip()
]

# --- Day 5: Silero TTS ---
TTS_MODEL: str = _get_str("TTS_MODEL", "v5_5_ru")
TTS_MODEL_URL: str = _get_str("TTS_MODEL_URL", "")        # пусто = официальный адрес Silero
TTS_SPEAKER: str = _get_str("TTS_SPEAKER", "xenia")       # aidar, baya, kseniya, xenia, eugene
TTS_DEVICE: str = _get_str("TTS_DEVICE", "auto")
TTS_CPU_THREADS: int = _get_int("TTS_CPU_THREADS", 4)
TTS_VOLUME: float = _get_float("TTS_VOLUME", 1.0)

# --- Озвучка: silero (по умолчанию) или f5 — F5-TTS (русская модель ESpeech, сервер f5/f5_server.py),
# при сбое F5 фраза идёт через Silero. ---
TTS_ENGINE: str = _get_str("TTS_ENGINE", "silero").lower()
F5_API_URL: str = _get_str("F5_API_URL", "http://127.0.0.1:8081").rstrip("/")
# Голос — папка voices/<id> на сервере F5 (FISH_REFERENCE_ID — старое имя, тоже понимается).
F5_REFERENCE_ID: str = _get_str("F5_REFERENCE_ID", "") or _get_str("FISH_REFERENCE_ID", "aura")
F5_TIMEOUT: float = _get_float("F5_TIMEOUT", 20.0)          # секунд на одну фразу
F5_SPEED: float = _get_float("F5_SPEED", 1.0)
# После сбоя F5 столько секунд сразу говорим через Silero, не дожидаясь таймаутов.
F5_RETRY_AFTER_S: int = _get_int("F5_RETRY_AFTER_S", 30)
# Сервер F5 поднимается сам, когда она заходит в войс (~20 с: WSL просыпается, модель грузится), и гасится
# через F5_IDLE_STOP_MIN минут после выхода — вне войса он зря держит ~5 ГБ оперативки (WSL) и ~1 ГБ
# видеопамяти.  F5_AUTO=0 — как раньше, запускать вручную (aura.bat → 9).
F5_AUTO: bool = _get_bool("F5_AUTO", True)
F5_IDLE_STOP_MIN: float = _get_float("F5_IDLE_STOP_MIN", 5.0)
F5_SCRIPT: Path = _PROJECT_ROOT / "f5" / "run_f5.sh"


# ---------------------------------------------------------------------------
# Character, social memory, liveliness
# ---------------------------------------------------------------------------

# Подробный характер (можно редактировать): persona.md рядом с main.py.
PERSONA_FILE: Path = _PROJECT_ROOT / _get_str("PERSONA_FILE", "persona.md")


def load_persona() -> str:
    """Текст persona.md (перечитывается при каждом ответе — правки без перезапуска)."""
    try:
        return PERSONA_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# Владелец бота: к нему отношение никогда не падает ниже OWNER_MIN_AFFINITY.
OWNER_USER_IDS: List[int] = _get_id_list("OWNER_USER_IDS")
OWNER_MIN_AFFINITY: float = _get_float("OWNER_MIN_AFFINITY", 0.35)
# Как быстро меняется отношение (1.0 — обычно, 0.5 — вдвое медленнее).
AFFINITY_SPEED: float = _get_float("AFFINITY_SPEED", 1.0)

# Сколько последних сообщений канала видит модель.
CHAT_HISTORY_LIMIT: int = _get_int("CHAT_HISTORY_LIMIT", 20)
# Чат: разбивать ответ из нескольких фраз на отдельные сообщения (как пишут люди).
# Рейдж-байт (оскорбления, «твою мать…», политота): шанс просто проигнорить первую провокацию.
# Дальше от того же человека игнорит всё чаще.
BAIT_IGNORE_CHANCE: float = _get_float("BAIT_IGNORE_CHANCE", 0.3)
# На сколько сразу падает отношение к человеку за каждую провокацию (шкала от -1 до 1).
# Шанс проигнорить неприятного человека (отношение -2.5 и ниже), даже если он обращается к ней.
# Чем хуже отношение, тем чаще игнор.
DISLIKED_IGNORE_CHANCE: float = _get_float("DISLIKED_IGNORE_CHANCE", 0.25)
BAIT_AFFINITY_PENALTY: float = _get_float("BAIT_AFFINITY_PENALTY", 0.15)
CHAT_SPLIT_MESSAGES: bool = _get_bool("CHAT_SPLIT_MESSAGES", True)
CHAT_SPLIT_CHANCE: float = _get_float("CHAT_SPLIT_CHANCE", 0.6)
# Чат: убирать точку в конце сообщения («ок.» → «ок»).  Многоточие, «!» и «?» остаются.
CHAT_DROP_FINAL_PERIOD: bool = _get_bool("CHAT_DROP_FINAL_PERIOD", True)
# Чат: скорость «печати» (символов в секунду) — от неё зависит, сколько висит «печатает…».
# Человек в чате печатает ~4-7 симв/с.  Время, пока модель думала, засчитывается в первое сообщение.
CHAT_TYPING_CPS: float = _get_float("CHAT_TYPING_CPS", 6.0)
# Максимум «печатания» одного сообщения, секунд.
CHAT_TYPING_MAX: float = _get_float("CHAT_TYPING_MAX", 18.0)
# Сколько людей описывать в промпте и сколько фактов о каждом.
SOCIAL_MAX_PEOPLE: int = _get_int("SOCIAL_MAX_PEOPLE", 6)
SOCIAL_FACTS_PER_PERSON: int = _get_int("SOCIAL_FACTS_PER_PERSON", 8)

# Рефлексия (обновление памяти и отношений) — когда накопилось N реплик и стало тихо.
REFLECT_MIN_LINES: int = _get_int("REFLECT_MIN_LINES", 10)
REFLECT_MAX_LINES: int = _get_int("REFLECT_MAX_LINES", 40)
REFLECT_QUIET_SECONDS: int = _get_int("REFLECT_QUIET_SECONDS", 45)

# Ollama: /api/chat (лучше держит роль) или старый /api/generate.
OLLAMA_API: str = _get_str("OLLAMA_API", "chat").lower()
OLLAMA_TOP_P: float = _get_float("OLLAMA_TOP_P", 0.95)
OLLAMA_REPEAT_PENALTY: float = _get_float("OLLAMA_REPEAT_PENALTY", 1.1)
# Держать модель в видеопамяти между запросами (иначе первая реплика после паузы — долгая).
OLLAMA_KEEP_ALIVE: str = _get_str("OLLAMA_KEEP_ALIVE", "60m")

# Иногда вместо ответа она просто ставит реакцию.
REACTION_PROBABILITY: float = _get_float("REACTION_PROBABILITY", 0.12)

# Часовой пояс — она знает, который час (ночью сонная, утром ворчит).
BOT_TIMEZONE: str = _get_str("BOT_TIMEZONE", "Europe/Moscow")

# Короткое «хм» / «ну...» сразу, пока она думает над ответом.
VOICE_FILLERS: bool = _get_bool("VOICE_FILLERS", False)
VOICE_FILLER_PROBABILITY: float = _get_float("VOICE_FILLER_PROBABILITY", 0.25)
# Не «обдумывать» разговоры, пока в войсе идёт живой разговор (секунд с последней фразы).
VOICE_REFLECT_PAUSE_S: int = _get_int("VOICE_REFLECT_PAUSE_S", 90)

# Голос: она слушает разговор и сама вмешивается, а не только на «Аура».
VOICE_PROACTIVE: bool = _get_bool("VOICE_PROACTIVE", True)
VOICE_PROACTIVE_THRESHOLD: float = _get_float("VOICE_PROACTIVE_THRESHOLD", 0.45)
# Не чаще одного «встревания» без приглашения раз в N секунд.
VOICE_PROACTIVE_COOLDOWN_S: int = _get_int("VOICE_PROACTIVE_COOLDOWN_S", 25)
# Сколько секунд после её реплики разговор считается продолжением разговора с ней.
VOICE_FOLLOWUP_WINDOW_S: int = _get_int("VOICE_FOLLOWUP_WINDOW_S", 25)
# Прогрев модели при входе в войс и в простое — быстрый первый ответ.
VOICE_WARMUP: bool = _get_bool("VOICE_WARMUP", True)

# Голос — скорость.  Голосовой промпт короче текстового: меньше истории и памяти.
VOICE_HISTORY_LINES: int = _get_int("VOICE_HISTORY_LINES", 10)
VOICE_MEMORY_SNIPPETS: int = _get_int("VOICE_MEMORY_SNIPPETS", 2)
VOICE_SOCIAL_PEOPLE: int = _get_int("VOICE_SOCIAL_PEOPLE", 3)
VOICE_SOCIAL_FACTS: int = _get_int("VOICE_SOCIAL_FACTS", 4)  # фактов о каждом человеке в голосовом промпте
# Отдельная (маленькая и быстрая) модель для войса, например gemma3:4b.  Пусто = та же, что и в чате.
VOICE_OLLAMA_MODEL: str = _get_str("VOICE_OLLAMA_MODEL", "")

# Озвучка: эмоции (темп/высота по настроению и знакам «!», «?», «...»).
TTS_EMOTIONAL: bool = _get_bool("TTS_EMOTIONAL", True)
# Живая речь в войсе: «ну», «хм», «ой», смешки, вздохи, паузы «...».  Silero произносит такое
# как заминку, поэтому по умолчанию включено только для F5.
VOICE_NATURAL_SPEECH: bool = _get_bool("VOICE_NATURAL_SPEECH", TTS_ENGINE == "f5")
# Первую длинную фразу отдавать в озвучку уже на запятой (быстрее старт). Для F5 даёт рваную паузу.
VOICE_SPLIT_ON_COMMA: bool = _get_bool("VOICE_SPLIT_ON_COMMA", TTS_ENGINE != "f5")
# Словарь произношений: «слово = как читать», + перед ударной гласной.
PRONUNCIATIONS_FILE: Path = _PROJECT_ROOT / _get_str("PRONUNCIATIONS_FILE", "pronunciations.txt")

# Картинки: Аура видит изображения из сообщений (модель должна уметь vision, например gemma3/gemma4).
VISION_ENABLED: bool = _get_bool("VISION_ENABLED", True)
# Отдельная модель для сообщений с картинками.  Пусто = та же, что OLLAMA_MODEL.
VISION_MODEL: str = _get_str("VISION_MODEL", "")
VISION_MAX_IMAGES: int = _get_int("VISION_MAX_IMAGES", 2)
# Картинка уменьшается до этого размера по длинной стороне (меньше = быстрее и меньше контекста).
VISION_MAX_SIDE: int = _get_int("VISION_MAX_SIDE", 1024)
# Файлы больше этого размера (МБ) не скачиваются.
VISION_MAX_MB: float = _get_float("VISION_MAX_MB", 15.0)
# Сколько секунд после картинки она ещё «видит» её в этом канале (для вопросов вроде «а что на ней?»).
VISION_FOLLOWUP_S: int = _get_int("VISION_FOLLOWUP_S", 180)
# Сколько кадров брать из гифки (равномерно по всей длине).  1 = только первый кадр.
VISION_GIF_FRAMES: int = _get_int("VISION_GIF_FRAMES", 6)

# Память на картинки: уменьшенные копии + подписи в data/images, чтобы возвращаться к ним через часы.
VISION_MEMORY_MB: float = _get_float("VISION_MEMORY_MB", 100.0)   # 0 = не хранить
VISION_MEMORY_DIR: Path = _PROJECT_ROOT / _get_str("VISION_MEMORY_DIR", "data/images")
# За сколько часов она вспоминает картинки из канала.
VISION_RECALL_HOURS: float = _get_float("VISION_RECALL_HOURS", 12.0)
# Когда спрашивают про картинку прямо («что было на скрине?») — искать её за столько часов
# (в этом канале и среди картинок, которые этот человек присылал где угодно, в том числе в личку).
VISION_ASK_RECALL_HOURS: float = _get_float("VISION_ASK_RECALL_HOURS", 168.0)
# Смотреть и запоминать картинки, даже когда она не отвечает на сообщение (подпись делается в фоне).
VISION_OBSERVE: bool = _get_bool("VISION_OBSERVE", True)

# Любимые картинки: то, что ей правда понравилось, она сохраняет себе в отдельную папку
# (в нормальном качестве, гифки анимированными).  Обычная чистка памяти на картинки их не трогает.
FAVORITES_ENABLED: bool = _get_bool("FAVORITES_ENABLED", True)
FAVORITES_DIR: Path = _PROJECT_ROOT / _get_str("FAVORITES_DIR", "data/favorites")
# Лимит папки любимых (МБ).  Сверх лимита уходят те, что понравились меньше всего.  0 = без лимита.
FAVORITES_MAX_MB: float = _get_float("FAVORITES_MAX_MB", 500.0)
# Оценка 0–10, с которой картинка попадает в любимые (модель ставит её, когда подписывает картинку).
# Меньше = сохраняет чаще.  «Аура, сохрани» — сохраняет всегда.
FAVORITES_MIN_SCORE: int = _get_int("FAVORITES_MIN_SCORE", 9)
# Её вкус — по нему модель решает, понравилась ли картинка.
FAVORITES_TASTE: str = _get_str(
    "FAVORITES_TASTE",
    "камни, минералы и кристаллы, кошки (особенно наглые), красивая природа и горы, фэнтези и магическая "
    "эстетика, таро, Genshin и Honkai: Star Rail, уютные вещи и по-настоящему смешные мемы",
)
# Она сама кидает картинки/гифки из своей коллекции любимых (только оттуда, без интернета).
MEDIA_SEND_ENABLED: bool = _get_bool("MEDIA_SEND_ENABLED", True)
# Не чаще одной картинки в канал за столько секунд (если не попросили прямо).
MEDIA_COOLDOWN_S: int = _get_int("MEDIA_COOLDOWN_S", 300)
# Сколько любимых показывать ей в промпте списком (лучшие первыми).
MEDIA_PROMPT_ITEMS: int = _get_int("MEDIA_PROMPT_ITEMS", 20)

# Реакции эмодзи сервера: доля реакций, для которых она выбирает кастомный эмодзи сервера по смыслу
# (остальные — обычные по настроению).  0 = только обычные.
SERVER_EMOJI_SHARE: float = _get_float("SERVER_EMOJI_SHARE", 0.6)
# Иногда ставит реакцию и на сообщение, на которое отвечает.
REACTION_WITH_REPLY_PROBABILITY: float = _get_float("REACTION_WITH_REPLY_PROBABILITY", 0.08)

# Напоминания: «Аура, напомни завтра в 6 про экзамен».
REMINDERS_ENABLED: bool = _get_bool("REMINDERS_ENABLED", True)

# Распорядок дня: она сова — ложится под утро, встаёт ближе к обеду.  Пока спит — не отвечает,
# проснувшись — разбирает личку.  Время каждый день немного своё (± SCHEDULE_JITTER_MIN минут).
SCHEDULE_ENABLED: bool = _get_bool("SCHEDULE_ENABLED", True)
SLEEP_TIME: str = _get_str("SLEEP_TIME", "04:00")
WAKE_TIME: str = _get_str("WAKE_TIME", "11:00")
SCHEDULE_JITTER_MIN: int = _get_int("SCHEDULE_JITTER_MIN", 40)
# Сколько часов после подъёма она сонная и ворчливая.
MORNING_HOURS: float = _get_float("MORNING_HOURS", 2.0)
# Перед сном пишет «всё, я спать» там, где разговаривала за последние столько минут.
GOODNIGHT_ACTIVE_MIN: int = _get_int("GOODNIGHT_ACTIVE_MIN", 20)

# Поиск в интернете (DuckDuckGo, без ключа): «загугли …» или когда для ответа нужны свежие факты.
WEB_SEARCH_ENABLED: bool = _get_bool("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_RESULTS: int = _get_int("WEB_SEARCH_RESULTS", 5)

# Её собственная жизнь: время от времени с ней «что-то случается» (последовательно, с памятью о прошлом).
LIFE_EVENTS_ENABLED: bool = _get_bool("LIFE_EVENTS_ENABLED", True)
# В среднем одно новое событие раз в столько часов (пока она не спит).
LIFE_EVENT_INTERVAL_H: float = _get_float("LIFE_EVENT_INTERVAL_H", 6.0)

# Чтение: её занятие.  Книги (.txt .fb2 .epub .pdf) и комиксы (.cbz или папка с картинками) — в LIBRARY_DIR.
READING_ENABLED: bool = _get_bool("READING_ENABLED", True)
LIBRARY_DIR: Path = _PROJECT_ROOT / _get_str("LIBRARY_DIR", "data/library")
# Примерно раз в столько минут (± треть) садится почитать, если не спит и не занята разговором.
READING_INTERVAL_MIN: float = _get_float("READING_INTERVAL_MIN", 45.0)
# Сколько читает за раз: символов текста / страниц комикса.
READING_CHUNK_CHARS: int = _get_int("READING_CHUNK_CHARS", 3500)
COMIC_PAGES_PER_SESSION: int = _get_int("COMIC_PAGES_PER_SESSION", 4)
# Запой: книга затянула (её оценка «не оторваться» ≥ BINGE_HOOK из 10) — читает подряд, примерно раз
# в BINGE_INTERVAL_MIN минут, до BINGE_MAX_H часов; если пора спать — засиживается до BINGE_STAY_UP_H часов.
BINGE_HOOK: int = _get_int("BINGE_HOOK", 8)
BINGE_INTERVAL_MIN: float = _get_float("BINGE_INTERVAL_MIN", 8.0)
BINGE_MAX_H: float = _get_float("BINGE_MAX_H", 3.0)
BINGE_STAY_UP_H: float = _get_float("BINGE_STAY_UP_H", 2.0)
# В запое отвечает не сразу: через столько секунд («от-до»), как человек, оторвавшийся от книги.
_bd = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", _get_str("BINGE_REPLY_DELAY_S", "60-240"))] or [60.0, 240.0]
BINGE_REPLY_DELAY_S: tuple = (min(_bd), max(_bd))
# Делиться впечатлениями о чтении в личку (тому, кто подарил книгу, иначе владельцу) — не чаще раза в столько часов.
READING_SHARE_ENABLED: bool = _get_bool("READING_SHARE_ENABLED", True)
READING_SHARE_MIN_H: float = _get_float("READING_SHARE_MIN_H", 2.0)
# Подаренные в личку файлы больше этого (МБ) не берёт.
BOOK_MAX_MB: float = _get_float("BOOK_MAX_MB", 50.0)

# Скучает: человек, к которому она хорошо относится, пропал на столько дней — она заметит, когда он вернётся.
MISS_AFTER_DAYS: float = _get_float("MISS_AFTER_DAYS", 4.0)
# Её сообщение в личке висит без ответа столько часов — слегка, по-доброму подколет за это.
IGNORED_AFTER_HOURS: float = _get_float("IGNORED_AFTER_HOURS", 12.0)
# Как часто она сама чем-то интересуется — после ответа задаёт вопрос (про сказанное, про человека,
# делится своим и спрашивает в ответ).  Не после прощаний и не два вопроса подряд.
CURIOSITY_DM: float = _get_float("CURIOSITY_DM", 0.45)      # в личке
CURIOSITY_CHAT: float = _get_float("CURIOSITY_CHAT", 0.3)   # в общем чате и в войсе

# Человек пишет очередью коротких сообщений — она ждёт, пока он допишет (пауза + «печатает…»),
# и отвечает одним сообщением на всё.  Дописал, пока она отвечала, — учтёт и это.
CHAT_BATCH_QUIET_S: float = _get_float("CHAT_BATCH_QUIET_S", 5.0)     # тишина после последнего сообщения
CHAT_BATCH_MAX_WAIT_S: float = _get_float("CHAT_BATCH_MAX_WAIT_S", 25.0)  # но не дольше этого сверх задержки
# Эмодзи и смайлики :) в её сообщениях (False — вычищаются).
CHAT_EMOJI: bool = _get_bool("CHAT_EMOJI", False)

# Дни рождения: узнаёт даты из разговоров и в свой день поздравляет в личке.
BIRTHDAYS_ENABLED: bool = _get_bool("BIRTHDAYS_ENABLED", True)
# Поздравляет тех, к кому относится не хуже этого (−10…10).
BIRTHDAY_MIN_AFFINITY: float = _get_float("BIRTHDAY_MIN_AFFINITY", 0)

