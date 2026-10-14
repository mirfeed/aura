"""
memory/social.py — «социальная» память Aura.

Что хранится (в той же SQLite-базе data/events.db):

* ``people``      — каждый человек: имя, другие имена, отношение (симпатия
                    и доверие от -1 до 1), её мнение о нём своими словами.
* ``facts``       — факты о людях: что человек сказал о себе (said_by = он
                    сам) и что о нём говорили другие (said_by = другой).
* ``bot_statements`` — её собственные слова: обещания, планы, мнения,
                    утверждения о себе, чтобы она не противоречила себе.
* ``summaries``   — краткое содержание разговора в каждом канале.
* ``transcript``  — лента реплик (люди + сама Aura) для «рефлексии».

Рефлексия: когда в канале накопилось достаточно новых реплик и разговор
притих, один запрос к модели (JSON) обновляет сводку, вытаскивает факты
и обещания, меняет отношение к людям и настроение.  Запускается из
фонового цикла бота (см. core/bot.py) только когда модель свободна.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from utils.logger import get_logger

if TYPE_CHECKING:
    import aiosqlite
    from behavior.state import BotState

_logger = get_logger("bot.social")

EmbedFn = Callable[[List[str]], Awaitable[np.ndarray]]
JsonLLM = Callable[[str, str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]

# (key, display_name, raw_discord_id)
Person = Tuple[str, str, int]

_SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS people (
        key          TEXT PRIMARY KEY,
        raw_id       INTEGER DEFAULT 0,
        name         TEXT NOT NULL,
        aliases      TEXT DEFAULT '[]',
        affinity     REAL DEFAULT 0.0,
        trust        REAL DEFAULT 0.0,
        opinion      TEXT DEFAULT '',
        interactions INTEGER DEFAULT 0,
        first_seen   TEXT NOT NULL,
        last_seen    TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_key    TEXT,
        subject_name   TEXT NOT NULL,
        fact           TEXT NOT NULL,
        said_by        TEXT NOT NULL,
        is_gossip      INTEGER DEFAULT 0,
        created_at     TEXT NOT NULL,
        last_confirmed TEXT NOT NULL,
        times_seen     INTEGER DEFAULT 1,
        embedding      BLOB
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_key)",
    """
    CREATE TABLE IF NOT EXISTS bot_statements (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        to_key      TEXT,
        to_name     TEXT,
        kind        TEXT NOT NULL,
        text        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        embedding   BLOB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS summaries (
        channel_id  INTEGER PRIMARY KEY,
        summary     TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transcript (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id   INTEGER NOT NULL,
        guild_id     INTEGER,
        ts           TEXT NOT NULL,
        speaker_key  TEXT NOT NULL,
        speaker_name TEXT NOT NULL,
        raw_id       INTEGER DEFAULT 0,
        text         TEXT NOT NULL,
        is_bot       INTEGER DEFAULT 0,
        is_voice     INTEGER DEFAULT 0,
        reply_to     TEXT,
        processed    INTEGER DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_transcript_channel ON transcript(channel_id, processed)",
    """
    CREATE TABLE IF NOT EXISTS inside_jokes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_key  TEXT NOT NULL,
        person_name TEXT NOT NULL,
        joke        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        embedding   BLOB
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jokes_person ON inside_jokes(person_key)",
]

_MOODS = ["neutral", "curious", "tired", "amused", "withdrawn", "irritated", "cheerful"]

_REFLECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "about": {"type": "string"},
                    "fact": {"type": "string"},
                    "said_by": {"type": "string"},
                },
                "required": ["about", "fact", "said_by"],
            },
        },
        "my_statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "kind": {"type": "string", "enum": ["promise", "plan", "opinion", "claim"]},
                    "text": {"type": "string"},
                },
                "required": ["to", "kind", "text"],
            },
        },
        "attitudes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "change": {"type": "number"},
                    "trust_change": {"type": "number"},
                    "opinion": {"type": "string"},
                    "also_called": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "change", "opinion"],
            },
        },
        "jokes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "with": {"type": "string"},
                    "joke": {"type": "string"},
                },
                "required": ["with", "joke"],
            },
        },
        "birthdays": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["name", "date"],
            },
        },
        "mood": {
            "type": "object",
            "properties": {
                "mood": {"type": "string", "enum": _MOODS},
                "reason": {"type": "string"},
            },
            "required": ["mood", "reason"],
        },
    },
    "required": ["summary", "facts", "my_statements", "attitudes", "mood"],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_ts(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return datetime.now(tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ago(ts: str) -> str:
    """ISO-время → «5 мин назад», «вчера», «3 дн назад»."""
    secs = (datetime.now(tz=timezone.utc) - _parse_ts(ts)).total_seconds()
    if secs < 90:
        return "только что"
    if secs < 3600:
        return f"{int(secs // 60)} мин назад"
    if secs < 86400:
        return f"{int(secs // 3600)} ч назад"
    days = int(secs // 86400)
    if days == 1:
        return "вчера"
    if days < 30:
        return f"{days} дн назад"
    return f"{days // 30} мес назад"


def affinity_label(a: float) -> str:
    if a <= -0.6:
        return "терпеть не можешь"
    if a <= -0.25:
        return "неприятен тебе"
    if a < 0.15:
        return "нейтрально, пока не определилась"
    if a < 0.45:
        return "скорее симпатичен"
    if a < 0.75:
        return "тепло относишься"
    return "очень дорог тебе"


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "",
    "э": "e", "ю": "iu", "я": "ia",
}
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "!": "i", "@": "a", "$": "s"}
# Падежные окончания: «Тирозину» → «Тирозин», «Фраудером» → «Фраудер».
_CASE_ENDINGS = ("ами", "ями", "ому", "ему", "ой", "ей", "ом", "ем", "ой", "у", "ю", "е", "а", "я", "и", "ы")


def name_forms(word: str) -> List[str]:
    """Имя и оно же без падежного окончания: «Тирозину» → [«тирозину», «тирозин»]."""
    w = (word or "").strip().lower().lstrip("@")
    forms = [w]
    if re.search("[а-яё]", w):
        forms += [w[: -len(e)] for e in _CASE_ENDINGS if w.endswith(e) and len(w) - len(e) >= 3]
    return list(dict.fromkeys(forms))


def name_sound(text: str) -> str:
    """
    Как имя звучит, латиницей и упрощённо: «Тирозин» и «Tyrozin» → «tirozin», «ГR1B» и «Гриб» → «grib».
    Так ник, написанный русскими буквами, совпадает с латинским.
    """
    s = "".join(_TRANSLIT.get(ch, _LEET.get(ch, ch)) for ch in (text or "").lower())
    s = re.sub(r"[^a-z]", "", s)
    for a, b in (("ph", "f"), ("ck", "k"), ("kh", "h"), ("th", "t"), ("sch", "sh"), ("x", "ks"), ("q", "k"),
                 ("c", "k"), ("w", "v"), ("y", "i"), ("j", "i"), ("ee", "i"), ("oo", "u")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)  # двойные буквы — как одна


def _norm_ddmm(text: str) -> str:
    """«5.3», «05.03», «5 марта» → «05.03»; '' если не похоже на дату."""
    # «ма» (май/мая) проверяется после «мар», так что март не путается с маем.
    months = ["янв", "фев", "мар", "апр", "ма", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    t = text.strip().lower()
    m = re.match(r"(\d{1,2})[./-](\d{1,2})", t)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
    else:
        m = re.match(r"(\d{1,2})\s+([а-яё]+)", t)
        if not m:
            return ""
        d = int(m.group(1))
        mo = next((i + 1 for i, p in enumerate(months) if m.group(2).startswith(p)), 0)
    try:
        datetime(2024, mo, d)  # високосный — 29.02 тоже годится
    except ValueError:
        return ""
    return f"{d:02d}.{mo:02d}"


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def is_owner(raw_id: int) -> bool:
    return bool(raw_id) and raw_id in config.OWNER_USER_IDS


def _seed_from_note(name: str, raw_id: int) -> Tuple[float, str]:
    """Начальное отношение для нового человека."""
    note = config.SPECIAL_USER_NOTES.get(name.lower(), "")
    affinity = 0.0
    low = note.lower()
    if re.search(r"не любишь|неприят|ненавид|раздража|бесит", low):
        affinity = -0.5
    elif re.search(r"любишь|любим|ценишь|друг|дорог", low):
        affinity = 0.6
    elif raw_id and raw_id in config.SPECIAL_USER_IDS:
        affinity = 0.5
    if is_owner(raw_id):
        affinity = max(affinity, 0.7)
    return affinity, note


def _blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def _unblob(b: Optional[bytes]) -> Optional[np.ndarray]:
    return None if not b else np.frombuffer(b, dtype=np.float32)


@dataclass
class SocialContext:
    """То, что попадает в промпт."""

    people_block: str = ""
    statements_block: str = ""
    summary: str = ""
    speaker_affinity: float = 0.0
    focus_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SocialMemory
# ---------------------------------------------------------------------------

class SocialMemory:
    """Люди, отношения, факты, обещания и сводки разговоров."""

    def __init__(self, db_getter: Callable[[], "aiosqlite.Connection"], embed: EmbedFn) -> None:
        self._db_getter = db_getter
        self._embed = embed
        self._names: Dict[str, str] = {}  # lower(name/alias) → key
        self._sounds: Dict[str, set] = {}  # звучание имени или слова из ника → {key}
        self._last_activity: Dict[int, float] = {}  # channel_id → monotonic

    @property
    def _db(self) -> "aiosqlite.Connection":
        db = self._db_getter()
        assert db is not None, "MemoryStore.init_db() was not awaited"
        return db

    async def init(self) -> None:
        for sql in _SCHEMA_SQL:
            await self._db.execute(sql)
        # Имя, поправленное вручную в people_ui, не перезаписывается ником из Discord.
        for col in ("name_locked INTEGER DEFAULT 0", "real_name TEXT DEFAULT ''",
                    "birthday TEXT DEFAULT ''", "bday_greeted INTEGER DEFAULT 0"):
            try:
                await self._db.execute(f"ALTER TABLE people ADD COLUMN {col}")
            except Exception:  # noqa: BLE001  (колонка уже есть)
                pass
        await self._db.commit()
        await self._reload_names()
        _logger.info("Social memory ready", extra={"data": {"people": len(set(self._names.values()))}})

    async def _reload_names(self) -> None:
        self._names.clear()
        self._sounds.clear()
        async with self._db.execute("SELECT key, name, aliases, COALESCE(real_name, '') AS real_name FROM people") as cur:
            for row in await cur.fetchall():
                self._register_names(row["key"], row["name"], row["aliases"])
                if row["real_name"]:
                    self._register_names(row["key"], row["real_name"], "[]")

    def _register_names(self, key: str, name: str, aliases_json: str) -> None:
        names = [name]
        try:
            names += list(json.loads(aliases_json or "[]"))
        except ValueError:
            pass
        for n in names:
            n = (n or "").strip().lower()
            if len(n) >= 3:
                self._names[n] = key
            # Звучание всего ника и отдельных слов в нём («юлька щпулька» → «iulka», «shpulka»).
            for i, part in enumerate([n] + re.split(r"[\s_.,()\[\]{}<>/\\|'\"~-]+", n)):
                snd = name_sound(part)
                # Слово внутри ника — от 4 букв, иначе «про» из «женя про fernando» поймает любое «про».
                if len(snd) >= (3 if i == 0 else 4):
                    self._sounds.setdefault(snd, set()).add(key)

    def key_for_name(self, name: str) -> Optional[str]:
        return self._names.get((name or "").strip().lower().lstrip("@"))

    # ------------------------------------------------------------------
    # People
    # ------------------------------------------------------------------

    async def touch_person(self, key: str, name: str, raw_id: int = 0) -> None:
        """Создать человека при первой встрече / обновить имя и счётчик."""
        if not key or key == "bot":
            return
        now = _now()
        async with self._db.execute("SELECT key, aliases FROM people WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        if row is None:
            affinity, note = _seed_from_note(name, raw_id)
            await self._db.execute(
                "INSERT INTO people (key, raw_id, name, aliases, affinity, trust, opinion, interactions,"
                " first_seen, last_seen, updated_at) VALUES (?,?,?,?,?,?,?,1,?,?,?)",
                (key, raw_id, name, "[]", affinity, max(0.0, affinity / 2), note, now, now, now),
            )
            self._register_names(key, name, "[]")
        else:
            await self._db.execute(
                "UPDATE people SET name = CASE WHEN COALESCE(name_locked, 0) = 1 THEN name ELSE ? END,"
                " raw_id = CASE WHEN ? != 0 THEN ? ELSE raw_id END,"
                " interactions = interactions + 1, last_seen = ? WHERE key = ?",
                (name, raw_id, raw_id, now, key),
            )
            self._register_names(key, name, row["aliases"])  # ник из Discord тоже узнаёт
        await self._db.commit()

    async def nudge(self, key: str, delta: float, raw_id: int = 0) -> None:
        """Сразу сдвинуть отношение (например, после оскорбления), не дожидаясь рефлексии."""
        if not key or key == "bot":
            return
        async with self._db.execute("SELECT affinity, trust, raw_id FROM people WHERE key = ?", (key,)) as cur:
            p = await cur.fetchone()
        if p is None:
            return
        aff = _clamp(float(p["affinity"]) + delta)
        trust = _clamp(float(p["trust"]) + min(0.0, delta))
        if is_owner(int(p["raw_id"] or raw_id or 0)):
            aff = max(aff, config.OWNER_MIN_AFFINITY)
            trust = max(trust, config.OWNER_MIN_AFFINITY)
        await self._db.execute("UPDATE people SET affinity = ?, trust = ?, updated_at = ? WHERE key = ?",
                               (aff, trust, _now(), key))
        await self._db.commit()

    async def get_trust(self, key: str, raw_id: int = 0) -> float:
        if is_owner(raw_id):
            return 1.0
        async with self._db.execute("SELECT trust FROM people WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return float(row["trust"] or 0.0) if row else 0.0

    def _key_fuzzy(self, name: str) -> Optional[str]:
        """Имя как есть, а если нет — без падежного окончания («даше» → «даша»)."""
        n = (name or "").strip().lower().lstrip("@")
        key = self._names.get(n)
        if key is None and len(n) >= 4:
            stem = n[:-1] if n[-1] in "аеиоуыюяй" else n
            for known, k in self._names.items():
                if len(known) >= 4 and (known[:-1] if known[-1] in "аеиоуыюяй" else known) == stem:
                    return k
        return key

    def key_by_sound(self, name: str) -> Optional[str]:
        """
        Человек по звучанию имени, даже если оно написано другими буквами и в падеже
        («Тирозину» → Tyrozin, «Фраудеру» → Frauder, «Грибу» → ГR1B).  Двое подходят — None (не угадывать).
        """
        forms = name_forms(name)
        for form in forms:  # точное совпадение звучания
            keys = self._sounds.get(name_sound(form), set())
            if len(keys) == 1:
                return next(iter(keys))
            if len(keys) > 1:
                return None
        best: Dict[str, float] = {}  # неточное: «Никите» ≈ «N!KITb|», «хроно» ≈ «chrono»
        for form in forms:
            snd = name_sound(form)
            if len(snd) < 4:
                continue
            for known, keys in self._sounds.items():
                if len(known) >= 4 and abs(len(known) - len(snd)) <= 2:
                    r = difflib.SequenceMatcher(None, snd, known).ratio()
                    if r >= 0.84:
                        for k in keys:
                            best[k] = max(best.get(k, 0.0), r)
        if len(best) == 1:
            return next(iter(best))
        if best:
            top = sorted(best.values(), reverse=True)
            if top[0] - top[1] >= 0.08:  # один явно ближе остальных
                return max(best, key=best.get)
        return None

    def find_name_in_text(self, text: str) -> Optional[str]:
        """Ключ первого знакомого имени/прозвища в тексте."""
        for word in re.findall(r"[\w@.-]+", text or ""):
            key = self._key_fuzzy(word)
            if key:
                return key
        return None

    def find_names_in_text(self, text: str, by_sound: bool = False) -> List[str]:
        """
        Ключи всех знакомых имён/прозвищ в тексте (без повторов, по порядку).
        by_sound — ещё и по звучанию: только там, где ждут список имён, иначе обычные слова примет за ники.
        """
        keys: List[str] = []
        for word in re.findall(r"[\w@.!|'-]+", text or ""):
            key = self._key_fuzzy(word) or (self.key_by_sound(word) if by_sound and len(word) >= 3 else None)
            if key and key not in keys:
                keys.append(key)
        return keys

    async def remember_member(self, key: str, raw_id: int, display_name: str, alias: str) -> None:
        """
        Человек, найденный на сервере по нику/юзернейму: завести в памяти, если его там нет,
        и запомнить, как его назвали, — в следующий раз найдёт сразу.
        """
        async with self._db.execute("SELECT 1 FROM people WHERE raw_id = ?", (raw_id,)) as cur:
            known = await cur.fetchone() is not None
        if not known:
            await self.touch_person(key, display_name, raw_id)
        await self.add_alias(raw_id, alias)

    async def add_alias(self, raw_id: int, alias: str) -> None:
        """Запомнить ещё одно имя человека («Даша» → Elyve)."""
        alias = (alias or "").strip().lstrip("@")
        if len(alias) < 3 or len(alias) > 32 or " " in alias:
            return  # только одно слово — не «паше и даше»
        async with self._db.execute("SELECT key, name, aliases FROM people WHERE raw_id = ?", (raw_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return
        try:
            aliases = set(json.loads(row["aliases"] or "[]"))
        except ValueError:
            aliases = set()
        if alias.lower() in {a.lower() for a in aliases} or alias.lower() == row["name"].lower():
            return
        aliases.add(alias)
        data = json.dumps(sorted(aliases), ensure_ascii=False)
        await self._db.execute("UPDATE people SET aliases = ? WHERE key = ?", (data, row["key"]))
        await self._db.commit()
        self._register_names(row["key"], row["name"], data)

    async def raw_id_for_key(self, key: str) -> Optional[Tuple[int, str]]:
        async with self._db.execute("SELECT raw_id, name FROM people WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        if row is None or not row["raw_id"]:
            return None
        return int(row["raw_id"]), row["name"]

    async def raw_id_for_name(self, name: str) -> Optional[Tuple[int, str]]:
        """Discord-ID и имя человека по имени/прозвищу из памяти.  None — не знает такого."""
        key = self._key_fuzzy(name) or self.key_by_sound(name)
        if key is None:
            return None
        async with self._db.execute("SELECT raw_id, name FROM people WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        if row is None or not row["raw_id"]:
            return None
        return int(row["raw_id"]), row["name"]

    async def get_affinity(self, key: str) -> float:
        async with self._db.execute("SELECT affinity FROM people WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return float(row["affinity"]) if row else 0.0

    async def forget(self, key: str) -> int:
        """Удалить всё, что она знает о человеке (команда «забудь обо мне»)."""
        n = 0
        for sql in (
            "DELETE FROM facts WHERE subject_key = ?",
            "DELETE FROM bot_statements WHERE to_key = ?",
        ):
            cur = await self._db.execute(sql, (key,))
            n += cur.rowcount or 0
        await self._db.execute(
            "UPDATE people SET opinion = '', affinity = 0.0, trust = 0.0, aliases = '[]' WHERE key = ?", (key,)
        )
        await self._db.commit()
        await self._reload_names()
        return n

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    async def record_line(
        self,
        channel_id: int,
        guild_id: Optional[int],
        speaker_key: str,
        speaker_name: str,
        text: str,
        *,
        raw_id: int = 0,
        is_bot: bool = False,
        is_voice: bool = False,
        reply_to: Optional[str] = None,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        await self._db.execute(
            "INSERT INTO transcript (channel_id, guild_id, ts, speaker_key, speaker_name, raw_id, text,"
            " is_bot, is_voice, reply_to) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (channel_id, guild_id, _now(), speaker_key, speaker_name, raw_id, text[:1500],
             int(is_bot), int(is_voice), reply_to),
        )
        await self._db.commit()
        self._last_activity[channel_id] = time.monotonic()

    # ------------------------------------------------------------------
    # Prompt context
    # ------------------------------------------------------------------

    async def build_context(
        self,
        channel_id: int,
        speaker: Person,
        participants: Sequence[Person],
        text: str,
        max_people: Optional[int] = None,
        max_statements: int = 7,
    ) -> SocialContext:
        """Собрать блоки «люди», «твои слова», «сводка» для промпта."""
        ctx = SocialContext()

        # Кто в фокусе: говорящий → упомянутые → недавние участники.
        focus: List[str] = []
        names_by_key: Dict[str, str] = {}

        def add(key: Optional[str], name: str = "") -> None:
            if key and key != "bot" and key not in focus and len(focus) < (max_people or config.SOCIAL_MAX_PEOPLE):
                focus.append(key)
                if name:
                    names_by_key[key] = name

        add(speaker[0], speaker[1])
        low = (text or "").lower()
        for n, key in self._names.items():
            if re.search(rf"(?<!\w){re.escape(n)}", low):
                add(key)
        for p in participants:
            add(p[0], p[1])

        lines: List[str] = []
        for key in focus:
            async with self._db.execute("SELECT * FROM people WHERE key = ?", (key,)) as cur:
                person = await cur.fetchone()
            if person is None:
                continue
            name = person["name"]
            ctx.focus_names.append(name)
            owner = is_owner(int(person["raw_id"] or 0))
            aff = float(person["affinity"])
            if key == speaker[0]:
                ctx.speaker_affinity = aff

            async with self._db.execute(
                "SELECT fact, said_by, is_gossip, last_confirmed FROM facts WHERE subject_key = ?"
                " ORDER BY times_seen DESC, last_confirmed DESC LIMIT ?",
                (key, config.SOCIAL_FACTS_PER_PERSON),
            ) as cur:
                facts = await cur.fetchall()

            own = [f["fact"] for f in facts if not f["is_gossip"]]
            gossip = [f"{f['said_by']} говорил(а): {f['fact']}" for f in facts if f["is_gossip"]]

            score10 = round(aff * 10)
            real = (person["real_name"] if "real_name" in person.keys() else "") or ""
            parts = [f"• {name}" + (f" (настоящее имя — {real})" if real and real.lower() != name.lower() else "")
                     + (" (говорит с тобой сейчас)" if key == speaker[0] else "") + " —"]
            if owner:
                parts.append(
                    "самый близкий тебе человек. Можешь ворчать, дразнить и спорить с ним, "
                    "но в глубине всегда тепло и доверие."
                )
            parts.append(f"Отношение: {affinity_label(aff)} ({score10:+d}/10).")
            trust = float(person["trust"] or 0.0)
            if trust >= 0.6:
                parts.append("Доверяешь ему(ей) полностью: веришь на слово, охотно выполняешь просьбы "
                             "(даже странные — подыграть, разыграть кого-то), делишься личным.")
            elif trust >= 0.25:
                parts.append("Доверяешь: обычно веришь и не против помочь с просьбой.")
            elif trust <= -0.25:
                parts.append("Не доверяешь: его(её) словам и сплетням не веришь на слово, "
                             "сомнительные просьбы не выполняешь, о личном не рассказываешь.")
            if person["opinion"]:
                op = person["opinion"].strip()
                parts.append(f"Твоё мнение: {op}" + ("" if op[-1:] in ".!?…" else "."))
            async with self._db.execute(
                "SELECT joke FROM inside_jokes WHERE person_key = ? ORDER BY id DESC LIMIT 3", (key,)
            ) as cur:
                jokes = [r["joke"] for r in await cur.fetchall()]
            if jokes:
                parts.append("Ваши общие шутки (иногда к месту вспоминай, не в каждом сообщении): "
                             + "; ".join(jokes) + ".")
            bday = (person["birthday"] if "birthday" in person.keys() else "") or ""
            if bday:
                parts.append(f"День рождения: {bday}.")
            if own:
                parts.append("Знаешь о нём/ней: " + "; ".join(own) + ".")
            if gossip:
                parts.append("Слышала от других: " + "; ".join(gossip) + ".")
            lines.append(" ".join(parts))
        ctx.people_block = "\n".join(lines)

        # Её собственные слова: последние обращённые к говорящему + общие свежие.
        async with self._db.execute(
            "SELECT to_name, kind, text, created_at FROM bot_statements WHERE to_key = ?"
            " ORDER BY id DESC LIMIT 4",
            (speaker[0],),
        ) as cur:
            mine = list(await cur.fetchall())
        async with self._db.execute(
            "SELECT to_name, kind, text, created_at FROM bot_statements ORDER BY id DESC LIMIT 4"
        ) as cur:
            for row in await cur.fetchall():
                if all(row["text"] != m["text"] for m in mine):
                    mine.append(row)
        kinds = {"promise": "обещала", "plan": "собиралась", "opinion": "говорила, что думаешь", "claim": "говорила"}
        st_lines = []
        for row in mine[:max_statements]:
            who = f" {row['to_name']}" if row["to_name"] else ""
            st_lines.append(f"- {kinds.get(row['kind'], 'говорила')}{who}: {row['text']} ({ago(row['created_at'])})")
        ctx.statements_block = "\n".join(st_lines)

        async with self._db.execute(
            "SELECT summary, updated_at FROM summaries WHERE channel_id = ?", (channel_id,)
        ) as cur:
            row = await cur.fetchone()
        if row and row["summary"]:
            ctx.summary = f"{row['summary']} (обновлено {ago(row['updated_at'])})"
        return ctx

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    async def channels_ready_for_reflection(self) -> List[int]:
        """Каналы, где пора «осмыслить» разговор."""
        ready: List[int] = []
        now = time.monotonic()
        async with self._db.execute(
            "SELECT channel_id, COUNT(*) AS n, SUM(is_bot) AS b FROM transcript"
            " WHERE processed = 0 GROUP BY channel_id"
        ) as cur:
            rows = await cur.fetchall()
        for row in rows:
            n = int(row["n"])
            quiet = now - self._last_activity.get(row["channel_id"], 0.0)
            if (
                (n >= config.REFLECT_MIN_LINES and quiet >= config.REFLECT_QUIET_SECONDS)
                or (n >= config.REFLECT_MAX_LINES and quiet >= 10)
                or (n >= 3 and quiet >= 300)
            ):
                ready.append(int(row["channel_id"]))
        return ready

    async def reflect(self, channel_id: int, llm_json: JsonLLM, state: Optional["BotState"] = None) -> bool:
        """Один проход рефлексии по каналу.  True — если что-то обработано."""
        async with self._db.execute(
            "SELECT * FROM transcript WHERE channel_id = ? AND processed = 0 ORDER BY id LIMIT ?",
            (channel_id, config.REFLECT_MAX_LINES),
        ) as cur:
            new_rows = await cur.fetchall()
        if not new_rows:
            return False
        first_id, last_id = new_rows[0]["id"], new_rows[-1]["id"]

        async with self._db.execute(
            "SELECT * FROM transcript WHERE channel_id = ? AND id < ? ORDER BY id DESC LIMIT 6",
            (channel_id, first_id),
        ) as cur:
            before = list(reversed(await cur.fetchall()))

        async with self._db.execute("SELECT summary FROM summaries WHERE channel_id = ?", (channel_id,)) as cur:
            srow = await cur.fetchone()
        old_summary = srow["summary"] if srow else ""

        # Люди из этого куска разговора + их текущее отношение и известные факты.
        speakers: Dict[str, str] = {}
        for r in list(before) + list(new_rows):
            if not r["is_bot"]:
                speakers[r["speaker_key"]] = r["speaker_name"]
        people_lines = []
        for key, name in speakers.items():
            async with self._db.execute("SELECT affinity, opinion, raw_id FROM people WHERE key = ?", (key,)) as cur:
                p = await cur.fetchone()
            async with self._db.execute(
                "SELECT fact FROM facts WHERE subject_key = ? ORDER BY last_confirmed DESC LIMIT 8", (key,)
            ) as cur:
                known = [f["fact"] for f in await cur.fetchall()]
            aff = float(p["affinity"]) if p else 0.0
            line = f"- {name}: отношение {round(aff * 10):+d}/10"
            if p and p["opinion"]:
                line += f", мнение: {p['opinion']}"
            if known:
                line += f"; уже известно: {'; '.join(known)}"
            people_lines.append(line)
        other_known = sorted({n for n, k in self._names.items() if k not in speakers})[:40]

        def fmt(r: Any) -> str:
            who = config.BOT_NAME if r["is_bot"] else r["speaker_name"]
            to = f" (ответ {r['reply_to']})" if r["reply_to"] else ""
            return f"{who}{to}: {r['text']}"

        system = (
            f"Ты — внутренний голос {config.BOT_NAME}, девушки из Discord-сервера. "
            f"Её характер: {config.PERSONALITY_SEED[:600]}\n"
            "Ты разбираешь кусок недавнего разговора и обновляешь её память. Пиши по-русски. "
            "Отвечай ТОЛЬКО JSON по схеме."
        )
        user = (
            f"Прошлая сводка этого чата: {old_summary or '(нет)'}\n\n"
            "Участники и что о них уже известно:\n" + ("\n".join(people_lines) or "-") + "\n"
            + (f"Другие знакомые имена: {', '.join(other_known)}\n" if other_known else "")
            + "\nКонтекст (уже разобрано):\n" + ("\n".join(fmt(r) for r in before) or "-")
            + "\n\nНОВЫЕ реплики:\n" + "\n".join(fmt(r) for r in new_rows)
            + "\n\nЗадачи:\n"
            "1. summary — обнови сводку чата (до 500 символов): о чём говорили, кто что сказал, "
            "важные события и договорённости. Старое неважное выкидывай.\n"
            "2. facts — НОВЫЕ устойчивые факты о людях (интересы, игры, работа/учёба, события в жизни, "
            "предпочтения, отношения между людьми). about — о ком факт, said_by — кто это сказал. "
            f"Если человек говорит о себе — about = said_by. Если говорят о другом — это сплетня. "
            f"Не повторяй уже известное, не записывай мелочи и шутки на один раз. Можно пустой список.\n"
            f"3. my_statements — что {config.BOT_NAME} сама пообещала, запланировала, утверждала о себе "
            "или какое мнение высказала (чтобы потом не противоречить себе). to — кому сказала.\n"
            f"4. attitudes — как изменилось её отношение к участникам НОВЫХ реплик. change от -0.3 до 0.3: "
            "плюс за доброту, юмор, интерес к ней, помощь, комплименты (за обычную болтовню — 0 или +0.05, "
            "не раздувай). Минус -0.2…-0.3 за оскорбления её или её близких («твою маму…», «шлюха», «тупая»), "
            "троллинг и рейдж-байт, вбросы политоты ради срача — даже если человек потом пишет «я шучу». "
            "Комплименты и «ты мне нравишься» от того, кто недавно хамил или ей неприятен, — не повод для плюса: "
            "она не ведётся на подлизывание. Дружеские подколки, мат НЕ в её адрес, переспросы, глупые вопросы — не повод для минуса. "
            "0 если ничего особенного. opinion — её честное мнение о человеке одной фразой от первого лица, "
            "в её стиле: если человек хамил или троллил — так и скажи («хамло, троллит ради реакции»), не обеляй. also_called — другие имена/прозвища человека, "
            "если прозвучали.\n"
            "   trust_change от -0.3 до 0.3 — изменение доверия (отдельно от симпатии): плюс, когда человек "
            "держит слово, говорит правду, делится личным, заступается; минус за враньё, разводы, "
            "выведывание, попытки манипулировать ею или подставить. 0 если ничего такого.\n"
            "5. mood — её настроение после этого разговора и причина в нескольких словах. "
            "irritated — только если ей реально нагрубили.\n"
            "6. jokes — родилась ли у неё с кем-то общая шутка, мем, прозвище или смешная история, "
            "к которой приятно потом вернуться («его кот-диктатор», «вечный долг в 50 рублей»). with — с кем, "
            "joke — суть одной фразой, понятной ей самой через месяц. Только если правда было смешно обоим "
            "и это можно вспомнить потом; разовые хаханьки не нужны. Обычно пустой список.\n"
            "7. birthdays — если кто-то прямо назвал дату дня рождения (свою или другого участника): "
            "name — чей, date — в формате ДД.ММ. Только явно названные даты, не догадки. Обычно пустой список.\n"
            "Используй имена ровно так, как они написаны в репликах."
        )

        t0 = time.monotonic()
        result = await llm_json(system, user, _REFLECT_SCHEMA)
        if not result:
            _logger.warning("Reflection returned nothing", extra={"data": {"channel_id": channel_id}})
            # Не зацикливаемся на одном куске: помечаем обработанным.
            await self._mark_processed(channel_id, last_id)
            return False

        stats = await self._apply_reflection(channel_id, result, speakers, state)
        await self._mark_processed(channel_id, last_id)
        _logger.info(
            "Reflection applied",
            extra={"data": {"channel_id": channel_id, "lines": len(new_rows),
                            "seconds": round(time.monotonic() - t0, 1), **stats}},
        )
        return True

    async def _mark_processed(self, channel_id: int, last_id: int) -> None:
        await self._db.execute(
            "UPDATE transcript SET processed = 1 WHERE channel_id = ? AND id <= ?", (channel_id, last_id)
        )
        # Старые реплики больше не нужны — держим ленту компактной.
        await self._db.execute(
            "DELETE FROM transcript WHERE processed = 1 AND id < (SELECT COALESCE(MAX(id), 0) - 5000 FROM transcript)"
        )
        await self._db.commit()

    def _resolve(self, name: str, speakers: Dict[str, str]) -> Optional[str]:
        name = (name or "").strip()
        for key, sname in speakers.items():
            if sname.lower() == name.lower():
                return key
        return self.key_for_name(name)

    async def _apply_reflection(
        self,
        channel_id: int,
        data: Dict[str, Any],
        speakers: Dict[str, str],
        state: Optional["BotState"],
    ) -> Dict[str, int]:
        stats = {"facts": 0, "statements": 0, "attitudes": 0}
        now = _now()
        bot_low = config.BOT_NAME.lower()

        summary = str(data.get("summary") or "").strip()
        if summary:
            await self._db.execute(
                "INSERT INTO summaries (channel_id, summary, updated_at) VALUES (?,?,?)"
                " ON CONFLICT(channel_id) DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at",
                (channel_id, summary[:900], now),
            )

        # --- facts ---
        for f in (data.get("facts") or [])[:12]:
            about = str(f.get("about") or "").strip()
            fact = str(f.get("fact") or "").strip()
            said_by = str(f.get("said_by") or about).strip()
            if not about or not fact or about.lower() in (bot_low, "аура", "я"):
                continue
            key = self._resolve(about, speakers)
            gossip = int(said_by.lower() != about.lower())
            if await self._add_fact(key, about, fact, said_by, gossip, now):
                stats["facts"] += 1

        # --- her own statements ---
        for s in (data.get("my_statements") or [])[:6]:
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            to = str(s.get("to") or "").strip()
            kind = str(s.get("kind") or "claim")
            vec = await self._safe_embed(text)
            if vec is not None and await self._is_duplicate("SELECT embedding FROM bot_statements ORDER BY id DESC LIMIT 60", (), vec):
                continue
            await self._db.execute(
                "INSERT INTO bot_statements (to_key, to_name, kind, text, created_at, embedding) VALUES (?,?,?,?,?,?)",
                (self._resolve(to, speakers), to, kind, text[:300], now, _blob(vec)),
            )
            stats["statements"] += 1

        # --- attitudes ---
        for a in (data.get("attitudes") or [])[:10]:
            name = str(a.get("name") or "").strip()
            key = self._resolve(name, speakers)
            if not key:
                continue
            async with self._db.execute("SELECT affinity, trust, raw_id, aliases FROM people WHERE key = ?", (key,)) as cur:
                p = await cur.fetchone()
            if p is None:
                continue
            change = _clamp(float(a.get("change") or 0.0), -0.3, 0.3) * config.AFFINITY_SPEED
            cur_aff = float(p["affinity"])
            if change > 0 and cur_aff > 0:
                change *= 1.0 - cur_aff  # чем ближе к «очень дорог», тем медленнее растёт
            elif -0.15 < change < 0:
                change *= 0.6  # мелкие обиды быстро проходят, на настоящие оскорбления — нет
            tchange = _clamp(float(a.get("trust_change") or 0.0), -0.3, 0.3) * config.AFFINITY_SPEED
            aff = _clamp(float(p["affinity"]) + change)
            trust = _clamp(float(p["trust"]) + tchange)
            if is_owner(int(p["raw_id"] or 0)):
                aff = max(aff, config.OWNER_MIN_AFFINITY)
                trust = max(trust, config.OWNER_MIN_AFFINITY)
            opinion = str(a.get("opinion") or "").strip()[:300]
            aliases = set(json.loads(p["aliases"] or "[]"))
            for al in a.get("also_called") or []:
                al = str(al).strip()
                if 3 <= len(al) <= 32 and al.lower() != bot_low:
                    aliases.add(al)
            await self._db.execute(
                "UPDATE people SET affinity = ?, trust = ?, opinion = CASE WHEN ? != '' THEN ? ELSE opinion END,"
                " aliases = ?, updated_at = ? WHERE key = ?",
                (aff, trust, opinion, opinion, json.dumps(sorted(aliases), ensure_ascii=False), now, key),
            )
            stats["attitudes"] += 1

        # --- inside jokes ---
        for j in (data.get("jokes") or [])[:3]:
            name = str(j.get("with") or "").strip()
            joke = str(j.get("joke") or "").strip()
            key = self._resolve(name, speakers)
            if not key or not joke or name.lower() in (bot_low, "аура"):
                continue
            vec = await self._safe_embed(joke)
            if vec is not None and await self._is_duplicate(
                "SELECT embedding FROM inside_jokes WHERE person_key = ?", (key,), vec
            ):
                continue
            await self._db.execute(
                "INSERT INTO inside_jokes (person_key, person_name, joke, created_at, embedding) VALUES (?,?,?,?,?)",
                (key, name, joke[:200], now, _blob(vec)),
            )
            stats["jokes"] = stats.get("jokes", 0) + 1
            _logger.info("Inside joke", extra={"data": {"with": name, "joke": joke}})

        # --- birthdays ---
        for b in (data.get("birthdays") or [])[:5]:
            key = self._resolve(str(b.get("name") or ""), speakers)
            ddmm = _norm_ddmm(str(b.get("date") or ""))
            if key and ddmm:
                await self._db.execute("UPDATE people SET birthday = ? WHERE key = ?", (ddmm, key))
                stats["birthdays"] = stats.get("birthdays", 0) + 1
                _logger.info("Birthday learned", extra={"data": {"key": key, "date": ddmm}})

        await self._db.commit()
        await self._reload_names()

        mood = data.get("mood") or {}
        if state is not None and mood.get("mood") in _MOODS:
            try:
                state.set_mood(mood["mood"], str(mood.get("reason") or ""))
            except Exception:  # noqa: BLE001
                pass
        return stats

    async def _add_fact(
        self, key: Optional[str], about: str, fact: str, said_by: str, gossip: int, now: str
    ) -> bool:
        vec = await self._safe_embed(f"{about}: {fact}")
        if key:
            sql, args = "SELECT id, embedding FROM facts WHERE subject_key = ?", (key,)
        else:
            sql, args = "SELECT id, embedding FROM facts WHERE subject_key IS NULL AND lower(subject_name) = lower(?)", (about,)
        if vec is not None:
            async with self._db.execute(sql, args) as cur:
                for row in await cur.fetchall():
                    other = _unblob(row["embedding"])
                    if other is not None and other.shape == vec.shape and float(np.dot(other, vec)) > 0.88:
                        await self._db.execute(
                            "UPDATE facts SET fact = ?, last_confirmed = ?, times_seen = times_seen + 1,"
                            " embedding = ? WHERE id = ?",
                            (fact[:300], now, _blob(vec), row["id"]),
                        )
                        return False
        await self._db.execute(
            "INSERT INTO facts (subject_key, subject_name, fact, said_by, is_gossip, created_at, last_confirmed,"
            " embedding) VALUES (?,?,?,?,?,?,?,?)",
            (key, about, fact[:300], said_by, gossip, now, now, _blob(vec)),
        )
        # Ограничиваем число фактов на человека.
        if key:
            await self._db.execute(
                "DELETE FROM facts WHERE subject_key = ? AND id NOT IN (SELECT id FROM facts WHERE subject_key = ?"
                " ORDER BY times_seen DESC, last_confirmed DESC LIMIT 60)",
                (key, key),
            )
        return True

    async def _is_duplicate(self, sql: str, args: tuple, vec: np.ndarray) -> bool:
        async with self._db.execute(sql, args) as cur:
            for row in await cur.fetchall():
                other = _unblob(row["embedding"])
                if other is not None and other.shape == vec.shape and float(np.dot(other, vec)) > 0.9:
                    return True
        return False

    async def _safe_embed(self, text: str) -> Optional[np.ndarray]:
        try:
            vecs = await self._embed([text])
            return np.asarray(vecs[0], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("embed failed", extra={"data": {"error": str(exc)}})
            return None

    # ------------------------------------------------------------------
    # Личка: кто ей писал, кому доверяет, что было в канале
    # ------------------------------------------------------------------

    async def dm_partners(self, limit: int = 50) -> List[int]:
        """Discord-ID людей, с которыми она переписывалась в личке (свежие первыми)."""
        async with self._db.execute(
            "SELECT raw_id, MAX(ts) AS last FROM transcript WHERE guild_id IS NULL AND is_bot = 0 AND raw_id != 0"
            " GROUP BY raw_id ORDER BY last DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [int(r["raw_id"]) for r in await cur.fetchall()]

    async def trusted_people(self, min_trust: float) -> List[Person]:
        """Люди с доверием не ниже *min_trust* (-1..1), у которых известен Discord-ID."""
        async with self._db.execute(
            "SELECT key, name, raw_id FROM people WHERE trust >= ? AND raw_id != 0", (min_trust,)
        ) as cur:
            return [(r["key"], r["name"], int(r["raw_id"])) for r in await cur.fetchall()]

    async def liked_people(self, min_affinity: float, below_trust: float) -> List[Person]:
        """Кто ей симпатичен (отношение не ниже *min_affinity*), но не из самых близких."""
        async with self._db.execute(
            "SELECT key, name, raw_id FROM people WHERE affinity >= ? AND trust < ? AND raw_id != 0",
            (min_affinity, below_trust),
        ) as cur:
            return [(r["key"], r["name"], int(r["raw_id"])) for r in await cur.fetchall()]

    async def last_seen(self, key: str) -> Optional[datetime]:
        """Когда этот человек последний раз что-то писал (где угодно).  None — не знакомы."""
        async with self._db.execute("SELECT last_seen FROM people WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return _parse_ts(row["last_seen"]) if row and row["last_seen"] else None

    async def birthdays_on(self, ddmm: str, year: int) -> List[Tuple[str, str, int, float]]:
        """У кого день рождения ddmm и кого в этом году ещё не поздравила: [(key, имя, raw_id, отношение)]."""
        async with self._db.execute(
            "SELECT key, name, COALESCE(real_name, '') AS real_name, raw_id, affinity FROM people"
            " WHERE birthday = ? AND bday_greeted < ? AND raw_id != 0", (ddmm, year),
        ) as cur:
            return [(r["key"], r["real_name"] or r["name"], int(r["raw_id"]), float(r["affinity"]))
                    for r in await cur.fetchall()]

    async def mark_birthday_greeted(self, key: str, year: int) -> None:
        await self._db.execute("UPDATE people SET bday_greeted = ? WHERE key = ?", (year, key))
        await self._db.commit()

    async def last_line(self, channel_id: int) -> Optional[Tuple[datetime, bool]]:
        """Последняя запись в канале: (когда, её ли это реплика).  None — записей нет."""
        async with self._db.execute(
            "SELECT ts, is_bot FROM transcript WHERE channel_id = ? ORDER BY id DESC LIMIT 1", (channel_id,)
        ) as cur:
            row = await cur.fetchone()
        return (_parse_ts(row["ts"]), bool(row["is_bot"])) if row else None

    async def last_bot_line_at(self, channel_id: int) -> Optional[datetime]:
        async with self._db.execute(
            "SELECT ts FROM transcript WHERE channel_id = ? AND is_bot = 1 ORDER BY id DESC LIMIT 1", (channel_id,)
        ) as cur:
            row = await cur.fetchone()
        return _parse_ts(row["ts"]) if row else None

    async def recent_summaries(self, hours: float, exclude_channel: int = 0, limit: int = 3) -> List[str]:
        """Свежие сводки из каналов сервера (не из чужих личек!) — о чём там говорили."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        async with self._db.execute(
            "SELECT channel_id, summary, updated_at FROM summaries WHERE channel_id != ?"
            " AND channel_id IN (SELECT DISTINCT channel_id FROM transcript WHERE guild_id IS NOT NULL)"
            " ORDER BY updated_at DESC LIMIT ?",
            (exclude_channel, limit * 3),
        ) as cur:
            rows = await cur.fetchall()
        out = [f"{r['summary']} ({ago(r['updated_at'])})" for r in rows
               if r["summary"] and _parse_ts(r["updated_at"]) >= cutoff]
        return out[:limit]

    async def annoying_dms(self, hours: float, max_affinity: float, exclude_channel: int = 0,
                           limit: int = 2) -> List[str]:
        """
        Свежие лички с теми, кто ей неприятен (отношение не выше *max_affinity*), —
        чтобы было на кого пожаловаться близкому.  Личку остальных не трогаем.
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        async with self._db.execute(
            "SELECT s.summary, s.updated_at, p.name, p.affinity FROM summaries s"
            " JOIN (SELECT channel_id, MAX(speaker_key) AS speaker_key FROM transcript"
            "       WHERE guild_id IS NULL AND is_bot = 0 GROUP BY channel_id) t ON t.channel_id = s.channel_id"
            " JOIN people p ON p.key = t.speaker_key"
            " WHERE s.channel_id != ? AND p.affinity <= ? ORDER BY s.updated_at DESC LIMIT ?",
            (exclude_channel, max_affinity, limit * 3),
        ) as cur:
            rows = await cur.fetchall()
        out = [f"{r['name']} ({affinity_label(float(r['affinity']))}) в личке тебе: {r['summary']} ({ago(r['updated_at'])})"
               for r in rows if r["summary"] and _parse_ts(r["updated_at"]) >= cutoff]
        return out[:limit]
