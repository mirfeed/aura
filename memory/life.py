"""
memory/life.py — её собственная жизнь.

Небольшие события из жизни Ауры (кошка Луна разбила кружку, нашла на барахолке
флюорит, сдала зачёт по кристаллографии).  Новое событие придумывает модель
время от времени — с оглядкой на прошлые, чтобы жизнь шла последовательно,
а не сочинялась заново в каждом ответе.  Последние события попадают в промпт:
она может к месту о них рассказать и не противоречит им.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from utils.logger import get_logger

if TYPE_CHECKING:
    import aiosqlite

_logger = get_logger("bot.life")

def _ago(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ""
    hours = (datetime.now(tz=timezone.utc) - dt).total_seconds() / 3600
    if hours < 1:
        return "только что"
    if hours < 20:
        return f"{int(hours)} ч назад"
    days = int(hours // 24) or 1
    return "вчера" if days == 1 else f"{days} дн назад"


class LifeStore:
    def __init__(self, db_getter: Callable[[], "aiosqlite.Connection"]) -> None:
        self._db_getter = db_getter
        self._recent: List[Dict[str, Any]] = []  # кэш последних событий (новые последними)

    @property
    def _db(self) -> "aiosqlite.Connection":
        return self._db_getter()

    async def init(self) -> None:
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS life_events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " created_at TEXT NOT NULL, text TEXT NOT NULL)"
        )
        await self._db.commit()
        await self._load()

    async def _load(self) -> None:
        async with self._db.execute("SELECT created_at, text FROM life_events ORDER BY id DESC LIMIT 30") as cur:
            rows = await cur.fetchall()
        self._recent = [{"created_at": r["created_at"], "text": r["text"]} for r in reversed(rows)]

    async def add(self, text: str) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        await self._db.execute("INSERT INTO life_events (created_at, text) VALUES (?, ?)", (now, text[:300]))
        await self._db.commit()
        self._recent.append({"created_at": now, "text": text[:300]})
        self._recent = self._recent[-30:]
        _logger.info("Life event", extra={"data": {"text": text}})

    def last_at(self) -> Optional[datetime]:
        if not self._recent:
            return None
        return datetime.fromisoformat(self._recent[-1]["created_at"])

    def history_lines(self, n: int = 15) -> List[str]:
        """Для генерации следующего события: что было раньше."""
        return [f"- ({_ago(e['created_at'])}) {e['text']}" for e in self._recent[-n:]]

    def prompt_block(self, n: int = 5) -> str:
        if not self._recent:
            return ""
        return ("Что у тебя самой было в жизни недавно (это правда твоя жизнь: можешь к месту рассказать, "
                "не вываливай всё сразу, не противоречь этому):\n"
                + "\n".join(f"- {e['text']} ({_ago(e['created_at'])})" for e in self._recent[-n:]))
