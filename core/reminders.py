"""
core/reminders.py — «Аура, напомни завтра в 6 про экзамен».

Просьбу разбирает модель (когда и о чём), напоминание хранится в events.db
(таблица reminders), а фоновый цикл бота отправляет его в срок: в тот же
канал с упоминанием человека или в личку.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from utils.logger import get_logger

if TYPE_CHECKING:
    import aiosqlite

_logger = get_logger("bot.reminders")

REMIND_RE = re.compile(r"\bнапомн(?:и|ишь|ите|ить)\b", re.I)

_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
           "сентября", "октября", "ноября", "декабря"]

PARSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "when": {"type": "string"},
        "what": {"type": "string"},
        "where": {"type": "string", "enum": ["here", "dm"]},
    },
    "required": ["ok", "when", "what", "where"],
}


def parse_prompt(text: str, author: str, now_local: datetime) -> tuple:
    """(system, user) для разбора просьбы моделью."""
    system = "Ты разбираешь просьбы о напоминаниях. Отвечай ТОЛЬКО JSON по схеме."
    user = (
        f"Сейчас: {_WEEKDAYS[now_local.weekday()]}, {now_local:%Y-%m-%d %H:%M}.\n"
        f"{author} написал(а): «{text}»\n\n"
        "Если это просьба напомнить о чём-то в определённое время — ok=true, when — когда напомнить, "
        "в формате ГГГГ-ММ-ДД ЧЧ:ММ по тому же времени, что «Сейчас». «Через 20 минут», «завтра в 6» "
        "(6 вечера, если по смыслу утро маловероятно), «в пятницу» (без времени — 12:00), «вечером» — 19:00, "
        "«утром» — 10:00. what — о чём напомнить, коротко, от второго лица («сдать отчёт», «позвонить маме»). "
        "where — dm, если просят написать в личку; иначе here.\n"
        "Если времени нет или это не просьба о напоминании — ok=false, остальные поля пустые."
    )
    return system, user


def human_when(due_local: datetime, now_local: datetime) -> str:
    """«сегодня в 18:00», «завтра в 9:30», «в пятницу, 17 октября, в 12:00»."""
    days = (due_local.date() - now_local.date()).days
    hm = f"{due_local.hour}:{due_local.minute:02d}"
    if days == 0:
        mins = int((due_local - now_local).total_seconds() // 60)
        return f"через {mins} мин" if mins < 60 else f"сегодня в {hm}"
    if days == 1:
        return f"завтра в {hm}"
    return f"в {_WEEKDAYS[due_local.weekday()]}, {due_local.day} {_MONTHS[due_local.month - 1]}, в {hm}"


class ReminderStore:
    """Напоминания в events.db."""

    def __init__(self, db_getter: Callable[[], "aiosqlite.Connection"]) -> None:
        self._db_getter = db_getter

    @property
    def _db(self) -> "aiosqlite.Connection":
        return self._db_getter()

    async def init(self) -> None:
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS reminders ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, user_name TEXT, channel_id INTEGER,"
            " is_dm INTEGER, due_at TEXT, text TEXT, request TEXT, created_at TEXT, done INTEGER DEFAULT 0)"
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(done, due_at)")
        await self._db.commit()

    async def add(self, user_id: int, user_name: str, channel_id: int, is_dm: bool,
                  due_utc: datetime, text: str, request: str) -> int:
        cur = await self._db.execute(
            "INSERT INTO reminders (user_id, user_name, channel_id, is_dm, due_at, text, request, created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (user_id, user_name, channel_id, int(is_dm), due_utc.isoformat(), text[:300], request[:500],
             datetime.now(tz=timezone.utc).isoformat()),
        )
        await self._db.commit()
        return int(cur.lastrowid)

    async def due(self) -> List[Dict[str, Any]]:
        now = datetime.now(tz=timezone.utc).isoformat()
        async with self._db.execute(
            "SELECT * FROM reminders WHERE done = 0 AND due_at <= ? ORDER BY due_at LIMIT 10", (now,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def mark_done(self, rid: int) -> None:
        await self._db.execute("UPDATE reminders SET done = 1 WHERE id = ?", (rid,))
        await self._db.commit()


def prefer_evening(due_utc: Optional[datetime], request: str, now_local: datetime) -> Optional[datetime]:
    """«в 6» без «утра» — это 18:00, а не 06:00 (так говорят люди).  «через …» и «6:30» не трогаем."""
    if due_utc is None:
        return None
    local = due_utc.astimezone(now_local.tzinfo)
    if (1 <= local.hour <= 7 and local.minute == 0
            and not re.search(r"утр|ночи|ночью|\d{1,2}[:.]\d{2}|через", request, re.I)):
        return due_utc + timedelta(hours=12)
    return due_utc


def parse_due(when: str, now_local: datetime) -> Optional[datetime]:
    """«ГГГГ-ММ-ДД ЧЧ:ММ» в местном времени → UTC.  None, если в прошлом или дальше года."""
    try:
        due = datetime.strptime(when.strip()[:16], "%Y-%m-%d %H:%M").replace(tzinfo=now_local.tzinfo)
    except ValueError:
        return None
    if due <= now_local - timedelta(minutes=1) or due > now_local + timedelta(days=366):
        return None
    return due.astimezone(timezone.utc)
