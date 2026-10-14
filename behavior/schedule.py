"""
behavior/schedule.py — распорядок дня Ауры.

Она сова: ложится под утро, встаёт ближе к обеду.  Время отхода ко сну и
подъёма каждый день немного своё (± SCHEDULE_JITTER_MIN), но для одного и
того же дня всегда одинаковое — случайность зависит только от даты, поэтому
состояние хранить не нужно.

Фазы:
  asleep  — спит: не отвечает, личку разберёт, когда проснётся;
  morning — первые часы после подъёма: сонная, ворчит, отвечает реже и короче;
  day     — обычный режим;
  night   — с 22:00 до сна: её время, бодрая и болтливая.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Optional, Tuple

import config


def _local_now() -> datetime:
    from pipeline.text import _local_now as now  # один источник часового пояса

    return now()


def _parse_hm(spec: str, default: time) -> time:
    try:
        h, m = (int(x) for x in spec.strip().split(":", 1))
        return time(h % 24, m % 60)
    except (ValueError, AttributeError):
        return default


def _sleep_window(day: date, tzinfo) -> Tuple[datetime, datetime]:
    """Когда она спит «в ночь на day»: (лечь, встать), с разбросом, одинаковым для этой даты."""
    rnd = random.Random(day.toordinal() * 7919 + 13)
    jitter = max(0, config.SCHEDULE_JITTER_MIN)
    bed_t = _parse_hm(config.SLEEP_TIME, time(4, 0))
    wake_t = _parse_hm(config.WAKE_TIME, time(11, 0))
    # Лечь после полуночи — это ночь на этот же день; лечь до полуночи — накануне вечером.
    bed_day = day if bed_t <= wake_t else day - timedelta(days=1)
    bed = datetime.combine(bed_day, bed_t, tzinfo) + timedelta(minutes=rnd.uniform(-jitter, jitter))
    wake = datetime.combine(day, wake_t, tzinfo) + timedelta(minutes=rnd.uniform(-jitter, jitter))
    return bed, wake


# Запой: она зачиталась и не может оторваться (выставляет бот, core/bot.py → _maybe_read).
# Пока запой — отвечает рассеяннее, а если пора спать — засиживается до BINGE_STAY_UP_H часов.
# Наутро после такой ночи встаёт позже и дольше сонная.
_binge: Dict[str, Any] = {"title": "", "until": None, "late_wake": None}


def set_binge(title: str, until: Optional[datetime]) -> None:
    """Начать (title, until) или закончить (\"\", None) запой."""
    _binge["title"], _binge["until"] = title, until


def binge_title(now: Optional[datetime] = None) -> str:
    """Что она сейчас читает взапой ('' — не читает)."""
    until = _binge["until"]
    if not _binge["title"] or until is None:
        return ""
    now = now or _local_now()
    return _binge["title"] if now < until.astimezone(now.tzinfo) else ""


def phase(now: datetime | None = None) -> str:
    """'asleep' | 'morning' | 'day' | 'night'."""
    if not config.SCHEDULE_ENABLED:
        return "day"
    now = now or _local_now()
    for d in (now.date(), now.date() + timedelta(days=1)):
        bed, wake = _sleep_window(d, now.tzinfo)
        if _binge["late_wake"] == wake.date():
            wake += timedelta(hours=1)  # засиделась с книгой — проспала
        if bed <= now < wake:
            if binge_title(now) and now < bed + timedelta(hours=config.BINGE_STAY_UP_H):
                _binge["late_wake"] = wake.date()
                return "night"  # «ещё одну главу…»
            return "asleep"
        morning_h = config.MORNING_HOURS + (1.5 if _binge["late_wake"] == wake.date() else 0)
        if wake <= now < wake + timedelta(hours=morning_h):
            return "morning"
    if now.hour >= 22 or now.hour < 6:
        return "night"
    return "day"


def is_asleep(now: datetime | None = None) -> bool:
    return phase(now) == "asleep"


def reply_factor(now: datetime | None = None) -> float:
    """Множитель вероятности ответить в общем чате."""
    f = {"asleep": 0.0, "morning": 0.6, "day": 1.0, "night": 1.2}[phase(now)]
    return f * (0.6 if binge_title(now) else 1.0)  # зачиталась — в чат заглядывает реже


def prompt_hint(now: datetime | None = None) -> str:
    """Подсказка в промпт: как она себя сейчас чувствует по времени суток."""
    hint = {
        "morning": "Ты недавно проснулась: сонная, без кофе ворчливая, отвечаешь короче и ленивее обычного.",
        "night": "Сейчас ночь — твоё время: ты бодрая, разговорчивая, охотно болтаешь.",
    }.get(phase(now), "")
    now = now or _local_now()
    if _binge["late_wake"] == now.date() and phase(now) == "morning":
        hint += " Ночью ты засиделась с книгой и не выспалась."
    title = binge_title(now)
    if title:
        hint += (" Ты сейчас увлечённо читаешь и отвечаешь коротко и немного рассеянно — но в ответе книгу "
                 "не упоминай и не оправдывайся ею, если о ней не спросили: просто отвечай по теме.")
    return hint.strip()
