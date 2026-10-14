"""utils/names.py — распознавание обращения к боту по имени (кириллицей и латиницей)."""

from __future__ import annotations

import re

import config


def mentions_bot_name(text: str) -> bool:
    """True, если в тексте есть имя бота или одно из VOICE_NAME_ALIASES («аура», «ауре», «aura»…)."""
    low = (text or "").lower()
    names = [config.BOT_NAME.lower()] + list(getattr(config, "VOICE_NAME_ALIASES", []))
    return any(re.search(rf"(?<!\w){re.escape(n)}", low) for n in names if n)


def is_bot_mentioned(message, bot_user_id: int) -> bool:
    """
    @упоминание бота: и как пользователя (@Aura), и как её роли.

    Discord в подсказке часто подставляет не пользователя, а роль бота
    с тем же именем (<@&id>) — такие упоминания тоже считаем обращением.
    """
    if any(getattr(m, "id", None) == bot_user_id for m in getattr(message, "mentions", []) or []):
        return True
    guild = getattr(message, "guild", None)
    me = getattr(guild, "me", None) if guild is not None else None
    if me is None:
        return False
    my_roles = {r.id for r in getattr(me, "roles", []) if not getattr(r, "is_default", lambda: False)()}
    return any(getattr(r, "id", None) in my_roles for r in getattr(message, "role_mentions", []) or [])
