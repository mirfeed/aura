"""
media.py — Аура кидает картинки из своей коллекции.

Коллекция — её любимые картинки (data/favorites): то, что ей понравилось в чате
или что её попросили сохранить.  Ничего из интернета.

В промпт уходит пронумерованный список коллекции; чтобы кинуть картинку,
модель пишет отдельной строкой метку [картинка: 3].  Метка вырезается из текста,
а файл уходит в чат следом за ответом.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import config
from utils.logger import get_logger

if TYPE_CHECKING:
    from memory.images import ImageMemory

_logger = get_logger("bot.media")

# [картинка: 3] / (гифка: кот на камнях) — номер из списка или описание.
_MARKER_RE = re.compile(
    r"[\[(]\s*(?:гифк[аиу]|гиф|gif|картинк[аиу]|пикч[аиу]|фот(?:о|ку)|мем)\s*[:：]\s*([^\]\)\n]{1,80})[\])]",
    re.I,
)
# Человек сам просит скинуть гифку/картинку — тогда без кулдауна.
_ASKED_RE = re.compile(
    r"(?:скин|кин|пришл|отправ|покаж|запост|дай)\w*.{0,30}(?:гиф|картин|пикч|мем|фот|арт|котик)", re.I)
_WORD_RE = re.compile(r"[a-zа-яё0-9]{3,}", re.I)


class MediaPicker:
    def __init__(self, image_memory: "ImageMemory") -> None:
        self._images = image_memory
        self._last_sent: Dict[int, float] = {}
        self._recent_paths: Dict[int, List[Path]] = {}

    # ------------------------------------------------------------------

    def library(self) -> List[Dict[str, Any]]:
        """Её коллекция: любимые картинки с файлами, лучшие первыми."""
        try:
            return self._images.favorite_files(limit=config.MEDIA_PROMPT_ITEMS)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Favorites list failed", extra={"data": {"error": str(exc)}})
            return []

    def prompt_block(self, items: List[Tuple[int, Dict[str, Any]]]) -> str:
        """Список коллекции для промпта; номера — как в library() (часть пунктов может быть скрыта)."""
        return ("Твоя коллекция любимых картинок (кинуть — метка [картинка: номер]):\n"
                + "\n".join(f"{i}. {'гифка' if it['is_gif'] else 'картинка'}: {it['caption'][:90] or 'без подписи'}"
                            for i, it in items))

    @staticmethod
    def asked(text: str) -> bool:
        return bool(_ASKED_RE.search(text or ""))

    def cooled_down(self, channel_id: int) -> bool:
        return time.monotonic() - self._last_sent.get(channel_id, -1e9) >= config.MEDIA_COOLDOWN_S

    @staticmethod
    def extract(text: str) -> Tuple[str, List[str]]:
        """Вырезать метки из ответа.  → (текст без меток, [что внутри метки])."""
        found = [m.group(1).strip(" \"'«»") for m in _MARKER_RE.finditer(text or "")]
        if not found:
            return (text or "").strip(), []  # меток нет — текст (и отступы в коде!) не трогать
        # Убрать только сами метки: строку, где была одна метка, — целиком, в остальных — лишний пробел на месте метки.
        lines = []
        for line in (text or "").splitlines():
            if _MARKER_RE.search(line):
                line = _MARKER_RE.sub("", line)
                if not line.strip():
                    continue
                line = re.sub(r"(?<=\S) {2,}(?=\S)", " ", line).rstrip()
            lines.append(line)
        return "\n".join(lines).strip(), found

    def pick(self, ref: str, channel_id: int) -> Optional[Path]:
        """Номер из списка или описание → файл.  Только из коллекции, ничего не выдумываем."""
        lib = self.library()
        if not lib:
            return None
        m = re.match(r"\s*№?\s*(\d{1,3})\b", ref)
        if m and 1 <= int(m.group(1)) <= len(lib):
            path = lib[int(m.group(1)) - 1]["path"]
        else:
            words = {w.lower()[:5] for w in _WORD_RE.findall(ref)}
            recent = self._recent_paths.get(channel_id, [])
            best, score = None, 0
            for it in lib:
                s = len(words & {w.lower()[:5] for w in _WORD_RE.findall(it["caption"])})
                if s > score or (s == score and s and best in recent):
                    best, score = it["path"], s
            path = best
        return path

    def mark_sent(self, channel_id: int, path: Path) -> None:
        self._last_sent[channel_id] = time.monotonic()
        recent = self._recent_paths.setdefault(channel_id, [])
        recent.append(path)
        del recent[:-10]

    def recently_sent(self, channel_id: int) -> List[Path]:
        return list(self._recent_paths.get(channel_id, []))
