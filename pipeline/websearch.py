"""
websearch.py — Аура может загуглить.

Поиск через DuckDuckGo (html-версия, без ключа): заголовки и сниппеты первых
результатов + текст первой страницы, чтобы было из чего ответить по делу.
"""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import Dict, List

import aiohttp

import config
from utils.logger import get_logger

_logger = get_logger("bot.websearch")

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0 Safari/537.36"}
_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_DROP_RE = re.compile(r"<(script|style|noscript|svg|header|footer|nav|aside|form)\b.*?</\1>", re.S | re.I)

# Явная просьба загуглить — ищем без лишних раздумий.
EXPLICIT_RE = re.compile(r"\b(?:за|по|про)?гугл\w*|\bпоищи\b|\bнайди\s+(?:в\s+(?:инете|интернете|гугле|сети))|"
                         r"\bпосмотри\s+в\s+(?:инете|интернете|гугле)", re.I)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub("", s))).strip()


def _real_url(href: str) -> str:
    """DuckDuckGo заворачивает ссылки в //duckduckgo.com/l/?uddg=…"""
    if "uddg=" in href:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        return q.get("uddg", [href])[0]
    return href if href.startswith("http") else "https:" + href


class WebSearch:
    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def search(self, query: str) -> List[Dict[str, str]]:
        """[{title, url, snippet}] — первые результаты; [] при сбое."""
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query, "kl": "ru-ru"})
        try:
            async with self._session.get(url, headers=_UA, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                page = await resp.text()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Search failed", extra={"data": {"query": query, "error": str(exc)}})
            return []
        out = []
        for href, title, snippet in _RESULT_RE.findall(page)[: config.WEB_SEARCH_RESULTS]:
            out.append({"title": _clean(title), "url": _real_url(html.unescape(href)), "snippet": _clean(snippet)})
        return out

    async def page_text(self, url: str, limit: int = 1500) -> str:
        """Основной текст страницы (грубо: без скриптов, меню и тегов), первые limit символов."""
        try:
            async with self._session.get(url, headers=_UA, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if "html" not in resp.headers.get("Content-Type", ""):
                    return ""
                raw = await resp.text(errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
        body = _DROP_RE.sub(" ", raw)
        paras = [_clean(p) for p in re.findall(r"<p\b[^>]*>(.*?)</p>", body, re.S | re.I)]
        text = " ".join(p for p in paras if len(p) > 40) or _clean(body)
        return text[:limit]

    async def lookup(self, query: str) -> str:
        """Готовый блок для промпта: сниппеты + кусок первой страницы.  '' — ничего не нашлось."""
        results = await self.search(query)
        if not results:
            return ""
        lines = [f"- {r['title']}: {r['snippet']}" for r in results]
        first = await self.page_text(results[0]["url"])
        if first:
            lines.append(f"\nСо страницы «{results[0]['title']}»: {first}")
        _logger.info("Web search", extra={"data": {"query": query, "results": len(results),
                                                    "page_chars": len(first)}})
        return "\n".join(lines)
