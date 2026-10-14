"""
memory/bookfinder.py — найти книгу, которую ей посоветовали.

Только легальные бесплатные источники (общественное достояние):
  * Викитека (ru.wikisource.org) — русская классика; книгу целиком собирает ws-export в epub;
  * Project Gutenberg (gutenberg.org) — классика на английском и других языках.
Современные книги, комиксы и мангу там не найти — тогда она честно скажет, что не нашла,
и попросит прислать файл.

Найденное сохраняется в библиотеку обычным .txt «Автор - Название.txt».
"""

from __future__ import annotations

import html
import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from memory.library import extract_text
from utils.logger import get_logger

_logger = get_logger("bot.bookfinder")

# Правила Викимедиа для ботов (https://w.wiki/4wJS): в User-Agent нужен контакт и библиотека.
# Контакт — ссылка на профиль бота в Discord (подставляется в BookFinder(contact=…)).
_UA = {"User-Agent": f"AuraBot/1.0 (+https://discord.com; personal Discord bot reading public-domain books) "
                     f"aiohttp/{aiohttp.__version__}"}
_WS_API = "https://{lang}.wikisource.org/w/api.php"
_WS_EXPORT = "https://ws-export.wmcloud.org/?format=epub&lang={lang}&page={page}"
# Какую редакцию брать, если у произведения их несколько (лучшие — первыми).
_WS_EDITION_PREF = ("/ВТ", "ПСС", "Версия 2", "Вариант")


def _words(s: str) -> set:
    return {w.lower()[:5] for w in re.findall(r"[\wё]{3,}", s or "")}


class BookFinder:
    def __init__(self, session: aiohttp.ClientSession, contact: str = "") -> None:
        self._session = session
        if contact:
            global _UA
            _UA = {"User-Agent": f"AuraBot/1.0 (+{contact}; personal Discord bot reading public-domain books) "
                                 f"aiohttp/{aiohttp.__version__}"}

    async def find(self, title: str, author: str, folder: Path) -> Optional[Dict[str, Any]]:
        """
        Найти и сохранить книгу.  → {"path", "source", "title", "author"} или None.
        Кириллица — сначала Викитека, латиница — сначала Gutenberg.
        """
        cyr = bool(re.search("[а-яё]", f"{title} {author}".lower()))
        order = (self._wikisource, self._gutenberg) if cyr else (self._gutenberg, self._wikisource)
        for source in order:
            found = None
            for attempt in (1, 2):  # сайты иногда отвечают таймаутом — вторая попытка
                try:
                    found = await source(title, author)
                    break
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("Book source failed", extra={"data": {
                        "source": source.__name__, "attempt": attempt, "error": repr(exc)[:200]}})
            if found:
                name = re.sub(r'[<>:"/\\|?*]+', "_", f"{found['author']} - {found['title']}" if found["author"]
                              else found["title"]).strip()[:150]
                path = folder / f"{name}.txt"
                n = 1
                while path.exists():
                    path = folder / f"{name} ({n}).txt"
                    n += 1
                path.write_text(found["text"], encoding="utf-8")
                _logger.info("Book found", extra={"data": {"title": found["title"], "source": found["source"],
                                                           "chars": len(found["text"])}})
                return {"path": path, "source": found["source"], "title": found["title"], "author": found["author"]}
        _logger.info("Book not found", extra={"data": {"title": title, "author": author}})
        return None

    # ------------------------------------------------------------------
    # Викитека
    # ------------------------------------------------------------------

    async def _wikisource(self, title: str, author: str, depth: int = 0) -> Optional[Dict[str, Any]]:
        for lang in ("ru", "en"):
            async with self._session.get(_WS_API.format(lang=lang), headers=_UA, timeout=aiohttp.ClientTimeout(total=15),
                                         params={"action": "query", "list": "search", "format": "json",
                                                 "srsearch": f"{title} {author}".strip(), "srlimit": "20",
                                                 "srnamespace": "0"}) as resp:
                if resp.status != 200:
                    continue
                hits = [h["title"] for h in (await resp.json()).get("query", {}).get("search", [])]
            exact = await self._ws_exact(lang, title, author)
            page = self._prefer_edition(exact, hits) if exact else self._pick_ws_page(hits, title, author)
            if not page:
                continue
            text, t2, a2 = await self._ws_export(lang, page)
            if not text:
                continue
            if self._pre_reform(text):
                if depth:
                    return None  # «Въ Мiрѣ Чудесъ» — читать неудобно, ищем другой перевод
                continue
            if depth == 0 and self._looks_like_index(text):
                # Страница-оглавление («Переводы и редакции: … — перевод …»): взять первый перевод
                # в современной орфографии.
                for line in text.splitlines():
                    if " — " in line and not re.search(r"дореформ|орфографи[яи]\s+до", line, re.I):
                        alt = line.split(" — ")[0].strip()
                        if alt and alt.lower() != title.lower():
                            found = await self._wikisource(alt, author, depth=1)
                            if found:
                                return found
                continue
            if len(text) < 1500:
                continue
            a2 = re.sub(r"^автор\s+|\s*\(\d{3,4}\s*[—-]\s*\d{3,4}\)\s*$", "", a2 or author).strip()
            return {"text": text, "title": re.sub(r"\s*\(.*?\)\s*$", "", t2 or page.split("/")[0]).strip(),
                    "author": a2, "source": "Викитека"}
        return None

    async def _ws_exact(self, lang: str, title: str, author: str) -> Optional[str]:
        """Страница «Название (Фамилия)» — так в Викитеке называют произведения.  None — такой нет."""
        surname = (author.split() or [""])[-1]
        if not surname:
            return None
        cand = f"{title[:1].upper()}{title[1:]} ({surname})"
        async with self._session.get(_WS_API.format(lang=lang), headers=_UA, timeout=aiohttp.ClientTimeout(total=15),
                                     params={"action": "query", "titles": cand, "redirects": "1",
                                             "format": "json"}) as resp:
            if resp.status != 200:
                return None
            pages = (await resp.json()).get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if int(pid) > 0:
                return p["title"]
        return None

    async def _ws_export(self, lang: str, page: str) -> tuple:
        """Страница Викитеки целиком (с подстраницами-главами) → (текст, название, автор)."""
        url = _WS_EXPORT.format(lang=lang, page=urllib.parse.quote(page.replace(" ", "_")))
        async with self._session.get(url, headers=_UA, timeout=aiohttp.ClientTimeout(total=420)) as resp:
            if resp.status != 200 or "epub" not in resp.headers.get("Content-Type", ""):
                return "", "", ""
            data = await resp.read()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "book.epub"
            p.write_bytes(data)
            text, t2, a2 = extract_text(p)
        return self._clean_ws(text), t2, a2

    @staticmethod
    def _pre_reform(text: str) -> bool:
        """Дореформенная орфография (ѣ, і, ѳ, ѵ) — такие издания пропускаем."""
        sample = text[:20000]
        return len(re.findall(r"[ѣіѳѵѢІѲѴ]", sample)) > max(20, len(sample) // 400)

    @staticmethod
    def _looks_like_index(text: str) -> bool:
        """Не текст, а список переводов/редакций."""
        return len(text) < 6000 and bool(re.search(
            r"Переводы и редакции|Редакции|Translations|сходными заголовками|may refer to", text)) \
            and len(re.findall(r" — ", text)) >= 2

    @staticmethod
    def _pick_ws_page(hits: List[str], title: str, author: str) -> Optional[str]:
        """Страница самого произведения: название совпадает; из редакций — по _WS_EDITION_PREF."""
        want, by = _words(title), _words(author)
        good = []
        for h in hits:
            main = h.split("/")[0]
            if not want or not want <= _words(main):
                continue
            if by and not (by & _words(h)) and "(" in main:
                continue  # «Хамелеон (Чехов)» — а просили другого автора
            good.append(h)
        if not good:
            return None
        good.sort(key=lambda h: 0 if by & _words(h.split("/")[0]) else 1)  # «… (Достоевский)» — первыми
        return BookFinder._prefer_edition(good[0].split("/")[0], good)

    @staticmethod
    def _prefer_edition(main: str, hits: List[str]) -> str:
        """У произведения несколько редакций («/Версия 2», «/Вариант ПСС») — взять одну, а не все сразу."""
        editions = [h for h in hits if h.startswith(main + "/")]
        for pref in _WS_EDITION_PREF:
            for h in editions:
                if pref in h:
                    return h
        return main

    @staticmethod
    def _clean_ws(text: str) -> str:
        """Убрать шапку Викитеки («автор…», «Экспортировано из Викитеки…») и хвост про лицензию."""
        m = re.search(r"Экспортировано из Викитеки[^\n]*\n|Exported from Wikisource[^\n]*\n", text)
        if m:
            text = text[m.end():]
        cut = re.search(r"\n(?:Об этой электронной книге|About this digital edition)", text)
        if cut:
            text = text[:cut.start()]
        return text.strip()

    # ------------------------------------------------------------------
    # Project Gutenberg
    # ------------------------------------------------------------------

    async def _gutenberg(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        q = urllib.parse.quote_plus(f"{title} {author}".strip())
        async with self._session.get(f"https://www.gutenberg.org/ebooks/search/?query={q}", headers=_UA,
                                     timeout=aiohttp.ClientTimeout(total=20)) as resp:
            page = await resp.text()
        want = _words(title)
        for m in re.finditer(r'<li class="booklink">.*?href="/ebooks/(\d+)".*?<span class="title">(.*?)</span>'
                             r'(?:\s*<span class="subtitle">(.*?)</span>)?', page, re.S):
            book_id, btitle, bauthor = m.group(1), html.unescape(m.group(2)), html.unescape(m.group(3) or "")
            if want and not want <= _words(btitle):
                continue
            async with self._session.get(f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8", headers=_UA,
                                         timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    continue
                text = await resp.text(errors="ignore")
            s = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", text)
            e = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG", text)
            text = text[s.end() if s else 0: e.start() if e else len(text)].strip()
            if len(text) < 1500:
                continue
            return {"text": text, "title": btitle.strip(), "author": bauthor.strip(), "source": "Project Gutenberg"}
        return None
