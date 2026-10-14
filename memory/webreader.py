"""
memory/webreader.py — читать по ссылке, как в браузере.

Ей кидают ссылку на «читалку» (ранобэ, фанфик, манхва, манга, веб-новелла) — она открывает
страницу, находит главное: текст главы или картинки-страницы комикса, а следующую главу —
по ссылке «Далее / Следующая / Next».  Главы подгружаются по ходу чтения (memory/library.py →
ensure_ahead), так что можно читать и длинные онгоинги.

Чего она НЕ делает: не обходит защиту сайтов (проверка «вы не робот», капча, вход, платные
главы).  Такой сайт она честно называет закрытым и просит прислать файл.
"""

from __future__ import annotations

import io
import os
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from utils.logger import get_logger

_logger = get_logger("bot.webreader")

# Страница-заслонка: защита от ботов / капча.
_BLOCKED_RE = re.compile(r"Проверка безопасности|Just a moment|Attention Required|Shields are up|captcha|"
                         r"not a robot|не робот|DDoS-Guard|Checking your browser|cf-browser-verification", re.I)
_NEXT_RE = re.compile(r"\bслед|\bдалее\b|\bвпер[её]д\b|\bnext\b|→|»|›|pg_next", re.I)
_PREV_RE = re.compile(r"\bпред|\bназад\b|\bprev|←|«|‹", re.I)
_CHAPTER_RE = re.compile(r"(?:глава|часть|эпизод|выпуск|chapter|episode|ch\.?|ep\.?|том)\s*(\d+|[IVXLC]+\b)", re.I)
_START_RE = re.compile(r"start reading|read first|first chapter|начать читать|читать с начала|"
                       r"первая глава|к первой главе|читать онлайн", re.I)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _num(s: str) -> int:
    """«12» → 12, «IV» → 4."""
    if s.isdigit():
        return int(s)
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch, 0)
        total, prev = (total - v, prev) if v < prev else (total + v, v)
    return total
_JUNK_IMG_RE = re.compile(r"logo|icon|avatar|banner|sprite|emoji|smile|button|\bads?\b|advert|pixel|counter|"
                          r"placeholder|loading|thumb", re.I)
_SKIP_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "form", "button", "select"}
_CODE_TAGS = {"script", "style", "noscript", "svg"}  # отсюда не берём даже ссылки
_BLOCKS = {"div", "article", "section", "main", "td", "body"}


def _prose_len(text: str) -> int:
    """Сколько в тексте связной прозы: строки от 60 символов (списки названий и меню не в счёт)."""
    return sum(len(line) for line in text.splitlines() if len(line.strip()) >= 60)


class _Page(HTMLParser):
    """Разбор страницы: блоки с текстом (для поиска основного), картинки, ссылки, заголовки."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: List[Dict[str, Any]] = [{"tag": "root", "parts": [], "depth": 0}]
        self.blocks: List[Tuple[int, int, str]] = []  # (длина, глубина, текст)
        self.images: List[str] = []
        self.links: List[Tuple[str, str]] = []  # (href, «подпись»: текст+title+class+rel)
        self.heads: List[str] = []
        self.meta: Dict[str, str] = {}
        self._skip = 0
        self._code = 0
        self._a: Optional[Dict[str, Any]] = None
        self._h: Optional[List[str]] = None
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag in _CODE_TAGS:
            self._code += 1
        if tag == "a" and not self._code:  # ссылки — отовсюду, кроме кода
            self._a = {"href": a.get("href", ""), "sig": " ".join(a.get(k, "") for k in ("title", "class", "rel", "aria-label")),
                       "text": []}
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta" and a.get("property", a.get("name", "")) in ("og:title", "og:site_name", "author"):
            self.meta[a.get("property") or a.get("name")] = a.get("content", "")
        if tag == "link" and "next" in a.get("rel", "").split():
            self.links.append((a.get("href", ""), "rel=next"))
        if self._skip:
            return
        if tag == "img":
            src = a.get("data-url") or a.get("data-src") or a.get("data-original") or a.get("data-lazy-src") or a.get("src", "")
            if src and not src.startswith("data:"):
                self.images.append(src)
        elif tag in ("h1", "h2"):
            self._h = []
        elif tag in ("p", "br", "li"):
            self.stack[-1]["parts"].append("\n")
        if tag in _BLOCKS:
            self.stack.append({"tag": tag, "parts": [], "depth": len(self.stack)})

    def handle_endtag(self, tag: str) -> None:
        if tag in _CODE_TAGS:
            self._code = max(0, self._code - 1)
        if tag == "a" and self._a is not None:
            self.links.append((self._a["href"], " ".join(self._a["text"]) + " " + self._a["sig"]))
            self._a = None
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if self._skip:
            return
        if tag in ("h1", "h2") and self._h is not None:
            self.heads.append(" ".join(self._h).strip())
            self._h = None
        elif tag == "p":
            self.stack[-1]["parts"].append("\n")
        if tag in _BLOCKS and len(self.stack) > 1:
            node = self.stack.pop()
            text = "".join(node["parts"])
            n = len(re.sub(r"\s+", "", text))
            if n:
                self.blocks.append((n, node["depth"], text))
            self.stack[-1]["parts"].append(text)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._a is not None and not self._code:
            self._a["text"].append(data.strip())
        if self._skip:
            return
        self.stack[-1]["parts"].append(data)
        if self._h is not None:
            self._h.append(data.strip())

    def main_text(self) -> str:
        """Основной текст: самый глубокий блок, в котором почти весь текст страницы (меню и комменты — мимо)."""
        if not self.blocks:
            return ""
        top = max(n for n, _, _ in self.blocks)
        best = max((b for b in self.blocks if b[0] >= top * 0.8), key=lambda b: b[1])
        text = re.sub(r"[ \t\xa0]+", " ", best[2])
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", re.sub(r" *\n *", "\n", text))
        return text.strip()


class WebReader:
    def __init__(self, session: aiohttp.ClientSession, contact: str = "") -> None:
        self._session = session
        self._ua = {"User-Agent": f"AuraBot/1.0 (+{contact or 'https://discord.com'}; personal Discord bot, "
                                  f"reads what friends share) aiohttp/{aiohttp.__version__}",
                    "Accept-Language": "ru,en;q=0.8"}

    async def _get_page(self, url: str) -> Tuple[int, str, str]:
        async with self._session.get(url, headers=self._ua, timeout=aiohttp.ClientTimeout(total=30),
                                     allow_redirects=True) as resp:
            return resp.status, await resp.text(errors="ignore"), str(resp.url)

    async def open(self, url: str) -> Dict[str, Any]:
        """
        Открыть ссылку.  → {"ok": True, "title", "author", "kind": "text"|"comic", "start_url", "site"}
        или {"ok": False, "reason": "..."} (почему не получилось — для её честного ответа).
        """
        url = self._webtoon_first_episode(url)
        try:
            status, page, final = await self._get_page(url)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"сайт не открылся ({type(exc).__name__})"}
        p = _Page()
        p.feed(page)
        if self._blocked(p) or status in (401, 429) or (status in (403, 503) and len(p.main_text()) < 2000):
            return {"ok": False, "reason": "сайт закрыт защитой от ботов (проверка «вы не робот») — "
                                           "обходить её нельзя"}
        if status >= 400:
            return {"ok": False, "reason": f"страница не открылась (ошибка {status})"}
        chapter = await self._chapter_from(p, final)
        if not chapter["text"] and not chapter["images"]:
            # Похоже на оглавление — найти первую главу.
            first = self._first_chapter_link(p, final)
            if first and first != final:
                return await self.open(first)
            return {"ok": False, "reason": "не нашла на странице ни текста главы, ни страниц комикса "
                                           "(может, нужен вход или сайт рисует всё скриптами)"}
        site = urllib.parse.urlparse(final).netloc.removeprefix("www.")
        title = self._clean_title(p.meta.get("og:title") or (p.heads[0] if p.heads else "") or p.title, site)
        author = p.meta.get("author", "")
        m = re.match(r"(.+?)\s*\(([^()]{3,40})\)$", title)
        if m and not author:  # «Преступление и наказание (Достоевский)»
            title, author = m.group(1).strip(), m.group(2).strip()
        return {"ok": True, "title": title or site, "author": author,
                "kind": "comic" if chapter["images"] else "text", "start_url": final, "site": site}

    async def read_chapter(self, url: str, kind: str) -> Optional[Dict[str, Any]]:
        """Одна глава: {"text" | "images": [bytes], "next_url", "heading"} или None (не открылась)."""
        try:
            status, page, final = await self._get_page(url)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Chapter fetch failed", extra={"data": {"url": url, "error": str(exc)}})
            return None
        p = _Page()
        p.feed(page)
        if status >= 400 or self._blocked(p):
            _logger.warning("Chapter blocked", extra={"data": {"url": url, "status": status}})
            return None
        ch = await self._chapter_from(p, final, want_images=(kind == "comic"))
        ch["next_url"] = self._next_link(p, final)
        ch["heading"] = p.heads[0] if p.heads else ""
        return ch

    @staticmethod
    def _blocked(p: "_Page") -> bool:
        """Страница-заслонка: в заголовке или в (коротком) видимом тексте — «проверка», «не робот»."""
        visible = p.main_text()
        return bool(_BLOCKED_RE.search(p.title) or (len(visible) < 3000 and _BLOCKED_RE.search(visible)))

    # ------------------------------------------------------------------

    async def _chapter_from(self, p: _Page, url: str, want_images: Optional[bool] = None) -> Dict[str, Any]:
        text = p.main_text()
        prose = _prose_len(text)
        images: List[bytes] = []
        if want_images is not False and (want_images or prose < 1500):
            images = await self._download_images(p.images, url)
        if images and (want_images or len(images) >= 2):
            return {"text": "", "images": images}
        return {"text": text if prose >= 400 else "", "images": []}

    async def _download_images(self, srcs: List[str], page_url: str, limit: int = 60) -> List[bytes]:
        """Картинки-страницы (крупные), по порядку.  Referer — сама страница (иначе хостинги не отдают)."""
        try:
            from PIL import Image
        except ImportError:
            return []
        out: List[bytes] = []
        seen = set()
        for src in srcs:
            full = urllib.parse.urljoin(page_url, src)
            if full in seen or _JUNK_IMG_RE.search(full) or full.lower().split("?")[0].endswith((".svg", ".ico")):
                continue
            seen.add(full)
            try:
                async with self._session.get(full, headers={**self._ua, "Referer": page_url},
                                             timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200 or not resp.headers.get("Content-Type", "").startswith("image"):
                        continue
                    raw = await resp.read()
                w, h = Image.open(io.BytesIO(raw)).size
            except Exception:  # noqa: BLE001
                continue
            if w >= 400 and h >= 300:  # страница/кусок ленты, а не значок
                out.append(raw)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _next_link(p: _Page, url: str) -> Optional[str]:
        host = urllib.parse.urlparse(url).netloc
        here = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        cands: List[Tuple[int, str]] = []
        for href, sig in p.links:
            if not href or href.startswith(("#", "javascript:")):
                continue
            if sig == "rel=next" or (_NEXT_RE.search(sig) and not _PREV_RE.search(sig)):
                full = urllib.parse.urljoin(url, href)
                path = urllib.parse.unquote(urllib.parse.urlparse(full).path)
                if (urllib.parse.urlparse(full).netloc != host or full.split("#")[0] == url.split("#")[0]
                        or re.search(r"/(?:Служебная|Special|Обсуждение|Talk|Участник|User|Файл|File):", path)):
                    continue
                # Ближе к адресу текущей главы — вероятнее та самая «следующая».
                common = len(os.path.commonprefix([here, path]))
                cands.append((common + (1000 if sig == "rel=next" else 0), full))
        if cands:
            return max(cands)[1]
        cur = _CHAPTER_RE.findall(urllib.parse.unquote(url).replace("_", " "))
        if cur:
            want = _num(cur[-1]) + 1
            base = urllib.parse.unquote(url).rsplit("/", 1)[0]
            for href, sig in p.links:
                full = urllib.parse.urljoin(url, href)
                m = _CHAPTER_RE.findall(urllib.parse.unquote(full).replace("_", " ") + " " + sig)
                if (m and _num(m[-1]) == want and urllib.parse.urlparse(full).netloc == host
                        and urllib.parse.unquote(full).rsplit("/", 1)[0] == base):
                    return full
        return None

    @staticmethod
    def _first_chapter_link(p: _Page, url: str) -> Optional[str]:
        """На странице-оглавлении — ссылка на главу с самым маленьким номером."""
        host = urllib.parse.urlparse(url).netloc
        for href, sig in p.links:  # кнопка «Начать читать» — лучше любой догадки
            full = urllib.parse.urljoin(url, href)
            if _START_RE.search(sig) and urllib.parse.urlparse(full).netloc == host:
                return full
        best, num = None, None
        for href, sig in p.links:
            m = _CHAPTER_RE.search(sig)
            full = urllib.parse.urljoin(url, href)
            if m and urllib.parse.urlparse(full).netloc == host:
                n = _num(m.group(1))
                if num is None or n < num:
                    best, num = full, n
        return best

    @staticmethod
    def _webtoon_first_episode(url: str) -> str:
        """WEBTOON: список эпизодов (новые сверху) → сразу первый эпизод."""
        m = re.match(r"(https://(?:www\.|m\.)?webtoons\.com/\w+/[^/]+/[^/]+)/list\?title_no=(\d+)", url)
        if m:
            return f"{m.group(1)}/episode-1/viewer?title_no={m.group(2)}&episode_no=1"
        return url

    @staticmethod
    def _clean_title(title: str, site: str) -> str:
        t = re.sub(r"\s+", " ", title or "").strip()
        t = t.split("/")[0].strip()  # «Книга/Часть I/Глава I»
        t = re.split(r"\s+[|–—-]\s+(?=[^|–—-]*$)", t)[0] if re.search(r"\s[|–—-]\s", t) else t
        t = _CHAPTER_RE.sub("", t)
        return t.strip(" .,:-–—|") or site
