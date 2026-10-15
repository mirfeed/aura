"""
memory/library.py — её книжная полка.

Книги и комиксы лежат в data/library (LIBRARY_DIR):
  .txt .fb2 .epub .pdf      — книги (читает текст кусками)
  .cbz или папка с картинками — комиксы (смотрит страницы через vision-модель)

Она читает в свободное время (core/bot.py → _maybe_read): кусок за раз, как человек.
После каждого сеанса — короткая запись в дневник чтения (что произошло и что она
об этом думает) и обновлённый пересказ «что было до сих пор».  Дальше места,
где остановилась, она книгу не знает — и не спойлерит.

Текст книги извлекается один раз и кэшируется в data/library/.cache/<id>.txt.

Память о книгах (book_notes) — как у человека:
  * пока читает — заметка на каждый сеанс (что было + что подумала);
  * дочитала — конспект (герои, ключевые моменты, концовка, что запомнилось), подробнее,
    если книга понравилась;
  * и забывание: у любимых книг заметки остаются все, у средних — часть, у скучных —
    только конспект;
  * в разговоре она вспоминает нужное по смыслу (recall), а не держит всё в голове.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import posixpath
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import config
from utils.logger import get_logger

if TYPE_CHECKING:
    import aiosqlite

_logger = get_logger("bot.library")

TEXT_EXTS = {".txt", ".fb2", ".epub", ".pdf"}
COMIC_EXTS = {".cbz"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
BOOK_EXTS = TEXT_EXTS | COMIC_EXTS


def _natural(s: str) -> list:
    """Сортировка «page2» раньше «page10»."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# Разговор о книгах вообще — слабая зацепка, чтобы вспомнить прочитанное.
_BOOK_TALK_RE = re.compile(r"книг|читал|читаешь|прочитал|роман|комикс|манг|сюжет|геро[йиея]|концовк|глав[аеуы]|"
                           r"автор|персонаж\w*\s+(?:из|в)\s+книг", re.I)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Извлечение текста
# ---------------------------------------------------------------------------

class _HtmlText(HTMLParser):
    """HTML → текст с абзацами (для epub)."""

    _BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "section", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in ("img", "image") and not self._skip:
            a = dict(attrs)
            src = a.get("src") or a.get("xlink:href") or a.get("href") or ""
            if src:
                self.parts.append(f"\n[[IMG:{src}]]\n")  # место иллюстрации в тексте
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_to_text(raw: str) -> str:
    p = _HtmlText()
    p.feed(raw)
    return html.unescape("".join(p.parts))


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251", "koi8-r", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\xa0 -   　]", " ", text)  # неразрывные и узкие пробелы
    text = text.replace("​", "").replace("﻿", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


_IMG_MARK_RE = re.compile(r"\[\[IMG:([^\]]+)\]\]")


def _read_fb2(path: Path) -> Tuple[str, str, str, Dict[str, bytes]]:
    root = ET.fromstring(path.read_bytes())  # кодировку берёт из <?xml encoding=…?>
    ns = {"fb": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"fb:{t}") if ns else (lambda t: t)
    title = root.findtext(f".//{q('title-info')}/{q('book-title')}", default="", namespaces=ns) or ""
    first = root.findtext(f".//{q('title-info')}/{q('author')}/{q('first-name')}", default="", namespaces=ns) or ""
    last = root.findtext(f".//{q('title-info')}/{q('author')}/{q('last-name')}", default="", namespaces=ns) or ""
    binaries: Dict[str, bytes] = {}
    for b in root.iter():
        if b.tag.split("}")[-1] == "binary" and b.get("id"):
            try:
                binaries[b.get("id")] = base64.b64decode((b.text or "").strip())
            except ValueError:
                pass

    def href(el: Any) -> str:
        return next((v for k, v in el.attrib.items() if k.endswith("href")), "").lstrip("#")

    parts: List[str] = []
    cover = root.find(f".//{q('title-info')}/{q('coverpage')}/{q('image')}", ns)
    if cover is not None and href(cover):
        parts.append(f"[[IMG:{href(cover)}]]")  # обложку видит первой, как у настоящей книги
    for body in root.findall(q("body"), ns):
        if body.get("name") == "notes":
            continue  # сноски — не читать подряд
        for el in body.iter():
            tag = el.tag.split("}")[-1]
            if tag in ("p", "v", "subtitle", "text-author"):
                parts.append("".join(el.itertext()).strip())
            elif tag == "empty-line":
                parts.append("")
            elif tag == "image" and href(el):
                parts.append(f"[[IMG:{href(el)}]]")
    return "\n".join(parts), title.strip(), f"{first} {last}".strip(), binaries


def _read_epub(path: Path) -> Tuple[str, str, str, Dict[str, bytes]]:
    images: Dict[str, bytes] = {}
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
        opf = ET.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        title = next((el.text or "" for el in opf.iter() if el.tag.endswith("}title")), "")
        author = next((el.text or "" for el in opf.iter() if el.tag.endswith("}creator")), "")
        manifest = {el.get("id"): el.get("href") for el in opf.iter() if el.tag.endswith("}item")}
        spine = [el.get("idref") for el in opf.iter() if el.tag.endswith("}itemref")]
        names = set(z.namelist())
        parts = []
        for idref in spine:
            href = manifest.get(idref)
            if not href:
                continue
            doc = urllib.parse.unquote(base + href)
            try:
                text = _html_to_text(_decode(z.read(doc)))
            except KeyError:
                continue

            def resolve(m: "re.Match") -> str:
                src = urllib.parse.unquote(m.group(1).split("#")[0])
                full = posixpath.normpath(posixpath.join(posixpath.dirname(doc), src))
                if full in names and full not in images:
                    images[full] = z.read(full)
                return f"[[IMG:{full}]]" if full in names else ""

            parts.append(_IMG_MARK_RE.sub(resolve, text))
    return "\n\n".join(parts), title.strip(), author.strip(), images


def _pdf_page_image(page: Any) -> Optional[bytes]:
    """Самая крупная картинка на странице PDF (иллюстрация или скан страницы)."""
    best, size = None, 0
    try:
        for im in page.images:
            w, h = im.image.size
            if w * h > size:
                best, size = im.image, w * h
    except Exception:  # noqa: BLE001  (экзотические форматы картинок)
        return None
    if best is None:
        return None
    buf = io.BytesIO()
    best.convert("RGB").save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _read_pdf(path: Path) -> Tuple[str, str, str, Dict[str, bytes]]:
    from pypdf import PdfReader  # pip install pypdf

    reader = PdfReader(str(path))
    meta = reader.metadata or {}
    parts: List[str] = []
    images: Dict[str, bytes] = {}
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        img = _pdf_page_image(page) if len(text) < 3000 else None  # плотная страница текста — без картинок
        if img:
            images[f"p{i}"] = img
            parts.append(f"[[IMG:p{i}]]")
        parts.append(text)
    return ("\n\n".join(parts), str(getattr(meta, "title", "") or ""), str(getattr(meta, "author", "") or ""),
            images)


def _good_illustration(raw: bytes) -> Optional[bytes]:
    """Иллюстрация, а не виньетка/линия/значок: от 250 px по меньшей стороне, без вытянутых полос.
    → уменьшенный JPEG или None."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if min(w, h) < 250 or max(w, h) / max(1, min(w, h)) > 4:
            return None
        img = img.convert("RGB")
        img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def extract_book(path: Path) -> Tuple[str, str, str, List[Tuple[int, bytes]]]:
    """
    (текст, название, автор, иллюстрации) — иллюстрации как (позиция в тексте, JPEG), по порядку.
    Мелочь (виньетки, разделители) и повторы отсеиваются.
    """
    ext = path.suffix.lower()
    images: Dict[str, bytes] = {}
    if ext == ".fb2":
        text, title, author, images = _read_fb2(path)
    elif ext == ".epub":
        text, title, author, images = _read_epub(path)
    elif ext == ".pdf":
        text, title, author, images = _read_pdf(path)
    else:
        text, title, author = _decode(path.read_bytes()), "", ""
    keys: List[str] = []

    def mark(m: "re.Match") -> str:
        keys.append(m.group(1))
        return f"\ue000{len(keys) - 1}\ue001"

    text = _tidy(_IMG_MARK_RE.sub(mark, text))  # метки → служебные символы, потом обычная чистка
    out: List[str] = []
    illus: List[Tuple[int, bytes]] = []
    seen = set()
    length, last = 0, 0
    for m in re.finditer(r"\ue000(\d+)\ue001", text):
        out.append(text[last:m.start()])
        length += m.start() - last
        last = m.end()
        raw = images.get(keys[int(m.group(1))])
        good = _good_illustration(raw) if raw else None
        if good and hashlib.md5(good).hexdigest() not in seen:
            seen.add(hashlib.md5(good).hexdigest())
            illus.append((length, good))  # точная позиция в итоговом тексте
    out.append(text[last:])
    return "".join(out).strip(), title, author, illus


def extract_text(path: Path) -> Tuple[str, str, str]:
    """(текст, название, автор) — название/автор пустые, если в файле их нет."""
    text, title, author, _ = extract_book(path)
    return text, title, author


def is_scanned_pdf(path: Path) -> bool:
    """PDF без текстового слоя (манга, комикс, скан книги) — читать глазами, как комикс."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = reader.pages[:10]
    text = sum(len(p.extract_text() or "") for p in pages)
    return text < 100 * len(pages) and any(_pdf_page_image(p) for p in pages[:3])


def comic_pages(path: Path) -> List[str]:
    """Страницы комикса по порядку (имена внутри .cbz, файлы в папке или страницы скан-PDF)."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [f"p{i}" for i in range(len(PdfReader(str(path)).pages))]
    if path.is_dir():
        return sorted((f.name for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=_natural)
    with zipfile.ZipFile(path) as z:
        return sorted((n for n in z.namelist() if Path(n).suffix.lower() in IMAGE_EXTS and not n.startswith("__MACOSX")),
                      key=_natural)


def comic_page_bytes(path: Path, name: str) -> bytes:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return _pdf_page_image(PdfReader(str(path)).pages[int(name[1:])]) or b""
    if path.is_dir():
        return (path / name).read_bytes()
    with zipfile.ZipFile(path) as z:
        return z.read(name)


_VOL_RE = re.compile(r"^(.*?)[\s,._:-]*(?:vol(?:ume)?\.?|том|т\.|книга|кн\.|part|часть|ч\.|#)\s*(\d{1,3})\b", re.I)


def series_of(title: str) -> Tuple[str, int]:
    """«Unnamed Memory vol 2 dark» → («unnamed memory», 2); не серия → ("", 0)."""
    m = _VOL_RE.match(title or "")
    if not m or not m.group(1).strip():
        return "", 0
    key = re.sub(r"[^\wё]+", " ", m.group(1).lower()).strip()
    return key, int(m.group(2))


def _title_from_name(path: Path) -> Tuple[str, str]:
    """«Автор - Название.fb2» → (название, автор); иначе имя файла."""
    stem = re.sub(r"[_]+", " ", path.stem if path.is_file() else path.name).strip()
    # «… dark», «(light)» — вариант оформления файла, а не часть названия
    stem = re.sub(r"[\s(\[]+(?:dark|light|night|mobile|print)[)\]]?$", "", stem, flags=re.I).strip()
    m = re.match(r"(.+?)\s+[-–—]\s+(.+)", stem)
    return (m.group(2).strip(), m.group(1).strip()) if m else (stem, "")


# ---------------------------------------------------------------------------
# Полка
# ---------------------------------------------------------------------------

class Library:
    def __init__(self, db_getter: Callable[[], "aiosqlite.Connection"], folder: Path,
                 embed: Optional[Callable[[List[str]], Any]] = None) -> None:
        self._db_getter = db_getter
        self._embed = embed
        # Все заметки с векторами — для вспоминания (перечитывается после записи).
        self._notes: Optional[List[Tuple[int, str, str, str, np.ndarray]]] = None
        self.dir = folder
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = self.dir / ".cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._text_cache: Dict[int, str] = {}
        # Для промпта (синхронно): текущая книга, дневник, прочитанные.
        self._snapshot: Dict[str, Any] = {}

    @property
    def _db(self) -> "aiosqlite.Connection":
        return self._db_getter()

    async def init(self) -> None:
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS books ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT UNIQUE, kind TEXT, title TEXT, author TEXT,"
            " total INTEGER DEFAULT 0, pos INTEGER DEFAULT 0, status TEXT DEFAULT 'unread',"
            " added_by TEXT DEFAULT '', added_at TEXT, started_at TEXT, finished_at TEXT, last_read_at TEXT,"
            " story TEXT DEFAULT '', opinion TEXT DEFAULT '', rating INTEGER DEFAULT 0, hook INTEGER DEFAULT 0)"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS reading_log ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, ts TEXT, pos_from INTEGER, pos_to INTEGER,"
            " summary TEXT, thought TEXT)"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS book_notes ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, kind TEXT, text TEXT, created_at TEXT,"
            " embedding BLOB)"
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_book_notes_book ON book_notes(book_id)")
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS book_images ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, pos INTEGER, file TEXT)"
        )
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_book_images ON book_images(book_id, pos)")
        for col in ("hook INTEGER DEFAULT 0", "url TEXT DEFAULT ''", "next_url TEXT DEFAULT ''",
                    "chapters INTEGER DEFAULT 0"):
            try:  # база из прошлой версии — без этих колонок
                await self._db.execute(f"ALTER TABLE books ADD COLUMN {col}")
            except Exception:  # noqa: BLE001
                pass
        await self._db.commit()
        await self.scan()

    # ------------------------------------------------------------------

    async def scan(self) -> List[str]:
        """Найти новые книги в папке.  Возвращает названия добавленных."""
        added: List[str] = []
        async with self._db.execute("SELECT file FROM books") as cur:
            known = {r["file"] for r in await cur.fetchall()}
        for entry in sorted(self.dir.iterdir(), key=lambda p: _natural(p.name)):
            if entry.name.startswith("."):
                continue
            rel = entry.name
            if rel in known:
                continue
            if entry.is_file() and entry.suffix.lower() in BOOK_EXTS:
                kind = "comic" if entry.suffix.lower() in COMIC_EXTS else "text"
            elif entry.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in entry.iterdir()):
                kind = "comic"
            else:
                continue
            title = await self._register(entry, kind, "")
            if title:
                added.append(title)
        if added:
            await self.refresh()
        return added

    async def _register(self, path: Path, kind: str, added_by: str) -> Optional[str]:
        title, author = _title_from_name(path)
        illus: List[Tuple[int, bytes]] = []
        try:
            if path.suffix.lower() == ".pdf" and is_scanned_pdf(path):
                kind = "comic"  # манга/комикс/скан книги в PDF — читать глазами, постранично
            if kind == "comic":
                total = len(comic_pages(path))
            else:
                text, t2, a2, illus = extract_book(path)
                if len(text) < 200:
                    _logger.warning("Book has no readable text", extra={"data": {"file": path.name}})
                    return None
                title, author, total = t2 or title, a2 or author, len(text)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Book not readable", extra={"data": {"file": path.name, "error": str(exc)}})
            return None
        if total <= 0:
            return None
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO books (file, kind, title, author, total, added_by, added_at) VALUES (?,?,?,?,?,?,?)",
            (path.name, kind, title[:200], author[:120], total, added_by, _now()),
        )
        await self._db.commit()
        if kind == "text" and cur.lastrowid:
            (self._cache_dir / f"{cur.lastrowid}.txt").write_text(text, encoding="utf-8")
            if illus:
                folder = self._cache_dir / f"{cur.lastrowid}_img"
                folder.mkdir(exist_ok=True)
                for n, (pos, raw) in enumerate(illus, 1):
                    (folder / f"{n:04d}.jpg").write_bytes(raw)
                    await self._db.execute("INSERT INTO book_images (book_id, pos, file) VALUES (?,?,?)",
                                           (cur.lastrowid, pos, f"{cur.lastrowid}_img/{n:04d}.jpg"))
                await self._db.commit()
        _logger.info("Book added", extra={"data": {"title": title, "author": author, "kind": kind,
                                                   "total": total, "illustrations": len(illus),
                                                   "added_by": added_by}})
        return title

    async def add_file(self, path: Path, added_by: str) -> Optional[Dict[str, Any]]:
        """Подаренная книга (уже сохранена в папку): зарегистрировать.  → строка книги или None."""
        kind = "comic" if path.suffix.lower() in COMIC_EXTS else "text"  # скан-PDF _register сам сделает комиксом
        title = await self._register(path, kind, added_by)
        if title is None:
            return None
        await self.refresh()
        async with self._db.execute("SELECT * FROM books WHERE file = ?", (path.name,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------

    async def add_web(self, url: str, title: str, author: str, kind: str, added_by: str) -> Dict[str, Any]:
        """Книга/комикс по ссылке: главы будут подгружаться по ходу чтения (ensure_ahead)."""
        cur = await self._db.execute(
            "INSERT INTO books (file, kind, title, author, total, added_by, added_at, url, next_url)"
            " VALUES (?,?,?,?,0,?,?,?,?)",
            (f"web:{url}", kind, title[:200], author[:120], added_by, _now(), url, url))
        bid = cur.lastrowid
        cache = f".cache/{bid}"
        if kind == "comic":
            (self.dir / cache).mkdir(parents=True, exist_ok=True)
            await self._db.execute("UPDATE books SET file = ? WHERE id = ?", (cache, bid))
        else:
            (self._cache_dir / f"{bid}.txt").write_text("", encoding="utf-8")
        await self._db.commit()
        await self.refresh()
        async with self._db.execute("SELECT * FROM books WHERE id = ?", (bid,)) as c:
            return dict(await c.fetchone())

    async def ensure_ahead(self, book: Dict[str, Any], reader: Any, max_chapters: int = 2) -> Dict[str, Any]:
        """
        Книга по ссылке: подгрузить следующие главы, если прочитанное почти кончилось.
        Возвращает обновлённую строку книги.
        """
        if not book.get("url"):
            return book
        for _ in range(max_chapters):
            need = config.COMIC_PAGES_PER_SESSION if book["kind"] == "comic" else config.READING_CHUNK_CHARS
            if book["total"] - book["pos"] >= need or not book.get("next_url"):
                break
            ch = await reader.read_chapter(book["next_url"], book["kind"])
            if ch is None:
                break  # глава не открылась — попробует в следующий раз
            added = 0
            if book["kind"] == "comic":
                folder = self.dir / book["file"]
                start = len(comic_pages(folder))
                for i, raw in enumerate(ch["images"]):
                    (folder / f"{start + i + 1:06d}.jpg").write_bytes(raw)
                added = len(ch["images"])
            elif ch["text"]:
                head = f"\n\n{ch['heading']}\n\n" if ch["heading"] else "\n\n"
                with open(self._cache_dir / f"{book['id']}.txt", "a", encoding="utf-8") as f:
                    f.write(head + ch["text"])
                added = len(head) + len(ch["text"])
                self._text_cache.pop(book["id"], None)
            book["total"] += added
            book["next_url"] = ch["next_url"] or ""
            book["chapters"] = (book.get("chapters") or 0) + 1
            await self._db.execute("UPDATE books SET total = ?, next_url = ?, chapters = ? WHERE id = ?",
                                   (book["total"], book["next_url"], book["chapters"], book["id"]))
            await self._db.commit()
            _logger.info("Web chapter loaded", extra={"data": {"book": book["title"], "chapter": book["chapters"],
                                                               "added": added, "has_next": bool(book["next_url"])}})
            if not added and not book["next_url"]:
                break
        return book

    async def current(self) -> Optional[Dict[str, Any]]:
        """Что она сейчас читает; если ничего — берёт следующую с полки (подаренные — первыми)."""
        async with self._db.execute(
            "SELECT * FROM books WHERE status = 'reading' ORDER BY last_read_at DESC LIMIT 1") as cur:
            row = await cur.fetchone()
        if row is None:
            async with self._db.execute(
                "SELECT * FROM books WHERE status = 'unread'"
                " ORDER BY (added_by != '') DESC, added_at ASC") as cur:
                rows = [dict(r) for r in await cur.fetchall()]
            if not rows:
                return None
            # Серия: начатую серию — продолжать следующим томом; тома — только по порядку.
            async with self._db.execute("SELECT title FROM books WHERE status = 'finished'"
                                        " ORDER BY finished_at DESC LIMIT 1") as cur:
                last = await cur.fetchone()
            last_series = series_of(last["title"])[0] if last else ""
            unread_vols: Dict[str, int] = {}
            for r in rows:
                key, vol = series_of(r["title"])
                if key:
                    unread_vols[key] = min(unread_vols.get(key, vol), vol)

            def ok(r: Dict[str, Any]) -> bool:  # не перепрыгивать через непрочитанный предыдущий том
                key, vol = series_of(r["title"])
                return not key or vol == unread_vols[key]

            candidates = [r for r in rows if ok(r)] or rows
            cont = [r for r in candidates if last_series and series_of(r["title"])[0] == last_series]
            row = (cont or candidates)[0]
            await self._db.execute("UPDATE books SET status = 'reading', started_at = ? WHERE id = ?",
                                   (_now(), row["id"]))
            await self._db.commit()
            await self.refresh()
            return row
        return dict(row)

    async def previous_volume(self, book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Предыдущий прочитанный том той же серии (чтобы помнить, что было раньше)."""
        key, vol = series_of(book.get("title", ""))
        if not key:
            return None
        best = None
        async with self._db.execute("SELECT * FROM books WHERE status = 'finished' AND id != ?", (book["id"],)) as cur:
            for r in await cur.fetchall():
                k, v = series_of(r["title"])
                if k == key and v < vol and (best is None or v > best[0]):
                    best = (v, dict(r))
        return best[1] if best else None

    async def switch_to(self, book_id: int) -> None:
        """Отложить текущую (место запоминается) и взяться за эту."""
        await self._db.execute("UPDATE books SET status = 'unread' WHERE status = 'reading'")
        await self._db.execute(
            "UPDATE books SET status = 'reading', started_at = COALESCE(started_at, ?), last_read_at = ? WHERE id = ?",
            (_now(), _now(), book_id))
        await self._db.commit()
        await self.refresh()

    async def find(self, text: str) -> Optional[Dict[str, Any]]:
        """Книга, о которой речь в тексте (по словам названия/автора)."""
        words = {w.lower()[:5] for w in re.findall(r"[\wё]{3,}", text or "")}
        best, score = None, 0
        async with self._db.execute("SELECT * FROM books WHERE status != 'finished'") as cur:
            rows = await cur.fetchall()
        for r in rows:
            have = {w.lower()[:5] for w in re.findall(r"[\wё]{3,}", f"{r['title']} {r['author']}")}
            s = len(words & have)
            if s > score:
                best, score = dict(r), s
        return best

    async def titles(self, limit: int = 15) -> List[str]:
        async with self._db.execute(
            "SELECT title, author, status FROM books ORDER BY added_at DESC LIMIT ?", (limit,)) as cur:
            return [f"«{r['title']}»" + (f" ({r['author']})" if r["author"] else "")
                    + {"finished": " — прочитана", "reading": " — читаешь"}.get(r["status"], "")
                    for r in await cur.fetchall()]

    def text_chunk(self, book: Dict[str, Any]) -> Tuple[str, int]:
        """Следующий кусок текста и позиция конца (по границе абзаца или предложения)."""
        text = self._text_cache.get(book["id"])
        if text is None:
            text = (self._cache_dir / f"{book['id']}.txt").read_text(encoding="utf-8")
            self._text_cache = {book["id"]: text}  # держим в памяти одну книгу
        start = book["pos"]
        end = min(len(text), start + config.READING_CHUNK_CHARS)
        if end < len(text):
            cut = text.rfind("\n", start + config.READING_CHUNK_CHARS // 2, end)
            if cut == -1:
                cut = max(text.rfind(". ", start, end), text.rfind("! ", start, end), text.rfind("? ", start, end))
                cut = cut + 1 if cut > start else end
            end = cut
        return text[start:end].strip(), end

    async def images_between(self, book_id: int, start: int, end: int, limit: int = 5) -> List[bytes]:
        """Иллюстрации, которые стоят в этом куске текста (по порядку, не больше limit)."""
        async with self._db.execute(
            "SELECT file FROM book_images WHERE book_id = ? AND pos >= ? AND pos < ? ORDER BY pos LIMIT ?",
            (book_id, start, end, limit)) as cur:
            files = [r["file"] for r in await cur.fetchall()]
        out = []
        for f in files:
            try:
                out.append((self._cache_dir / f).read_bytes())
            except OSError:
                pass
        return out

    def comic_chunk(self, book: Dict[str, Any]) -> Tuple[List[bytes], int]:
        path = self.dir / book["file"]
        pages = comic_pages(path)
        names = pages[book["pos"]: book["pos"] + config.COMIC_PAGES_PER_SESSION]
        return [comic_page_bytes(path, n) for n in names], book["pos"] + len(names)

    async def save_session(self, book: Dict[str, Any], pos_to: int, summary: str, thought: str, story: str,
                           hook: int = 0) -> bool:
        """Записать сеанс чтения.  True — книга дочитана."""
        now = _now()
        await self._db.execute(
            "INSERT INTO reading_log (book_id, ts, pos_from, pos_to, summary, thought) VALUES (?,?,?,?,?,?)",
            (book["id"], now, book["pos"], pos_to, summary[:600], thought[:300]))
        finished = pos_to >= book["total"] and not book.get("next_url")
        if summary:
            frac = pos_to / max(1, book["total"])
            where = "начало" if frac < 0.34 else "середина" if frac < 0.67 else "ближе к концу"
            await self._add_note(book["id"], "chunk",
                                 f"({where}) {summary}" + (f" Ты подумала: {thought}" if thought else ""))
        await self._db.execute(
            "UPDATE books SET pos = ?, story = ?, last_read_at = ?, status = ?, finished_at = ?, hook = ? WHERE id = ?",
            (pos_to, (story or book["story"])[:1500], now, "finished" if finished else "reading",
             now if finished else None, int(hook or 0), book["id"]))
        await self._db.commit()
        await self.refresh()
        return finished

    async def add_note(self, book_id: int, kind: str, text: str) -> None:
        await self._add_note(book_id, kind, text)

    async def _add_note(self, book_id: int, kind: str, text: str) -> None:
        vec = None
        if self._embed is not None:
            try:
                vec = np.asarray((await self._embed([text]))[0], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("Note embedding failed", extra={"data": {"error": str(exc)}})
        await self._db.execute(
            "INSERT INTO book_notes (book_id, kind, text, created_at, embedding) VALUES (?,?,?,?,?)",
            (book_id, kind, text[:700], _now(), vec.tobytes() if vec is not None else None))
        await self._db.commit()
        self._notes = None

    async def store_konspekt(self, book: Dict[str, Any], data: Dict[str, Any], rating: int) -> int:
        """
        Конспект дочитанной книги + забывание мелочей.  Чем больше понравилась (rating), тем больше
        заметок по ходу чтения она помнит.  Возвращает, сколько записей конспекта сохранено.
        """
        n = 0
        bid = book["id"]
        if data.get("gist"):
            await self._add_note(bid, "konspekt", f"О чём книга: {data['gist']}")
            n += 1
        for c in data.get("characters") or []:
            if c.get("name"):
                await self._add_note(bid, "character", f"Герой {c['name']} — {c.get('who', '')}")
                n += 1
        for m in data.get("moments") or []:
            if m:
                await self._add_note(bid, "moment", f"Запомнилось: {m}")
                n += 1
        if data.get("ending"):
            await self._add_note(bid, "ending", f"Чем кончилось: {data['ending']}")
            n += 1
        if data.get("feeling"):
            await self._add_note(bid, "feeling", f"Что ты чувствовала, читая: {data['feeling']}")
            n += 1
        # Забывание: заметки по ходу чтения — у любимых все, у средних часть, у скучных ни одной.
        async with self._db.execute(
            "SELECT id FROM book_notes WHERE book_id = ? AND kind = 'chunk' ORDER BY id", (bid,)) as cur:
            chunks = [r["id"] for r in await cur.fetchall()]
        keep_every = 1 if rating >= 8 else 2 if rating >= 5 else 0
        drop = [cid for i, cid in enumerate(chunks) if not keep_every or i % keep_every]
        if drop:
            await self._db.execute(f"DELETE FROM book_notes WHERE id IN ({','.join('?' * len(drop))})", drop)
            await self._db.commit()
            self._notes = None
        _logger.info("Book konspekt", extra={"data": {"book": book["title"], "rating": rating, "notes": n,
                                                      "chunk_notes_kept": len(chunks) - len(drop)}})
        return n

    async def notes_for(self, book_id: int, kind: str = "chunk", limit: int = 400) -> List[str]:
        async with self._db.execute(
            "SELECT text FROM book_notes WHERE book_id = ? AND kind = ? ORDER BY id LIMIT ?",
            (book_id, kind, limit)) as cur:
            return [r["text"] for r in await cur.fetchall()]

    async def recall(self, query: str, context: str = "", limit: int = 5) -> List[str]:
        """
        Вспомнить из прочитанного то, что относится к разговору.  Как у человека — нужна зацепка:
        назвали книгу, автора или героя (сильная), заговорили о книгах вообще (слабая); без зацепки
        вспоминается только то, что очень похоже по смыслу.  context — пара предыдущих реплик:
        «а что там было с брошью?» посреди разговора о книге — тоже про неё.  [] — ничего не вспомнилось.
        """
        if self._embed is None or len((query or "").strip()) < 8:
            return []
        if self._notes is None:
            notes = []
            async with self._db.execute(
                "SELECT n.id, n.kind, n.text, n.embedding, b.title, b.author FROM book_notes n"
                " JOIN books b ON b.id = n.book_id WHERE n.embedding IS NOT NULL") as cur:
                for r in await cur.fetchall():
                    notes.append((r["id"], r["kind"], r["text"], f"{r['title']}  {r['author']}",
                                  np.frombuffer(r["embedding"], dtype=np.float32)))
            self._notes = notes
        if not self._notes:
            return []
        q = np.asarray((await self._embed([query]))[0], dtype=np.float32)
        talk = f"{context}\n{query}"
        words = {w.lower()[:5] for w in re.findall(r"[\wё]{4,}", talk)}
        book_talk = bool(_BOOK_TALK_RE.search(talk))
        # Чем книгу можно «назвать»: слова названия, автора и имена героев из конспекта.
        hooks: Dict[str, set] = {}
        for _id, kind, text, book, vec in self._notes:
            h = hooks.setdefault(book, {w.lower()[:5] for w in re.findall(r"[\wё]{4,}", book)})
            m = re.match(r"Герой (\S+)", text) if kind == "character" else None
            if m:
                h.add(m.group(1).lower()[:5])
        scored = []
        for _id, kind, text, book, vec in self._notes:
            if vec.shape != q.shape:
                continue
            s = float(np.dot(q, vec))
            named = bool(words & hooks[book])
            need = 0.30 if named else 0.40 if book_talk else 0.62
            if s >= need:
                scored.append((s + (0.25 if named else 0.0), f"«{book.split('  ')[0].strip()}»: {text}", named, kind))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [line for _, line, _, _ in scored[:limit]]

    async def set_verdict(self, book_id: int, opinion: str, rating: int) -> None:
        await self._db.execute("UPDATE books SET opinion = ?, rating = ? WHERE id = ?",
                               (opinion[:400], max(1, min(10, int(rating or 0))) if rating else 0, book_id))
        await self._db.commit()
        await self.refresh()

    async def last_session_at(self) -> Optional[datetime]:
        async with self._db.execute("SELECT MAX(ts) AS t FROM reading_log") as cur:
            row = await cur.fetchone()
        return datetime.fromisoformat(row["t"]) if row and row["t"] else None

    # ------------------------------------------------------------------
    # Для промпта
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        """Обновить снимок для промпта (промпт строится синхронно)."""
        snap: Dict[str, Any] = {}
        async with self._db.execute(
            "SELECT * FROM books WHERE status = 'reading' ORDER BY last_read_at DESC LIMIT 1") as cur:
            book = await cur.fetchone()
        if book is not None:
            snap["book"] = dict(book)
            prev = await self.previous_volume(dict(book))
            snap["prev"] = prev
            async with self._db.execute(
                "SELECT summary, thought, ts FROM reading_log WHERE book_id = ? ORDER BY id DESC LIMIT 2",
                (book["id"],)) as cur:
                snap["recent"] = [dict(r) for r in reversed(await cur.fetchall())]
        async with self._db.execute(
            "SELECT title, author, opinion, rating FROM books WHERE status = 'finished'"
            " ORDER BY finished_at DESC LIMIT 4") as cur:
            snap["finished"] = [dict(r) for r in await cur.fetchall()]
        async with self._db.execute("SELECT COUNT(*) AS n FROM books WHERE status = 'unread'") as cur:
            snap["unread"] = (await cur.fetchone())["n"]
        self._snapshot = snap

    def mentions_my_book(self, text: str) -> bool:
        """
        В тексте — название или герой книги, которую она читает или прочитала?
        Имена героев — слова с большой буквы из её пересказа сюжета (не в начале предложения).
        """
        snap = self._snapshot
        books = ([snap["book"]] if snap.get("book") else []) + list(snap.get("finished", []))
        keys: set = set()
        for b in books:
            keys |= {w.lower()[:5] for w in re.findall(r"[\wё]{4,}", b.get("title") or "")}
            story = b.get("story") or ""
            keys |= {m.group(1).lower()[:5] for m in re.finditer(r"(?<=[\wё,] )([A-ZА-ЯЁ][a-zа-яё]{3,})", story)}
        keys -= {"книга", "глава", "часть", "volum", "dark"}
        words = {w.lower()[:5] for w in re.findall(r"[\wё]{4,}", text or "")}
        return bool(words & keys)

    def prompt_block(self) -> str:
        snap = self._snapshot
        lines: List[str] = []
        book = snap.get("book")
        if book:
            frac = book["pos"] / max(1, book["total"])
            where = ("только начала" if frac < 0.08 else "прочитала совсем немного" if frac < 0.2 else
                     "прочитала где-то треть" if frac < 0.4 else "дошла примерно до середины" if frac < 0.6 else
                     "прочитала больше половины" if frac < 0.85 else "почти дочитала")
            what = "комикс" if book["kind"] == "comic" else "книгу"
            by = f" ({book['author']})" if book["author"] else ""
            gift = f", её тебе подарил(а) {book['added_by']}" if book["added_by"] else ""
            if book.get("url"):
                gift = (f", читаешь онлайн по ссылке от {book['added_by']}" if book["added_by"] else ", читаешь онлайн")
                gift += f", прочитала глав: {book.get('chapters') or 0}"
            lines.append(f"Ты сейчас читаешь {what} «{book['title']}»{by}{gift} — {where}. "
                         "О том, сколько прочитала, говори как человек («только начала», «на середине»), "
                         "без процентов и цифр.")
            prev = snap.get("prev")
            if prev:
                lines.append(f"Это продолжение «{prev['title']}», которую ты уже прочитала. Что было там: "
                             f"{prev['story'][:900]}")
            if book["story"]:
                lines.append(f"Что там было до сих пор: {book['story']}")
            for r in snap.get("recent", []):
                lines.append(f"Недавно прочитала: {r['summary']} Ты подумала: {r['thought']}")
            lines.append("Дальше этого места ты книгу НЕ знаешь — не выдумывай продолжение и не спойли; "
                         "о книге говори к месту, как человек, который её сейчас читает. Даже если тебе кажется, "
                         "что ты что-то знаешь об этой истории (что будет дальше, другие тома, аниме, что пишут в "
                         "интернете), — ты этого не читала: говори только о том, что есть в пересказе и заметках выше. "
                         "Спросят про то, до чего не дочитала, — «не знаю, я ещё не дошла».")
        for f in snap.get("finished", []):
            by = f" ({f['author']})" if f["author"] else ""
            verdict = f" — {f['opinion']}" if f["opinion"] else ""
            score = f" [{f['rating']}/10]" if f["rating"] else ""
            lines.append(f"Прочитала: «{f['title']}»{by}{score}{verdict}")
        if snap.get("unread"):
            lines.append(f"На полке ещё {snap['unread']} непрочитанных.")
        return ("Чтение — твоё любимое занятие:\n" + "\n".join(lines)) if lines else ""

    def prompt_brief(self) -> str:
        """Коротко, когда разговор не о книгах: что читает — и не тащить это в ответ."""
        snap = self._snapshot
        book = snap.get("book")
        titles = [f"«{f['title']}»" for f in snap.get("finished", [])]
        if not book and not titles:
            return ""
        parts = []
        if book:
            parts.append(f"Ты сейчас читаешь «{book['title']}».")
        if titles:
            parts.append("Уже прочитала: " + ", ".join(titles) + ".")
        return (" ".join(parts) + " Сейчас разговор не о книгах — сама о них не заговаривай, не вставляй в ответ "
                "героев и сюжет и не оправдывайся тем, что зачиталась. Спросят — тогда расскажешь.")
