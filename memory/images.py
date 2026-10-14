"""
images.py — память на картинки.

Каждая картинка из чата сохраняется уменьшенной копией (JPEG) в data/images/
вместе с короткой подписью, которую делает сама модель.  Так Аура может
вернуться к картинке через несколько часов: подпись попадает в промпт, а
если о картинке спрашивают — ей снова показывают саму картинку.

Общий размер ограничен VISION_MEMORY_MB: самые старые картинки удаляются.

Любимые картинки: то, что ей по-настоящему понравилось (или что её попросили
сохранить), копируется в отдельную папку (data/favorites) в нормальном качестве.
Их запись в памяти чистка не трогает, а сама папка чистится только по своему
лимиту FAVORITES_MAX_MB — первыми уходят те, что понравились меньше всего.
"""

from __future__ import annotations

import base64
import io
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

_logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zа-яё0-9]{3,}", re.I)
_DM_RE = re.compile(r"\bв\s+(?:лс|личк|личн|дм|dm)", re.I)


class ImageMemory:
    def __init__(self, folder: Path, max_mb: float, side: int = 640,
                 fav_dir: Optional[Path] = None, fav_max_mb: float = 0) -> None:
        self._dir = folder
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = int(max_mb * 1024 * 1024)
        self._fav_dir = fav_dir
        if fav_dir is not None:
            fav_dir.mkdir(parents=True, exist_ok=True)
        self._fav_max_bytes = int(fav_max_mb * 1024 * 1024)
        self._side = side
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self._dir / "index.db"), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS images ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, channel_id INTEGER, author TEXT,"
            " caption TEXT DEFAULT '', files TEXT, is_gif INTEGER, bytes INTEGER)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_images_channel ON images(channel_id, ts)")
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(images)")}
        if "author_id" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN author_id INTEGER DEFAULT 0")
        if "is_dm" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN is_dm INTEGER DEFAULT 0")
        if "favorite" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN favorite INTEGER DEFAULT 0")
            self._db.execute("ALTER TABLE images ADD COLUMN fav_score INTEGER DEFAULT 0")
            self._db.execute("ALTER TABLE images ADD COLUMN fav_file TEXT DEFAULT ''")
        self._db.commit()

    # ------------------------------------------------------------------

    def add(self, channel_id: int, author: str, frames_b64: List[str], is_gif: bool,
            author_id: int = 0, is_dm: bool = False) -> Optional[int]:
        """Сохранить картинку (или кадры гифки).  Возвращает id записи."""
        blobs = [self._shrink(base64.b64decode(f)) for f in frames_b64]
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO images (ts, channel_id, author, files, is_gif, bytes, author_id, is_dm)"
                " VALUES (?, ?, ?, '[]', ?, 0, ?, ?)",
                (time.time(), channel_id, author, int(is_gif), int(author_id or 0), int(is_dm)),
            )
            img_id = cur.lastrowid
            names = []
            for k, blob in enumerate(blobs):
                name = f"{img_id}_{k}.jpg"
                (self._dir / name).write_bytes(blob)
                names.append(name)
            self._db.execute("UPDATE images SET files = ?, bytes = ? WHERE id = ?",
                             (json.dumps(names), sum(len(b) for b in blobs), img_id))
            self._db.commit()
        self._prune()
        return img_id

    def set_caption(self, img_id: int, caption: str) -> None:
        with self._lock:
            self._db.execute("UPDATE images SET caption = ? WHERE id = ?", (caption.strip()[:300], img_id))
            self._db.commit()

    def recent(self, channel_id: int, hours: float, limit: int = 5,
               author_id: int = 0) -> List[Dict[str, Any]]:
        """
        Последние картинки за hours часов, новые первыми: из этого канала
        и (если задан author_id) все, что этот человек сам присылал где угодно, в том числе в личку.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, author, caption, files, is_gif, channel_id, is_dm, favorite FROM images"
                " WHERE (channel_id = ? OR (? != 0 AND author_id = ?)) AND ts >= ? ORDER BY ts DESC LIMIT ?",
                (channel_id, author_id, author_id, time.time() - hours * 3600, limit),
            ).fetchall()
        return [self._row(r, channel_id) for r in rows]

    def favorites(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Любимые картинки (за всё время), новые первыми."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, author, caption, files, is_gif, channel_id, is_dm, favorite FROM images"
                " WHERE favorite = 1 ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._row(r, 0) for r in rows]

    def favorite_files(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Любимые, у которых есть файл в папке любимых: [{caption, path, is_gif}], лучшие первыми."""
        if self._fav_dir is None:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT caption, fav_file, is_gif FROM images WHERE favorite = 1 AND fav_file != ''"
                " ORDER BY fav_score DESC, ts DESC LIMIT ?", (limit,),
            ).fetchall()
        out = []
        for caption, name, is_gif in rows:
            path = self._fav_dir / name
            if path.exists():
                out.append({"caption": caption or "", "path": path, "is_gif": bool(is_gif)})
        return out

    @staticmethod
    def _row(r: Any, channel_id: int) -> Dict[str, Any]:
        return {"id": r[0], "ts": r[1], "author": r[2], "caption": r[3] or "",
                "files": json.loads(r[4] or "[]"), "is_gif": bool(r[5]),
                "here": r[6] == channel_id, "is_dm": bool(r[7]), "favorite": bool(r[8])}

    def is_favorite(self, img_id: int) -> bool:
        with self._lock:
            row = self._db.execute("SELECT favorite FROM images WHERE id = ?", (img_id,)).fetchone()
        return bool(row and row[0])

    def mark_favorite(self, img_id: int, score: int, frames_b64: Optional[List[str]] = None) -> Optional[Path]:
        """
        Сохранить картинку в любимые: копия в папку любимых (кадры в нормальном качестве,
        если переданы, иначе — уменьшенная копия из памяти).  Гифка сохраняется анимированной.
        """
        if self._fav_dir is None:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT ts, author, caption, files, is_gif, favorite, fav_file FROM images WHERE id = ?",
                (img_id,)).fetchone()
        if not row:
            return None
        ts, author, caption, files, is_gif, already, fav_file = row
        if already and fav_file and (self._fav_dir / fav_file).exists():
            with self._lock:
                self._db.execute("UPDATE images SET fav_score = MAX(fav_score, ?) WHERE id = ?", (score, img_id))
                self._db.commit()
            return self._fav_dir / fav_file
        if frames_b64:
            blobs = [base64.b64decode(f) for f in frames_b64]
        else:
            blobs = [(self._dir / n).read_bytes() for n in json.loads(files or "[]") if (self._dir / n).exists()]
        if not blobs:
            return None
        slug = re.sub(r"[^\w]+", "_", (caption or author or "").lower()).strip("_")[:40]
        stem = time.strftime("%Y-%m-%d", time.localtime(ts)) + f"_{img_id}" + (f"_{slug}" if slug else "")
        data, ext = self._encode_fav(blobs, bool(is_gif))
        name = f"{stem}.{ext}"
        (self._fav_dir / name).write_bytes(data)
        with self._lock:
            self._db.execute("UPDATE images SET favorite = 1, fav_score = ?, fav_file = ? WHERE id = ?",
                             (int(score), name, img_id))
            self._db.commit()
        _logger.info("Image saved to favorites", extra={"data": {"id": img_id, "file": name, "score": score}})
        self._prune_favorites()
        return self._fav_dir / name

    def best_match(self, rows: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Картинки, о которых, похоже, речь.  Есть совпадение слов с подписью — одна лучшая;
        нет — две самые свежие (модель сама поймёт, какая из них).
        «в лс / в личке» — сначала картинки из лички.
        """
        if not rows:
            return []
        if _DM_RE.search(text or ""):
            dm = [r for r in rows if r["is_dm"]]
            rows = dm or rows
        words = {w.lower()[:5] for w in _WORD_RE.findall(text or "")}
        best, score = None, 0
        for row in rows:
            cap = {w.lower()[:5] for w in _WORD_RE.findall(row["caption"])}
            s = len(words & cap)
            if s > score:
                best, score = row, s
        return [best] if best else rows[:2]

    def load(self, row: Dict[str, Any]) -> List[str]:
        out = []
        for name in row["files"]:
            try:
                out.append(base64.b64encode((self._dir / name).read_bytes()).decode("ascii"))
            except OSError:
                pass
        return out

    # ------------------------------------------------------------------

    def _shrink(self, raw: bytes) -> bytes:
        try:
            from PIL import Image
        except ImportError:
            return raw
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.thumbnail((self._side, self._side))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        except Exception:  # noqa: BLE001
            return raw

    @staticmethod
    def _encode_fav(blobs: List[bytes], is_gif: bool) -> "tuple[bytes, str]":
        """Одна картинка → JPEG как есть; кадры гифки → анимированный GIF."""
        if len(blobs) == 1 or not is_gif:
            return blobs[0], "jpg"
        try:
            from PIL import Image
            frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
            buf = io.BytesIO()
            frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=350, loop=0)
            return buf.getvalue(), "gif"
        except Exception:  # noqa: BLE001
            return blobs[0], "jpg"

    def _prune_favorites(self) -> None:
        """Папка любимых больше своего лимита — убрать те, что понравились меньше всего (и старее)."""
        if self._fav_dir is None or self._fav_max_bytes <= 0:
            return
        with self._lock:
            sizes = {f.name: f.stat().st_size for f in self._fav_dir.iterdir() if f.is_file()}
            total = sum(sizes.values())
            removed = 0
            while total > self._fav_max_bytes:
                row = self._db.execute(
                    "SELECT id, fav_file FROM images WHERE favorite = 1 AND fav_file != ''"
                    " ORDER BY fav_score ASC, ts ASC LIMIT 1").fetchone()
                if not row:
                    break
                try:
                    (self._fav_dir / row[1]).unlink()
                except OSError:
                    pass
                total -= sizes.get(row[1], 0)
                # из любимых выпала — дальше живёт как обычная картинка в памяти
                self._db.execute("UPDATE images SET favorite = 0, fav_file = '' WHERE id = ?", (row[0],))
                removed += 1
            if removed:
                self._db.commit()
                _logger.info("Favorites pruned", extra={"data": {"removed": removed, "bytes": total}})

    def _prune(self) -> None:
        """Удалять самые старые картинки, пока общий размер больше лимита.  Любимые не трогать."""
        with self._lock:
            total = self._db.execute("SELECT COALESCE(SUM(bytes), 0) FROM images").fetchone()[0]
            removed = 0
            while total > self._max_bytes:
                row = self._db.execute("SELECT id, files, bytes FROM images WHERE favorite = 0"
                                       " ORDER BY ts ASC LIMIT 1").fetchone()
                if not row:
                    break
                for name in json.loads(row[1] or "[]"):
                    try:
                        (self._dir / name).unlink()
                    except OSError:
                        pass
                self._db.execute("DELETE FROM images WHERE id = ?", (row[0],))
                total -= row[2] or 0
                removed += 1
            if removed:
                self._db.commit()
                _logger.info("Image memory pruned", extra={"data": {"removed": removed, "bytes": total}})


def ago_text(ts: float) -> str:
    """«5 минут назад», «2 часа назад»."""
    minutes = max(1, int((time.time() - ts) / 60))
    if minutes < 60:
        return f"{minutes} мин назад"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} ч назад"
    return f"{hours // 24} дн назад"
