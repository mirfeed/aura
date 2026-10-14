"""
memory/store.py — persistent memory system for the Discord companion bot.

Two storage backends work together:

SQLite (via aiosqlite)
    ``events`` table  : every message seen, for auditing and relationship stats.
    ``relationships`` table : per-author interaction statistics used to compute
    a relationship score that modulates reply probability.

ChromaDB (synchronous client, always run via executor)
    Semantic vector store.  Each stored event is embedded with
    SentenceTransformer(all-MiniLM-L6-v2) and later retrieved by cosine
    similarity so the bot can recall relevant past exchanges.

Threading rules
---------------
* aiosqlite is async-native — call directly from async methods.
* SentenceTransformer.encode() and all ChromaDB calls are synchronous and
  CPU-bound; every such call is dispatched through
  ``loop.run_in_executor(None, ...)`` so it never stalls the event loop.

Failure safety
--------------
* recall(), log_event(), and get_relationship_score() each wrap their
  entire body in try/except Exception.  Memory failures never crash the bot.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import aiosqlite
import chromadb
from sentence_transformers import SentenceTransformer

import config
from memory.social import SocialMemory, ago
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.message_handler import MessageContext

_logger = get_logger("bot.memory")

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_EVENTS: str = """
CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id        TEXT    NOT NULL,
    author_name      TEXT    NOT NULL,
    channel_id       INTEGER NOT NULL,
    guild_id         INTEGER,
    content_summary  TEXT    NOT NULL,
    full_content     TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    relevance_score  REAL    DEFAULT 0.0,
    bot_replied      INTEGER DEFAULT 0
)
"""

_CREATE_RELATIONSHIPS: str = """
CREATE TABLE IF NOT EXISTS relationships (
    author_id         TEXT PRIMARY KEY,
    author_name       TEXT    NOT NULL,
    interaction_count INTEGER DEFAULT 0,
    replied_count     INTEGER DEFAULT 0,
    last_seen         TEXT    NOT NULL,
    first_seen        TEXT    NOT NULL
)
"""

_INSERT_EVENT: str = """
INSERT INTO events
    (author_id, author_name, channel_id, guild_id,
     content_summary, full_content, timestamp, relevance_score, bot_replied)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RELATIONSHIP: str = """
SELECT interaction_count, replied_count, first_seen, last_seen
FROM relationships WHERE author_id = ?
"""

_INSERT_RELATIONSHIP: str = """
INSERT INTO relationships
    (author_id, author_name, interaction_count, replied_count, last_seen, first_seen)
VALUES (?, ?, 1, ?, ?, ?)
"""

_UPDATE_RELATIONSHIP: str = """
UPDATE relationships
SET author_name       = ?,
    interaction_count = interaction_count + 1,
    replied_count     = replied_count + ?,
    last_seen         = ?
WHERE author_id = ?
"""


class MemoryStore:
    """
    Persistent memory combining SQLite event log and ChromaDB vector store.

    Instantiate once in ``main.py``.  Immediately after construction call
    ``await memory.init_db()`` to open the async database connection and
    create tables.  Call ``await memory.close()`` before process exit.

    Parameters
    ----------
    None — all settings are read from ``config``.
    """

    def __init__(self) -> None:
        """
        Create data directories, load the embedding model, and open the
        ChromaDB client.

        The aiosqlite connection is deliberately NOT opened here because
        ``__init__`` is synchronous and cannot ``await``.  Call
        ``await init_db()`` immediately after construction.
        """
        # Ensure directories exist before anything tries to write to them.
        config.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        _logger.info(
            "Loading sentence embedding model",
            extra={"data": {"model": config.EMBEDDING_MODEL}},
        )
        # Blocking — acceptable here because __init__ runs before the bot
        # connects to Discord and before any async events fire.
        self._embedder: SentenceTransformer = SentenceTransformer(
            config.EMBEDDING_MODEL
        )

        # ChromaDB persistent client (synchronous).
        self._chroma: chromadb.PersistentClient = chromadb.PersistentClient(
            path=str(config.CHROMA_PERSIST_DIR)
        )
        # Своя коллекция на каждую модель эмбеддингов: векторы разных моделей
        # несовместимы.  При смене модели старые воспоминания переносятся.
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", config.EMBEDDING_MODEL.split("/")[-1]).strip("_").lower()
        coll_name = "bot_memory" if slug == "all_minilm_l6_v2" else f"bot_memory_{slug}"[:60]
        self._collection = self._chroma.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )
        if coll_name != "bot_memory" and self._collection.count() == 0:
            self._migrate_from("bot_memory")

        # Set in init_db(); None until then.
        self._db: Optional[aiosqlite.Connection] = None
        self.social: SocialMemory = SocialMemory(lambda: self._db, self.embed)

        _logger.info(
            "MemoryStore ready (SQLite not yet open — call init_db())",
            extra={
                "data": {
                    "chroma_dir": str(config.CHROMA_PERSIST_DIR),
                    "sqlite_path": str(config.SQLITE_DB_PATH),
                    "collection_docs": self._collection.count(),
                }
            },
        )

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """
        Open the aiosqlite connection and create tables if absent.

        Must be called (and awaited) from ``main.py`` immediately after
        constructing MemoryStore, before the bot starts handling messages.
        """
        self._db = await aiosqlite.connect(config.SQLITE_DB_PATH)
        self._db.row_factory = aiosqlite.Row

        async with self._db.cursor() as cur:
            await cur.execute(_CREATE_EVENTS)
            await cur.execute(_CREATE_RELATIONSHIPS)
        await self._db.commit()

        _logger.info(
            "SQLite tables ready",
            extra={"data": {"path": str(config.SQLITE_DB_PATH)}},
        )
        await self.social.init()

    # ------------------------------------------------------------------
    # Embeddings / migration
    # ------------------------------------------------------------------

    async def embed(self, texts: List[str]):
        """Нормализованные эмбеддинги (через executor — encode блокирующий)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._embedder.encode(texts, normalize_embeddings=True)
        )

    def _migrate_from(self, old_name: str) -> None:
        """Переэмбеддить воспоминания из старой коллекции новой моделью."""
        try:
            old = self._chroma.get_collection(old_name)
        except Exception:  # noqa: BLE001
            return
        total = old.count()
        if total == 0:
            return
        _logger.info("Migrating memories to new embedding model",
                     extra={"data": {"from": old_name, "count": total, "model": config.EMBEDDING_MODEL}})
        offset = 0
        while offset < total:
            batch = old.get(limit=256, offset=offset, include=["documents", "metadatas"])
            ids, docs, metas = batch["ids"], batch["documents"], batch["metadatas"]
            if not ids:
                break
            vecs = self._embedder.encode(list(docs), normalize_embeddings=True)
            self._collection.add(ids=ids, documents=docs, metadatas=metas,
                                 embeddings=[v.tolist() for v in vecs])
            offset += len(ids)
        _logger.info("Migration done", extra={"data": {"count": self._collection.count()}})

    async def log_bot_reply(self, context: "MessageContext", response: str) -> None:
        """Запомнить её собственный ответ (в ленту разговора и в векторную память)."""
        try:
            await self.social.record_line(
                context.channel_id, context.guild_id, "bot", config.BOT_NAME, response,
                is_bot=True, is_voice=context.is_voice, reply_to=context.author_display_name,
            )
            vecs = await self.embed([response])
            ts = datetime.now(tz=timezone.utc).isoformat()
            meta = {
                "author_id": "bot",
                "author_name": config.BOT_NAME,
                "to_author_id": context.author_id,
                "to_author_name": context.author_display_name,
                "channel_id": str(context.channel_id),
                "timestamp": ts,
                "bot_replied": 1,
            }
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._collection.add(
                ids=[f"bot_{context.message_id}_{int(datetime.now().timestamp() * 1000)}"],
                documents=[response[:300]],
                embeddings=[vecs[0].tolist()],
                metadatas=[meta],
            ))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("log_bot_reply failed", extra={"data": {"error": str(exc)}})

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def log_event(
        self,
        context: "MessageContext",
        bot_replied: bool,
    ) -> None:
        """
        Persist a message event to SQLite and add its embedding to ChromaDB.

        Safe to call without ``await``-ing the result; failures are caught
        and logged so the bot never crashes due to a storage error.

        Parameters
        ----------
        context:
            Fully populated message context for the incoming message.
        bot_replied:
            True if the bot sent a reply.
        """
        try:
            await self._log_event_unsafe(context, bot_replied)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "log_event failed — event not persisted",
                extra={"data": {"error": str(exc)}},
            )

    async def note_incoming(self, context: "MessageContext") -> None:
        """Записать входящую реплику в ленту разговора (до ответа — чтобы порядок был верный)."""
        try:
            await self.social.touch_person(
                context.author_id, context.author_display_name, getattr(context, "raw_author_id", 0)
            )
            text = context.content
            if context.attachments and not text:
                text = "[прислал(а) вложение]"
            await self.social.record_line(
                context.channel_id, context.guild_id, context.author_id, context.author_display_name, text,
                raw_id=getattr(context, "raw_author_id", 0), is_voice=context.is_voice,
                reply_to=getattr(context, "reply_to_name", None),
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("social log failed", extra={"data": {"error": str(exc)}})

    async def recall(self, context: "MessageContext") -> List[str]:
        """
        Retrieve semantically similar past messages for prompt injection.

        Embeds the current message content and queries ChromaDB for the
        top-K most similar past messages from the same author.

        Parameters
        ----------
        context:
            Current message context.

        Returns
        -------
        List[str]
            Formatted snippet strings such as
            ``"Alice (previously): did you see the game last night"``.
            Returns an empty list on any failure or when the collection is
            empty.
        """
        try:
            return await self._recall_unsafe(context)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "recall failed — returning empty memory",
                extra={"data": {"error": str(exc)}},
            )
            return []

    async def get_relationship_score(self, author_id: str) -> float:
        """
        Compute a 0.0–1.0 relationship score for *author_id*.

        Higher scores indicate more frequent, replied-to interactions.
        A recency decay factor reduces the score for authors not seen
        recently relative to ``config.RELATIONSHIP_DECAY_DAYS``.

        Parameters
        ----------
        author_id:
            The anonymised (hashed) author ID stored in MessageContext.

        Returns
        -------
        float
            Clamped score in [0.0, 1.0]; 0.0 if unknown or on error.
        """
        try:
            return await self._get_relationship_score_unsafe(author_id)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "get_relationship_score failed — returning 0.0",
                extra={"data": {"error": str(exc)}},
            )
            return 0.0

    async def close(self) -> None:
        """
        Close the aiosqlite connection cleanly.

        ChromaDB's PersistentClient flushes automatically; no explicit
        close is needed or available.
        """
        if self._db is not None:
            await self._db.close()
            self._db = None
        _logger.info("MemoryStore closed", extra={"data": {}})

    # ------------------------------------------------------------------
    # Private implementation — no exception handling, callers wrap
    # ------------------------------------------------------------------

    async def _log_event_unsafe(
        self,
        context: "MessageContext",
        bot_replied: bool,
    ) -> None:
        """Core log_event logic; lets exceptions propagate to log_event."""
        assert self._db is not None, "init_db() was not awaited before log_event()"

        timestamp: str = datetime.now(tz=timezone.utc).isoformat()
        content_summary: str = context.content[:300]
        guild_id: Optional[int] = context.guild_id

        # --- SQLite: insert event ---
        async with self._db.cursor() as cur:
            await cur.execute(
                _INSERT_EVENT,
                (
                    context.author_id,
                    context.author_display_name,
                    context.channel_id,
                    guild_id,
                    content_summary,
                    context.content,
                    timestamp,
                    context.relevance_score,
                    int(bot_replied),
                ),
            )

            # --- SQLite: upsert relationship ---
            async with self._db.execute(
                _SELECT_RELATIONSHIP, (context.author_id,)
            ) as sel:
                existing = await sel.fetchone()

            if existing is None:
                await cur.execute(
                    _INSERT_RELATIONSHIP,
                    (
                        context.author_id,
                        context.author_display_name,
                        int(bot_replied),   # replied_count
                        timestamp,          # last_seen
                        timestamp,          # first_seen
                    ),
                )
            else:
                await cur.execute(
                    _UPDATE_RELATIONSHIP,
                    (
                        context.author_display_name,
                        int(bot_replied),   # replied_count delta
                        timestamp,          # last_seen
                        context.author_id,
                    ),
                )

        await self._db.commit()

        # --- ChromaDB: embed and store (both blocking → executor) ---
        loop = asyncio.get_event_loop()

        if len(context.content.strip()) < 4:
            return  # «ок», «ага» — в векторной памяти бесполезны
        raw_embedding = await self.embed([context.content])
        embedding: List[float] = raw_embedding[0].tolist()

        chroma_id: str = f"{context.author_id}_{context.message_id}"
        metadata: Dict[str, object] = {
            "author_id": context.author_id,
            "author_name": context.author_display_name,
            "channel_id": str(context.channel_id),
            "timestamp": timestamp,
            "bot_replied": int(bot_replied),
        }

        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=[chroma_id],
                documents=[content_summary],
                embeddings=[embedding],
                metadatas=[metadata],
            ),
        )

        _logger.debug(
            "Event logged to SQLite + ChromaDB",
            extra={
                "data": {
                    "chroma_id": chroma_id,
                    "bot_replied": bot_replied,
                }
            },
        )

    async def _recall_unsafe(self, context: "MessageContext") -> List[str]:
        """
        Core recall logic; lets exceptions propagate to recall().

        Ищем похожее в двух местах: что говорил этот же человек и что
        вообще звучало в этом канале (включая её собственные ответы).
        Свежие сообщения (они и так есть в истории чата) и далёкие по
        смыслу совпадения отбрасываются.
        """
        loop = asyncio.get_event_loop()
        count: int = await loop.run_in_executor(None, self._collection.count)
        if count == 0 or len(context.content.strip()) < 3:
            return []

        query_embedding: List[float] = (await self.embed([context.content]))[0].tolist()
        k = min(config.MEMORY_TOP_K * 3, count)

        def _q(where):
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        queries = [{"author_id": context.author_id}, {"channel_id": str(context.channel_id)}]
        results = []
        for where in queries:
            try:
                results.append(await loop.run_in_executor(None, lambda w=where: _q(w)))
            except Exception:  # noqa: BLE001
                continue

        now = datetime.now(tz=timezone.utc)
        skip_recent_s = config.MEMORY_SKIP_RECENT_MINUTES * 60
        seen_docs = set()
        scored = []
        for res in results:
            for doc, meta, dist, rid in zip(
                res.get("documents", [[]])[0], res.get("metadatas", [[]])[0],
                res.get("distances", [[]])[0], res.get("ids", [[]])[0],
            ):
                if rid.endswith(str(context.message_id)) or doc in seen_docs:
                    continue
                if dist is None or dist > config.MEMORY_MAX_DISTANCE:
                    continue
                ts = str(meta.get("timestamp", ""))
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if (now - dt).total_seconds() < skip_recent_s:
                        continue
                except ValueError:
                    pass
                seen_docs.add(doc)
                if meta.get("author_id") == "bot":
                    line = f"[{ago(ts)}] ты ответила {meta.get('to_author_name', 'кому-то')}: {doc}"
                else:
                    line = f"[{ago(ts)}] {meta.get('author_name', 'кто-то')}: {doc}"
                scored.append((dist, line))

        scored.sort(key=lambda x: x[0])
        return [line for _, line in scored[: config.MEMORY_TOP_K]]

    async def _get_relationship_score_unsafe(self, author_id: str) -> float:
        """Core relationship score logic; lets exceptions propagate."""
        assert self._db is not None, (
            "init_db() was not awaited before get_relationship_score()"
        )

        async with self._db.execute(_SELECT_RELATIONSHIP, (author_id,)) as cur:
            row = await cur.fetchone()

        if row is None:
            return 0.0

        interaction_count: int = row["interaction_count"]
        replied_count: int = row["replied_count"]
        # Затухание — от последней встречи, а не от знакомства: давние
        # друзья не должны «остывать», а пропавшие надолго — да.
        first_seen_str: str = row["last_seen"]

        try:
            first_seen_dt = datetime.fromisoformat(first_seen_str)
        except ValueError:
            first_seen_dt = datetime.now(tz=timezone.utc)
        if first_seen_dt.tzinfo is None:
            first_seen_dt = first_seen_dt.replace(tzinfo=timezone.utc)

        days_known = max(1, (datetime.now(tz=timezone.utc) - first_seen_dt).days)
        decay = max(
            0.1,
            1.0 - (days_known / max(config.RELATIONSHIP_DECAY_DAYS, 1)) * 0.5,
        )
        reply_ratio: float = replied_count / max(interaction_count, 1)
        raw: float = (min(interaction_count, 50) / 50) * 0.6 + reply_ratio * 0.4
        score: float = min(1.0, raw * decay)

        _logger.debug(
            "Relationship score",
            extra={
                "data": {
                    "author_id": author_id,
                    "interactions": interaction_count,
                    "reply_ratio": round(reply_ratio, 3),
                    "days_known": days_known,
                    "decay": round(decay, 3),
                    "score": round(score, 3),
                }
            },
        )
        return score
