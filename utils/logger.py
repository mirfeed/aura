"""
utils/logger.py — structured JSON logging for the Discord companion bot.

Every log record is a valid JSON object containing:
    timestamp, level, module, event, data (dict)

Author IDs are anonymised with the first 12 hex chars of their SHA-256
digest before they reach any log sink.

Log sinks:
    • Console (stderr)
    • LOG_DIR/bot.log  — RotatingFileHandler, 5 MB max, 3 backups
"""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Format every LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise *record* to a compact JSON string."""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "event": record.getMessage(),
            "data": getattr(record, "data", {}),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured with the project's JSON handlers.

    Safe to call multiple times with the same *name* — handlers are only
    attached once.
    """
    import config  # local import to avoid circular deps at module level

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — don't add duplicate handlers.
        return logger

    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    formatter = JsonFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # Rotating file handler
    log_dir: Path = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = log_dir / "bot.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Don't propagate to root logger — we own our own handlers.
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Author ID anonymisation
# ---------------------------------------------------------------------------

def anonymise_author_id(author_id: int) -> str:
    """
    Return the first 12 hex characters of the SHA-256 digest of *author_id*.

    This is a one-way transform sufficient for log correlation without
    storing raw Discord user IDs.
    """
    digest = hashlib.sha256(str(author_id).encode()).hexdigest()
    return digest[:12]


# ---------------------------------------------------------------------------
# Structured helper functions
# ---------------------------------------------------------------------------

_logger = get_logger("bot.events")


def _log(level: int, event: str, data: Dict[str, Any]) -> None:
    """Emit a structured log record at *level* carrying *event* and *data*."""
    record = _logger.makeRecord(
        name=_logger.name,
        level=level,
        fn="",
        lno=0,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.data = data  # type: ignore[attr-defined]
    _logger.handle(record)


def log_message_received(
    message_id: int,
    author_id: int,
    author_display_name: str,
    channel_id: int,
    guild_id: Optional[int],
    content_length: int,
) -> None:
    """Log that a Discord message was received and passed for processing."""
    _log(
        logging.INFO,
        "message_received",
        {
            "message_id": message_id,
            "author_id_hash": anonymise_author_id(author_id),
            "author_display_name": author_display_name,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "content_length": content_length,
        },
    )


def log_relevance(
    message_id: int,
    score: float,
    breakdown: Dict[str, float],
) -> None:
    """Log the relevance scoring breakdown for a message."""
    _log(
        logging.DEBUG,
        "relevance_scored",
        {
            "message_id": message_id,
            "score": round(score, 4),
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
        },
    )


def log_decision(
    message_id: int,
    should_reply: bool,
    reason: str,
    delay_seconds: float,
    typing_duration: float,
) -> None:
    """Log the reply decision made by the behaviour engine."""
    _log(
        logging.INFO,
        "reply_decision",
        {
            "message_id": message_id,
            "should_reply": should_reply,
            "reason": reason,
            "delay_seconds": round(delay_seconds, 3),
            "typing_duration": round(typing_duration, 3),
        },
    )


def log_reply_sent(
    message_id: int,
    channel_id: int,
    response_length: int,
) -> None:
    """Log that a reply was successfully sent to Discord."""
    _log(
        logging.INFO,
        "reply_sent",
        {
            "message_id": message_id,
            "channel_id": channel_id,
            "response_length": response_length,
        },
    )
