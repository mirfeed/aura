"""
voice/dave_patch.py — DAVE (E2EE) decryption for discord-ext-voice-recv.

С марта 2026 Discord требует сквозное шифрование DAVE во всех голосовых
каналах.  discord.py 2.7+ (с пакетом ``davey``) умеет *отправлять* звук
через DAVE, но discord-ext-voice-recv 0.5.x выпущен раньше и не умеет
*расшифровывать* входящий звук: он отдаёт в Opus-декодер зашифрованные
кадры, и на выходе получается шум/треск.

Этот модуль подменяет ``PacketDecoder._decode_packet`` так, чтобы перед
Opus-декодированием каждый кадр с маркером DAVE (последние 2 байта 0xFAFA)
расшифровывался через ``voice_client._connection.dave_session``.
Кадры, которые расшифровать нельзя, превращаются в «потерю пакета»
(Opus PLC) — вместо шума будет короткая тишина.

Вызовите :func:`apply` один раз при старте (до подключения к войсу).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from utils.logger import get_logger

_logger = get_logger("bot.voice.dave")

_DAVE_MAGIC = b"\xfa\xfa"
_applied = False
_fail_count = 0
# Счётчики для диагностики «она не слышит»: сколько кадров пришло и что с ними стало.
_stats = {"dave": 0, "ok": 0, "plain": 0, "no_user": 0, "fail": 0, "last_error": ""}


def stats() -> dict:
    return dict(_stats)


def reset_stats() -> None:
    _stats.update(dave=0, ok=0, plain=0, no_user=0, fail=0, last_error="")


def _dave_decrypt(decoder: Any, data: Optional[bytes]) -> Optional[bytes]:
    """Return plain Opus bytes, or None if the frame must be dropped."""
    global _fail_count

    if not data:
        return data

    # Not a DAVE frame (silence frames, passthrough during transitions).
    if len(data) < 3 or data[-2:] != _DAVE_MAGIC:
        _stats["plain"] += 1
        return data
    _stats["dave"] += 1

    try:
        vc = decoder.sink.voice_client
        conn = vc._connection
        session = getattr(conn, "dave_session", None)
        if session is None or not getattr(conn, "dave_protocol_version", 0):
            return data

        user_id = decoder._cached_id or vc._get_id_from_ssrc(decoder.ssrc)
        if user_id is None:
            _stats["no_user"] += 1
            return None

        import davey  # type: ignore

        out = session.decrypt(int(user_id), davey.MediaType.audio, data)
        _stats["ok"] += 1
        return out
    except Exception as exc:  # noqa: BLE001
        _fail_count += 1
        _stats["fail"] += 1
        _stats["last_error"] = str(exc)[:200]
        # Не засоряем лог: первые 5 ошибок и потом каждая 500-я.
        if _fail_count <= 5 or _fail_count % 500 == 0:
            _logger.warning(
                "DAVE decrypt failed — dropping frame",
                extra={"data": {"error": str(exc), "failures": _fail_count}},
            )
        return None


def apply() -> None:
    """Monkey-patch discord-ext-voice-recv's PacketDecoder (idempotent)."""
    global _applied
    if _applied:
        return

    from discord.ext.voice_recv.opus import PacketDecoder

    def _decode_packet(self: Any, packet: Any) -> Tuple[Any, bytes]:
        assert self._decoder is not None
        data: Optional[bytes] = None
        if packet:
            data = _dave_decrypt(self, packet.decrypted_data)
        try:
            pcm = self._decoder.decode(data, fec=False)
        except Exception:  # noqa: BLE001 — corrupt frame → concealment
            pcm = self._decoder.decode(None, fec=False)
        return packet, pcm

    PacketDecoder._decode_packet = _decode_packet  # type: ignore[assignment]
    _applied = True
    _logger.info("DAVE receive patch applied to voice_recv", extra={"data": {}})
