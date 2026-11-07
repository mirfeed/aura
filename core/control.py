"""
core/control.py — локальное управление ботом для пульта (tools/panel.py).

Слушает только 127.0.0.1:CONTROL_PORT.  Каждый запрос должен нести заголовок X-Aura-Token с ключом
из data/control.token (бот пишет новый ключ при каждом запуске), так что ни сайт в браузере, ни
чужая программа без доступа к папке проекта ею не покомандуют.

  GET  /status                     — что с ней сейчас: фаза дня, книга, войс, настроение…
  POST /wake      {"minutes": 60}  — разбудить (null — до обычного подъёма)
  POST /sleep     {"minutes": 120, "quiet": false} — уложить спать (null — до обычного подъёма)
  POST /auto                       — вернуть обычный распорядок
  POST /distract                   — оторвать от книги
  POST /read                       — почитать сейчас
  POST /set       {"key": "READING_ENABLED", "value": false} — переключить настройку на лету
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from aiohttp import web

import config
from behavior import schedule
from core import game_mode
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.bot import CompanionBot

_logger = get_logger("bot.control")

# Переключатели, которые пульт может менять на лету (только bool).  Остальное — страница настроек.
TOGGLES = {
    "SCHEDULE_ENABLED", "READING_ENABLED", "READING_SHARE_ENABLED", "LIFE_EVENTS_ENABLED", "BIRTHDAYS_ENABLED",
    "PROACTIVE_FRIENDS", "PROACTIVE_TRUSTED", "DM_CATCHUP", "REMINDERS_ENABLED", "WEB_SEARCH_ENABLED",
    "VISION_ENABLED", "VISION_OBSERVE", "MEDIA_SEND_ENABLED", "FAVORITES_ENABLED", "CHAT_EMOJI",
    "CHAT_SPLIT_MESSAGES", "VOICE_ENABLED", "VOICE_PROACTIVE", "F5_AUTO", "TEST_MODE",
}
# Эти создаются при запуске — переключение подействует только после перезапуска.
_RESTART_ALWAYS = {"VOICE_ENABLED", "F5_AUTO", "MEDIA_SEND_ENABLED", "FAVORITES_ENABLED"}


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


class ControlServer:
    def __init__(self, bot: "CompanionBot") -> None:
        self.bot = bot
        self.token = secrets.token_urlsafe(24)
        self._runner: Optional[web.AppRunner] = None

    async def start(self) -> None:
        app = web.Application(middlewares=[self._auth])
        app.add_routes([
            web.get("/status", self.status),
            web.post("/wake", self.wake),
            web.post("/sleep", self.sleep),
            web.post("/auto", self.auto),
            web.post("/distract", self.distract),
            web.post("/read", self.read),
            web.post("/set", self.set_value),
        ])
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", config.CONTROL_PORT).start()
        config.CONTROL_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.CONTROL_TOKEN_FILE.write_text(self.token, encoding="utf-8")
        config.BOT_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        _logger.info("Control server listening", extra={"data": {"port": config.CONTROL_PORT}})

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @web.middleware
    async def _auth(self, request: web.Request, handler: Any) -> web.StreamResponse:
        if not secrets.compare_digest(request.headers.get("X-Aura-Token", ""), self.token):
            return web.json_response({"error": "forbidden"}, status=403)
        return await handler(request)

    @staticmethod
    async def _body(request: web.Request) -> Dict[str, Any]:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------

    async def status(self, _: web.Request) -> web.Response:
        bot = self.bot
        now = schedule._local_now()
        ov = schedule.override(now)
        plan = schedule.plan(now)
        book: Dict[str, Any] = {}
        lib = bot._library
        if lib is not None:
            try:
                cur = await lib.current()
                if cur:
                    book = {"title": cur.get("title", ""), "pos": cur.get("pos", 0), "total": cur.get("total", 0),
                            "kind": cur.get("kind", "")}
                last = await lib.last_session_at()
                book["last_session"] = _iso(last)
            except Exception:  # noqa: BLE001
                pass
        voice: Dict[str, Any] = {"enabled": bot._voice is not None, "channels": []}
        if bot._voice is not None:
            for gid in list(getattr(bot._voice, "_sessions", {})):
                guild = bot.get_guild(gid)
                vc = guild.voice_client if guild is not None else None
                voice["channels"].append(getattr(getattr(vc, "channel", None), "name", str(gid)))
        state = bot._state
        pipe = bot._pipeline
        return web.json_response({
            "ok": True,
            "pid": os.getpid(),
            "user": str(bot.user) if bot.user else "",
            "ready": bot.is_ready(),
            "latency_ms": round(bot.latency * 1000) if bot.latency == bot.latency else None,
            "guilds": len(bot.guilds),
            "started_at": _iso(bot._started_at),
            "phase": schedule.phase(now),
            "override": {"mode": ov.get("mode", ""), "until": _iso(ov.get("until"))} if ov else None,
            "plan": {"bed": _iso(plan.get("bed")), "wake": _iso(plan.get("wake"))} if plan else None,
            "schedule_enabled": config.SCHEDULE_ENABLED,
            "reading": {
                "enabled": lib is not None and config.READING_ENABLED,
                "now": bot.reading_now,
                "since": _iso(bot._read_started),
                "binge": schedule.binge_title(now),
                "paused_min": max(0, round((bot._read_pause_until - time.monotonic()) / 60)),
                "queued": bot._read_now,
                "book": book,
            },
            "busy": pipe.busy,
            "idle_s": round(pipe.seconds_since_last_request),
            "mood": state.mood.value,
            "mood_reason": state.mood_reason,
            "energy": round(state.energy, 2),
            "voice": voice,
            "game_mode": game_mode.active(),
            "toggles": {k: bool(getattr(config, k, False)) for k in sorted(TOGGLES)},
        })

    async def wake(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        now = schedule._local_now()
        minutes = body.get("minutes")
        until = now + timedelta(minutes=float(minutes)) if minutes else schedule.next_wake(now)
        was_asleep = schedule.is_asleep(now)
        schedule.wake_up(until)
        phase = await self.bot.check_phase()
        _logger.info("Woken up from panel", extra={"data": {"until": until.isoformat(), "was_asleep": was_asleep}})
        return web.json_response({"ok": True, "phase": phase, "until": until.isoformat(), "was_asleep": was_asleep})

    async def sleep(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        now = schedule._local_now()
        minutes = body.get("minutes")
        until = now + timedelta(minutes=float(minutes)) if minutes else schedule.next_wake(now)
        if not minutes and until <= now + timedelta(minutes=5):
            until = now + timedelta(hours=8)  # «до подъёма», а подъём вот-вот — значит, на нормальный сон
        self.bot._quiet_goodnight = bool(body.get("quiet")) and not schedule.is_asleep(now)
        schedule.put_to_sleep(until)
        phase = await self.bot.check_phase()
        _logger.info("Put to sleep from panel", extra={"data": {"until": until.isoformat()}})
        return web.json_response({"ok": True, "phase": phase, "until": until.isoformat()})

    async def auto(self, _: web.Request) -> web.Response:
        schedule.clear_override()
        phase = await self.bot.check_phase()
        _logger.info("Schedule back to auto", extra={"data": {"phase": phase}})
        return web.json_response({"ok": True, "phase": phase})

    async def distract(self, _: web.Request) -> web.Response:
        return web.json_response({"ok": True, **self.bot.distract("отвлекли с пульта")})

    async def read(self, _: web.Request) -> web.Response:
        if self.bot._library is None or not config.READING_ENABLED:
            return web.json_response({"ok": False, "error": "чтение выключено"})
        if schedule.is_asleep():
            return web.json_response({"ok": False, "error": "она спит"})
        self.bot.read_now()
        return web.json_response({"ok": True})

    async def set_value(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        key, value = str(body.get("key", "")), body.get("value")
        if key not in TOGGLES or not isinstance(value, bool):
            return web.json_response({"ok": False, "error": f"нельзя менять {key}"}, status=400)
        setattr(config, key, value)
        bot = self.bot
        live = key not in _RESTART_ALWAYS
        if key == "READING_ENABLED":
            live = bot._library is not None or not value
            if not value:
                bot.distract("чтение выключили")
        elif key == "LIFE_EVENTS_ENABLED":
            live = bot._life is not None or not value
        elif key == "REMINDERS_ENABLED":
            live = bot._reminders is not None or not value
        elif key == "WEB_SEARCH_ENABLED":
            live = bot._pipeline._web is not None or not value
        elif key == "SCHEDULE_ENABLED":
            await bot.check_phase()
        _logger.info("Setting changed from panel", extra={"data": {"key": key, "value": value, "live": live}})
        return web.json_response({"ok": True, "live": live})
