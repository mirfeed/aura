"""
main.py — entry point for the Discord AI companion bot.

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy()) is set
before anything else; this is required for discord.py to operate correctly
on Windows 11.
"""

from __future__ import annotations

import asyncio
import sys

# ── MUST be the very first asyncio call ────────────────────────────────────
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
# ───────────────────────────────────────────────────────────────────────────

import aiohttp  # noqa: E402

import config  # noqa: E402  (after policy set, before discord import)
from behavior.engine import DecisionEngine  # noqa: E402
from behavior.state import BotState  # noqa: E402
from core.bot import CompanionBot  # noqa: E402
from memory.store import MemoryStore  # noqa: E402
from pipeline.text import TextPipeline  # noqa: E402
from utils.logger import get_logger  # noqa: E402

try:  # Day 4/5 voice is optional — the bot still runs without its packages.
    import davey  # noqa: E402,F401  (DAVE E2EE — обязателен для войса с 2026)
    import faster_whisper  # noqa: E402,F401
    from discord.ext import voice_recv  # noqa: E402,F401
    from voice.manager import VoiceManager  # noqa: E402
    _voice_import_error = None
except Exception as _exc:  # noqa: BLE001
    VoiceManager = None  # type: ignore[assignment,misc]
    _voice_import_error = _exc

_logger = get_logger("bot.main")


STOP_FLAG = config.MODELS_DIR.parent / "stop.flag"


async def _watch_stop_flag(bot: "CompanionBot") -> None:
    """stop.bat кладёт data/stop.flag — выходим аккуратно (с выходом из войса), а не убиваем процесс."""
    while not bot.is_closed():
        await asyncio.sleep(2)
        if STOP_FLAG.exists():
            STOP_FLAG.unlink(missing_ok=True)
            _logger.info("Stop flag found — shutting down", extra={"data": {}})
            await bot.close()
            return


async def main() -> None:
    """
    Construct all dependencies, start the bot, and handle shutdown.

    The aiohttp.ClientSession is created inside the running event loop
    (never at module level) and injected into TextPipeline.  It is closed
    cleanly in the finally block regardless of how the bot exits.

    All collaborators are instantiated here and passed explicitly; no
    module-level globals are used.
    """
    # Validate required config.
    if not config.DISCORD_TOKEN:
        _logger.error(
            "DISCORD_TOKEN is not set — cannot start",
            extra={"data": {}},
        )
        sys.exit(1)

    _logger.info(
        "Starting bot",
        extra={
            "data": {
                "bot_name": config.BOT_NAME,
                "personality_seed": config.PERSONALITY_SEED,
                "base_response_probability": config.BASE_RESPONSE_PROBABILITY,
                "ollama_model": config.OLLAMA_MODEL,
                "ollama_base_url": config.OLLAMA_BASE_URL,
                "ollama_num_ctx": config.OLLAMA_NUM_CTX,
                "max_response_tokens": config.MAX_RESPONSE_TOKENS,
                "guild_ids": config.GUILD_IDS,
                "channel_whitelist": config.CHANNEL_WHITELIST,
            }
        },
    )

    # aiohttp session must be created inside the running event loop.
    session = aiohttp.ClientSession()

    # MemoryStore: constructor is sync (loads embedder, opens ChromaDB).
    # init_db() opens the aiosqlite connection and must be awaited here.
    memory = MemoryStore()
    await memory.init_db()

    try:
        # Construct shared state and collaborators.
        state = BotState()
        engine = DecisionEngine()
        pipeline = TextPipeline(session=session)

        bot = CompanionBot(state=state, engine=engine, pipeline=pipeline, memory=memory)

        if config.VOICE_ENABLED:
            if VoiceManager is None:
                _logger.warning(
                    "Voice disabled — packages missing (pip install -r requirements.txt)",
                    extra={"data": {"error": str(_voice_import_error)}},
                )
            else:
                bot._voice = VoiceManager(client=bot, state=state, pipeline=pipeline, memory=memory)

        STOP_FLAG.unlink(missing_ok=True)  # старый флаг от прошлого раза не должен сразу погасить бота
        asyncio.create_task(_watch_stop_flag(bot))
        await bot.start(config.DISCORD_TOKEN)

    except KeyboardInterrupt:
        _logger.info("KeyboardInterrupt received — shutting down", extra={"data": {}})

    finally:
        # Shutdown order: Discord → memory → HTTP session.
        if "bot" in dir() and not bot.is_closed():  # type: ignore[possibly-undefined]
            await bot.close()
        await memory.close()
        if not session.closed:
            await session.close()
        _logger.info("Bot has shut down cleanly", extra={"data": {}})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Suppress the traceback for a clean Ctrl-C exit in the terminal.
        pass
