"""
core/game_mode.py — игровой режим: Аура только в чате, модель частично на CPU.

Включается из aura.bat (пункт G): там кладётся или убирается data/game_mode.flag, бот подхватывает
это на лету (core/bot.py, _game_mode_loop).  Проверка — просто наличие файла, так что её можно звать
перед каждым запросом.
"""

from __future__ import annotations

from typing import Any, Dict

import config


def active() -> bool:
    return config.GAME_MODE_FLAG.exists()


def ollama_options() -> Dict[str, Any]:
    """
    Добавка к options каждого запроса к Ollama.  Должна быть одинаковой во всех запросах:
    другой num_gpu — и Ollama перезагружает модель.
    """
    if not active():
        return {}
    return {"num_gpu": config.GAME_MODE_GPU_LAYERS, "num_thread": config.GAME_MODE_THREADS}
