"""
voice/stt.py — Day 4: распознавание речи через faster-whisper.

* Модель грузится лениво, в отдельном потоке (``load()``), чтобы не
  блокировать event loop.
* ``WHISPER_DEVICE=auto`` — пробуем CUDA (float16), при любой ошибке
  (нет видеокарты, нет cuBLAS/cuDNN) откатываемся на CPU (int8).
* На Windows автоматически подключаются DLL из pip-пакетов
  ``nvidia-cublas-cu12`` и ``nvidia-cudnn-cu12``, если они установлены.
* Все вызовы модели идут через один поток — модель не потокобезопасна.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

import config
from utils.logger import get_logger

_logger = get_logger("bot.voice.stt")

# Типичные «галлюцинации» Whisper на тишине/шуме (особенно на русском).
_HALLUCINATIONS: List[str] = [
    "продолжение следует",
    "субтитры",
    "редактор субтитров",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "ставьте лайки",
    "до новых встреч",
    "dimatorzok",
    "amara.org",
    "thanks for watching",
    "thank you for watching",
    "music",
    "♪",
]


def _register_cuda_dlls() -> None:
    """Сделать DLL из pip-пакетов nvidia-* видимыми для CTranslate2 (Windows)."""
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            mod = __import__(pkg, fromlist=["__path__"])
        except Exception:  # noqa: BLE001
            continue
        for base in getattr(mod, "__path__", []):
            bin_dir = Path(base) / "bin"
            if bin_dir.is_dir():
                try:
                    os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _collapse_repeats(text: str) -> str:
    """«трухим-трухим-трухим-трухим» → «трухим-трухим»; если весь текст — повтор, пусто."""
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 6 and len(set(words)) <= max(1, len(words) // 4):
        return ""  # почти одно и то же слово много раз — галлюцинация
    return re.sub(r"\b(\w+)(?:[\s,\-]+\1\b){2,}", r"\1 \1", text, flags=re.IGNORECASE)


def _is_hallucination(text: str) -> bool:
    low = text.lower().strip(" .!?…,")
    if not low:
        return True
    return any(h in low for h in _HALLUCINATIONS)


class SpeechToText:
    """Асинхронная обёртка над faster-whisper."""

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._load_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def load(self) -> None:
        async with self._load_lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_sync)

    def _load_sync(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        _register_cuda_dlls()
        download_root = str(config.MODELS_DIR / "whisper")
        wanted = config.WHISPER_DEVICE.lower()
        attempts = []
        if wanted in ("auto", "cuda", "gpu"):
            # int8_float16: почти та же точность, вдвое меньше видеопамяти (её делим с gemma).
            ct = config.WHISPER_COMPUTE_TYPE if config.WHISPER_COMPUTE_TYPE != "auto" else "int8_float16"
            attempts.append(("cuda", ct))
        if wanted != "cuda":
            ct = config.WHISPER_COMPUTE_TYPE if config.WHISPER_COMPUTE_TYPE != "auto" else "int8"
            attempts.append(("cpu", ct))

        last_exc: Optional[Exception] = None
        for device, compute_type in attempts:
            t0 = time.monotonic()
            try:
                model = WhisperModel(
                    config.WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                    download_root=download_root,
                )
                # Пробный прогон: ошибки cuDNN/cuBLAS всплывают только здесь.
                segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="ru")
                list(segs)
                self._model = model
                self._device = device
                _logger.info(
                    "Whisper loaded",
                    extra={"data": {
                        "model": config.WHISPER_MODEL,
                        "device": device,
                        "compute_type": compute_type,
                        "seconds": round(time.monotonic() - t0, 1),
                    }},
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                _logger.warning(
                    "Whisper failed to load on device — trying next",
                    extra={"data": {"device": device, "error": str(exc)}},
                )
        raise RuntimeError(f"Whisper could not be loaded: {last_exc}")

    async def transcribe(self, audio: np.ndarray, names: Optional[List[str]] = None) -> str:
        """
        Распознать фразу (float32, 16 кГц, моно).  Пустая строка — ничего.

        *names* — ники людей в канале: подсказываем их Whisper'у, чтобы он
        писал «Кактусик», а не «как тусик».
        """
        if self._model is None:
            await self.load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_sync, audio, names or [])

    def _transcribe_sync(self, audio: np.ndarray, names: List[str]) -> str:
        t0 = time.monotonic()
        language = config.WHISPER_LANGUAGE or None
        hint_names = [n for n in names if n and len(n) <= 32][:8]
        prompt = config.WHISPER_INITIAL_PROMPT
        if hint_names:
            prompt = f"{prompt} В канале: {', '.join(hint_names)}."
        hotwords = ", ".join([config.BOT_NAME, "Аура"] + hint_names + config.WHISPER_HOTWORDS)
        kwargs = dict(
            language=language,
            beam_size=config.WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
            vad_filter=False,  # речь уже нарезана нашим VAD
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.2,  # «трухим-трухим-трухим» → отбросить и перераспознать
            repetition_penalty=1.05,
        )
        try:
            segments, _info = self._model.transcribe(audio, hotwords=hotwords, **kwargs)
        except TypeError:  # старый faster-whisper без hotwords
            segments, _info = self._model.transcribe(audio, **kwargs)
        parts: List[str] = []
        for seg in segments:
            if seg.no_speech_prob > 0.7 or seg.avg_logprob < -1.2:
                continue
            txt = _collapse_repeats(seg.text.strip())
            if txt and not _is_hallucination(txt):
                parts.append(txt)
        text = " ".join(parts).strip()
        _logger.debug(
            "Transcribed",
            extra={"data": {
                "chars": len(text),
                "audio_seconds": round(audio.size / 16000, 2),
                "latency_seconds": round(time.monotonic() - t0, 2),
                "device": self._device,
            }},
        )
        return text

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
