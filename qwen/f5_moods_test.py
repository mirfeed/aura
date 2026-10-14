"""Одна фраза каждым настроением голоса qwen3 через F5 → D:\\ai\\qwen\\out\\f5_moods\\<настроение>.wav"""
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, "/mnt/d/ai/f5")
import f5_server  # noqa: E402

OUT = Path("/mnt/d/ai/qwen/out/f5_moods")
OUT.mkdir(parents=True, exist_ok=True)
LINE = "Да ладно, серьёзно? Ну ты даёшь, я бы так не смогла!"

engine = f5_server.Engine()
engine.synthesize("Привет.", "qwen3", "neutral", 1.0)  # прогрев
for mood in ["neutral", "cheerful", "amused", "curious", "irritated", "tired", "withdrawn"]:
    t0 = time.monotonic()
    path = OUT / f"{mood}.wav"
    path.write_bytes(engine.synthesize(LINE, "qwen3", mood, 1.0))
    print(f"{mood:10s} {sf.info(str(path)).duration:.1f} с звука за {time.monotonic() - t0:.1f} с", flush=True)
