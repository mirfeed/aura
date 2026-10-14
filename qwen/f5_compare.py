"""
Те же реплики голосами qwen1 / qwen3, но через F5 (образец — 7 с из Qwen VoiceDesign).
Запуск из Ubuntu:  ~/f5-tts/.venv/bin/python /mnt/d/ai/qwen/f5_compare.py
Результат — D:\\ai\\qwen\\out\\f5_<голос>_<n>.wav
"""
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "/mnt/d/ai/f5")
import f5_server  # noqa: E402

OUT = Path("/mnt/d/ai/qwen/out")
LINES = [
    "Ой, привет! Я тут, слушаю.",
    "Да ладно, серьёзно? Ну ты даёшь, я бы так не смогла!",
    "Слушай, это долгая история. Короче, Луна опять уронила горшок, и теперь у меня на учебнике "
    "по минералогии огромное пятно.",
]

t0 = time.monotonic()
engine = f5_server.Engine()
print(f"F5 загружен за {time.monotonic() - t0:.1f} с")
engine.synthesize("Привет, это проверка голоса.", "qwen1", "neutral", 1.0)  # прогрев

for voice in ("qwen1", "qwen3"):
    print(f"== {voice}")
    for i, line in enumerate(LINES, 1):
        t1 = time.monotonic()
        wav = engine.synthesize(line, voice, "neutral", 1.0)
        gen_s = time.monotonic() - t1
        path = OUT / f"f5_{voice}_{i}.wav"
        path.write_bytes(wav)
        audio_s = sf.info(str(path)).duration
        print(f"  {i}: {audio_s:4.1f} с звука за {gen_s:4.1f} с (x{audio_s / gen_s:.2f})")

print(f"пик видеопамяти: {torch.cuda.max_memory_allocated() / 2**30:.2f} ГБ")
