"""
Образцы настроений для F5 голосом 3 (клон Qwen3-TTS 0.6B-Base по 7-секундному образцу).
Запуск из Ubuntu:  ~/qwen-tts/.venv/bin/python /mnt/d/ai/qwen/make_moods.py
Результат — D:\\ai\\qwen\\out\\moods\\<настроение>_<дубль>.wav; лучший дубль выбирает pick_moods.py.
"""
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = Path("/mnt/d/ai/qwen/out/moods")
OUT.mkdir(parents=True, exist_ok=True)
TEXTS = Path("/mnt/d/ai/f5/texts")
REF = Path("/mnt/d/ai/qwen/out/ref_3_cool_sarcastic.wav")
REF_TEXT = "Привет! Ну наконец-то ты зашёл. Я тут как раз сидела с кофе и разбирала свою коллекцию камней."
TAKES = 3
MOODS = ["cheerful", "amused", "curious", "irritated", "tired", "withdrawn"]

model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0", dtype=torch.bfloat16)
ref, sr = sf.read(REF, dtype="float32")
prompt = model.create_voice_clone_prompt(ref_audio=(ref, sr), ref_text=REF_TEXT)

for mood in MOODS:
    text = (TEXTS / f"{mood}.txt").read_text(encoding="utf-8").strip()
    for k in range(1, TAKES + 1):
        t0 = time.monotonic()
        wavs, out_sr = model.generate_voice_clone(text=text, language="russian", voice_clone_prompt=prompt)
        sf.write(OUT / f"{mood}_{k}.wav", wavs[0], out_sr)
        print(f"{mood}_{k}: {len(wavs[0]) / out_sr:.1f} с за {time.monotonic() - t0:.1f} с", flush=True)
