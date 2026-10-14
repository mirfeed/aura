"""
Тест Qwen3-TTS 0.6B-Base: клон голосов 1 и 3 из VoiceDesign (живой режим без описания).
Запуск из Ubuntu:  ~/qwen-tts/.venv/bin/python /mnt/d/ai/qwen/clone_test.py
Результат — D:\\ai\\qwen\\out\\clone_<голос>_<n>.wav и образцы ref_<голос>.wav.
"""
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = Path("/mnt/d/ai/qwen/out")
REF_TEXT = "Привет! Ну наконец-то ты зашёл. Я тут как раз сидела с кофе и разбирала свою коллекцию камней."
# Конец третьего предложения по таймкодам Whisper (+0.2 с запаса).
REFS = {"1_warm_playful": 7.08 + 0.2, "3_cool_sarcastic": 6.70 + 0.2}

# Типичные реплики в войсе: короткая, средняя, длинная.
LINES = [
    "Ой, привет! Я тут, слушаю.",
    "Да ладно, серьёзно? Ну ты даёшь, я бы так не смогла!",
    "Слушай, это долгая история. Короче, Луна опять уронила горшок, и теперь у меня на учебнике "
    "по минералогии огромное пятно.",
]

t0 = time.monotonic()
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0", dtype=torch.bfloat16)
print(f"модель 0.6B загружена за {time.monotonic() - t0:.1f} с, видеопамять {torch.cuda.memory_allocated() / 2**30:.2f} ГБ")

for voice, cut_s in REFS.items():
    wav, sr = sf.read(OUT / f"{voice}.wav", dtype="float32")
    ref = wav[: int(cut_s * sr)]
    sf.write(OUT / f"ref_{voice}.wav", ref, sr)
    t1 = time.monotonic()
    prompt = model.create_voice_clone_prompt(ref_audio=(ref, sr), ref_text=REF_TEXT)  # один раз на голос
    print(f"== {voice}: образец {len(ref) / sr:.1f} с, подготовка {time.monotonic() - t1:.1f} с")
    for i, line in enumerate(LINES, 1):
        torch.cuda.synchronize()
        t2 = time.monotonic()
        wavs, out_sr = model.generate_voice_clone(text=line, language="russian", voice_clone_prompt=prompt)
        torch.cuda.synchronize()
        gen_s = time.monotonic() - t2
        audio_s = len(wavs[0]) / out_sr
        sf.write(OUT / f"clone_{voice}_{i}.wav", wavs[0], out_sr)
        print(f"  {i}: {audio_s:4.1f} с звука за {gen_s:4.1f} с (x{audio_s / gen_s:.2f})  «{line[:40]}…»")

print(f"пик видеопамяти: {torch.cuda.max_memory_allocated() / 2**30:.2f} ГБ")
