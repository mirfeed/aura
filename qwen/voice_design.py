"""
Тест Qwen3-TTS VoiceDesign: голос Ауры по текстовому описанию, без образца.
Запуск из Ubuntu:  ~/qwen-tts/.venv/bin/python /mnt/d/ai/qwen/voice_design.py
Результат — D:\\ai\\qwen\\out\\<вариант>.wav (слушать из Windows).
"""
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = Path("/mnt/d/ai/qwen/out")
OUT.mkdir(parents=True, exist_ok=True)

# Одни и те же реплики для всех вариантов — так голоса легче сравнивать на слух.
TEXT = ("Привет! Ну наконец-то ты зашёл. Я тут как раз сидела с кофе и разбирала свою коллекцию камней. "
        "Слушай, а ты знал, что аметист — это просто кварц с примесью железа? Вот и я раньше не знала, ха-ха. "
        "Да ладно, серьёзно? Ой, всё, не смеши меня!")

VOICES = {
    "1_warm_playful": "A young woman around twenty, warm and slightly playful voice, medium pitch, clear diction, "
                      "speaks casually with a hint of teasing irony, natural conversational pace.",
    "2_soft_bright": "A soft, bright young female voice, about twenty years old, gentle and friendly, a little shy, "
                     "a light smile in the voice, calm pace.",
    "3_cool_sarcastic": "A confident young woman in her early twenties, slightly low and smooth timbre, dry sarcastic "
                        "humor, relaxed and cool delivery, unhurried.",
    "4_lively": "An energetic cheerful girl about nineteen, higher pitch, lively and expressive, animated speech, "
                "laughs easily.",
    "5_ru_description": "Молодая девушка двадцати лет, тёплый живой голос, немного насмешливая интонация, "
                        "говорит непринуждённо, как с друзьями в голосовом чате.",
}

t0 = time.monotonic()
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda:0", dtype=torch.bfloat16)
print(f"модель загружена за {time.monotonic() - t0:.1f} с, видеопамять: "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} ГБ")
langs = model.get_supported_languages()
lang = next((l for l in langs if "russian" in str(l).lower() or str(l).lower() in ("ru", "rus")), None)
print("языки:", langs, "→", lang)

for name, instruct in VOICES.items():
    t1 = time.monotonic()
    wavs, sr = model.generate_voice_design(text=TEXT, instruct=instruct, language=lang)
    gen_s = time.monotonic() - t1
    audio_s = len(wavs[0]) / sr
    sf.write(OUT / f"{name}.wav", wavs[0], sr)
    print(f"{name}: {audio_s:.1f} с звука за {gen_s:.1f} с (x{audio_s / gen_s:.2f} от реального времени)")

print(f"пик видеопамяти: {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ")
