"""
HTTP-сервер озвучки F5-TTS (русская модель ESpeech) для Aura.

    POST /v1/tts  {"text": "...", "reference_id": "aura", "mood": "cheerful"}  ->  audio/wav (24 кГц, моно)
    GET  /v1/health

Голоса лежат в папке VOICES_DIR (по умолчанию D:\\ai\\f5\\voices):

    voices/<голос>/neutral.wav + neutral.txt     обязательный образец (до 12 секунд!) и его точный текст
    voices/<голос>/cheerful.wav + cheerful.txt   необязательные: по одному на настроение бота
    (cheerful, amused, curious, irritated, tired, withdrawn) — эмоцию F5 берёт из образца.

Если образца для настроения нет, берётся neutral.  Файлы перечитываются при изменении,
перезапуск не нужен.  Ударения ставит RUAccent; слова, где «+» уже стоит, он не трогает.
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT

MODEL_REPO = os.getenv("F5_MODEL_REPO", "ESpeech/ESpeech-TTS-1_RL-V2")
MODEL_FILE = os.getenv("F5_MODEL_FILE", "espeech_tts_rlv2.pt")
VOCAB_REPO = os.getenv("F5_VOCAB_REPO", "ESpeech/ESpeech-TTS-1_podcaster")
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
VOICES_DIR = Path(os.getenv("F5_VOICES_DIR", "/mnt/d/ai/f5/voices"))
NFE_STEP = int(os.getenv("F5_NFE_STEP", "32"))       # больше — чище, но медленнее (16–48)
CROSS_FADE = float(os.getenv("F5_CROSS_FADE", "0.15"))
TAIL_SILENCE = float(os.environ.get("F5_TAIL_SILENCE", "0.25"))   # секунд тишины после фразы
LISTEN = os.getenv("F5_LISTEN", "0.0.0.0:8080")
MAX_REF_SECONDS = 12.0

STRESSED_WORD = r"[А-Яа-яЁё]*\+[А-Яа-яЁё+]*"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)



def _patch_onnx_token_type_ids() -> None:
    """Модель омографов RUAccent требует token_type_ids, а свежий токенизатор
    их не отдаёт, и RUAccent падает на фразах с омографами. Дописываем нули."""
    import onnxruntime as ort

    original = ort.InferenceSession.run
    if getattr(original, "_aura_patched", False):
        return

    def run(self, output_names, input_feed, *args, **kwargs):
        if isinstance(input_feed, dict) and "token_type_ids" not in input_feed:
            needed = {i.name for i in self.get_inputs()}
            if "token_type_ids" in needed and "input_ids" in input_feed:
                input_feed = dict(input_feed)
                input_feed["token_type_ids"] = np.zeros_like(input_feed["input_ids"])
        return original(self, output_names, input_feed, *args, **kwargs)

    run._aura_patched = True
    ort.InferenceSession.run = run


class Engine:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"device: {device}")
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        try:
            vocab_path = hf_hub_download(MODEL_REPO, "vocab.txt")
        except Exception:  # noqa: BLE001
            vocab_path = hf_hub_download(VOCAB_REPO, "vocab.txt")
        self.model = load_model(DiT, MODEL_CFG, model_path, vocab_file=vocab_path).to(device)
        self.vocoder = load_vocoder().to(device)
        log(f"model loaded: {MODEL_REPO}/{MODEL_FILE}")

        _patch_onnx_token_type_ids()
        from ruaccent import RUAccent

        self.accent = RUAccent()
        self.accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
        log("RUAccent loaded")

        self.lock = threading.Lock()
        self._refs: dict[Path, tuple[float, str, str]] = {}

    def stress(self, text: str) -> str:
        try:
            return self.accent.process_all(text, skip_regex=STRESSED_WORD)
        except Exception as exc:  # noqa: BLE001
            log(f"RUAccent failed on phrase, trying by parts: {exc}")
        parts = re.split(r"(?<=[,.!?;:…])\s+", text)
        out = []
        for part in parts:
            try:
                out.append(self.accent.process_all(part, skip_regex=STRESSED_WORD))
            except Exception as exc:  # noqa: BLE001
                log(f"RUAccent failed, part as is: {exc}")
                out.append(part)
        return " ".join(out)

    def reference(self, voice: str, mood: str) -> tuple[str, str]:
        """(путь к подготовленному образцу, текст образца с ударениями)."""
        folder = VOICES_DIR / voice
        for name in (mood, "neutral"):
            wav = folder / f"{name}.wav"
            txt = folder / f"{name}.txt"
            if wav.is_file() and txt.is_file():
                break
        else:
            raise HTTPException(404, f"нет образца: {folder}/neutral.wav + neutral.txt")
        mtime = max(wav.stat().st_mtime, txt.stat().st_mtime)
        cached = self._refs.get(wav)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        info = sf.info(str(wav))
        if info.duration > MAX_REF_SECONDS:
            log(f"WARNING {wav.name}: {info.duration:.1f} s > {MAX_REF_SECONDS:.0f} s — "
                "F5 обрежет звук, а текст останется полным, речь испортится. Обрежьте образец.")
        ref_text = self.stress(txt.read_text(encoding="utf-8").strip())
        ref_audio, ref_text = preprocess_ref_audio_text(str(wav), ref_text, show_info=lambda *_: None)
        self._refs[wav] = (mtime, ref_audio, ref_text)
        log(f"reference {voice}/{wav.stem}: {info.duration:.1f} s | {ref_text}")
        return ref_audio, ref_text

    def synthesize(self, text: str, voice: str, mood: str, speed: float) -> bytes:
        ref_audio, ref_text = self.reference(voice, mood)
        text = re.sub(r"[\s,;:—-]+$", "", text.strip())
        if text and text[-1] not in ".!?":
            text += "."
        # Лишний пробел в конце даёт F5 чуть больше времени на последнее слово.
        gen_text = self.stress(text) + " "
        with self.lock:
            t0 = time.monotonic()
            wave, sr, _ = infer_process(
                ref_audio, ref_text, gen_text, self.model, self.vocoder,
                cross_fade_duration=CROSS_FADE, nfe_step=NFE_STEP, speed=speed,
                show_info=lambda *_: None, progress=None,
            )
        if wave is None:
            raise HTTPException(400, "пустой текст")
        audio = np.asarray(wave, dtype=np.float32)
        # Конец фразы у F5 обрывается резко: мягко гасим последние 60 мс
        # и добавляем немного тишины, чтобы речь «договаривалась».
        fade = min(audio.size, int(sr * 0.06))
        if fade:
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32) ** 2
        audio = np.concatenate([audio, np.zeros(int(sr * TAIL_SILENCE), dtype=np.float32)])
        log(f"{voice}/{mood}: {audio.size / sr:.1f} s audio in {time.monotonic() - t0:.2f} s | {gen_text}")
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()


class TTSRequest(BaseModel):
    text: str
    reference_id: str | None = "aura"
    mood: str = "neutral"
    speed: float = 1.0
    format: str = "wav"

    model_config = {"extra": "ignore"}   # лишние поля в запросе просто игнорируются


app = FastAPI(title="Aura F5-TTS")
engine: Engine | None = None


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok" if engine else "loading"}


@app.post("/v1/tts")
def tts(req: TTSRequest) -> Response:
    if engine is None:
        raise HTTPException(503, "model is loading")
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    voice = re.sub(r"[^\w\-]", "", req.reference_id or "aura") or "aura"
    data = engine.synthesize(req.text, voice, (req.mood or "neutral").lower(), req.speed)
    return Response(content=data, media_type="audio/wav")


def main() -> None:
    global engine
    engine = Engine()
    try:   # прогрев, чтобы первый ответ в войсе был быстрым
        engine.synthesize("Привет, это проверка голоса.", "aura", "neutral", 1.0)
    except Exception as exc:  # noqa: BLE001
        log(f"warmup skipped: {getattr(exc, 'detail', exc)}")
    host, port = LISTEN.rsplit(":", 1)
    log(f"ready on http://{host}:{port}/v1/tts")
    uvicorn.run(app, host=host, port=int(port), log_level="warning")


if __name__ == "__main__":
    main()
