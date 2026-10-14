"""
Выбрать лучший дубль каждого настроения: Whisper сверяет речь с текстом (F5 нужен точный текст образца).
Запуск из Windows:  python qwen\\pick_moods.py  → f5\\voices\\qwen3\\<настроение>.wav + .txt
"""
import difflib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

from voice.stt import _register_cuda_dlls  # noqa: E402

TAKES = ROOT / "qwen" / "out" / "moods"
TEXTS = ROOT / "f5" / "texts"
VOICE = ROOT / "f5" / "voices" / "qwen3"


def words(s: str) -> list:
    return re.findall(r"[а-яa-z0-9]+", s.lower().replace("ё", "е"))


_register_cuda_dlls()
model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="int8_float16",
                     download_root=str(config.MODELS_DIR / "whisper"))
moods = sorted({p.stem.rsplit("_", 1)[0] for p in TAKES.glob("*_*.wav")})
for mood in moods:
    target = (TEXTS / f"{mood}.txt").read_text(encoding="utf-8").strip()
    scored = []
    for take in sorted(TAKES.glob(f"{mood}_*.wav")):
        segs, _ = model.transcribe(str(take), language="ru", beam_size=5)
        heard = " ".join(s.text.strip() for s in segs)
        score = difflib.SequenceMatcher(None, words(target), words(heard)).ratio()
        scored.append((score, take, heard))
    score, best, heard = max(scored, key=lambda x: x[0])
    shutil.copy(best, VOICE / f"{mood}.wav")
    (VOICE / f"{mood}.txt").write_text(target, encoding="utf-8")
    others = ", ".join(f"{t.stem[-1]}:{s:.2f}" for s, t, _ in scored)
    print(f"{mood:10s} → {best.name}  совпадение {score:.2f}  (дубли {others})")
    if score < 0.9:
        print(f"           слышно: «{heard}»")
