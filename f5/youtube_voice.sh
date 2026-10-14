#!/usr/bin/env bash
# Образец голоса из видео на YouTube (только для теста).
# Запуск в Ubuntu:
#   bash /mnt/d/ai/f5/youtube_voice.sh "ССЫЛКА" НАЧАЛО КОНЕЦ [голос] [настроение]
# Пример:
#   bash /mnt/d/ai/f5/youtube_voice.sh "https://youtu.be/xxxx" 12:05 12:13
# Берите 6–10 секунд, где говорит один человек. Тихую фоновую музыку скрипт уберёт сам (Demucs).
# Результат: D:\ai\f5\voices\<голос>\<настроение>.wav + .txt (по умолчанию podcast/neutral).
# Голос Ауры (voices/aura) не трогается. Проверка: bash /mnt/d/ai/f5/test_f5.sh podcast
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,9p "$0"; exit 1; }
URL="$1"; START="$2"; END="$3"; VOICE="${4:-podcast}"; MOOD="${5:-neutral}"
F5="/mnt/d/ai/f5"
PY="$HOME/f5-tts/.venv/bin/python"
OUT="$F5/voices/$VOICE"
TMP="$(mktemp -d)"
mkdir -p "$OUT"

if [ ! -x "$HOME/f5-tts/.venv/bin/yt-dlp" ]; then
  echo "Ставлю yt-dlp (один раз)..."
  (cd "$HOME/f5-tts" && uv pip install -q -U yt-dlp)
fi

if [ -f "$URL" ]; then
  # Вместо ссылки можно дать готовый файл (НАЧАЛО/КОНЕЦ тогда не важны, пишите 0 0).
  cp "$URL" "$TMP/clip.${URL##*.}"
else
  echo "Скачиваю отрывок $START–$END..."
  "$HOME/f5-tts/.venv/bin/yt-dlp" -q --no-playlist -f bestaudio \
    --download-sections "*$START-$END" --force-keyframes-at-cuts \
    -o "$TMP/clip.%(ext)s" "$URL"
fi
SRC="$(ls "$TMP"/clip.* | head -1)"

# Фоновую музыку убираем нейросетью Demucs (оставляет только голос). Отключить: NO_DEMUCS=1
BRIGHT=""
if [ "${NO_DEMUCS:-0}" != 1 ]; then
  if ! "$PY" -c "import demucs" 2>/dev/null; then
    echo "Ставлю Demucs для отделения голоса от музыки (один раз)..."
    (cd "$HOME/f5-tts" && uv pip install -q demucs) || true
  fi
  echo "Отделяю голос от музыки..."
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$SRC" -t 15 -ac 2 -ar 44100 "$TMP/mix.wav"
  if timeout 300 "$PY" - "$TMP/mix.wav" "$TMP/vocals.wav" <<'PYEOF' 2>"$TMP/demucs.log"; then
import sys
import numpy as np, soundfile as sf, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
wav, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
x = torch.from_numpy(wav.T.copy())
ref = x.mean(0)
x = (x - ref.mean()) / (ref.std() + 1e-8)
model = get_model("htdemucs")
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = apply_model(model, x[None], device=dev, progress=False)[0]
voc = out[model.sources.index("vocals")] * (ref.std() + 1e-8) + ref.mean()
sf.write(sys.argv[2], voc.cpu().numpy().T, sr)
print("ok")
PYEOF
    SRC="$TMP/vocals.wav"
    BRIGHT=",highshelf=f=3000:g=5"   # Demucs приглушает верхи, возвращаем
  else
    echo "Demucs не сработал, беру звук как есть:"; tail -3 "$TMP/demucs.log"
  fi
fi

ffmpeg -nostdin -hide_banner -loglevel error -y -i "$SRC" -t 15 -af \
  "highpass=f=70${BRIGHT},silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.25,areverse,loudnorm=I=-18:TP=-2" \
  -ac 1 -ar 24000 "$OUT/$MOOD.wav"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/$MOOD.wav")
printf 'Образец: %s  (%.1f c)\n' "$OUT/$MOOD.wav" "$dur"
if awk -v d="$dur" 'BEGIN{exit !(d>12)}'; then
  echo "ВНИМАНИЕ: длиннее 12 секунд — F5 обрежет звук. Возьмите отрывок короче."
fi

if [ -n "${TEXT:-}" ]; then
  printf '%s\n' "$TEXT" > "$OUT/$MOOD.txt"; echo "Текст: $TEXT"
else
echo "Расшифровываю текст (в первый раз скачается Whisper, ~1,5 ГБ)..."
"$PY" - "$OUT/$MOOD.wav" "$OUT/$MOOD.txt" <<'PYEOF'
import sys
from f5_tts.infer.utils_infer import transcribe
text = transcribe(sys.argv[1], language="russian").strip()
open(sys.argv[2], "w", encoding="utf-8").write(text + "\n")
print("Текст:", text)
PYEOF
fi
rm -rf "$TMP"
echo
echo "Проверь текст в $OUT/$MOOD.txt — он должен совпадать со звуком слово в слово (поправь, если нужно)."
echo "Потом: bash $F5/test_f5.sh $VOICE   -> D:\\ai\\f5\\test"
