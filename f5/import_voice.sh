#!/usr/bin/env bash
# Импорт присланных образцов голоса (от подруги и т. п.) для F5.
# 1) Положи файлы в D:\ai\f5\incoming\ и назови по настроению:
#    neutral, cheerful, amused, curious, irritated, tired, withdrawn  (расширение любое: .m4a .mp3 .ogg .wav .opus)
# 2) Запусти в Ubuntu:  bash /mnt/d/ai/f5/import_voice.sh
# Скрипт обрежет тишину, выровняет громкость, сделает 24 кГц моно и положит в voices/aura вместе с текстом.
set -uo pipefail
F5="/mnt/d/ai/f5"
VOICE="${VOICE:-aura}"
IN="$F5/incoming"
OUT="$F5/voices/$VOICE"
mkdir -p "$IN" "$OUT"

found=0
for f in "$IN"/*; do
  [ -f "$f" ] || continue
  mood="$(basename "${f%.*}" | tr 'A-Z' 'a-z')"
  txt="$F5/texts/$mood.txt"
  if [ ! -f "$txt" ]; then
    echo "Пропуск $(basename "$f"): имя должно быть одним из neutral, cheerful, amused, curious, irritated, tired, withdrawn"
    continue
  fi
  ffmpeg -hide_banner -loglevel error -y -i "$f" -af \
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2,areverse,loudnorm=I=-18:TP=-2" \
    -ac 1 -ar 24000 "$OUT/$mood.wav" || { echo "Не удалось прочитать $(basename "$f")"; continue; }
  cp "$txt" "$OUT/$mood.txt"
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/$mood.wav")
  printf '%-10s %5.1f c  <- %s\n' "$mood" "$dur" "$(basename "$f")"
  if awk -v d="$dur" 'BEGIN{exit !(d>12)}'; then
    echo "           ВНИМАНИЕ: длиннее 12 секунд — попроси перезаписать чуть быстрее."
  fi
  found=1
done
[ "$found" = 1 ] || echo "В $IN нет подходящих файлов."
echo "Сервер подхватит образцы сам. Проверка: bash $F5/test_f5.sh"
