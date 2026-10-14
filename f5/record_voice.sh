#!/usr/bin/env bash
# Запись образца голоса для F5 под одно настроение.
#   bash /mnt/d/ai/f5/record_voice.sh cheerful        (или neutral, amused, curious, irritated, tired, withdrawn)
#   bash /mnt/d/ai/f5/record_voice.sh all             (все по очереди)
# Читай текст с той эмоцией, которую записываешь, — F5 перенимает интонацию образца.
set -uo pipefail
F5="/mnt/d/ai/f5"
VOICE="${VOICE:-aura}"
OUT="$F5/voices/$VOICE"
mkdir -p "$OUT" "$F5/raw"

# Имена файлов остаются английскими (так их зовёт бот), на экране — по-русски.
declare -A RU=(
  [neutral]="спокойно, обычным голосом"
  [cheerful]="весело, радостно"
  [amused]="со смехом, тебе забавно"
  [curious]="с любопытством, заинтересованно"
  [irritated]="раздражённо"
  [tired]="устало, сонно"
  [withdrawn]="отстранённо, тихо, без желания говорить"
)

record() {
  local mood="$1" txt="$F5/texts/$1.txt"
  [ -f "$txt" ] || { echo "Нет текста $txt"; return 1; }
  echo
  echo "================ ${RU[$mood]:-$mood} ================"
  echo "Прочитай ${RU[$mood]:-$mood}:"
  echo
  echo "    $(cat "$txt")"
  echo
  read -r -p "Enter — начать запись, потом q — остановить. " _
  ffmpeg -hide_banner -loglevel error -stats -y -f pulse -thread_queue_size 4096 -fragment_size 19200 \
    -i default -ac 1 -ar 48000 "$F5/raw/$mood.wav"
  # обрезать тишину по краям, выровнять громкость, 24 кГц моно — как у модели
  ffmpeg -hide_banner -loglevel error -y -i "$F5/raw/$mood.wav" -af \
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2,areverse,loudnorm=I=-18:TP=-2" \
    -ac 1 -ar 24000 "$OUT/$mood.wav"
  cp "$txt" "$OUT/$mood.txt"
  local dur
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/$mood.wav")
  echo "Сохранено: voices/$VOICE/$mood.wav (${dur%.*} c)"
  if awk -v d="$dur" 'BEGIN{exit !(d>12)}'; then
    echo "ВНИМАНИЕ: длиннее 12 секунд — запиши чуть быстрее, иначе F5 испортит речь."
  fi
}

if [ "${1:-}" = "all" ]; then
  for m in neutral cheerful amused curious irritated tired withdrawn; do record "$m"; done
else
  record "${1:?Укажи настроение: neutral, cheerful, amused, curious, irritated, tired, withdrawn или all}"
fi
echo
echo "Сервер подхватит новые образцы сам. Проверка: bash $F5/test_f5.sh"
