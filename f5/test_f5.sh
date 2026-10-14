#!/usr/bin/env bash
# Проверка сервера F5: одна фраза под разные настроения -> D:\ai\f5\test\<mood>.wav
# Запуск (сервер должен работать):  bash /mnt/d/ai/f5/test_f5.sh [голос]
set -uo pipefail
VOICE="${1:-aura}"
OUT="/mnt/d/ai/f5/test"
mkdir -p "$OUT"
PHRASE="Слушай, ты правда думал, что я не замечу? Ладно, давай ещё раз, только теперь нормально. Мирф+ид, ты где?"
for mood in neutral cheerful amused curious irritated tired withdrawn; do
  start=$(date +%s.%N)
  if curl -sS -f -X POST http://127.0.0.1:8081/v1/tts -H 'Content-Type: application/json' \
      -d "{\"text\": \"$PHRASE\", \"reference_id\": \"$VOICE\", \"mood\": \"$mood\"}" -o "$OUT/$mood.wav"; then
    printf '%-10s %s c\n' "$mood" "$(awk -v s="$start" -v e="$(date +%s.%N)" 'BEGIN{printf "%.1f", e-s}')"
  else
    echo "$mood: ошибка"
  fi
done
echo "Готово: D:\\ai\\f5\\test  (настроения без своего образца звучат как neutral)"
