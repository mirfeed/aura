#!/usr/bin/env bash
# Одна и та же фраза с разными маркерами эмоций голосом из references/<id>.
# Запуск из Ubuntu (сервер Fish должен работать):  bash /mnt/d/ai/fish/test_emotions.sh [aura]
# Результат: D:\ai\fish\emotions\NN_<маркер>.wav
set -uo pipefail
REF="${1:-aura}"
OUT="/mnt/d/ai/fish/emotions"
mkdir -p "$OUT"

PHRASE="Слушай, ты правда думал, что я не замечу? Ладно, давай ещё раз, только теперь нормально."

# Сначала маркеры, которые бот ставит сам по настроению, потом остальные для сравнения.
MARKERS=(
  ""             # без маркера (нейтрально)
  "(joyful)"     # cheerful
  "(amused)"     # amused
  "(curious)"    # curious
  "(frustrated)" # irritated
  "(angry)"      # irritated + восклицание
  "(excited)"    # восклицание
  "(soft tone)"  # tired / withdrawn / многоточие
  "(sad)"
  "(sarcastic)"
  "(whispering)"
  "(shouting)"
  "(laughing)"
  "(sighing)"
)

i=0
for m in "${MARKERS[@]}"; do
  i=$((i + 1))
  name="${m//[() ]/}"; name="${name:-neutral}"
  file="$OUT/$(printf '%02d' "$i")_${name}.wav"
  text="${m:+$m }$PHRASE"
  start=$(date +%s.%N)
  if curl -sS -f -X POST http://127.0.0.1:8080/v1/tts \
      -H 'Content-Type: application/json' \
      -d "{\"text\": \"$text\", \"format\": \"wav\", \"reference_id\": \"$REF\", \"use_memory_cache\": \"on\"}" \
      -o "$file"; then
    printf '%-16s %5.1f c  -> %s\n' "${m:-(нет)}" "$(awk -v s="$start" -v e="$(date +%s.%N)" 'BEGIN{print e-s}')" "$(basename "$file")"
  else
    echo "${m:-(нет)}: ошибка запроса"
  fi
done
echo "Готово: D:\\ai\\fish\\emotions"
