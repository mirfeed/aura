#!/usr/bin/env bash
# Проверка сервера: синтезирует фразу голосом Ауры в D:\ai\fish\test.wav
set -euo pipefail
OUT="/mnt/d/ai/fish/test.wav"
REF="${1:-aura}"
time curl -sS -f -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"(joyful) Привет! Я Аура, и это мой новый голос. (soft tone) Ну... как тебе?\", \"format\": \"wav\", \"reference_id\": \"$REF\", \"use_memory_cache\": \"on\"}" \
  -o "$OUT"
echo "Сохранено: D:\\ai\\fish\\test.wav"
