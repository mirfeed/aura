#!/usr/bin/env bash
# Сервер Fish Speech для Aura: http://127.0.0.1:8080/v1/tts (из Windows адрес тот же).
# Запуск из Ubuntu:  bash /mnt/d/ai/fish/run_fish.sh   (или run_fish.bat из Windows)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/fish-speech"

# --compile: синтез в разы быстрее, но первый запуск компилирует несколько минут.
# Без него: FISH_COMPILE=0 bash run_fish.sh
COMPILE_FLAG="--compile"
[ "${FISH_COMPILE:-1}" = "0" ] && COMPILE_FLAG=""

exec uv run tools/api_server.py \
  --listen 0.0.0.0:8080 \
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --decoder-config-name modded_dac_vq \
  $COMPILE_FLAG
