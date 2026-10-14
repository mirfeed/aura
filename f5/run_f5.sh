#!/usr/bin/env bash
# Сервер F5-TTS для Aura: http://127.0.0.1:8081/v1/tts (из Windows адрес тот же).
# Запуск из Ubuntu:  bash /mnt/d/ai/f5/run_f5.sh   (или aura.bat → 9 из Windows)
# Качество/скорость: F5_NFE_STEP=16 (быстрее) … 48 (чище), по умолчанию 32.
set -euo pipefail
export F5_LISTEN="${F5_LISTEN:-0.0.0.0:8081}"
export F5_VOICES_DIR="${F5_VOICES_DIR:-/mnt/d/ai/f5/voices}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$HOME/f5-tts/.venv/bin/python" /mnt/d/ai/f5/f5_server.py
