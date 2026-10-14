#!/usr/bin/env bash
# Тест Qwen3-TTS (VoiceDesign: голос по текстовому описанию) в WSL2 Ubuntu.
# Запуск из Ubuntu:  bash /mnt/d/ai/qwen/setup_qwen.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
Q_DIR="$HOME/qwen-tts"
rm -rf "$Q_DIR/.venv"   # прошлая попытка через python -m venv не создалась
mkdir -p "$Q_DIR"
cd "$Q_DIR"

echo "== 1/2 Python-окружение"
uv venv --python 3.12
PY="$Q_DIR/.venv/bin/python"
# torch той же версии, что у F5: берётся из кэша uv.
uv pip install --python "$PY" torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" qwen-tts soundfile
"$PY" -c "import torch, qwen_tts; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| qwen_tts ok')"

echo "== 2/2 Модель VoiceDesign 1.7B"
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'))"
echo "== Готово"
