#!/usr/bin/env bash
# Установка сервера F5-TTS (русская модель ESpeech RL-V2) в WSL2 Ubuntu.
# Запуск из Ubuntu:  bash /mnt/d/ai/f5/setup_f5.sh
# Fish уже ставил системные пакеты и uv, поэтому здесь только Python-окружение и модели.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
F5_DIR="$HOME/f5-tts"
mkdir -p "$F5_DIR"
cd "$F5_DIR"

echo "== 1/3 Python-окружение"
[ -d .venv ] || uv venv --python 3.12
PY="$F5_DIR/.venv/bin/python"
# torch той же версии, что у Fish: берётся из кэша uv, заново ~3 ГБ не качается.
uv pip install --python "$PY" torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" f5-tts ruaccent fastapi uvicorn soundfile "torchcodec==0.7.*"
"$PY" -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "== 2/3 Модели (ESpeech, Vocos, RUAccent)"
"$PY" - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt")
try:
    hf_hub_download("ESpeech/ESpeech-TTS-1_RL-V2", "vocab.txt")
except Exception:
    hf_hub_download("ESpeech/ESpeech-TTS-1_podcaster", "vocab.txt")
from f5_tts.infer.utils_infer import load_vocoder
load_vocoder(device="cpu")
from ruaccent import RUAccent
a = RUAccent(); a.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
print(a.process_all("Привет, мы проверяем ударения в словах замок и молоко."))
PY

echo "== 3/3 Готово"
echo "Образцы голоса: D:\\ai\\f5\\voices\\aura\\neutral.wav + neutral.txt (до 12 секунд)."
echo "Запуск сервера: bash /mnt/d/ai/f5/run_f5.sh"
