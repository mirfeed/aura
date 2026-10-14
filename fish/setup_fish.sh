#!/usr/bin/env bash
# Установка сервера Fish Speech (OpenAudio S1-mini) в WSL2 Ubuntu.
# Запуск из Ubuntu:  bash /mnt/d/ai/fish/setup_fish.sh
# Повторный запуск безопасен: уже сделанные шаги пропускаются.
set -euo pipefail

FISH_DIR="$HOME/fish-speech"
# Последняя версия fish-speech с OpenAudio S1-mini (дальше репозиторий перешёл на S2, ему нужно 24 ГБ).
FISH_COMMIT="d3df505"
MODEL_DIR="checkpoints/openaudio-s1-mini"

echo "== 1/5 Системные пакеты"
sudo apt-get update
sudo apt-get install -y git curl build-essential python3-dev portaudio19-dev libsox-dev ffmpeg

echo "== 2/5 uv (менеджер Python)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== 3/5 fish-speech ($FISH_COMMIT)"
if [ ! -d "$FISH_DIR/.git" ]; then
  git clone https://github.com/fishaudio/fish-speech.git "$FISH_DIR"
fi
cd "$FISH_DIR"
git fetch -q origin
git checkout -q "$FISH_COMMIT"
uv sync --python 3.12 --extra cu128

echo "== 4/5 Видеокарта"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
uv run python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "== 5/5 Модель OpenAudio S1-mini"
if [ ! -f "$MODEL_DIR/codec.pth" ]; then
  echo "Модель закрыта соглашением. Если ещё не сделал:"
  echo "  1) открой https://huggingface.co/fishaudio/openaudio-s1-mini и прими условия"
  echo "  2) создай токен (Read): https://huggingface.co/settings/tokens"
  uvx --from 'huggingface_hub[cli]' hf auth login
  uvx --from 'huggingface_hub[cli]' hf download fishaudio/openaudio-s1-mini --local-dir "$MODEL_DIR"
fi

mkdir -p references/aura
echo
echo "Готово. Образец голоса положи в $FISH_DIR/references/aura/ : sample.wav и sample.lab (текст записи)."
echo "Запуск сервера: bash /mnt/d/ai/fish/run_fish.sh"
