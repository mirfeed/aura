@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist logs mkdir logs
set LOG=logs\setup.log
echo === voice setup %date% %time% > %LOG%
echo Устанавливаю пакеты для голоса, лог: %LOG%
python --version >> %LOG% 2>&1
python -m pip install "discord.py[voice]>=2.7.1" "davey>=0.1.6" "discord-ext-voice-recv==0.5.2a179" "faster-whisper>=1.2.0" "silero-vad>=6.0" "num2words>=0.5.13" >> %LOG% 2>&1
echo --- pip exit code %errorlevel% >> %LOG%
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
  echo --- nvidia-smi >> %LOG%
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv >> %LOG% 2>&1
  python -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" >> %LOG% 2>&1
  echo --- cuda libs exit code %errorlevel% >> %LOG%
) else (
  echo --- no nvidia-smi >> %LOG%
)
echo --- import check >> %LOG%
python -c "import discord,davey,faster_whisper,silero_vad,num2words,torch;from discord.ext import voice_recv;print('OK discord',discord.__version__,'davey',davey.__version__,'torch',torch.__version__,'torch_cuda',torch.cuda.is_available())" >> %LOG% 2>&1
echo --- ollama >> %LOG%
curl -s -m 5 http://localhost:11434/api/tags >> %LOG% 2>&1
echo. >> %LOG%
echo --- running python processes >> %LOG%
tasklist /fi "imagename eq python.exe" >> %LOG% 2>&1
echo === DONE >> %LOG%
echo Готово. Окно можно закрыть.
timeout /t 5 >nul
