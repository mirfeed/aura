@echo off
chcp 65001 >nul
title Aura bot
cd /d "%~dp0"
if not exist logs mkdir logs

rem Ollama: если не отвечает — запустить в отдельном свёрнутом окне и дождаться.
curl -s -m 2 http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo Запускаю Ollama...
    start "Ollama" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    for /l %%i in (1,1,30) do (
        timeout /t 1 /nobreak >nul
        curl -s -m 1 http://localhost:11434/api/version >nul 2>&1 && goto ollama_ok
    )
    echo Ollama не запустилась за 30 секунд — бот всё равно стартует, но отвечать не сможет.
)
:ollama_ok

echo === start %date% %time% >> logs\console.log
echo Aura запущена. Чтобы остановить — закройте это окно или нажмите Ctrl+C.
python main.py >> logs\console.log 2>&1
echo === exit code %errorlevel% %date% %time% >> logs\console.log
