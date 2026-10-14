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

set CRASHES=0
:run
echo === start %date% %time% >> logs\console.log
echo [%time:~0,8%] Aura работает. Остановить — aura.bat, пункт 3 (так она аккуратно выйдет из войса).
python main.py >> logs\console.log 2>&1
set CODE=%errorlevel%
echo === exit code %CODE% %date% %time% >> logs\console.log
if %CODE%==0 (
    echo [%time:~0,8%] Aura остановлена.
    timeout /t 3 >nul
    exit /b 0
)

rem Упала — перезапустить, но не больше 5 раз подряд (например, если неверный токен).
set /a CRASHES+=1
if %CRASHES% GEQ 5 (
    echo Aura упала %CRASHES% раз подряд — не перезапускаю. Смотри logs\console.log
    pause
    exit /b %CODE%
)
echo [%time:~0,8%] Aura упала (код %CODE%) — перезапуск через 15 секунд... (%CRASHES%/5)
timeout /t 15 /nobreak >nul
goto run
