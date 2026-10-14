@echo off
chcp 65001 >nul
title Aura settings
cd /d "%~dp0"
echo Настройки Ауры: http://127.0.0.1:8765  (закройте окно, чтобы остановить)
python tools\settings_ui.py
pause
