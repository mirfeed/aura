@echo off
chcp 65001 >nul
title Aura: отношения
cd /d "%~dp0"
echo Редактор отношений: http://127.0.0.1:8766  (закройте окно, чтобы остановить)
python tools\people_ui.py
pause
