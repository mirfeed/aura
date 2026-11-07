@echo off
rem Пульт Ауры: окно с управлением, сам бот работает скрыто (значок в трее).
rem Старое меню в консоли: scripts\bot.ps1 (start | stop | restart | status | log | unload | game).
cd /d "%~dp0"
start "" pythonw "%~dp0tools\panel.py"
