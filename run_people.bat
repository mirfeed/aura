@echo off
chcp 65001 >nul
title Aura: people
cd /d "%~dp0"
python tools\people_report.py --facts %*
pause
