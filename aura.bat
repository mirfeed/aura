@echo off
chcp 65001 >nul
title Aura — пульт
cd /d "%~dp0"
set PS=powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bot.ps1

:menu
cls
echo  ==============================
echo           А У Р А
echo  ==============================
call %PS% status
echo.
echo   1  Запустить
echo   2  Перезапустить
echo   3  Остановить (аккуратно, с выходом из войса)
echo   4  Лог — последние события
echo   5  Выгрузить модель из видеопамяти
echo.
echo   6  Настройки (в браузере)
echo   7  Отношения с людьми (в браузере)
echo   8  Отчёт о людях
echo   9  Сервер голоса F5
echo.
echo   0  Выход
echo.
choice /c 1234567890 /n /m "Выбери: "
set k=%errorlevel%
echo.
if %k%==1 call %PS% start
if %k%==2 call %PS% restart
if %k%==3 call %PS% stop
if %k%==4 call %PS% log
if %k%==5 call %PS% unload
if %k%==6 start "Aura: настройки" cmd /c "python tools\settings_ui.py"
if %k%==7 start "Aura: отношения" cmd /c "python tools\people_ui.py"
if %k%==8 python tools\people_report.py --facts
if %k%==9 start "Aura: F5 голос" cmd /c "wsl -- bash /mnt/d/ai/f5/run_f5.sh & pause"
if %k%==10 exit /b
echo.
pause
goto menu
