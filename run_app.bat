@echo off
chcp 65001 >nul
title Aura Gold AI
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python not found
        pause
        exit /b
    )
    set "PYTHON=python"
)

if not exist "gold_bundle.pkl" (
    echo Model bundle not found. Run: %PYTHON% train_gold_pro.py
    pause
    exit /b
)

powershell.exe -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/health | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    start "Aura Gold Server" /D "%~dp0" "%PYTHON%" -m uvicorn main:app --app-dir "%~dp0" --host 127.0.0.1 --port 8000
)

set "READY="
for /l %%I in (1,1,30) do (
    powershell.exe -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/health | Out-Null; exit 0 } catch { exit 1 }"
    if not errorlevel 1 (
        set "READY=1"
        goto :server_ready
    )
    timeout /t 1 /nobreak >nul
)

echo Server did not become ready. Check the Aura Gold Server window for details.
pause
exit /b 1

:server_ready
start "" "%~dp0index.html"
exit