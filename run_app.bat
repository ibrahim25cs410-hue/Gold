@echo off
chcp 65001 >nul
title Aura Gold AI
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
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

curl.exe --fail --silent http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
    start "Aura Gold Server" /D "%~dp0" "%PYTHON%" -m uvicorn main:app --app-dir "%~dp0" --host 127.0.0.1 --port 8000
)

timeout /t 5 /nobreak >nul
start "" "%~dp0index.html"
exit