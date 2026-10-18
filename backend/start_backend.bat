@echo off
REM ================================================================
REM  Legal Triage — Backend Startup Script
REM  Starts FastAPI with uvicorn, MUST be run from the "backend" dir
REM ================================================================

cd /d "%~dp0"

echo.
echo  ========================================
echo   Legal Triage — Backend API Server
echo  ========================================
echo.

REM The shared project virtual environment is one level above backend.
if not exist "..\.venv\Scripts\python.exe" (
    echo  [ERROR] Virtual environment not found.
    echo  Please create ..\.venv and install backend\requirements.txt
    pause
    exit /b 1
)

echo  [OK] Project virtual environment found
echo  [>>] Starting FastAPI at http://127.0.0.1:8000
echo  [>>] API docs: http://127.0.0.1:8000/docs
echo.
echo  Make sure LM Studio is running at http://127.0.0.1:1234
echo.

..\.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
