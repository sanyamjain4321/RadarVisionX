@echo off
REM ═══════════════════════════════════════════════════════════
REM  SAR Ship Detector — Start FastAPI Backend (Auto-Restart)
REM  Run from repo root: start_backend.bat
REM ═══════════════════════════════════════════════════════════

echo.
echo  ══════════════════════════════════════════
echo   SAR Ship Detection Backend (FastAPI)
echo   Auto-restart on crash enabled
echo  ══════════════════════════════════════════
echo.

cd backend

echo  Installing / verifying dependencies...
pip install -r requirements.txt --quiet

echo.
echo  Backend starting on http://localhost:8000
echo  Press Ctrl+C to stop
echo.

:RESTART
echo  [%TIME%] Starting backend process...
python main.py
echo.
echo  [%TIME%] Backend exited — restarting in 3 seconds...
echo  (Press Ctrl+C to stop)
timeout /t 3 /nobreak >nul
goto RESTART
