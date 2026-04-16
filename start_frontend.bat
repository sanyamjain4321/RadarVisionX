@echo off
REM ═══════════════════════════════════════════════════════════
REM  SAR Ship Detector — Start Frontend + ONNX Model Server
REM  Run from repo root: start_frontend.bat
REM ═══════════════════════════════════════════════════════════

echo.
echo  ══════════════════════════════════════════
echo   SAR Frontend + ONNX Server (port 5500)
echo  ══════════════════════════════════════════
echo.
echo  Open in browser: http://localhost:5500
echo  Press Ctrl+C to stop
echo.
python backend/serve_onnx.py
