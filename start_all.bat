@echo off
setlocal
REM ═══════════════════════════════════════════════════════════════
REM  SAR Ship Detector — start_all.bat (FIXED)
REM  Fully automated: dataset → train → backend → open UI
REM  Run from repo root:  start_all.bat
REM ═══════════════════════════════════════════════════════════════

echo.
echo  ══════════════════════════════════════════════════════
echo   SAR Ship Detector v2.0 — Full Auto-Start (Fixed)
echo  ══════════════════════════════════════════════════════
echo.

REM ── Step 1: Check for trained SAR model ───────────────────────
if exist "models\yolov8_sar.pt" (
    echo  [1/4] SAR model found: models\yolov8_sar.pt
    goto :start_backend
)

echo  [1/4] No SAR model found — starting training pipeline...
echo.

REM ── Step 2: Install training dependencies ─────────────────────
echo  Installing training dependencies...
pip install -r training\requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: pip install failed. Ensure Python is in PATH.
    pause & exit /b 1
)

REM ── Step 3: Prepare dataset ────────────────────────────────────
echo.
echo  [2/4] Preparing dataset...
echo  ─────────────────────────────────────────────────────
cd training

REM Check if dataset already exists (images AND labels)
python -c "from pathlib import Path; d=Path('dataset/images/train'); l=Path('dataset/labels/train'); ok=d.exists() and l.exists() and any(d.iterdir()) and any(l.iterdir()); exit(0 if ok else 1)" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo  Dataset already exists - skipping preparation
    goto :do_train
)

echo  No dataset found - generating synthetic SAR data...
python prepare_ssdd.py --synthetic
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Synthetic generation failed! Trying again with verbose output...
    python prepare_ssdd.py --synthetic
    if %ERRORLEVEL% NEQ 0 (
        echo  ═══════════════════════════════════════════
        echo  ERROR: Dataset preparation completely failed.
        echo  Fix: pip install numpy Pillow
        echo  Then re-run this script.
        echo  ═══════════════════════════════════════════
        cd ..
        pause & exit /b 1
    )
)

REM Verify dataset
python -c "from pathlib import Path; d=Path('dataset/images/train'); l=Path('dataset/labels/train'); n=len(list(d.glob('*.jpg'))+list(d.glob('*.png'))); print(f'  Dataset: {n} train images'); exit(0 if n>0 else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Dataset verification failed.
    cd ..
    pause & exit /b 1
)

:do_train
REM ── Step 4: Train YOLOv8 ──────────────────────────────────────
echo.
echo  [3/4] Training YOLOv8 SAR model (10 quick epochs)...
echo  For full accuracy, run: start_training.bat  (50 epochs)
echo  ─────────────────────────────────────────────────────
python train.py --epochs 10
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Training failed — see output above.
    echo  Try: python training\prepare_ssdd.py --synthetic
    cd ..
    pause & exit /b 1
)
cd ..

REM ── Verify model was saved ─────────────────────────────────────
if not exist "models\yolov8_sar.pt" (
    echo  WARNING: Training finished but model not found in models\
    echo  Continuing with COCO pretrained model (yolov8n.pt if available)
    echo  CFAR detection will still work without YOLO.
)

:start_backend
REM ── Step 5: Install backend dependencies ──────────────────────
echo.
echo  [4/4] Starting services...
echo  ─────────────────────────────────────────────────────
echo  Installing backend dependencies...
pip install -r backend\requirements.txt --quiet

REM ── Step 6: Start FastAPI backend in background ────────────────
echo  Starting FastAPI backend on http://localhost:8000 ...
start "SAR Backend" /B cmd /c "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 | tee ..\backend.log"

REM ── Step 7: Start frontend server in background ───────────────
echo  Starting frontend server on http://localhost:5500 ...
start "SAR Frontend" /B cmd /c "python backend\serve_onnx.py 2>&1 | tee frontend.log"

REM ── Step 8: Wait for services to start ────────────────────────
echo  Waiting for services to start...
timeout /t 5 /nobreak >nul

REM ── Step 9: Open browser ──────────────────────────────────────
echo  Opening browser...
start http://localhost:5500

echo.
echo  ══════════════════════════════════════════════════════
echo   All services running!
echo.
echo   Frontend: http://localhost:5500
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
echo   CFAR detection works with or without the backend.
echo   For YOLO pipeline: ensure backend shows connected in UI.
echo.
echo   Close this window to stop all services.
echo  ══════════════════════════════════════════════════════
echo.
pause
