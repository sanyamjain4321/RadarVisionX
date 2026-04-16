@echo off
REM ═══════════════════════════════════════════════════════════
REM  SAR Ship Detector — Training Pipeline (FIXED)
REM  Run from repo root: start_training.bat
REM ═══════════════════════════════════════════════════════════

echo.
echo  ══════════════════════════════════════════
echo   SAR YOLOv8 Training Pipeline (Fixed)
echo  ══════════════════════════════════════════
echo.
echo  Installing training dependencies...
cd training
pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: pip install failed. Check your Python/pip installation.
    cd ..
    pause
    exit /b 1
)

echo.
echo  Step 1: Validate / Prepare dataset
echo  ─────────────────────────────────────────

REM Check if dataset already exists with images AND labels
python -c "from pathlib import Path; d=Path('dataset/images/train'); l=Path('dataset/labels/train'); ok=d.exists() and l.exists() and any(d.iterdir()) and any(l.iterdir()); exit(0 if ok else 1)" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo  [OK] Dataset already exists - skipping preparation
    goto :train
)

echo  No dataset found - auto-generating synthetic SAR dataset...
echo  (This creates 200 synthetic images for pipeline testing)
echo.
python prepare_ssdd.py --synthetic
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Synthetic dataset generation failed!
    echo  Check output above. Common fixes:
    echo    pip install numpy Pillow
    cd ..
    pause
    exit /b 1
)

REM Verify dataset was actually created
python -c "from pathlib import Path; d=Path('dataset/images/train'); l=Path('dataset/labels/train'); ok=d.exists() and l.exists() and any(d.iterdir()) and any(l.iterdir()); exit(0 if ok else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Dataset files not found after generation!
    echo  Check disk space and permissions.
    cd ..
    pause
    exit /b 1
)
echo  Dataset ready!

:train
echo.
echo  Step 2: Validate dataset structure
echo  ─────────────────────────────────────────
python validate_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Dataset validation FAILED - see output above
    echo  Fix the issues then re-run this script.
    cd ..
    pause
    exit /b 1
)

echo.
echo  Step 3: Train YOLOv8 SAR model
echo  ─────────────────────────────────────────
echo  Using synthetic dataset for quick pipeline test.
echo  For real accuracy, use --ssdd-dir with the real SSDD dataset.
echo.
python train.py --epochs 10 --quick
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Training FAILED - see output above.
    echo  Common fixes:
    echo    pip install ultralytics
    echo    python prepare_ssdd.py --synthetic  ^(re-generate dataset^)
    cd ..
    pause
    exit /b 1
)

echo.
echo  ══════════════════════════════════════════
echo  Training complete!
echo  Model saved to: models\yolov8_sar.pt
echo.
echo  Next steps:
echo    Start the backend: start_backend.bat
echo    Open the UI:       start_frontend.bat
echo  ══════════════════════════════════════════
cd ..
pause
