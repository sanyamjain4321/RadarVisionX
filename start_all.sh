#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  SAR Ship Detector — start_all.sh (Linux / macOS)
#  Fully automated: dataset → train → export → backend → open UI
#  Usage:  chmod +x start_all.sh && ./start_all.sh
# ═══════════════════════════════════════════════════════════════

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo ""
echo "══════════════════════════════════════════════════════"
echo " SAR Ship Detector v2.0 — Full Auto-Start"
echo "══════════════════════════════════════════════════════"
echo ""

# ── Step 1: Check for trained SAR model ───────────────────────
if [ -f "models/yolov8_sar.pt" ]; then
    echo "[1/4] SAR model found: models/yolov8_sar.pt"
else
    echo "[1/4] No SAR model — starting training pipeline..."

    # Install training deps
    pip install -r training/requirements.txt -q

    # Prepare dataset
    echo ""
    echo "[2/4] Preparing dataset..."
    cd training
    python prepare_ssdd.py
    cd "$REPO"

    # Train
    echo ""
    echo "[3/4] Training YOLOv8 (30 epochs)..."
    echo "This may take 10–90 minutes on your hardware."
    cd training
    python train.py --epochs 30
    cd "$REPO"

    if [ ! -f "models/yolov8_sar.pt" ]; then
        echo "ERROR: Training finished but model not found in models/"
        exit 1
    fi
    echo "Training complete! → models/yolov8_sar.pt"
fi

# ── Step 4: Install backend deps ──────────────────────────────
echo ""
echo "[4/4] Starting services..."
pip install -r backend/requirements.txt -q

# ── Start FastAPI backend ──────────────────────────────────────
echo "Starting backend on http://localhost:8000..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd "$REPO"

# ── Start frontend server ──────────────────────────────────────
echo "Starting frontend on http://localhost:5500..."
python backend/serve_onnx.py &
FRONTEND_PID=$!

# ── Wait and open browser ─────────────────────────────────────
sleep 3
echo ""
echo "Opening browser..."
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5500
elif command -v open &>/dev/null; then
    open http://localhost:5500
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo " All services running!"
echo ""
echo " Frontend: http://localhost:5500"
echo " Backend:  http://localhost:8000"
echo " API Docs: http://localhost:8000/docs"
echo ""
echo " Press Ctrl+C to stop all services."
echo "══════════════════════════════════════════════════════"

# Wait and clean up on Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
