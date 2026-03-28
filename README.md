# SAR Ship Detection — Production YOLO Integration

## Quick Start (3 commands)

```bat
cd C:\Users\malvi\Downloads\hackathon_repo_v2
pip install -r backend/requirements.txt -q
start_all.bat
```

`start_all.bat` does everything automatically:
1. Checks if `models/yolov8_sar.pt` exists
2. If not → prepares dataset → trains YOLOv8 → exports ONNX
3. Starts FastAPI backend (`localhost:8000`)
4. Starts frontend server (`localhost:5500`)
5. Opens browser

---

## Project Structure

```
hackathon_repo_v2/
├── index.html                  ← Frontend entry point
├── css/styles.css
├── js/
│   ├── yolo.js                 ← YOLO inference (real model only, no simulation)
│   ├── ui.js                   ← UI + YOLO state machine + training controls
│   ├── cfar.js                 ← 2D OS-CFAR detector
│   ├── ai_model.js             ← Online logistic regression filter
│   ├── scene.js                ← Synthetic SAR scene generator
│   └── metrics.js              ← Metrics dashboard
├── backend/
│   ├── main.py                 ← FastAPI app (health, detect, train routes)
│   ├── requirements.txt
│   ├── serve_onnx.py           ← CORS static server (needed for browser ONNX)
│   ├── model/
│   │   └── yolo_inference.py   ← YOLOv8 inference wrapper (SAR preprocessing)
│   └── routes/
│       ├── detect.py           ← POST /detect
│       └── train.py            ← POST /train/start, GET /train/status
├── training/
│   ├── train.py                ← YOLOv8 training script (argparse, auto ONNX export)
│   ├── prepare_ssdd.py         ← Dataset prep (auto-download or synthetic)
│   ├── data.yaml               ← Dataset config
│   └── requirements.txt        ← Training dependencies
├── models/
│   ├── yolov8_sar.pt           ← SAR fine-tuned weights (after training)
│   └── yolov8_sar.onnx         ← Browser-ready ONNX (after training)
├── start_all.bat               ← All-in-one: train → backend → frontend
├── start_all.sh                ← Same for Linux/macOS
├── start_backend.bat           ← Backend only
└── start_frontend.bat          ← Frontend only
```

---

## Pipeline Architecture

```
SAR Image
    │
    ▼
2D OS-CFAR ──────────────────── Candidate ROIs (30–80 detections)
    │
    ▼
YOLOv8 (real trained model)
  • Backend mode:  POST /detect → FastAPI → yolov8_sar.pt
  • ONNX mode:     onnxruntime-web → yolov8_sar.onnx (browser-side)
  │
  ├── Match YOLO boxes ↔ CFAR ROIs (IoU ≥ 0.10)
  ├── Enrich CFAR dets with YOLO confidence (blended)
  └── Add YOLO-only detections (ships CFAR missed)
    │
    ▼
AI Logistic Regression Filter
  • Features: conf, area, aspect ratio, elongation, center dist
  • Online training: accumulates samples, retrains every 4+ detections
    │
    ▼
Final Output: bounding boxes + TP/FP/FN metrics
```

---

## Step-by-Step Manual Setup

### 1. Prerequisites

```
Python 3.9–3.11
pip (latest)
4 GB RAM minimum (8 GB recommended for training)
GPU optional — CPU training works (~30 min for 30 epochs on modern CPU)
```

### 2. Install dependencies

```bat
pip install -r backend/requirements.txt
pip install -r training/requirements.txt
```

### 3. Prepare dataset

**Option A — Synthetic (fast, pipeline test only):**
```bat
cd training
python prepare_ssdd.py --synthetic
```
Generates 200 SAR-like images in ~5 seconds.

**Option B — Real SSDD dataset (recommended for accuracy):**
```bat
cd training
python prepare_ssdd.py --ssdd-dir /path/to/ssdd
```
Or let it auto-download:
```bat
python prepare_ssdd.py
```

### 4. Train YOLOv8

```bat
cd training
python train.py --epochs 30          # ~30 min on CPU
python train.py --epochs 50          # better accuracy
python train.py --quick              # 2 epochs (smoke test only)
```

After training:
- `models/yolov8_sar.pt` — PyTorch weights (used by backend)
- `models/yolov8_sar.onnx` — ONNX model (used by browser)

### 5. Start backend

```bat
start_backend.bat
```
Or manually:
```bat
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: http://localhost:8000/health

Expected response:
```json
{
  "status": "ok",
  "model": {
    "loaded": true,
    "is_sar": true,
    "model_type": "SAR Fine-tuned",
    "name": "yolov8_sar"
  }
}
```

### 6. Start frontend

```bat
start_frontend.bat
```
Or manually:
```bat
python backend/serve_onnx.py
```

Open: **http://localhost:5500**

> ⚠ Do NOT open `index.html` as `file://` — onnxruntime-web requires HTTP.

---

## Backend API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check + model info |
| `/detect` | POST | YOLO inference on image |
| `/train/start` | POST | Start background training |
| `/train/status` | GET | Poll training progress |
| `/train/cancel` | POST | Abort training |
| `/train/logs` | GET | Last N training log lines |
| `/docs` | GET | Auto-generated Swagger UI |

### POST /detect

```
Content-Type: multipart/form-data
Fields:
  image:       image file (PNG/JPG)
  conf_thresh: float  (default 0.20)
  iou_thresh:  float  (default 0.45)
```

Response:
```json
{
  "detections": [
    {"x1":120,"y1":80,"x2":145,"y2":100,"confidence":0.82,"class_name":"ship","source":"yolo-sar"}
  ],
  "inference_ms": 145,
  "model_used": "SAR Fine-tuned",
  "model_is_sar": true
}
```

### POST /train/start

```json
{ "epochs": 30, "quick": false }
```

### GET /train/status

```json
{
  "status": "running",
  "progress": 45,
  "epoch": 14,
  "total_epochs": 30,
  "message": "Training epoch 14/30…",
  "elapsed_s": 420.5
}
```

---

## UI Usage

1. Open **http://localhost:5500**
2. The UI auto-pings the backend on load
3. If SAR model found → YOLO badge shows **READY (green)**
4. If no SAR model → Training panel appears → click **▶ Start YOLOv8 Training**
5. Click **▶ Arm Detector** (CFAR pipeline initialisation)
6. Click **▶ Run CFAR → YOLO → AI**

### YOLO Status States

| Badge | Meaning |
|-------|---------|
| CHECKING | Pinging backend on startup |
| NO MODEL | Backend online but only COCO model available — train needed |
| TRAINING | training in progress (shows epoch/progress bar) |
| LOADING | ONNX model being loaded into browser |
| READY (green) | SAR model ready — full pipeline active |
| ERROR | Something failed — check log panel |

---

## Troubleshooting

### "Backend Offline" badge
```bat
cd backend
uvicorn main:app --reload
```
Check: http://localhost:8000/health

### ONNX model not loading
- Open the page via HTTP, not `file://`
- Run `start_frontend.bat` or `python backend/serve_onnx.py`
- Confirm `models/yolov8_sar.onnx` exists (run training first)

### Training fails immediately
```bat
cd training
python prepare_ssdd.py --synthetic   # generate test dataset first
python train.py --quick              # 2-epoch smoke test
```

### Out of memory during training
```bat
python train.py --batch 4 --model yolov8n --epochs 20
```

### CORS error in browser console
- You opened `index.html` as `file://` — use the HTTP server instead:
```bat
start_frontend.bat
```

---

## Model Performance (expected after 30 epochs on SSDD)

| Metric | COCO pretrained | SAR fine-tuned (30ep) |
|--------|----------------|----------------------|
| mAP@50 | ~0.15 (boat class only) | ~0.55–0.75 |
| Precision | ~0.30 | ~0.70–0.85 |
| Recall | ~0.25 | ~0.65–0.80 |
| Inference (CPU) | ~300ms | ~300ms |
| Inference (GPU) | ~25ms | ~25ms |

*Results vary with dataset size and training epochs.*

---

## Key Design Decisions

- **No simulation**: YOLO runs ONLY as a real trained model. If no model → error + training prompt.
- **SAR preprocessing**: Log-scale + CLAHE on every image before YOLO (matches training preprocessing).
- **Model priority**: `yolov8_sar.pt` > `yolov8n.pt` (COCO fallback). Frontend shows warning if COCO only.
- **Auto-reload**: After training via `/train/start`, the backend reloads the new model automatically (no restart needed).
- **CFAR as ROI generator**: CFAR reduces full-image search space; YOLO validates candidates; AI filter removes remaining FPs.
