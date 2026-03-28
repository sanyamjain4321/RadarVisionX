"""
SAR Ship Detection Backend — FastAPI  (Hardened Edition)
=========================================================
Endpoints:
  GET  /health          → backend status + model info (always responds)
  POST /detect          → YOLO inference → JSON bounding boxes
  GET  /train/status    → training progress (poll every 2s)
  POST /train/start     → kick off background YOLOv8 training
  POST /train/cancel    → abort training
  GET  /train/logs      → last N lines of training stdout
"""

import logging
import sys
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("sar-backend")

from routes.detect import router as detect_router
from routes.train  import router as train_router

app = FastAPI(
    title="SAR Ship Detection API",
    description="YOLOv8-based SAR ship detection — CFAR → YOLO → AI-LR pipeline.",
    version="2.1.0",
)

# ── CORS: allow the frontend (served from any origin) ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,          # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        ms = round((time.time() - t0) * 1000)
        logger.info(f"← {request.method} {request.url.path} [{response.status_code}] {ms}ms")
        return response
    except Exception as exc:
        ms = round((time.time() - t0) * 1000)
        logger.error(f"✗ {request.method} {request.url.path} CRASHED after {ms}ms — {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"}
        )

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(detect_router, tags=["detection"])
app.include_router(train_router,  tags=["training"])


# ── Health endpoint (always responds — never crashes) ─────────────────────────
@app.get("/health")
async def health():
    """
    Liveness + model status check.
    Frontend polls this every 2 s to show Online/Offline badge and model type.
    Always returns 200 — even if model failed to load.
    """
    try:
        from model.yolo_inference import get_model_info
        model_info = get_model_info()
        model_loaded = model_info.get("loaded", False)
    except Exception as exc:
        logger.warning(f"/health: model info unavailable — {exc}")
        model_info  = {"loaded": False, "error": str(exc), "model_type": "Not loaded"}
        model_loaded = False

    try:
        from routes.train import _state as train_state
        training = {
            "status":   train_state["status"],
            "progress": train_state["progress"],
            "message":  train_state["message"],
        }
    except Exception:
        training = {"status": "idle", "progress": 0, "message": ""}

    return {
        "status":       "ok",
        "model_loaded": model_loaded,
        "model":        model_info,
        "training":     training,
    }



# ── Startup: pre-warm the model so first /detect is fast ──────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("═" * 52)
    logger.info("  SAR Ship Detection Backend v2.1 starting…")
    logger.info("═" * 52)
    try:
        from model.yolo_inference import _get_model
        _get_model()
        logger.info("✓ YOLO model pre-loaded successfully.")
    except Exception as exc:
        logger.warning(f"⚠ Model pre-load failed (will retry on first request): {exc}")
    logger.info("✓ Backend ready — listening on http://0.0.0.0:8080")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        access_log=True,
    )
