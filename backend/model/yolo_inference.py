"""
YOLOv8 Inference Wrapper — Production SAR Edition (Hardened)
==============================================================
Model priority (highest to lowest):
  1. ../models/yolov8_sar.pt   — YOUR fine-tuned SAR model (best!)
  2. ../models/yolov8n.pt      — COCO pretrained (fallback, suboptimal)
  3. yolov8n.pt                — Auto-download from Ultralytics (last resort)

SAR preprocessing pipeline:
  Raw image → Log-scale compression → CLAHE contrast → 3-channel RGB → YOLO

CLAHE is critical: SAR ships often appear as tiny bright spots against
dark ocean backscatter. Without contrast enhancement, even SAR-trained
models miss dim targets.
"""

import time
import logging
import os
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("sar-backend.yolo")

# ── Model state ──────────────────────────────────────────────────────────────
_yolo_model        = None
_loaded_model_path = None
_load_error        = None         # last exception string if model failed

# ── Path resolution ───────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.resolve()
MODELS_DIR  = REPO_ROOT / "models"

# Ordered model preference: SAR-trained > COCO general
_MODEL_CANDIDATES = [
    MODELS_DIR / "yolov8_sar.pt",       # fine-tuned on SSDD (best!)
    MODELS_DIR / "yolov8n.pt",           # COCO pretrained (local models/ cache)
    REPO_ROOT   / "yolov8n.pt",          # pre-bundled model at repo root
    "yolov8n.pt",                         # Ultralytics auto-download (last resort)
]



def _resolve_model_path() -> str:
    """Return the best available model path."""
    for candidate in _MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return str(p)
    # Fall through to auto-download (last item is a filename, not a path)
    return str(_MODEL_CANDIDATES[-1])


def _reload_model():
    """Force-reload the model (called after training completes)."""
    global _yolo_model, _loaded_model_path, _load_error
    _yolo_model        = None
    _loaded_model_path = None
    _load_error        = None
    _get_model()   # re-resolve and load


def _get_model():
    """
    Lazy loader — loads once, reuses across all requests.
    Retries on every call if model is not yet loaded (no permanent failure cache).
    Never raises: on failure, sets _load_error and returns None.
    """
    global _yolo_model, _loaded_model_path, _load_error

    # Already loaded and healthy — fast path
    if _yolo_model is not None:
        return _yolo_model

    # ── FIX: Removed permanent failure cache ──────────────────────────
    # Previously: if _load_error is not None: return None
    # This caused YOLO to be permanently disabled after any single load
    # failure (e.g. ultralytics not yet installed at startup, missing file
    # on first request, etc.).  We now always attempt to (re)load so that:
    #   • installing ultralytics after server start works automatically
    #   • a transient file-lock or import error is retried naturally
    #   • the backend does NOT need a restart to recover YOLO
    # ─────────────────────────────────────────────────────────────────

    print("🔄 Attempting to load YOLO model...")
    logger.info("[YOLO] Attempting model load...")

    try:
        from ultralytics import YOLO
    except ImportError:
        _load_error = "ultralytics not installed. Run: pip install ultralytics"
        print(f"❌ YOLO load failed: {_load_error}")
        logger.error(f"[YOLO] {_load_error}")
        return None

    path = _resolve_model_path()
    print(f"🔄 Loading model from: {path}")
    logger.info(f"[YOLO] Loading model: {path}")
    try:
        _yolo_model        = YOLO(path)
        _loaded_model_path = path
        _load_error        = None   # clear any previous error on success
        print(f"✅ YOLO model loaded successfully → {path}")
        logger.info(f"[YOLO] Model ready — classes: {_yolo_model.names}")
        return _yolo_model
    except Exception as e:
        _load_error = str(e)
        print(f"❌ YOLO load failed: {e}")
        logger.error(f"[YOLO] Model load FAILED: {e}")
        return None


def _is_sar_model() -> bool:
    """True if we loaded the SAR fine-tuned model (single class: ship)."""
    if _loaded_model_path is None:
        return False
    return "sar" in Path(_loaded_model_path).stem.lower()


# ── SAR Preprocessing ─────────────────────────────────────────────────────────

def apply_sar_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    SAR-optimised preprocessing chain:
      1. Convert to grayscale (SAR is amplitude/intensity, not colour)
      2. Log1p-scale compression (collapses the huge dynamic range of SAR)
      3. CLAHE contrast enhancement (makes faint ships visible)
      4. Convert to 3-channel RGB (YOLO expects 3-channel input)
    """
    # Step 1: to grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = image.copy()

    # Step 2: log-scale compression
    img_f   = np.float32(image)
    img_log = np.log1p(img_f)
    lo, hi  = img_log.min(), img_log.max()
    if hi > lo:
        img_log = 255.0 * (img_log - lo) / (hi - lo)
    img_log = np.clip(img_log, 0, 255).astype(np.uint8)

    # Step 3: CLAHE
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_log)

    # Step 4: to 3-channel RGB
    return cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_yolo_inference(
    image_bytes: bytes,
    conf_thresh: float = 0.20,
    iou_thresh:  float = 0.45,
):
    """
    Run YOLO inference on uploaded image bytes.

    Returns:
        (list[dict], inference_ms: int)
        Each dict: {x1, y1, x2, y2, confidence, class, source}

    Raises:
        ValueError  — image decode failed
        RuntimeError — model not available or inference crashed
    """
    t0 = time.time()

    # 1. Decode image
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")

    np_img = np.array(pil_img)

    # 2. SAR preprocessing
    try:
        prep_img = apply_sar_preprocessing(np_img)
    except Exception as e:
        logger.warning(f"SAR preprocessing failed, using raw image: {e}")
        prep_img = np_img   # fallback to raw

    # 3. Load model (lazy, safe — retries on every call if needed)
    model = _get_model()
    if model is None:
        err_msg = _load_error or "Model not loaded"
        print(f"❌ YOLO unavailable: {err_msg}")
        raise RuntimeError(f"YOLO model unavailable: {err_msg}")

    sar_model = _is_sar_model()

    # 4. Inference — wrap in try/except so a bad image never crashes the server
    try:
        print(f"🚀 Running YOLO inference — conf={conf_thresh} iou={iou_thresh} sar={sar_model}")
        logger.info(f"[YOLO] Running inference — conf={conf_thresh} iou={iou_thresh}")
        results = model.predict(
            source=prep_img,
            imgsz=640,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"[YOLO] Inference crashed: {e}")
        raise RuntimeError(f"Inference failed: {e}")

    # 5. Parse boxes
    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes  = result.boxes
        if boxes is not None:
            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf    = float(boxes.conf[i].cpu().numpy())
                cls_id  = int(boxes.cls[i].cpu().numpy())
                cls_name = model.names.get(cls_id, "unknown")

                if not sar_model:
                    if cls_id not in (8,):
                        continue
                    cls_name = "ship"

                detections.append({
                    "x1":         int(x1),
                    "y1":         int(y1),
                    "x2":         int(x2),
                    "y2":         int(y2),
                    "confidence": round(conf, 3),
                    "class":      "ship",
                    "class_raw":  cls_name,
                    "source":     "yolo-sar" if sar_model else "yolo-coco",
                })

    inference_ms = round((time.time() - t0) * 1000)
    print(f"✅ YOLO inference done — {len(detections)} detections in {inference_ms}ms")
    logger.info(f"[YOLO] Inference done — {len(detections)} detections in {inference_ms}ms")
    return detections, inference_ms


# ── Health info ───────────────────────────────────────────────────────────────

def get_model_info() -> dict:
    """Return model status for the /health endpoint. Never raises."""
    try:
        model = _get_model()
        if model is None:
            return {
                "loaded":     False,
                "error":      _load_error or "Model not loaded",
                "model_type": "Not loaded",
                "is_sar":     False,
            }
        path  = _loaded_model_path or "unknown"
        is_sar = _is_sar_model()
        return {
            "loaded":     True,
            "weights":    path,
            "name":       Path(path).stem,
            "is_sar":     is_sar,
            "model_type": "SAR Fine-tuned" if is_sar else "COCO Pretrained",
            "nc":         len(model.names),
            "names":      model.names,
        }
    except Exception as e:
        return {
            "loaded":     False,
            "error":      str(e),
            "model_type": "Not loaded",
            "is_sar":     False,
        }
