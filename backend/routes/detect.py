"""
Detection Endpoint (Hardened)
==============================
POST /detect        — accepts image file + optional conf/iou params

Request (multipart/form-data):
    image:      image file (PNG/JPG/TIFF)
    mode:       "yolo" | "cfar" | "hybrid"  (informational, logged only)
    conf_thresh: float  0.05–0.90  (default 0.20)
    iou_thresh:  float  0.10–0.90  (default 0.45)

Response (JSON):
    {
      "detections": [{x1,y1,x2,y2,confidence,class_name,source}],
      "inference_ms": 142,
      "model_used": "SAR Fine-tuned",
      "model_name": "yolov8_sar",
      "model_is_sar": true
    }
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from model.yolo_inference import run_yolo_inference, get_model_info

router = APIRouter()
logger = logging.getLogger("sar-backend.detect")


# ── Response schemas ──────────────────────────────────────────────────────────

class DetectionBox(BaseModel):
    x1:         int
    y1:         int
    x2:         int
    y2:         int
    confidence: float
    class_name: str
    source:     str

class DetectionResponse(BaseModel):
    detections:   List[DetectionBox]
    inference_ms: int
    model_used:   str
    model_name:   str
    model_is_sar: bool


# ── /detect ───────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectionResponse)
async def detect(
    image:       UploadFile = File(...),
    mode:        str   = Form("yolo"),
    conf_thresh: float = Form(0.20),
    iou_thresh:  float = Form(0.45),
):
    """
    Run YOLO inference on an uploaded SAR image.
    Returns structured detection results. Never crashes the server.
    """
    logger.info(f"/detect called — mode={mode} conf={conf_thresh} iou={iou_thresh} file={image.filename}")

    # Validate content type (be permissive — canvas.toBlob gives image/png)
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/") and content_type != "application/octet-stream":
        raise HTTPException(400, f"Expected an image file, got: {image.content_type}")

    # Clamp thresholds
    conf_thresh = max(0.05, min(0.90, conf_thresh))
    iou_thresh  = max(0.10, min(0.90, iou_thresh))

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file uploaded")

    logger.info(f"/detect — image size: {len(image_bytes)} bytes")

    try:
        detections, ms = run_yolo_inference(
            image_bytes,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
    except ValueError as e:
        logger.error(f"/detect — image decode error: {e}")
        raise HTTPException(400, f"Image decode error: {e}")
    except RuntimeError as e:
        logger.error(f"/detect — model not available: {e}")
        raise HTTPException(503, f"Model not available: {e}")
    except Exception as e:
        import traceback
        logger.error(f"/detect — unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Inference error: {e}")

    # Get model info for response
    info       = get_model_info()
    model_used = info.get("model_type", "Unknown")
    model_name = info.get("name", "unknown")
    is_sar     = info.get("is_sar", False)

    logger.info(f"/detect — {len(detections)} detections in {ms}ms using {model_used}")

    output_dets = [
        DetectionBox(
            x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
            confidence=d["confidence"],
            class_name=d.get("class", "ship"),
            source=d.get("source", "yolo"),
        )
        for d in detections
    ]

    return DetectionResponse(
        detections=output_dets,
        inference_ms=ms,
        model_used=model_used,
        model_name=model_name,
        model_is_sar=is_sar,
    )
