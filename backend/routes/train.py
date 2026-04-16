"""
Training Endpoint — FastAPI (Hardened)
========================================
Exposes training as an API so the frontend can trigger/monitor it.

GET  /train/status      → current training state (idle|running|done|error)
POST /train/start       → kick off background training
POST /train/cancel      → abort running training
GET  /train/logs        → last N lines of training stdout
GET  /train/dataset     → dataset validation status
"""

import sys
import subprocess
import threading
import time
import re
import logging
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
logger = logging.getLogger("sar-backend.train")

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "status":        "idle",   # idle | running | done | error
    "progress":      0,
    "epoch":         0,
    "total_epochs":  0,
    "message":       "No training started yet",
    "error":         None,
    "log_lines":     [],
    "started_at":    None,
    "finished_at":   None,
    "model_ready":   False,
}

_process: Optional[subprocess.Popen] = None
_lock = threading.Lock()

REPO_ROOT    = Path(__file__).parent.parent.resolve()
TRAINING_DIR = REPO_ROOT / "training"
MODELS_DIR   = REPO_ROOT / "models"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_log(line: str):
    logger.info(f"[train] {line}")
    with _lock:
        _state["log_lines"].append(line)
        if len(_state["log_lines"]) > 200:
            _state["log_lines"] = _state["log_lines"][-200:]


def _count_dataset_files():
    """Return (n_train_images, n_val_images, n_train_labels, n_val_labels)."""
    def _count(d, *exts):
        if not d.exists(): return 0
        return sum(len(list(d.glob(f"*{e}"))) for e in exts)

    imgs_train  = TRAINING_DIR / "dataset" / "images"  / "train"
    imgs_val    = TRAINING_DIR / "dataset" / "images"  / "val"
    lbls_train  = TRAINING_DIR / "dataset" / "labels"  / "train"
    lbls_val    = TRAINING_DIR / "dataset" / "labels"  / "val"

    return {
        "train_images": _count(imgs_train, ".jpg", ".png", ".jpeg"),
        "val_images":   _count(imgs_val,   ".jpg", ".png", ".jpeg"),
        "train_labels": _count(lbls_train, ".txt"),
        "val_labels":   _count(lbls_val,   ".txt"),
        "train_path":   str(imgs_train),
        "val_path":     str(imgs_val),
    }


def _dataset_is_ready():
    stats = _count_dataset_files()
    # Need both images AND labels to train
    return stats["train_images"] > 0 and stats["train_labels"] > 0


def _run_subprocess_safe(cmd, cwd, timeout=None):
    """
    Run a subprocess, capture all output, return (returncode, output_lines).
    Works on Windows: uses CREATE_NO_WINDOW to prevent popup.
    Handles encoding gracefully (utf-8 with errors='replace').
    """
    kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",   # never crash on weird characters
    )
    # Windows: suppress console window popup
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **kwargs)
    lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                _add_log(line)
    except Exception as e:
        _add_log(f"[stderr read error] {e}")
    proc.wait()
    return proc.returncode, lines, proc


# ── Background training worker ────────────────────────────────────────────────

def _run_training_process(epochs: int, quick: bool):
    global _process
    with _lock:
        _state["status"]       = "running"
        _state["progress"]     = 2
        _state["epoch"]        = 0
        _state["total_epochs"] = epochs
        _state["message"]      = "Preparing dataset…"
        _state["error"]        = None
        _state["log_lines"]    = []
        _state["started_at"]   = time.time()
        _state["model_ready"]  = False

    python = sys.executable
    _add_log(f"[trainer] Python: {python}")
    _add_log(f"[trainer] Training dir: {TRAINING_DIR}")
    _add_log(f"[trainer] Epochs: {epochs}  Quick: {quick}")

    # ── Step 1: Dataset preparation ───────────────────────────────────────────
    _add_log("[trainer] Step 1/2: Checking dataset…")
    dataset_train = TRAINING_DIR / "dataset" / "images" / "train"

    if not dataset_train.exists() or not _dataset_is_ready():
        _add_log("[trainer] No usable dataset found — generating synthetic SAR data…")
        stats_before = _count_dataset_files()
        _add_log(f"[trainer] Before: images={stats_before['train_images']} labels={stats_before['train_labels']}")
        with _lock:
            _state["message"]  = "Generating synthetic dataset…"
            _state["progress"] = 5

        # Try up to 2 times with different approaches
        for attempt in range(2):
            _add_log(f"[trainer] Synthetic generation attempt {attempt+1}/2…")
            rc, lines, _ = _run_subprocess_safe(
                [python, str(TRAINING_DIR / "prepare_ssdd.py"), "--synthetic"],
                cwd=TRAINING_DIR,
            )

            _add_log(f"[trainer] prepare_ssdd.py exit code: {rc}")
            for line in lines[-15:]:
                _add_log(f"[trainer] prepare: {line}")

            if _dataset_is_ready():
                break
            _add_log(f"[trainer] Attempt {attempt+1} did not produce dataset, retrying…")

        # Verify dataset was actually created regardless of exit code
        if not _dataset_is_ready():
            stats = _count_dataset_files()
            err_msg = (
                f"Dataset preparation failed — "
                f"images={stats['train_images']}, labels={stats['train_labels']} "
                f"(expected both > 0). "
                f"Run manually: python training/prepare_ssdd.py --synthetic"
            )
            _add_log(f"[trainer] ✗ {err_msg}")
            with _lock:
                _state["status"]  = "error"
                _state["error"]   = err_msg
                _state["message"] = err_msg
            return
        else:
            stats = _count_dataset_files()
            _add_log(
                f"[trainer] ✓ Dataset ready — "
                f"{stats['train_images']} train images, "
                f"{stats['val_images']} val images, "
                f"{stats['train_labels']} labels"
            )
    else:
        stats = _count_dataset_files()
        _add_log(f"[trainer] ✓ Dataset already exists — {stats['train_images']} train images, {stats['train_labels']} labels")

    with _lock:
        _state["progress"] = 10
        _state["message"]  = "Starting YOLOv8 training…"

    # ── Step 2: Training ──────────────────────────────────────────────────────
    cmd = [python, str(TRAINING_DIR / "train.py"), f"--epochs={epochs}"]
    if quick:
        cmd.append("--quick")

    _add_log(f"[trainer] Step 2/2: {' '.join(cmd)}")

    # Windows-safe subprocess
    kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(TRAINING_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    _process = subprocess.Popen(cmd, **kwargs)
    epoch_re  = re.compile(r"Epoch\s+(\d+)/(\d+)", re.IGNORECASE)

    try:
        for line in _process.stdout:
            line = line.rstrip()
            if not line: continue
            _add_log(line)

            m = epoch_re.search(line)
            if m:
                ep, total = int(m.group(1)), int(m.group(2))
                pct = 10 + int(ep / max(total, 1) * 85)
                with _lock:
                    _state["epoch"]        = ep
                    _state["total_epochs"] = total
                    _state["progress"]     = pct
                    _state["message"]      = f"Training epoch {ep}/{total}…"

            # Also detect error messages from training
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['error:', 'traceback', 'exception', 'failed']):
                _add_log(f"[trainer] ⚠ Detected error in training output")
    except Exception as e:
        _add_log(f"[trainer] stdout read error: {e}")

    try:
        _process.wait(timeout=30)
    except Exception:
        _add_log("[trainer] Process wait timed out — killing")
        try:
            _process.kill()
        except Exception:
            pass
    rc = _process.returncode
    _process = None

    # ── Done ──────────────────────────────────────────────────────────────────
    sar_pt   = MODELS_DIR / "yolov8_sar.pt"
    sar_onnx = MODELS_DIR / "yolov8_sar.onnx"
    model_ok = sar_pt.exists()

    _add_log(f"[trainer] train.py exit code: {rc}  model_ok={model_ok}")

    with _lock:
        _state["finished_at"] = time.time()
        if rc == 0 and model_ok:
            _state["status"]      = "done"
            _state["progress"]    = 100
            _state["model_ready"] = True
            _state["message"]     = (
                f"✓ Training complete! Model: {sar_pt.name}"
                + (" + ONNX ready" if sar_onnx.exists() else "")
            )
        elif model_ok:
            # Training completed but exit code != 0 (Windows quirk)
            _state["status"]      = "done"
            _state["progress"]    = 100
            _state["model_ready"] = True
            _state["message"]     = f"✓ Model saved (exit={rc}) — {sar_pt.name}"
        else:
            _state["status"]  = "error"
            _state["error"]   = f"Process exited with code {rc} — model not saved"
            _state["message"] = "Training failed — check /train/logs"

    if model_ok:
        try:
            from model.yolo_inference import _reload_model
            _reload_model()
            _add_log("[trainer] ✓ YOLO inference module reloaded with new SAR model")
        except Exception as e:
            _add_log(f"[trainer] ⚠ Reload failed (restart backend): {e}")


# ── Dataset validation endpoint ───────────────────────────────────────────────

@router.get("/train/dataset")
async def train_dataset():
    """Return dataset stats and whether it's ready for training."""
    stats = _count_dataset_files()
    has_images = stats["train_images"] > 0
    has_labels = stats["train_labels"] > 0
    ready = has_images and has_labels
    
    if ready:
        msg = f"✓ {stats['train_images']} train / {stats['val_images']} val images, {stats['train_labels']} labels"
    elif has_images and not has_labels:
        msg = (f"⚠ {stats['train_images']} images found but NO labels. "
               "Run: python training/prepare_ssdd.py --synthetic")
    else:
        msg = ("No dataset found. Click 'Start Training' to auto-generate synthetic data, "
               "or run: python training/prepare_ssdd.py --synthetic")
    return {
        "ready":  ready,
        "stats":  stats,
        "message": msg,
    }


@router.get("/train/validate")
async def train_validate():
    """Run the dataset validator and return structured results."""
    import subprocess
    python = sys.executable
    validate_script = TRAINING_DIR / "validate_dataset.py"
    
    if not validate_script.exists():
        return {"ok": False, "message": "validate_dataset.py not found", "lines": []}
    
    try:
        result = subprocess.run(
            [python, str(validate_script)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(TRAINING_DIR), timeout=15
        )
        lines = (result.stdout + result.stderr).splitlines()
        ok = result.returncode == 0
        return {"ok": ok, "lines": lines, "returncode": result.returncode}
    except Exception as e:
        return {"ok": False, "message": str(e), "lines": []}


# ── Route schemas ─────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = 30
    quick:  bool = False

class TrainStatus(BaseModel):
    status:        str
    progress:      int
    epoch:         int
    total_epochs:  int
    message:       str
    error:         Optional[str]
    model_ready:   bool
    elapsed_s:     Optional[float]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/train/status", response_model=TrainStatus)
async def train_status():
    with _lock:
        s = dict(_state)
    elapsed = None
    if s["started_at"]:
        end = s["finished_at"] or time.time()
        elapsed = round(end - s["started_at"], 1)
    return TrainStatus(
        status=s["status"],
        progress=s["progress"],
        epoch=s["epoch"],
        total_epochs=s["total_epochs"],
        message=s["message"],
        error=s["error"],
        model_ready=s["model_ready"],
        elapsed_s=elapsed,
    )


@router.post("/train/start")
async def train_start(req: TrainRequest, background_tasks: BackgroundTasks):
    with _lock:
        if _state["status"] == "running":
            return {"ok": False, "message": "Training already running"}

    logger.info(f"Training requested: epochs={req.epochs} quick={req.quick}")
    background_tasks.add_task(_run_training_process, req.epochs, req.quick)
    return {
        "ok": True,
        "message": f"Training started: {req.epochs} epochs (quick={req.quick})",
        "epochs": req.epochs,
    }


@router.post("/train/cancel")
async def train_cancel():
    global _process
    with _lock:
        if _state["status"] != "running" or _process is None:
            return {"ok": False, "message": "No training running"}
        try:
            _process.terminate()
        except Exception:
            pass
        _state["status"]   = "idle"
        _state["message"]  = "Training cancelled"
        _state["progress"] = 0
    return {"ok": True, "message": "Training cancelled"}


@router.get("/train/logs")
async def train_logs(n: int = 50):
    with _lock:
        lines = list(_state["log_lines"][-n:])
    return {"lines": lines, "total": len(_state["log_lines"])}
