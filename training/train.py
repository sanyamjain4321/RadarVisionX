#!/usr/bin/env python3
"""
SAR Ship Detection — YOLOv8 Training Script (Hardened)
=======================================================
Usage (from repo root):
    cd training
    python train.py                        # 50 epochs, auto-device
    python train.py --epochs 100 --batch 8
    python train.py --epochs 2 --quick     # smoke-test (2 epochs)
    python train.py --model yolov8s        # larger model

After training:
    • best.pt  → copied to  ../models/yolov8_sar.pt
    • best.onnx→ copied to  ../models/yolov8_sar.onnx  (browser-ready)
"""

import os
import sys
import glob
import shutil
import argparse
import traceback
from pathlib import Path


# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
REPO_ROOT   = SCRIPT_DIR.parent
MODELS_DIR  = REPO_ROOT / "models"
DATA_YAML   = SCRIPT_DIR / "data.yaml"
DATASET_DIR = SCRIPT_DIR / "dataset"


def check_dataset():
    """Verify the dataset exists, return n_train count. Raises SystemExit if missing."""
    train_imgs   = DATASET_DIR / "images" / "train"
    train_labels = DATASET_DIR / "labels" / "train"
    val_imgs     = DATASET_DIR / "images" / "val"

    print(f"[check_dataset] Looking for dataset at: {DATASET_DIR}")
    print(f"[check_dataset] Train images dir: {train_imgs} (exists={train_imgs.exists()})")

    if not train_imgs.exists() or not any(train_imgs.iterdir()):
        print("\n" + "═" * 60)
        print("⚠  SAR DATASET NOT FOUND — AUTO-GENERATING SYNTHETIC DATA")
        print("═" * 60)
        print(f"Expected: {train_imgs}")
        print()

        # AUTO-FIX: Generate synthetic data instead of exiting
        print("⟳ Running prepare_ssdd.py --synthetic...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "prepare_ssdd.py"), "--synthetic"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(SCRIPT_DIR), timeout=120,
            )
            print(result.stdout)
            if result.stderr:
                print(f"[stderr] {result.stderr}")

            # Re-check if dataset was generated
            if not train_imgs.exists() or not any(train_imgs.iterdir()):
                print("✗ Auto-generation failed. Manual steps:")
                print("    python prepare_ssdd.py --synthetic")
                sys.exit(1)
            print("✓ Synthetic dataset generated successfully!")
        except Exception as e:
            print(f"✗ Auto-generation error: {e}")
            print("    Run manually: python prepare_ssdd.py --synthetic")
            sys.exit(1)

    # Count files - handle various image extensions
    img_extensions = ["*.jpg", "*.png", "*.jpeg", "*.ppm"]
    n_train = sum(len(list(train_imgs.glob(ext))) for ext in img_extensions)
    n_val = 0
    if val_imgs.exists():
        n_val = sum(len(list(val_imgs.glob(ext))) for ext in img_extensions)
    n_labels = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0

    print(f"✓ Dataset found — {n_train} train images, {n_val} val images, {n_labels} train labels")

    if n_labels == 0:
        print("⚠ No label files found — trying to regenerate...")
        try:
            import subprocess
            subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "prepare_ssdd.py"), "--synthetic"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(SCRIPT_DIR), timeout=120,
            )
            n_labels = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0
        except Exception:
            pass

        if n_labels == 0:
            print("✗ Still no label files. Run: python prepare_ssdd.py --synthetic")
            sys.exit(1)

    if n_labels < n_train * 0.5:
        print(f"⚠ Label count ({n_labels}) is much less than image count ({n_train})")
        print("  Some images may be missing labels — this is OK if expected")

    return n_train


def patch_data_yaml():
    """Write an absolute-path data.yaml so Ultralytics finds dataset regardless of cwd."""
    abs_yaml     = SCRIPT_DIR / "data_abs.yaml"
    # Use forward slashes even on Windows — Ultralytics handles both
    dataset_abs  = str(DATASET_DIR).replace("\\", "/")

    # Verify paths exist before writing yaml
    train_path = DATASET_DIR / "images" / "train"
    val_path   = DATASET_DIR / "images" / "val"

    if not train_path.exists():
        raise FileNotFoundError(f"Train images directory not found: {train_path}")

    # Auto-create val directory if missing (copy 20% of train)
    if not val_path.exists() or not any(val_path.iterdir() if val_path.exists() else []):
        print("⚠ Validation directory missing — creating from train data...")
        val_path.mkdir(parents=True, exist_ok=True)
        val_lbl = DATASET_DIR / "labels" / "val"
        val_lbl.mkdir(parents=True, exist_ok=True)

        train_imgs = list(train_path.glob("*"))
        n_val = max(1, len(train_imgs) // 5)
        import random
        random.seed(42)
        val_candidates = random.sample(train_imgs, min(n_val, len(train_imgs)))
        for img in val_candidates:
            try:
                shutil.copy2(img, val_path / img.name)
                # Copy matching label too
                lbl_src = DATASET_DIR / "labels" / "train" / (img.stem + ".txt")
                if lbl_src.exists():
                    shutil.copy2(lbl_src, val_lbl / lbl_src.name)
            except Exception as e:
                print(f"  ⚠ Could not copy {img.name} to val: {e}")
        print(f"  ✓ Created {len(val_candidates)} validation samples")

    with open(abs_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_abs}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: ship\n")
    print(f"✓ data_abs.yaml written: path={dataset_abs}")
    return abs_yaml


def export_onnx(model, save_dir: Path):
    """Export best.pt → best.onnx."""
    print("\n" + "─" * 50)
    print("⟳  Exporting model to ONNX…")
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            opset=12,
            dynamic=False,
        )
        print(f"✓  ONNX exported: {onnx_path}")
        return Path(onnx_path)
    except Exception as e:
        print(f"⚠  ONNX export failed (non-fatal): {e}")
        print("   Install onnx: pip install onnx onnxsim")
        return None


def copy_weights(best_pt: Path, onnx_path):
    """Copy trained weights → models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)

    dst_pt = MODELS_DIR / "yolov8_sar.pt"
    shutil.copy2(best_pt, dst_pt)
    print(f"✓  Weights → {dst_pt}")

    if onnx_path and Path(onnx_path).exists():
        dst_onnx = MODELS_DIR / "yolov8_sar.onnx"
        shutil.copy2(onnx_path, dst_onnx)
        print(f"✓  ONNX    → {dst_onnx}")
        return dst_pt, dst_onnx

    return dst_pt, None


def print_next_steps(dst_pt, dst_onnx):
    print("\n" + "═" * 60)
    print("🎉  TRAINING COMPLETE — NEXT STEPS")
    print("═" * 60)
    print(f"\n  Weights:  {dst_pt}")
    if dst_onnx:
        print(f"  ONNX:     {dst_onnx}")
    print()
    print("1. Restart the FastAPI backend so it picks up the new model:")
    print("   start_backend.bat  (or Ctrl+C and re-run)")
    print()
    print("2. In the UI:")
    print("   • Pipeline: CFAR → YOLO → AI")
    print("   • YOLO Mode: Backend API")
    print("   • Click ▶ Arm Detector → ▶ Run")
    print("\n" + "═" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 SAR Ship Detection Training")
    parser.add_argument("--model",    default="yolov8n",  help="Base model: yolov8n/s/m (default: yolov8n)")
    parser.add_argument("--epochs",   type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch",    type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz",    type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device",   default="",  help="Device: '' (auto), 'cpu', '0' (GPU 0)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--quick",    action="store_true", help="Smoke test: 2 epochs only")
    parser.add_argument("--no-onnx",  action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.batch  = 4
        print("⚡ QUICK MODE — 2 epochs (smoke test)")

    print("=" * 60)
    print(" SAR Ship Detection — YOLOv8 Training")
    print("=" * 60)
    print(f"  Model:    {args.model}.pt")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Imgsz:    {args.imgsz}")
    print(f"  Device:   {args.device or 'auto'}")
    print(f"  Script:   {SCRIPT_DIR}")
    print(f"  Dataset:  {DATASET_DIR}")
    print()

    # 1. Check dataset (auto-generates if missing)
    try:
        n_train = check_dataset()
    except SystemExit:
        raise
    except Exception as e:
        print(f"✗ Dataset check error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Reduce batch size for small datasets to avoid DataLoader errors
    if n_train < 20 and args.batch > 4:
        print(f"⚠ Small dataset ({n_train} images) — reducing batch size to 4")
        args.batch = 4

    # 2. Patch data.yaml with absolute paths
    try:
        abs_yaml = patch_data_yaml()
    except Exception as e:
        print(f"✗ data.yaml error: {e}")
        traceback.print_exc()
        sys.exit(1)
    print(f"✓ Data config: {abs_yaml}")

    # 3. Import Ultralytics
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"✗ ultralytics not installed: {e}")
        print("    pip install ultralytics>=8.2.0")
        print("    Or: pip install -r training/requirements.txt")
        sys.exit(1)

    # 4. Load model
    model_file = f"{args.model}.pt"
    # Try local models first, then fallback to auto-download
    model_path = None
    for candidate in [
        REPO_ROOT / model_file,          # repo root (bundled)
        MODELS_DIR / model_file,         # models/ directory
        model_file,                       # auto-download from Ultralytics
    ]:
        if Path(candidate).exists() or str(candidate) == model_file:
            model_path = str(candidate)
            break

    print(f"\n⟳  Loading base model: {model_path}")
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load base model {model_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Train (with GPU→CPU fallback)
    print(f"\n⟳  Starting training — {args.epochs} epochs…\n")
    results = None
    device  = args.device

    train_kwargs = dict(
        data=str(abs_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="SAR_Ship_Runs",
        name="yolov8_sar_finetune",
        patience=args.patience,
        # SAR-specific augmentations
        degrees=45.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_s=0.0,
        hsv_h=0.0,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        save=True,
        exist_ok=True,
        verbose=True,
    )

    for attempt_device in [device, "cpu"]:
        try:
            print(f"⟳  Device: '{attempt_device or 'auto'}'")
            results = model.train(device=attempt_device, **train_kwargs)
            print(f"✓  Training complete on device='{attempt_device or 'auto'}'")
            break
        except RuntimeError as e:
            errmsg = str(e)
            if ("CUDA" in errmsg or "cuda" in errmsg or "device" in errmsg.lower()) \
                    and attempt_device != "cpu":
                print(f"⚠ GPU failed ({e.__class__.__name__}: {errmsg[:120]}) — retrying on CPU…")
                try:
                    model = YOLO(model_path)
                except Exception:
                    pass
                continue
            print(f"✗ Training RuntimeError: {e}")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"✗ Training crashed: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            sys.exit(1)

    if results is None:
        print("✗ Training could not complete on any device.")
        sys.exit(1)

    # 6. Find best.pt
    best_pt = Path("SAR_Ship_Runs/yolov8_sar_finetune/weights/best.pt")
    if not best_pt.exists():
        # Also check relative to training dir
        alt_best = SCRIPT_DIR / best_pt
        if alt_best.exists():
            best_pt = alt_best
        else:
            candidates = sorted(glob.glob("SAR_Ship_Runs/*/weights/best.pt"))
            if not candidates:
                candidates = sorted(glob.glob(str(SCRIPT_DIR / "SAR_Ship_Runs/*/weights/best.pt")))
            if candidates:
                best_pt = Path(candidates[-1])
            else:
                last_candidates = sorted(glob.glob("SAR_Ship_Runs/*/weights/last.pt"))
                if not last_candidates:
                    last_candidates = sorted(glob.glob(str(SCRIPT_DIR / "SAR_Ship_Runs/*/weights/last.pt")))
                if last_candidates:
                    best_pt = Path(last_candidates[-1])
                    print(f"⚠ best.pt not found, using last.pt: {best_pt}")
                else:
                    print("✗ Could not find best.pt or last.pt. Check SAR_Ship_Runs/")
                    sys.exit(1)

    print(f"\n✓ Best weights: {best_pt}")

    # 7. Export ONNX
    onnx_path = None
    if not args.no_onnx:
        try:
            best_model = YOLO(str(best_pt))
            onnx_path  = export_onnx(best_model, best_pt.parent)
        except Exception as e:
            print(f"⚠ ONNX export skipped: {e}")

    # 8. Copy to models/
    try:
        dst_pt, dst_onnx = copy_weights(best_pt, onnx_path)
    except Exception as e:
        print(f"✗ Failed to copy weights: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 9. Print next steps
    print_next_steps(dst_pt, dst_onnx)


if __name__ == "__main__":
    main()
