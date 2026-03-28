#!/usr/bin/env python3
"""
SSDD Dataset Preparation for YOLOv8  (Hardened Edition)
========================================================
Downloads and converts the SSDD (SAR Ship Detection Dataset) to YOLO format.

Usage:
    cd training
    python prepare_ssdd.py              # full auto (downloads SSDD)
    python prepare_ssdd.py --ssdd-dir /path/to/existing/ssdd  # local copy
    python prepare_ssdd.py --val-ratio 0.15  # 85/15 split
    python prepare_ssdd.py --synthetic  # generate synthetic dataset for testing

Dataset sources:
    SSDD: ~1160 SAR images, ~2456 ship instances
    PascalVOC XML annotations → YOLO .txt format

Output structure (training/dataset/):
    dataset/
    ├── images/
    │   ├── train/   (*.jpg)
    │   └── val/     (*.jpg)
    └── labels/
        ├── train/   (*.txt)
        └── val/     (*.txt)
"""

import os
import sys
import glob
import shutil
import random
import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "dataset"
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"

# Public mirrors for SSDD
SSDD_URLS = [
    "https://huggingface.co/datasets/keremberke/sar-ship-detection/resolve/main/data.zip",
    "https://public.roboflow.com/ds/xFPvIRaXgr?key=j3C5XNnwAo",
]

# ── Conversion ────────────────────────────────────────────────────────────────

def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list:
    """Convert a PascalVOC XML annotation to a list of YOLO lines."""
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f"  ⚠ XML parse error {xml_path.name}: {e}")
        return []
    except Exception as e:
        print(f"  ⚠ Cannot read {xml_path.name}: {e}")
        return []

    root = tree.getroot()
    lines = []

    size_el = root.find("size")
    if size_el is not None:
        try:
            w = int(size_el.find("width").text or 0)
            h = int(size_el.find("height").text or 0)
        except (AttributeError, ValueError, TypeError):
            w, h = img_w, img_h
    else:
        w, h = img_w, img_h

    if w <= 0 or h <= 0:
        return lines

    for obj in root.findall("object"):
        try:
            name_el = obj.find("name")
            if name_el is None or name_el.text is None:
                continue
            name = name_el.text.strip().lower()
            if name not in ("ship", "vessel", "boat"):
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin_el = bndbox.find("xmin")
            ymin_el = bndbox.find("ymin")
            xmax_el = bndbox.find("xmax")
            ymax_el = bndbox.find("ymax")

            if any(el is None or el.text is None for el in [xmin_el, ymin_el, xmax_el, ymax_el]):
                print(f"  ⚠ Missing bbox coords in {xml_path.name}")
                continue

            xmin = float(xmin_el.text)
            ymin = float(ymin_el.text)
            xmax = float(xmax_el.text)
            ymax = float(ymax_el.text)

            # Validate coordinates
            if xmin >= xmax or ymin >= ymax:
                print(f"  ⚠ Invalid bbox in {xml_path.name}: ({xmin},{ymin},{xmax},{ymax})")
                continue

            cx = (xmin + xmax) / 2.0 / w
            cy = (ymin + ymax) / 2.0 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))

            if bw > 0 and bh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        except (AttributeError, TypeError, ValueError) as e:
            print(f"  ⚠ Skipping malformed object in {xml_path.name}: {e}")
            continue

    return lines


def build_dataset_from_ssdd(ssdd_root: Path, val_ratio: float = 0.20):
    """
    Given an SSDD root (containing JPEGImages/ and Annotations/),
    convert and split into YOLO dataset/ directory.
    """
    img_dir = None
    ann_dir = None

    # Find image and annotation directories (various SSDD packaging layouts)
    for candidate in ["JPEGImages", "images", "Images", "image"]:
        p = ssdd_root / candidate
        if p.exists():
            img_dir = p
            break
    for candidate in ["Annotations", "annotations", "labels", "Labels"]:
        p = ssdd_root / candidate
        if p.exists():
            ann_dir = p
            break

    if img_dir is None or ann_dir is None:
        print(f"✗ Could not find images or annotations in {ssdd_root}")
        try:
            contents = [x.name for x in ssdd_root.iterdir() if x.is_dir()]
            print(f"  Contents: {contents}")
        except Exception:
            print(f"  (Could not list directory contents)")
        return False

    img_files = sorted(glob.glob(str(img_dir / "*.jpg")) +
                       glob.glob(str(img_dir / "*.png")) +
                       glob.glob(str(img_dir / "*.jpeg")))

    if not img_files:
        print(f"✗ No images found in {img_dir}")
        return False

    print(f"  Found {len(img_files)} images in {img_dir.name}/")
    print(f"  Found annotations in {ann_dir.name}/")

    # Shuffle and split
    random.seed(42)
    random.shuffle(img_files)
    split = max(1, int(len(img_files) * (1 - val_ratio)))
    train_imgs = img_files[:split]
    val_imgs   = img_files[split:]
    print(f"  Split: {len(train_imgs)} train / {len(val_imgs)} val")

    # Create output dirs
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DATASET_DIR / d).mkdir(parents=True, exist_ok=True)

    ok_count = 0
    skip_count = 0

    for split_name, split_files in [("train", train_imgs), ("val", val_imgs)]:
        img_out = DATASET_DIR / "images" / split_name
        lbl_out = DATASET_DIR / "labels" / split_name

        for img_path_str in split_files:
            img_path = Path(img_path_str)
            stem = img_path.stem

            # Find matching XML
            xml_path = ann_dir / f"{stem}.xml"
            if not xml_path.exists():
                skip_count += 1
                continue

            # Convert annotation first (skip files with parse errors)
            try:
                yolo_lines = voc_to_yolo(xml_path, 0, 0)
            except Exception as e:
                print(f"  ⚠ Failed to parse {xml_path.name}: {e}")
                skip_count += 1
                continue

            # Skip images with no valid ship annotations
            if not yolo_lines:
                skip_count += 1
                continue

            # Copy image
            dst_img = img_out / img_path.name
            try:
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
            except Exception as e:
                print(f"  ⚠ Failed to copy {img_path.name}: {e}")
                skip_count += 1
                continue

            lbl_path = lbl_out / f"{stem}.txt"
            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            ok_count += 1
            if ok_count % 100 == 0:
                print(f"  ... processed {ok_count} files so far")

    print(f"\n  ✓ Converted {ok_count} image-annotation pairs")
    if skip_count:
        print(f"  ⚠ Skipped {skip_count} (no matching XML or no ships)")

    return ok_count > 0


def try_download_ssdd():
    """Attempt to download SSDD from known mirrors. Returns path to extracted root or None."""
    try:
        import requests
    except ImportError:
        print("  ✗ requests not installed — run: pip install requests")
        return None

    DOWNLOAD_DIR.mkdir(exist_ok=True)
    zip_path = DOWNLOAD_DIR / "ssdd.zip"

    for url in SSDD_URLS:
        print(f"  Trying: {url}")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    print(f"    HTTP {r.status_code} — skipping")
                    continue
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"    {pct:.0f}%", end="\r")
            print(f"\n  ✓ Downloaded to {zip_path}")
            # Extract
            extract_dir = DOWNLOAD_DIR / "ssdd_extracted"
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            # Find root
            subdirs = [p for p in extract_dir.rglob("Annotations") if p.is_dir()]
            if subdirs:
                return subdirs[0].parent
            return extract_dir
        except Exception as e:
            print(f"    ✗ {e}")
            continue

    return None


def print_manual_instructions():
    """Print clear manual download instructions."""
    print()
    print("═" * 60)
    print("📥  MANUAL DATASET SETUP INSTRUCTIONS")
    print("═" * 60)
    print()
    print("OPTION A — SSDD (recommended):")
    print("  1. Download from GitHub:")
    print("     https://github.com/TencentYoutuResearch/ObjectDetection-SSDD")
    print("     (Use the GitHub 'Download ZIP' button or git clone)")
    print()
    print("  2. Extract and note the folder containing JPEGImages/ and Annotations/")
    print()
    print("  3. Run this script with --ssdd-dir:")
    print("     python prepare_ssdd.py --ssdd-dir /path/to/ssdd")
    print()
    print("OPTION B — Use your own YOLO-format dataset:")
    print(f"  Place images in:  {DATASET_DIR / 'images' / 'train'}")
    print(f"  Place images in:  {DATASET_DIR / 'images' / 'val'}")
    print(f"  Place labels in:  {DATASET_DIR / 'labels' / 'train'}")
    print(f"  Place labels in:  {DATASET_DIR / 'labels' / 'val'}")
    print(f"  (Labels = YOLO .txt format: cls cx cy w h, one box per line)")
    print()
    print("OPTION C — Quick smoke-test without a real dataset:")
    print("  python prepare_ssdd.py --synthetic")
    print("  (Generates 200 synthetic SAR-like grayscale images for pipeline testing)")
    print("═" * 60)


def generate_synthetic_dataset(n_train=160, n_val=40):
    """
    Generate synthetic SAR-like images for smoke-testing the training pipeline.
    Not useful for real accuracy — just validates the training script works.

    HARDENED: handles missing numpy/Pillow by using pure-Python fallback.
    """
    use_numpy = True
    use_pil = True

    try:
        import numpy as np
    except ImportError:
        print("⚠ numpy not installed — using pure-Python image generation (slower)")
        use_numpy = False

    try:
        from PIL import Image
    except ImportError:
        print("⚠ Pillow not installed — using raw PPM format (no compression)")
        use_pil = False

    print(f"\n⟳  Generating {n_train + n_val} synthetic SAR images…")

    total_written = 0
    total_errors = 0

    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating {n} {split} images in {img_dir}...")

        for i in range(n):
            W, H = 640, 640

            try:
                if use_numpy:
                    import numpy as np
                    # SAR-like background: Rayleigh-distributed speckle
                    speckle = np.random.rayleigh(30, (H, W)).astype(np.float32)
                    speckle = np.clip(speckle, 0, 255).astype(np.uint8)
                    img = np.stack([speckle] * 3, axis=-1)
                else:
                    # Pure Python fallback - simple random noise
                    import math
                    img_data = []
                    for _y in range(H):
                        row = []
                        for _x in range(W):
                            # Approximate Rayleigh with exponential
                            u = random.random() or 1e-9
                            v = int(min(255, max(0, -math.log(u) * 30)))
                            row.append((v, v, v))
                        img_data.append(row)

                # Add 1–5 synthetic ship blobs (bright ellipses)
                labels = []
                n_ships = random.randint(1, 5)
                for _ in range(n_ships):
                    cx = random.uniform(0.1, 0.9)
                    cy = random.uniform(0.1, 0.9)
                    bw = random.uniform(0.02, 0.07)
                    bh = random.uniform(0.01, 0.04)
                    # Draw bright blob
                    x1 = int((cx - bw / 2) * W)
                    y1 = int((cy - bh / 2) * H)
                    x2 = int((cx + bw / 2) * W)
                    y2 = int((cy + bh / 2) * H)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    brightness = random.randint(180, 255)

                    if use_numpy:
                        img[y1:y2, x1:x2] = brightness
                    else:
                        for _y in range(y1, y2):
                            for _x in range(x1, x2):
                                img_data[_y][_x] = (brightness, brightness, brightness)

                    labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # Save image
                img_filename = f"synth_{split}_{i:04d}.jpg"
                img_path = img_dir / img_filename

                if use_numpy and use_pil:
                    from PIL import Image
                    pil_img = Image.fromarray(img)
                    pil_img.save(img_path, quality=85)
                elif use_numpy:
                    # Save as PPM (no Pillow needed), then rename to .jpg
                    # Actually save as PNG using pure cv2 if available, else PPM
                    try:
                        import cv2
                        cv2.imwrite(str(img_path), img)
                    except ImportError:
                        # Last resort: save as .ppm (PPM format)
                        ppm_path = img_dir / f"synth_{split}_{i:04d}.ppm"
                        with open(ppm_path, 'wb') as f:
                            f.write(f"P6\n{W} {H}\n255\n".encode())
                            f.write(img.tobytes())
                        # Rename to .jpg extension (content is PPM but YOLO can often handle it)
                        img_path = img_dir / f"synth_{split}_{i:04d}.png"
                        ppm_path.rename(img_path)
                else:
                    # Pure Python: write as PPM then rename
                    img_path = img_dir / f"synth_{split}_{i:04d}.ppm"
                    with open(img_path, 'wb') as f:
                        f.write(f"P6\n{W} {H}\n255\n".encode())
                        for row in img_data:
                            for r, g, b in row:
                                f.write(bytes([r, g, b]))

                # Save label
                lbl_filename = f"synth_{split}_{i:04d}.txt"
                lbl_path = lbl_dir / lbl_filename
                with open(lbl_path, "w") as f:
                    f.write("\n".join(labels) + "\n")

                total_written += 1
                if total_written % 40 == 0:
                    print(f"  ... {total_written}/{n_train + n_val} generated")

            except Exception as e:
                print(f"  ⚠ Failed to generate image {i}: {e}")
                total_errors += 1
                # Still try to create at least a label file
                try:
                    lbl_path = lbl_dir / f"synth_{split}_{i:04d}.txt"
                    with open(lbl_path, "w") as f:
                        f.write("0 0.500000 0.500000 0.050000 0.030000\n")
                except Exception:
                    pass

    if total_written == 0:
        print("✗ No synthetic images were generated — check permissions and disk space")
        print(f"  Dataset dir: {DATASET_DIR}")
        print(f"  Errors: {total_errors}")
        return False

    print(f"\n✓ Synthetic dataset: {total_written} files in {DATASET_DIR}")
    if total_errors:
        print(f"⚠ {total_errors} errors during generation (non-fatal)")
    print("⚠  Synthetic data is for pipeline testing only — train on real SSDD for accuracy!")

    # Verify what was actually written
    _verify_dataset()
    return True


def _verify_dataset():
    """Quick verification of generated dataset."""
    for split in ["train", "val"]:
        img_d = DATASET_DIR / "images" / split
        lbl_d = DATASET_DIR / "labels" / split
        n_img = len(list(img_d.glob("*"))) if img_d.exists() else 0
        n_lbl = len(list(lbl_d.glob("*.txt"))) if lbl_d.exists() else 0
        print(f"  {split}: {n_img} images, {n_lbl} labels")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare SSDD dataset for YOLOv8 training")
    parser.add_argument("--ssdd-dir",   type=str, default=None,
                        help="Path to existing SSDD root (with JPEGImages/ and Annotations/)")
    parser.add_argument("--val-ratio",  type=float, default=0.20,
                        help="Fraction of data for validation (default: 0.20)")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Generate synthetic SAR images for pipeline testing")
    parser.add_argument("--no-download", action="store_true",
                        help="Do not attempt auto-download; show manual instructions")
    args = parser.parse_args()

    print("═" * 60)
    print(" SAR Ship Dataset Preparation")
    print("═" * 60)

    # ── Synthetic mode ────────────────────────────────────────────────────────
    if args.synthetic:
        ok = generate_synthetic_dataset()
        if ok:
            print("\n✓ Done! Now run:  python train.py --epochs 2 --quick")
            sys.exit(0)
        else:
            print("\n✗ Synthetic generation failed")
            sys.exit(1)

    # ── Use provided local SSDD directory ─────────────────────────────────────
    if args.ssdd_dir:
        ssdd_root = Path(args.ssdd_dir)
        if not ssdd_root.exists():
            print(f"✗ --ssdd-dir not found: {ssdd_root}")
            sys.exit(1)
        print(f"\n⟳  Using local SSDD at: {ssdd_root}")
        ok = build_dataset_from_ssdd(ssdd_root, args.val_ratio)
        if ok:
            print(f"\n✓ Dataset ready at: {DATASET_DIR}")
            print("Next step:  python train.py")
        else:
            print("✗ Dataset preparation failed — check the directory structure")
        sys.exit(0 if ok else 1)

    # ── Check if dataset already exists ───────────────────────────────────────
    train_dir = DATASET_DIR / "images" / "train"
    if train_dir.exists():
        try:
            files = list(train_dir.iterdir())
            if files:
                n = len([f for f in files if f.suffix.lower() in ('.jpg', '.png', '.jpeg', '.ppm')])
                print(f"\n✓ Dataset already exists — {n} train images at {DATASET_DIR}")
                print("  (Delete training/dataset/ to re-prepare)")
                print("\nNext step:  python train.py")
                sys.exit(0)
        except Exception:
            pass

    # ── Auto-download ─────────────────────────────────────────────────────────
    if not args.no_download:
        print("\n⟳  Attempting auto-download from public mirrors…")
        ssdd_root = try_download_ssdd()

        if ssdd_root:
            print(f"\n⟳  Building YOLO dataset from {ssdd_root}…")
            ok = build_dataset_from_ssdd(ssdd_root, args.val_ratio)
            if ok:
                print(f"\n✓ Dataset ready at: {DATASET_DIR}")
                print("Next step:  python train.py")
                sys.exit(0)
        print("\n  Auto-download failed — falling back to synthetic generation…")
        # AUTO-FALLBACK: generate synthetic data instead of failing
        print("\n⟳  Auto-generating synthetic dataset for testing…")
        ok = generate_synthetic_dataset()
        if ok:
            print("\n✓ Synthetic dataset ready! Run: python train.py --epochs 2 --quick")
            sys.exit(0)

    # ── Manual instructions ───────────────────────────────────────────────────
    print_manual_instructions()


if __name__ == "__main__":
    random.seed(42)
    main()
