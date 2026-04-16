#!/usr/bin/env python3
"""
Dataset Validation Script (Hardened)
======================================
Checks training/dataset/ structure and reports issues.
Auto-generates synthetic dataset if nothing is found.

Usage:
    cd training
    python validate_dataset.py
    python validate_dataset.py --auto-fix    # auto-generate if missing
"""
import os
import sys
import argparse
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "dataset"


def validate(auto_fix=False):
    print("=" * 60)
    print(" SAR Dataset Validator")
    print("=" * 60)
    print(f"\nDataset path: {DATASET_DIR}\n")

    issues = []
    warnings = []

    # Check directory structure
    required_dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val",
    ]
    for d in required_dirs:
        full = DATASET_DIR / d
        exists = full.exists()
        if exists:
            try:
                count = len(list(full.glob("*")))
            except Exception:
                count = 0
            print(f"  {'✓' if count > 0 else '⚠'} {d:25s} → {count} files")
            if count == 0:
                warnings.append(f"{d} exists but is EMPTY")
        else:
            print(f"  ✗ {d:25s} → MISSING")
            issues.append(f"Missing directory: {d}")

    # Count files
    print()
    img_exts = ["*.jpg", "*.png", "*.jpeg", "*.ppm"]
    def count_files(d, *patterns):
        p = DATASET_DIR / d
        if not p.exists():
            return 0
        return sum(len(list(p.glob(pat))) for pat in patterns)

    n_train_img = count_files("images/train", *img_exts)
    n_val_img   = count_files("images/val",   *img_exts)
    n_train_lbl = count_files("labels/train", "*.txt")
    n_val_lbl   = count_files("labels/val",   "*.txt")

    print(f"  Train images : {n_train_img}")
    print(f"  Val images   : {n_val_img}")
    print(f"  Train labels : {n_train_lbl}")
    print(f"  Val labels   : {n_val_lbl}")

    # Cross-check images ↔ labels
    if n_train_img > 0 and n_train_lbl == 0:
        issues.append("Train images found but NO label files!")
    elif n_train_img == 0:
        issues.append("No training images found!")

    if n_train_img > 0 and n_train_lbl > 0:
        match_ratio = n_train_lbl / n_train_img
        if match_ratio < 0.5:
            warnings.append(f"Only {match_ratio*100:.0f}% of images have labels")

    # Check label format (sample first 20)
    lbl_dir = DATASET_DIR / "labels" / "train"
    if lbl_dir.exists():
        bad_labels = []
        for lbl_file in list(lbl_dir.glob("*.txt"))[:20]:
            try:
                with open(lbl_file) as f:
                    lines = [l.strip() for l in f if l.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        bad_labels.append(f"{lbl_file.name}: wrong columns ({len(parts)} instead of 5)")
                        break
                    try:
                        cls_id = int(parts[0])
                        vals = [float(p) for p in parts[1:]]
                        if any(v < 0 or v > 1 for v in vals):
                            bad_labels.append(f"{lbl_file.name}: values out of [0,1] range")
                            break
                    except ValueError as ve:
                        bad_labels.append(f"{lbl_file.name}: parse error — {ve}")
                        break
            except Exception as e:
                bad_labels.append(f"{lbl_file.name}: read error — {e}")
        if bad_labels:
            print(f"\n  ⚠ Label format issues (first {len(bad_labels)}):")
            for b in bad_labels[:5]:
                print(f"    {b}")
            warnings.extend(bad_labels)

    # data.yaml check
    print()
    for yaml_name in ["data_abs.yaml", "data.yaml"]:
        yaml_path = SCRIPT_DIR / yaml_name
        if yaml_path.exists():
            print(f"  ✓ {yaml_name} found")
            try:
                with open(yaml_path) as f:
                    content = f.read()
                if "path:" not in content:
                    warnings.append(f"{yaml_name} missing 'path:' field")
                if "train:" not in content:
                    warnings.append(f"{yaml_name} missing 'train:' field")
                if "nc:" not in content:
                    warnings.append(f"{yaml_name} missing 'nc:' field")
            except Exception as e:
                warnings.append(f"Could not read {yaml_name}: {e}")
        else:
            if yaml_name == "data.yaml":
                warnings.append("data.yaml not found (will be auto-created by train.py)")

    # Summary
    print("\n" + "=" * 60)
    if issues:
        print(f"❌ DATASET NOT READY — {len(issues)} issue(s):")
        for i in issues:
            print(f"   • {i}")
        print()

        # AUTO-FIX: generate synthetic dataset if requested
        if auto_fix:
            print("⟳ AUTO-FIX: Generating synthetic dataset...")
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(SCRIPT_DIR / "prepare_ssdd.py"), "--synthetic"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    cwd=str(SCRIPT_DIR), timeout=120,
                )
                print(result.stdout)
                if result.returncode == 0:
                    print("✓ Synthetic dataset generated! Run validate again to verify.")
                    return True
                else:
                    print(f"✗ Generation failed (exit={result.returncode})")
                    if result.stderr:
                        print(f"  stderr: {result.stderr[:500]}")
            except Exception as e:
                print(f"✗ Auto-fix error: {e}")
        else:
            print("FIX: Run one of:")
            print("  python prepare_ssdd.py --synthetic          # quick test dataset")
            print("  python prepare_ssdd.py --ssdd-dir <path>   # real SSDD dataset")
            print("  python validate_dataset.py --auto-fix       # auto-generate")
        print("=" * 60)
        return False
    else:
        print(f"✅ DATASET READY — {n_train_img} train / {n_val_img} val images")
        if warnings:
            print(f"\n⚠ Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"   • {w}")
        print("\nNext step: python train.py")
        print("=" * 60)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SAR dataset")
    parser.add_argument("--auto-fix", action="store_true",
                        help="Auto-generate synthetic dataset if validation fails")
    args = parser.parse_args()
    ok = validate(auto_fix=args.auto_fix)
    sys.exit(0 if ok else 1)
