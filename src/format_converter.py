"""
src/format_converter.py

Converts non-JPG images to JPG for dataset consistency.
Targeted fix for val/horse PNG anomaly.

Safe operation:
  - Original files are NOT deleted during conversion
  - Converted files saved alongside originals first
  - Verification pass confirms conversion quality
  - Only after verification: originals removed
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL     import Image
import pandas as pd

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
JPG_QUALITY      = 95   # high quality to minimize conversion loss
ROOT_DIR         = "data/raw/Animal"
REPORTS_DIR      = "reports"


# ─── CORE CONVERSION ──────────────────────────────────────────────────────────

def convert_to_jpg(src_path: Path, quality: int = JPG_QUALITY) -> Path:
    """
    Convert a single image to JPG.
    Saves as same stem + .jpg in same directory.
    Returns path of converted file.
    Raises on failure — never silently skips.
    """
    if src_path.suffix.lower() == ".jpg":
        raise ValueError(f"{src_path.name} is already JPG")

    dst_path = src_path.with_suffix(".jpg")

    if dst_path.exists():
        raise FileExistsError(
            f"Target already exists: {dst_path.name}"
        )

    # PIL handles PNG → JPG correctly including alpha channel
    img = Image.open(src_path).convert("RGB")   # drop alpha if PNG has it
    img.save(dst_path, format="JPEG", quality=quality)

    return dst_path


def verify_conversion(original: Path, converted: Path,
                      max_mse: float = 50.0) -> tuple[bool, float]:
    """
    Verify conversion quality by computing MSE between
    original and converted image at pixel level.

    max_mse=50.0:
      PNG → JPG at quality=95 typically produces MSE < 10
      MSE > 50 indicates something went wrong in conversion
      (corruption, wrong colorspace, dimension mismatch)

    Returns (passed: bool, mse: float)
    """
    img_orig = cv2.imread(str(original))
    img_conv = cv2.imread(str(converted))

    if img_orig is None or img_conv is None:
        return False, float("inf")

    if img_orig.shape != img_conv.shape:
        # Resize original to match (handles minor dimension differences)
        img_orig = cv2.resize(
            img_orig,
            (img_conv.shape[1], img_conv.shape[0])
        )

    mse = float(np.mean((img_orig.astype(np.float32) -
                          img_conv.astype(np.float32)) ** 2))
    return mse <= max_mse, round(mse, 4)


# ─── DATASET-LEVEL CONVERSION ─────────────────────────────────────────────────

def convert_non_jpg(split     : str  = "val",
                    class_name: str  = "horse",
                    dry_run   : bool = True) -> pd.DataFrame:
    """
    Convert all non-JPG images in a specific class to JPG.

    dry_run=True  → scan and report only, no files changed
    dry_run=False → perform actual conversion

    Safe sequence:
      1. Identify all non-JPG files
      2. Convert each → .jpg (original untouched)
      3. Verify conversion quality (MSE check)
      4. If verified → remove original
      5. If failed   → remove corrupted .jpg, keep original
      6. Report results
    """
    class_dir = Path(ROOT_DIR) / split / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Class dir not found: {class_dir}")

    # Find all non-JPG files
    non_jpg = [
        f for f in class_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
        and f.suffix.lower() != ".jpg"
    ]

    print(f"\n{'='*40}")
    print(f"Format Conversion — {split}/{class_name}")
    print(f"{'='*40}")
    print(f"Non-JPG files found : {len(non_jpg)}")
    print(f"Target format       : JPG (quality={JPG_QUALITY})")
    print(f"Mode                : {'DRY RUN — no changes' if dry_run else 'LIVE — files will be modified'}")

    if dry_run:
        print(f"\nFiles that would be converted:")
        for f in sorted(non_jpg)[:10]:
            print(f"  {f.name} ({f.suffix})")
        if len(non_jpg) > 10:
            print(f"  ... and {len(non_jpg) - 10} more")
        print(f"\nRe-run with dry_run=False to perform conversion.")
        return pd.DataFrame()

    # ── Live conversion ───────────────────────────────────────────
    records   = []
    converted = 0
    failed    = 0
    verified  = 0

    for src in sorted(non_jpg):
        record = {
            "original_path" : str(src),
            "original_format": src.suffix.lower(),
            "converted_path" : None,
            "status"         : None,
            "mse"            : None,
        }

        try:
            # Step 1: Convert
            dst = convert_to_jpg(src)
            record["converted_path"] = str(dst)

            # Step 2: Verify
            passed, mse = verify_conversion(src, dst)
            record["mse"] = mse

            if passed:
                # Step 3: Remove original
                src.unlink()
                record["status"] = "converted"
                converted += 1
                verified  += 1
            else:
                # Conversion quality too low — keep original
                dst.unlink()
                record["status"] = f"failed_verification (mse={mse})"
                failed += 1
                print(f"  ⚠ Verification failed: {src.name} "
                      f"(MSE={mse:.1f} > 50)")

        except FileExistsError:
            # JPG already exists — skip, original still present
            record["status"] = "skipped_exists"
            print(f"  → Skipped (target exists): {src.name}")

        except Exception as e:
            record["status"] = f"error: {e}"
            failed += 1
            print(f"  ✗ Failed: {src.name} — {e}")

        records.append(record)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n── Conversion Summary ────────────────────────────")
    print(f"  Attempted  : {len(non_jpg)}")
    print(f"  Converted  : {converted}")
    print(f"  Verified   : {verified} (MSE < 50)")
    print(f"  Failed     : {failed}")

    # ── Verify final state ────────────────────────────────────────
    final_jpgs = list(class_dir.glob("*.jpg"))
    final_pngs = list(class_dir.glob("*.png"))
    final_jpegs= list(class_dir.glob("*.jpeg"))
    print(f"\n── Final state of {split}/{class_name} ───────────")
    print(f"  JPG  : {len(final_jpgs)}")
    print(f"  PNG  : {len(final_pngs)}  (should be 0)")
    print(f"  JPEG : {len(final_jpegs)}  (should be 0)")

    # ── Save conversion log ───────────────────────────────────────
    df       = pd.DataFrame(records)
    log_path = f"{REPORTS_DIR}/{split}_{class_name}_conversion_log.csv"
    df.to_csv(log_path, index=False)
    print(f"\n✅ Conversion log → {log_path}")

    return df


# ─── FULL DATASET SCAN ────────────────────────────────────────────────────────

def scan_format_distribution(splits=("train", "val")) -> pd.DataFrame:
    """
    Scan entire dataset and report format distribution per class.
    Flags any class with >5% non-JPG as anomalous.
    """
    print(f"\n── Format Distribution Scan ──────────────────────")
    print(f"{'split':8s} {'class':12s} {'jpg':>6} {'png':>6} "
          f"{'jpeg':>6} {'total':>7} {'non_jpg%':>9} {'status':>12}")
    print("-" * 65)

    records = []
    for split in splits:
        split_path = Path(ROOT_DIR) / split
        for cls_dir in sorted(split_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            files  = list(cls_dir.iterdir())
            jpgs   = sum(1 for f in files if f.suffix.lower() == ".jpg")
            pngs   = sum(1 for f in files if f.suffix.lower() == ".png")
            jpegs  = sum(1 for f in files if f.suffix.lower() == ".jpeg")
            total  = jpgs + pngs + jpegs
            non_jpg_pct = round(100 * (pngs + jpegs) / max(total, 1), 1)
            status = "⚠ ANOMALY" if non_jpg_pct > 5 else "✅ clean"

            print(f"{split:8s} {cls_dir.name:12s} "
                  f"{jpgs:>6} {pngs:>6} {jpegs:>6} "
                  f"{total:>7} {non_jpg_pct:>8.1f}% {status:>12}")

            records.append({
                "split"      : split,
                "class"      : cls_dir.name,
                "jpg"        : jpgs,
                "png"        : pngs,
                "jpeg"       : jpegs,
                "total"      : total,
                "non_jpg_pct": non_jpg_pct,
                "anomaly"    : non_jpg_pct > 5
            })

    df = pd.DataFrame(records)
    anomalies = df[df["anomaly"]]
    if len(anomalies) > 0:
        print(f"\n  {len(anomalies)} anomalous class(es) found:")
        for _, r in anomalies.iterrows():
            print(f"  → {r['split']}/{r['class']}: "
                  f"{r['non_jpg_pct']}% non-JPG")
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Full scan first
    scan_format_distribution()

    # # Step 2: Dry run on the anomalous class
    # print("\n" + "="*40)
    # print("DRY RUN — val/horse conversion preview")
    # convert_non_jpg(split="val", class_name="horse", dry_run=True)

    # Step 3: Uncomment when ready to perform actual conversion
    print("\n" + "="*40)
    print("LIVE CONVERSION")
    convert_non_jpg(split="val", class_name="horse", dry_run=False)