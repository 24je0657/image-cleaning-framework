# run_pipeline.py
"""
General-purpose Image Cleaning Pipeline.

Works with ANY dataset where:
  - Folders = class names
  - Images are JPG/PNG/JPEG

Usage examples:
    # Original animal dataset
    python run_pipeline.py --data_dir data/raw/Animal --output_dir data_clean/Animal

    # Any new dataset
    python run_pipeline.py --data_dir /path/to/flowers --output_dir /path/to/flowers_clean

    # Skip heavy steps if already done
    python run_pipeline.py --data_dir data/raw/Animal --output_dir data_clean/Animal \\
                           --skip-embeddings

    # Tune thresholds for a specific dataset
    python run_pipeline.py --data_dir /path/to/dataset --output_dir /path/to/clean \\
                           --blur-percentile 10 --contamination 0.03
"""

import argparse
import time
import sys
import shutil
import json
from pathlib import Path

sys.path.insert(0, ".")

from src.config import PipelineConfig


# ─── ARGS ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="General-purpose Image Data Cleaning Framework",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── Required ──────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to input dataset\n"
             "(expects: data_dir/[split]/class_name/images\n"
             "      or: data_dir/class_name/images)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Path where cleaned dataset will be written"
    )

    # ── Skip flags ────────────────────────────────────────────────
    parser.add_argument("--skip-embeddings",
                        action="store_true",
                        help="Skip embedding extraction if cache exists")
    parser.add_argument("--skip-preprocessing",
                        action="store_true",
                        help="Skip format scan and validation")

    # ── Module toggles ────────────────────────────────────────────
    parser.add_argument("--no-duplicates", action="store_true")
    parser.add_argument("--no-blur",       action="store_true")
    parser.add_argument("--no-noise",      action="store_true")
    parser.add_argument("--no-outliers",   action="store_true")
    parser.add_argument("--no-mislabels",  action="store_true")

    # ── Tunable thresholds ────────────────────────────────────────
    parser.add_argument("--blur-percentile", type=int,   default=5,
                        help="Per-class blur threshold percentile (default: 5)")
    parser.add_argument("--contamination",   type=float, default=0.05,
                        help="Isolation Forest contamination (default: 0.05)")
    parser.add_argument("--cosine-threshold",type=float, default=0.97,
                        help="Near-duplicate cosine threshold (default: 0.97)")

    return parser.parse_args()


# ─── STEP RUNNER ──────────────────────────────────────────────────────────────

def run_step(name, fn, *args, **kwargs):
    print(f"\n{'#'*60}")
    print(f"#  {name}")
    print(f"{'#'*60}")
    t0 = time.time()
    try:
        result  = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n✅  {name} — complete ({elapsed:.1f}s)")
        return result, True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n❌  {name} — FAILED ({elapsed:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return None, False


# ─── EXPORT CLEAN IMAGES ──────────────────────────────────────────────────────

def export_clean_images(cfg: PipelineConfig, split: str) -> dict:
    """
    Read master report and copy CLEAN images to output_dir.
    Preserves class folder structure.
    Returns export summary.
    """
    import pandas as pd

    master_path = cfg.report_path(split, "master_report")
    if not master_path.exists():
        print(f"  Master report not found: {master_path}")
        return {}

    df       = pd.read_csv(master_path)
    clean_df = df[df["verdict"] == "CLEAN"]

    out_root = cfg.output_split_path(split)
    out_root.mkdir(parents=True, exist_ok=True)

    copied  = 0
    failed  = 0
    per_cls = {}

    for _, row in clean_df.iterrows():
        src = Path(row["file_path"])
        # Preserve class subfolder
        cls     = row.get("class", src.parent.name)
        dst_dir = out_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst     = dst_dir / src.name

        try:
            shutil.copy2(src, dst)
            copied += 1
            per_cls[cls] = per_cls.get(cls, 0) + 1
        except Exception as e:
            print(f"  Copy failed: {src.name} — {e}")
            failed += 1

    total   = len(df)
    removed = total - len(clean_df)

    print(f"\n── Export summary ─────────────────────────────────")
    print(f"  Input images    : {total:,}")
    print(f"  Clean exported  : {copied:,}  ({100*copied/total:.1f}%)")
    print(f"  Removed         : {removed:,}  ({100*removed/total:.1f}%)")
    print(f"  Failed copies   : {failed}")
    print(f"  Output location : {out_root}")
    print(f"\n  Per class:")
    for cls in sorted(per_cls):
        cls_total = len(df[df["class"] == cls]) if "class" in df else "?"
        print(f"    {cls:15s}: {per_cls[cls]:4d} clean / {cls_total}")

    return {
        "total"  : total,
        "copied" : copied,
        "removed": removed,
        "failed" : failed,
        "per_cls": per_cls,
        "out_dir": str(out_root)
    }


# ─── SINGLE SPLIT PIPELINE ────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig, split: str) -> dict:
    summary = {"split": split, "steps": {}, "success": True}

    split_path = cfg.split_path(split)
    label      = split if split else "all"

    # ── Step 1: Format scan ───────────────────────────────────────
    if not args.skip_preprocessing:
        from src.format_converter import scan_format_distribution
        _, ok = run_step(
            f"Step 1 — Format Distribution Scan ({label})",
            scan_format_distribution,
            splits=[split] if split else [""]
        )
        summary["steps"]["format_scan"] = ok

    # ── Step 2: Image validation ──────────────────────────────────
    if not args.skip_preprocessing:
        from src.preprocessing import validate_dataset
        df_valid, ok = run_step(
            f"Step 2 — Image Validation ({label})",
            validate_dataset,
            str(cfg.data_dir), split
        )
        if ok and df_valid is not None:
            print(f"    Invalid images : {(~df_valid['valid']).sum()}")
        summary["steps"]["validation"] = ok

    # ── Step 3: Embeddings ────────────────────────────────────────
    emb_path = cfg.embedding_path(split, "embeddings")
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_embeddings and emb_path.exists():
        print(f"\n{'#'*60}")
        print(f"#  Step 3 — Embeddings ({label}) — SKIPPED (cache exists)")
        print(f"{'#'*60}")
        from src.embeddings import load_embeddings
        embeddings, labels, paths = load_embeddings(split)
        summary["steps"]["embeddings"] = True
    else:
        from src.embeddings import extract_embeddings, load_embeddings
        _, ok = run_step(
            f"Step 3 — ResNet50 Embedding Extraction ({label})",
            extract_embeddings, split
        )
        summary["steps"]["embeddings"] = ok
        if not ok:
            print("    Embeddings failed — cannot continue")
            summary["success"] = False
            return summary
        embeddings, labels, paths = load_embeddings(split)

    # ── Step 4: Duplicates ────────────────────────────────────────
    if cfg.run_duplicates:
        import pandas as pd
        import numpy  as np
        from src.duplicates import (find_exact_duplicates,
                                    find_near_duplicates)

        def run_duplicates():
            df_e, flagged  = find_exact_duplicates(paths, split)
            df_n, near_f   = find_near_duplicates(embeddings, paths, split)

            df_combined = df_e.copy()
            if not df_n.empty:
                df_combined["near_duplicate"] = df_combined[
                    "file_path"].isin(df_n["file_path"])
                df_n_dedup = (
                    df_n.sort_values(by="cosine_score", ascending=False)
                    .drop_duplicates(subset="file_path", keep="first")
                    .set_index("file_path")
                )
                for col in ["cosine_score","near_duplicate_of","dup_confidence"]:
                    if col in df_n_dedup.columns:
                        df_combined[col] = df_combined["file_path"].map(
                            df_n_dedup[col])
            else:
                df_combined["near_duplicate"] = False

            out = cfg.report_path(split, "duplicates_report")
            df_combined.to_csv(out, index=False)

            exact_col = ("Exact_duplicates"
                         if "Exact_duplicates" in df_combined.columns
                         else "exact_duplicate")
            print(f"    Exact : {df_combined[exact_col].astype(bool).sum()}")
            print(f"    Near  : {df_combined['near_duplicate'].astype(bool).sum()}")

        _, ok = run_step(
            f"Step 4 — Duplicate Detection ({label})",
            run_duplicates
        )
        summary["steps"]["duplicates"] = ok

    # ── Step 5: Blur ──────────────────────────────────────────────
    if cfg.run_blur:
        from src.quality import detect_blur
        _, ok = run_step(
            f"Step 5 — Blur Detection ({label}) [adaptive p{cfg.blur_percentile}]",
            detect_blur, split,
            use_adaptive=True,
            percentile=cfg.blur_percentile
        )
        summary["steps"]["blur"] = ok

    # ── Step 6: Noise ─────────────────────────────────────────────
    if cfg.run_noise:
        from src.quality import detect_noise , refine_noise_flags
        _, ok_noise = run_step(
            f"Step 6 — Noise Detection ({label}) [adaptive]",
            detect_noise, split, use_adaptive=True
        )
        if ok_noise:
            _, ok_refine = run_step(
                f"Step 6b — Noise Flag Refinement ({label})",
                refine_noise_flags, split
            )
        else:
            ok_refine = False
        summary["steps"]["noise"] = ok_noise and ok_refine

    # ── Step 7: Outliers ──────────────────────────────────────────
    if cfg.run_outliers:
        from src.outliers import detect_outliers
        _, ok = run_step(
            f"Step 7 — Outlier Detection ({label})",
            detect_outliers, split
        )
        summary["steps"]["outliers"] = ok

    # ── Step 8: Mislabels ─────────────────────────────────────────
    if cfg.run_mislabels:
        from src.mislabel import detect_mislabels
        _, ok = run_step(
            f"Step 8 — Mislabel Detection ({label})",
            detect_mislabels, split
        )
        summary["steps"]["mislabels"] = ok

    # ── Step 9: Decision engine ───────────────────────────────────
    from src.decision_engine import build_master_report
    df_master, ok = run_step(
        f"Step 9 — Decision Engine ({label})",
        build_master_report, split
    )
    summary["steps"]["decision_engine"] = ok

    if ok and df_master is not None:
        total  = len(df_master)
        clean  = int((df_master["verdict"] == "CLEAN").sum())
        remove = int(df_master["verdict"].str.startswith("REMOVE").sum())
        review = int(df_master["verdict"].str.startswith("REVIEW").sum())
        summary["results"] = {
            "total"     : total,
            "clean"     : clean,
            "remove"    : remove,
            "review"    : review,
            "clean_pct" : round(100 * clean  / total, 2),
            "remove_pct": round(100 * remove / total, 2),
            "review_pct": round(100 * review / total, 2),
        }

    # ── Step 10: Export clean images ──────────────────────────────
    export_result, ok = run_step(
        f"Step 10 — Export Clean Images ({label})",
        export_clean_images, cfg, split
    )
    summary["steps"]["export"]  = ok
    summary["export"]           = export_result

    failed = [k for k, v in summary["steps"].items() if not v]
    if failed:
        summary["success"] = False

    return summary


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    global args
    args = parse_args()

    # ── Build config ──────────────────────────────────────────────
    cfg = PipelineConfig(
        data_dir         = Path(args.data_dir),
        output_dir       = Path(args.output_dir),
        run_duplicates   = not args.no_duplicates,
        run_blur         = not args.no_blur,
        run_noise        = not args.no_noise,
        run_outliers     = not args.no_outliers,
        run_mislabels    = not args.no_mislabels,
        blur_percentile  = args.blur_percentile,
        contamination    = args.contamination,
        cosine_threshold = args.cosine_threshold,
    )

    # Create output directories
    cfg.output_dir.mkdir(parents=True,  exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    print("\n" + "=" * 60)
    print("    INTELLIGENT IMAGE DATA CLEANING FRAMEWORK")
    print("=" * 60)
    print(cfg.summary())
    print(f"\n  Modules          : "
          f"{'dup ' if cfg.run_duplicates else ''}"
          f"{'blur ' if cfg.run_blur else ''}"
          f"{'noise ' if cfg.run_noise else ''}"
          f"{'outlier ' if cfg.run_outliers else ''}"
          f"{'mislabel' if cfg.run_mislabels else ''}")
    print(f"  Skip embeddings  : {args.skip_embeddings}")
    print("=" * 60)

    # Save config for reproducibility
    cfg.to_json(cfg.reports_dir / "pipeline_config.json")

    # Run pipeline for each split
    t_start   = time.time()
    summaries = {}

    for split in cfg.splits:
        label = split if split else "all"
        print(f"\n\n{'='*60}")
        print(f"    PROCESSING: {label.upper()}")
        print(f"{'='*60}")
        summaries[split] = run_pipeline(cfg, split)

    # ── Final summary ─────────────────────────────────────────────
    t_total = time.time() - t_start

    print(f"\n\n{'='*60}")
    print(f"    PIPELINE COMPLETE  —  {t_total:.1f}s total")
    print(f"{'='*60}")

    all_ok = True
    for split, summary in summaries.items():
        label   = split if split else "all"
        steps   = summary.get("steps",   {})
        failed  = [k for k, v in steps.items() if not v]
        results = summary.get("results", {})
        export  = summary.get("export",  {})

        print(f"\n  {label.upper()}:")
        if failed:
            print(f"    ❌ Failed steps : {', '.join(failed)}")
            all_ok = False
        else:
            print(f"    ✅ All steps passed")

        if results:
            print(f"    Total    : {results['total']:>6,}")
            print(f"    Clean    : {results['clean']:>6,}  ({results['clean_pct']}%)")
            print(f"    Remove   : {results['remove']:>6,}  ({results['remove_pct']}%)")
            print(f"    Review   : {results['review']:>6,}  ({results['review_pct']}%)")

        if export:
            print(f"    Exported : {export.get('copied',0):>6,} images → {export.get('out_dir','')}")

    print(f"\n  Clean dataset → {cfg.output_dir}")
    print(f"  Reports       → {cfg.reports_dir}")
    print(f"  Dashboard     → streamlit run app.py")
    print(f"  API           → python run_api.py")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()