"""
run_pipeline.py
Master pipeline script — runs all 9 cleaning modules in order.

Usage:
    python run_pipeline.py --split both --skip-embeddings
    python run_pipeline.py --split train
    python run_pipeline.py --split val --skip-embeddings

Dependency order:
    1. format_scan     → format distribution report
    2. preprocessing   → validation report
    3. embeddings      → .npy cache (required by steps 4, 7, 8)
    4. duplicates      → exact + near-dup report
    5. blur            → blur report (adaptive per-class threshold)
    6. noise           → noise report (adaptive per-class threshold)
    7. outliers        → isolation forest report
    8. mislabels       → 3-method voting report
    9. decision_engine → master report (reads all above)
"""

import argparse
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, ".")


# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Image Data Cleaning Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Dataset split to process (default: both)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding extraction if .npy cache already exists"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip format scan and image validation"
    )
    parser.add_argument(
        "--no-adaptive-blur",
        action="store_true",
        help="Use global blur threshold instead of adaptive per-class"
    )
    parser.add_argument(
        "--no-adaptive-noise",
        action="store_true",
        help="Use global noise threshold instead of adaptive per-class"
    )
    return parser.parse_args()


# ─── STEP RUNNER ──────────────────────────────────────────────────────────────

def run_step(name: str, fn, *args, **kwargs):
    """
    Run one pipeline step with timing and error handling.
    Returns (result, success_bool).
    Never raises — catches and reports all exceptions.
    """
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
        print(f"\n❌  {name} — FAILED ({elapsed:.1f}s)")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None, False


# ─── SINGLE SPLIT PIPELINE ────────────────────────────────────────────────────

def run_pipeline(split: str, args) -> dict:
    """
    Run all 9 steps for one split.
    Returns summary dict: steps passed/failed + final results.
    """
    summary = {
        "split"  : split,
        "steps"  : {},
        "results": {},
        "success": True,
    }

    adaptive_blur  = not args.no_adaptive_blur
    adaptive_noise = not args.no_adaptive_noise

    # ── Step 1: Format distribution scan ─────────────────────────
    if not args.skip_preprocessing:
        from src.format_converter import scan_format_distribution
        _, ok = run_step(
            f"Step 1 — Format Distribution Scan ({split})",
            scan_format_distribution,
            splits=[split]
        )
        summary["steps"]["format_scan"] = ok

    # ── Step 2: Image validation ──────────────────────────────────
    if not args.skip_preprocessing:
        from src.preprocessing import validate_dataset
        df_valid, ok = run_step(
            f"Step 2 — Image Validation ({split})",
            validate_dataset,
            "data/raw/Animal", split
        )
        if ok and df_valid is not None:
            invalid = int((~df_valid["valid"]).sum())
            print(f"    Invalid images : {invalid}")
        summary["steps"]["validation"] = ok

    # ── Step 3: Embedding extraction ──────────────────────────────
    emb_path = Path(f"embeddings/{split}_embeddings.npy")

    if args.skip_embeddings and emb_path.exists():
        print(f"\n{'#'*60}")
        print(f"#  Step 3 — Embeddings ({split}) — SKIPPED (cache exists)")
        print(f"{'#'*60}")
        from src.embeddings import load_embeddings
        embeddings, labels, paths = load_embeddings(split)
        summary["steps"]["embeddings"] = True
    else:
        from src.embeddings import extract_embeddings, load_embeddings
        _, ok = run_step(
            f"Step 3 — ResNet50 Embedding Extraction ({split})",
            extract_embeddings, split
        )
        summary["steps"]["embeddings"] = ok
        if not ok:
            print("    Embeddings failed — cannot continue. "
                  "Steps 4, 7, 8 depend on embeddings.")
            summary["success"] = False
            return summary
        embeddings, labels, paths = load_embeddings(split)

    # ── Step 4: Duplicate detection ───────────────────────────────
    import pandas as pd
    import numpy  as np
    from src.duplicates import find_exact_duplicates, find_near_duplicates

    def run_duplicates():
        df_exact, flagged     = find_exact_duplicates(paths, split)
        df_near, near_flagged = find_near_duplicates(
            embeddings, paths, split
        )

        df_combined = df_exact.copy()

        if not df_near.empty:

        # A file can have multiple near-duplicate matches.
        # Aggregate them so each file_path appears only once.
         confidence_order = {
             "POSSIBLE": 1,
             "LIKELY": 2,
             "DEFINITE": 3
         }

         near_summary = (
             df_near.groupby("file_path")
             .agg(
                 cosine_score=("cosine_score", "max"),
                 near_duplicate_of=(
                     "near_duplicate_of",
                     lambda x: "; ".join(x.dropna().astype(str).unique())
                ),
                 dup_confidence=(
                     "dup_confidence",
                     lambda x: max(
                         x.dropna(),
                         key=lambda v: confidence_order.get(v, 0),
                         default=np.nan
                     )
                 )
             )
             .reset_index()
         )

         # Every file now has at most one row.
         near_indexed = near_summary.set_index("file_path")

         df_combined["near_duplicate"] = df_combined["file_path"].isin(
             near_indexed.index
         )

         df_combined["cosine_score"] = df_combined["file_path"].map(
            near_indexed["cosine_score"]
         )

         df_combined["near_dup_of"] = df_combined["file_path"].map(
             near_indexed["near_duplicate_of"]
         )

         df_combined["dup_confidence"] = df_combined["file_path"].map(
             near_indexed["dup_confidence"]
         )

        else:
         df_combined["near_duplicate"] = False
         df_combined["cosine_score"] = np.nan
         df_combined["near_dup_of"] = np.nan
         df_combined["dup_confidence"] = np.nan

        out = f"reports/{split}_duplicates_report.csv"
        df_combined.to_csv(out, index=False)

        exact_col = (
            "Exact_duplicates"
            if "Exact_duplicates" in df_combined.columns
            else "exact_duplicate"
        )
        print(f"    Exact duplicates : "
              f"{int(df_combined[exact_col].astype(bool).sum())}")
        print(f"    Near duplicates  : "
              f"{int(df_combined['near_duplicate'].astype(bool).sum())}")
        print(f"    Report saved     : {out}")

    _, ok = run_step(
        f"Step 4 — Duplicate Detection ({split})",
        run_duplicates
    )
    summary["steps"]["duplicates"] = ok

    # ── Step 5: Blur detection ────────────────────────────────────
    from src.quality import detect_blur
    _, ok = run_step(
        f"Step 5 — Blur Detection ({split}) "
        f"[{'adaptive' if adaptive_blur else 'global'}]",
        detect_blur,
        split,
        use_adaptive=adaptive_blur
    )
    summary["steps"]["blur"] = ok

    # ── Step 6: Noise detection + refinement ───────────────────────
    from src.quality import detect_noise, refine_noise_flags

    def run_noise_pipeline():
        # First detect noise using the AutoEncoder
        detect_noise(
            split,
            use_adaptive=adaptive_noise
        )

        # Then refine the noise flags using blur ratio + outlier evidence
        return refine_noise_flags(split)

    _, ok = run_step(
        f"Step 6 — Noise Detection ({split}) "
        f"[{'adaptive' if adaptive_noise else 'global'}]",
        run_noise_pipeline
    )

    summary["steps"]["noise"] = ok

    # ── Step 7: Outlier detection ─────────────────────────────────
    from src.outliers import detect_outliers
    _, ok = run_step(
        f"Step 7 — Outlier Detection ({split})",
        detect_outliers, split
    )
    summary["steps"]["outliers"] = ok

    # ── Step 8: Mislabel detection ────────────────────────────────
    from src.mislabel import detect_mislabels
    _, ok = run_step(
        f"Step 8 — Mislabel Detection ({split})",
        detect_mislabels, split
    )
    summary["steps"]["mislabels"] = ok

    # ── Step 9: Decision engine ───────────────────────────────────
    from src.decision_engine import build_master_report
    df_master, ok = run_step(
        f"Step 9 — Decision Engine ({split})",
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

    failed = [k for k, v in summary["steps"].items() if not v]
    if failed:
        summary["success"] = False

    return summary


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    splits = ["train", "val"] if args.split == "both" else [args.split]

    print("\n" + "=" * 60)
    print("    INTELLIGENT IMAGE DATA CLEANING FRAMEWORK")
    print("=" * 60)
    print(f"  Splits           : {', '.join(splits)}")
    print(f"  Skip embeddings  : {args.skip_embeddings}")
    print(f"  Skip preprocess  : {args.skip_preprocessing}")
    print(f"  Adaptive blur    : {not args.no_adaptive_blur}")
    print(f"  Adaptive noise   : {not args.no_adaptive_noise}")
    print("=" * 60)

    t_start   = time.time()
    summaries = {}

    for split in splits:
        print(f"\n\n{'=' * 60}")
        print(f"    PROCESSING: {split.upper()}")
        print(f"{'=' * 60}")
        summaries[split] = run_pipeline(split, args)

    # ── Final summary ─────────────────────────────────────────────
    t_total = time.time() - t_start

    print(f"\n\n{'=' * 60}")
    print(f"    PIPELINE COMPLETE  —  {t_total:.1f}s total")
    print(f"{'=' * 60}")

    all_ok = True
    for split, summary in summaries.items():
        steps   = summary.get("steps", {})
        failed  = [k for k, v in steps.items() if not v]
        results = summary.get("results", {})

        print(f"\n  {split.upper()}:")

        if failed:
            print(f"    ❌ Failed steps : {', '.join(failed)}")
            all_ok = False
        else:
            print(f"    ✅ All steps passed")

        if results:
            print(f"    Total    : {results['total']:>6,}")
            print(f"    Clean    : {results['clean']:>6,}  "
                  f"({results['clean_pct']}%)")
            print(f"    Remove   : {results['remove']:>6,}  "
                  f"({results['remove_pct']}%)")
            print(f"    Review   : {results['review']:>6,}  "
                  f"({results['review_pct']}%)")

    print(f"\n  Reports   →  reports/")
    print(f"  Dashboard →  streamlit run app.py")
    print(f"  API       →  python run_api.py")
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()