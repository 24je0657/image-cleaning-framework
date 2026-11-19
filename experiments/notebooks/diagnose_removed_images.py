# notebooks/diagnose_removed_images.py
"""
For each "wrongly removed" image identified during visual inspection,
find out EXACTLY which flags caused its removal.
Diagnose before changing anything.
"""

import sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from PIL     import Image
from pathlib import Path

sys.path.insert(0, ".")

SPLIT      = "train"
OUTPUT_DIR = Path("notebooks/diagnosis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_img(path, size=(180, 180)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size)
        return img
    except:
        return None


def diagnose_class(cls: str, split: str = SPLIT, n: int = 20):
    """
    For a given class:
    1. Get all REMOVED images
    2. Show exactly which flags triggered removal
    3. Show blur score vs class threshold
    4. Show whether blur was the PRIMARY cause or a secondary flag
    """
    master_df = pd.read_csv(f"reports/{split}_master_report.csv")
    blur_df   = pd.read_csv(f"reports/{split}_blur_report.csv")

    # All removed images for this class
    removed = master_df[
        (master_df["class"]   == cls) &
        (master_df["verdict"].str.startswith("REMOVE"))
    ].copy()

    if removed.empty:
        print(f"  {cls}: nothing removed")
        return

    # Merge blur scores
    removed = removed.merge(
        blur_df[["file_path","blur_score","threshold_used","is_blurry"]],
        on="file_path", how="left"
    )

    # ── Flag breakdown ────────────────────────────────────────────
    flag_cols = {
        "Exact_duplicates" : "exact_dup",
        "exact_duplicate"  : "exact_dup",
        "near_duplicate"   : "near_dup",
        "is_blurry"        : "blurry",
        "is_noisy"         : "noisy",
        "is_outlier"       : "outlier",
        "is_mislabeled"    : "mislabeled",
    }

    # Normalise column names
    active_flags = {}
    for col, label in flag_cols.items():
        if col in removed.columns:
            active_flags[label] = removed[col].astype(bool)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS: {cls.upper()} — {split}")
    print(f"{'='*60}")
    print(f"  Total removed : {len(removed)}")
    print(f"\n  Flag breakdown (images removed due to each flag):")

    # Count how many removed images have each flag
    for label, series in active_flags.items():
        count = series.sum()
        pct   = 100 * count / len(removed)
        print(f"    {label:15s}: {count:4d}  ({pct:.1f}% of removed)")

    # ── Primary cause analysis ────────────────────────────────────
    print(f"\n  Primary removal cause (verdict column):")
    verdict_counts = removed["verdict"].value_counts()
    for v, c in verdict_counts.items():
        print(f"    {v:35s}: {c}")

    # ── Multi-flag analysis ───────────────────────────────────────
    print(f"\n  Multi-flag breakdown:")
    for n_flags in sorted(removed["total_flags"].unique()):
        subset = removed[removed["total_flags"] == n_flags]
        print(f"    {n_flags} flags: {len(subset)} images")
        if n_flags > 1:
            # Show which flag combinations occur
            combo_col = [
                label for label, series in active_flags.items()
            ]
            combos = (
                removed[removed["total_flags"] == n_flags]
                .apply(
                    lambda r: "+".join([
                        label for label, series in active_flags.items()
                        if bool(r.get(
                            [c for c,l in flag_cols.items()
                             if l==label and c in r.index][0]
                            if any(c in r.index
                                   for c,l in flag_cols.items()
                                   if l==label)
                            else "dummy", False
                        ))
                    ]),
                    axis=1
                )
                .value_counts()
            )
            for combo, cnt in combos.head(5).items():
                print(f"      {combo:40s}: {cnt}")

    # ── Blur-specific deep dive ───────────────────────────────────
    if "is_blurry" in removed.columns:
        blur_removed = removed[removed["is_blurry"] == True]
        only_blur    = removed[
            (removed["is_blurry"] == True) &
            (removed["total_flags"] == 1)
        ]
        print(f"\n  Blur deep dive:")
        print(f"    Removed WITH blur flag    : {len(blur_removed)}")
        print(f"    Removed ONLY due to blur  : {len(only_blur)}")
        if len(blur_removed) > 0:
            print(f"    Blur score range (removed): "
                  f"{blur_removed['blur_score'].min():.1f} – "
                  f"{blur_removed['blur_score'].max():.1f}")
            print(f"    Threshold used            : "
                  f"{blur_removed['threshold_used'].iloc[0]:.1f}")

        if len(only_blur) > 0:
            print(f"\n  ← These {len(only_blur)} images were removed")
            print(f"    ONLY because of blur (no other flags)")
            print(f"    These are the ones to visually inspect")
            print(f"    Blur scores: "
                  f"{sorted(only_blur['blur_score'].values)}")

    return removed


def visualise_removal_cause(cls: str, split: str = SPLIT,
                             n: int = 15):
    """
    Grid showing removed images annotated with ALL their flags.
    Color code by primary removal cause.
    """
    master_df = pd.read_csv(f"reports/{split}_master_report.csv")
    blur_df   = pd.read_csv(f"reports/{split}_blur_report.csv")

    removed = master_df[
        (master_df["class"] == cls) &
        (master_df["verdict"].str.startswith("REMOVE"))
    ].copy()

    removed = removed.merge(
        blur_df[["file_path","blur_score","threshold_used"]],
        on="file_path", how="left"
    )

    sample = removed.sample(
        n=min(n, len(removed)), random_state=42
    )

    ncols  = 5
    nrows  = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.8, nrows * 3.2)
    )
    fig.suptitle(
        f"{cls.upper()} — REMOVED images with flag breakdown\n"
        f"({split}, {len(removed)} total removed)",
        fontsize=11, fontweight="bold"
    )
    axes = np.array(axes).flatten()

    # Color per primary verdict
    VERDICT_COLORS = {
        "duplicate"  : "#C62828",
        "blurry"     : "#1565C0",
        "mislabeled" : "#6A1B9A",
        "high priority": "#E65100",
        "near duplicate": "#AD1457",
    }

    for i, (_, row) in enumerate(sample.iterrows()):
        img = load_img(row["file_path"])
        ax  = axes[i]

        if img:
            ax.imshow(img)

        # Build full flag list
        flag_cols_map = {
            "Exact_duplicates": "dup",
            "exact_duplicate" : "dup",
            "near_duplicate"  : "near",
            "is_blurry"       : "blur",
            "is_noisy"        : "noise",
            "is_outlier"      : "outlier",
            "is_mislabeled"   : "mislabel",
        }
        flags = [
            lbl for col, lbl in flag_cols_map.items()
            if col in row.index and bool(row[col])
        ]
        flags = list(dict.fromkeys(flags))  # deduplicate

        blur_score = row.get("blur_score", "?")
        blur_thresh = row.get("threshold_used", "?")
        verdict = row["verdict"].replace("REMOVE -", "")

        #Border color by verdict
        border_color = "#C62828"  # default red
        for key , color in VERDICT_COLORS.items():
            if key in verdict.lower():
                border_color = color
                break

        score_str = (f"{blur_score:.0f}" if isinstance(blur_score, float)
                     else "?")
        thresh_str = (f"{blur_thresh:.0f}" if isinstance(blur_thresh, float)
                     else "?")

        ax.set_title(
            f"{Path(row['file_path']).name}\n"
            f"verdict: {verdict}\n"
            f"flags: {'+'.join(flags) if flags else '—'}\n"
            f"blur={score_str} (t={thresh_str}) "
            f"p={row.get('priority_score','?')}",
            fontsize=5.5, pad=3            
        )

        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)



    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / f"{split}_{cls}_removal_cause.png"
    fig.savefig(out, dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n  Saved → {out.name}")
    return out


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        choices=["train","val"])
    parser.add_argument("--classes", nargs="+",
                        default=["cat","dog","horse","elephant","lion"])
    args = parser.parse_args()

    all_diagnoses = {}

    for cls in args.classes:
        removed_df = diagnose_class(cls, args.split)
        if removed_df is not None:
            all_diagnoses[cls] = removed_df
            visualise_removal_cause(cls, args.split)

    # ── Cross-class summary ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CROSS-CLASS DIAGNOSIS SUMMARY — {args.split.upper()}")
    print(f"{'='*60}")
    print(f"\n  {'class':12s} {'removed':>8} {'blur_only':>10} "
          f"{'dup':>6} {'outlier':>8} {'mislabel':>9}")
    print("  " + "-" * 57)

    for cls, df in all_diagnoses.items():
        flag_map = {
            "Exact_duplicates": "exact_dup",
            "exact_duplicate" : "exact_dup",
            "near_duplicate"  : "near_dup",
            "is_blurry"       : "blurry",
            "is_outlier"      : "outlier",
            "is_mislabeled"   : "mislabeled",
        }

        def safe_flag_sum(col):
            for c in flag_map:
                if flag_map[c] == col and c in df.columns:
                    return int(df[c].astype(bool).sum())
            return 0

        blur_only = int(
            df[
                (df.get("is_blurry", False).astype(bool)) &
                (df["total_flags"] == 1)
            ].shape[0]
            if "is_blurry" in df.columns else 0
        )
        dup     = safe_flag_sum("exact_dup")
        outlier = safe_flag_sum("outlier")
        mislabel= safe_flag_sum("mislabeled")

        print(f"  {cls:12s} {len(df):>8} {blur_only:>10} "
              f"{dup:>6} {outlier:>8} {mislabel:>9}")

    print(f"\n  'blur_only' = removed exclusively due to blur flag")
    print(f"  If blur_only is low → blur threshold is NOT the main cause")
    print(f"  If blur_only is high → blur threshold IS too aggressive")

    
