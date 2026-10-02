# notebooks/visual_inspection.py
"""
Randomly sample and display images from both
original and clean dataset side by side per class.

Helps answer:
  - Do removed images actually look bad?
  - Do kept images actually look good?
  - Are there any obvious mistakes?

Run: python notebooks/visual_inspection.py
"""

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL     import Image
from pathlib import Path

sys.path.insert(0, ".")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SPLIT         = "train"
CLASSES       = ["cat", "dog", "elephant", "horse", "lion"]
N_PER_CLASS   = 10          # images to show per class
RANDOM_SEED   = 42
OUTPUT_DIR    = Path("notebooks/visual_inspection_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_RAW      = Path("data/raw/Animal")
DATA_CLEAN    = Path("data_clean/Animal")
REPORTS_DIR   = Path("reports")

random.seed(RANDOM_SEED)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_img(path, size=(160, 160)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size)
        return img
    except:
        return None


def get_verdict(file_path: str, master_df: pd.DataFrame) -> str:
    row = master_df[master_df["file_path"] == file_path]
    if row.empty:
        return "UNKNOWN"
    return row.iloc[0]["verdict"]


def get_flags(file_path: str, master_df: pd.DataFrame) -> list:
    row = master_df[master_df["file_path"] == file_path]
    if row.empty:
        return []
    r       = row.iloc[0]
    flags   = []
    col_map = {
        "Exact_duplicates" : "duplicate",
        "exact_duplicate"  : "duplicate",
        "near_duplicate"   : "near-dup",
        "is_blurry"        : "blurry",
        "is_noisy"         : "noisy",
        "is_outlier"       : "outlier",
        "is_mislabeled"    : "mislabeled",
    }
    for col, label in col_map.items():
        if col in r.index and bool(r[col]):
            flags.append(label)
    return flags


# ─── INSPECTION 1: REMOVED IMAGES (did we remove the right ones?) ──────────────

def inspect_removed(split=SPLIT, n_per_class=N_PER_CLASS):
    """
    Show a grid of REMOVED images per class.
    Question: do these actually look like bad images?
    """
    master_df = pd.read_csv(REPORTS_DIR / f"{split}_master_report.csv")
    removed   = master_df[master_df["verdict"].str.startswith("REMOVE")]

    print(f"\nInspecting REMOVED images ({split})")
    print(f"Total removed: {len(removed)}")

    for cls in CLASSES:
        cls_removed = removed[removed["class"] == cls]
        if cls_removed.empty:
            print(f"  {cls}: nothing removed")
            continue

        sample = cls_removed.sample(
            n=min(n_per_class, len(cls_removed)),
            random_state=RANDOM_SEED
        )

        ncols = 5
        nrows = (len(sample) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.5, nrows * 2.8)
        )
        fig.suptitle(
            f"REMOVED — {cls.upper()}  ({split})  "
            f"[{len(cls_removed)} total removed from this class]",
            fontsize=11, fontweight="bold", color="#C62828"
        )
        axes = np.array(axes).flatten()

        for i, (_, row) in enumerate(sample.iterrows()):
            img   = load_img(row["file_path"])
            flags = get_flags(row["file_path"], master_df)
            ax    = axes[i]

            if img:
                ax.imshow(img)
            else:
                ax.set_facecolor("#FFE0E0")
                ax.text(0.5, 0.5, "load error",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

            verdict_short = row["verdict"].replace("REMOVE — ", "")
            ax.set_title(
                f"{Path(row['file_path']).name}\n"
                f"❌ {verdict_short}\n"
                f"flags: {', '.join(flags) if flags else '—'}",
                fontsize=6.5, color="#B71C1C", pad=3
            )
            ax.axis("off")

            # Red border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#C62828")
                spine.set_linewidth(2)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = OUTPUT_DIR / f"{split}_{cls}_removed.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {cls}: {len(sample)} removed images → {out.name}")


# ─── INSPECTION 2: CLEAN IMAGES (did we keep the right ones?) ──────────────────

def inspect_clean(split=SPLIT, n_per_class=N_PER_CLASS):
    """
    Show a random sample of CLEAN images per class.
    Question: do these look genuinely good quality?
    """
    master_df = pd.read_csv(REPORTS_DIR / f"{split}_master_report.csv")
    clean     = master_df[master_df["verdict"] == "CLEAN"]

    print(f"\nInspecting CLEAN images ({split})")
    print(f"Total clean: {len(clean)}")

    for cls in CLASSES:
        cls_clean = clean[clean["class"] == cls]
        sample    = cls_clean.sample(
            n=min(n_per_class, len(cls_clean)),
            random_state=RANDOM_SEED
        )

        ncols = 5
        nrows = (len(sample) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.5, nrows * 2.8)
        )
        fig.suptitle(
            f"CLEAN — {cls.upper()}  ({split})  "
            f"[{len(cls_clean)} total clean in this class]",
            fontsize=11, fontweight="bold", color="#1B5E20"
        )
        axes = np.array(axes).flatten()

        for i, (_, row) in enumerate(sample.iterrows()):
            img = load_img(row["file_path"])
            ax  = axes[i]

            if img:
                ax.imshow(img)
            else:
                ax.set_facecolor("#E8F5E9")
                ax.text(0.5, 0.5, "load error",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)

            ax.set_title(
                f"{Path(row['file_path']).name}\n"
                f"✅ clean",
                fontsize=6.5, color="#2E7D32", pad=3
            )
            ax.axis("off")

            # Green border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#2E7D32")
                spine.set_linewidth(1.5)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = OUTPUT_DIR / f"{split}_{cls}_clean.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {cls}: {len(sample)} clean images → {out.name}")


# ─── INSPECTION 3: SIDE BY SIDE (before vs after per class) ───────────────────

def inspect_before_after(split=SPLIT, n_per_class=6):
    """
    Show 6 CLEAN vs 6 REMOVED side by side per class.
    Most informative view — directly compares kept vs removed.
    Question: is the boundary between kept/removed sensible?
    """
    master_df = pd.read_csv(REPORTS_DIR / f"{split}_master_report.csv")
    clean     = master_df[master_df["verdict"] == "CLEAN"]
    removed   = master_df[master_df["verdict"].str.startswith("REMOVE")]

    print(f"\nInspecting Before vs After ({split})")

    for cls in CLASSES:
        cls_clean   = clean[clean["class"] == cls].sample(
            n=min(n_per_class, len(clean[clean["class"] == cls])),
            random_state=RANDOM_SEED
        )
        cls_removed = removed[removed["class"] == cls]

        if cls_removed.empty:
            print(f"  {cls}: no removed images — skipping before/after")
            continue

        cls_removed = cls_removed.sample(
            n=min(n_per_class, len(cls_removed)),
            random_state=RANDOM_SEED
        )

        n       = max(len(cls_clean), len(cls_removed))
        fig     = plt.figure(figsize=(n * 2.6, 6.5))
        gs      = gridspec.GridSpec(
            2, n,
            figure=fig,
            hspace=0.45, wspace=0.08
        )

        fig.suptitle(
            f"{cls.upper()}  —  Before vs After Cleaning  ({split})\n"
            f"Top row: KEPT (clean)  |  Bottom row: REMOVED",
            fontsize=11, fontweight="bold"
        )

        # Top row — clean
        for i, (_, row) in enumerate(cls_clean.iterrows()):
            ax  = fig.add_subplot(gs[0, i])
            img = load_img(row["file_path"])
            if img:
                ax.imshow(img)
            ax.set_title(
                f"✅ KEPT\n{Path(row['file_path']).name}",
                fontsize=6, color="#1B5E20", pad=3
            )
            ax.axis("off")
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor("#2E7D32")
                sp.set_linewidth(2)

        # Bottom row — removed
        for i, (_, row) in enumerate(cls_removed.iterrows()):
            ax     = fig.add_subplot(gs[1, i])
            img    = load_img(row["file_path"])
            flags  = get_flags(row["file_path"], master_df)
            reason = row["verdict"].replace("REMOVE — ", "")
            if img:
                ax.imshow(img)
            ax.set_title(
                f"❌ {reason}\n"
                f"{', '.join(flags[:2]) if flags else '—'}",
                fontsize=6, color="#B71C1C", pad=3
            )
            ax.axis("off")
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor("#C62828")
                sp.set_linewidth(2)

        out = OUTPUT_DIR / f"{split}_{cls}_before_after.png"
        fig.savefig(out, dpi=130, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"  {cls}: before/after → {out.name}")


# ─── INSPECTION 4: REVIEW IMAGES (borderline cases) ───────────────────────────

def inspect_review(split=SPLIT, n_per_class=8):
    """
    Show REVIEW images — the borderline cases the pipeline is uncertain about.
    Question: do these look like they need human judgment?
    """
    master_df = pd.read_csv(REPORTS_DIR / f"{split}_master_report.csv")
    review    = master_df[master_df["verdict"].str.startswith("REVIEW")]

    print(f"\nInspecting REVIEW images ({split})")
    print(f"Total review: {len(review)}")

    for cls in CLASSES:
        cls_review = review[review["class"] == cls]
        if cls_review.empty:
            continue

        sample = cls_review.sample(
            n=min(n_per_class, len(cls_review)),
            random_state=RANDOM_SEED
        )

        ncols = 4
        nrows = (len(sample) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.8, nrows * 3.0)
        )
        fig.suptitle(
            f"REVIEW — {cls.upper()}  ({split})  "
            f"[{len(cls_review)} borderline images]",
            fontsize=11, fontweight="bold", color="#E65100"
        )
        axes = np.array(axes).flatten()

        for i, (_, row) in enumerate(sample.iterrows()):
            img   = load_img(row["file_path"])
            flags = get_flags(row["file_path"], master_df)
            ax    = axes[i]

            if img:
                ax.imshow(img)

            verdict_short = row["verdict"].replace("REVIEW — ", "")
            ax.set_title(
                f"{Path(row['file_path']).name}\n"
                f"👁 {verdict_short}\n"
                f"score={row.get('priority_score', '?')} "
                f"flags={', '.join(flags) if flags else '—'}",
                fontsize=6, color="#BF360C", pad=3
            )
            ax.axis("off")

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#FF6D00")
                spine.set_linewidth(1.5)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = OUTPUT_DIR / f"{split}_{cls}_review.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {cls}: {len(sample)} review images → {out.name}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="train",
                        choices=["train", "val"])
    parser.add_argument("--mode",
                        default="all",
                        choices=["removed", "clean",
                                 "before_after", "review", "all"],
                        help="Which inspection to run")
    parser.add_argument("--n",      default=10, type=int,
                        help="Images per class to show")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  VISUAL INSPECTION — {args.split.upper()}")
    print(f"{'='*60}")
    print(f"  Output → {OUTPUT_DIR}/")

    if args.mode in ("removed", "all"):
        inspect_removed(args.split, args.n)

    if args.mode in ("clean", "all"):
        inspect_clean(args.split, args.n)

    if args.mode in ("before_after", "all"):
        inspect_before_after(args.split, args.n // 2)

    if args.mode in ("review", "all"):
        inspect_review(args.split, args.n)

    print(f"\n{'='*60}")
    print(f"  Done — open {OUTPUT_DIR}/ to review images")
    print(f"\n  Files generated:")
    for f in sorted(OUTPUT_DIR.glob(f"{args.split}_*.png")):
        print(f"    {f.name}")
    print(f"{'='*60}")