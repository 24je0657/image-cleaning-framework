"""
Fast controlled experiment using cached ResNet50 embeddings.

Classifier: Logistic Regression (multinomial)
  = equivalent to nn.Linear(2048, 5) with softmax
  = direct equivalent of frozen ResNet50 + single FC layer

Path matching: normalised before comparison to avoid
  format mismatches between embedding paths and master report.
"""

import sys
import time
import random
import json
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib  import Path
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score

sys.path.insert(0, ".")

CLASSES     = ["cat", "dog", "elephant", "horse", "lion"]
RANDOM_SEED = 42
RESULTS_DIR = Path("notebooks/experiment_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_split(split: str):
    embs   = np.load(f"embeddings/{split}_embeddings.npy")
    labels = np.load(f"embeddings/{split}_labels.npy",
                     allow_pickle=True)
    paths  = np.load(f"embeddings/{split}_file_paths.npy",
                     allow_pickle=True)
    return embs, labels, paths


# ─── PATH NORMALISATION ───────────────────────────────────────────────────────

def normalise_path(p: str) -> str:
    """
    Normalise path to filename only for comparison.
    Handles mismatches between:
      embeddings: 'data/raw/Animal/train/cat/cat1.jpg'
      master    : 'data/raw/Animal/train/cat/cat1.jpg'
    If prefixes differ, fall back to filename comparison.
    """
    return str(Path(p).as_posix())


def build_path_index(paths: np.ndarray) -> dict:
    """
    Build two lookup dicts:
      full path → array index
      filename  → array index  (fallback)
    """
    full_idx = {normalise_path(str(p)): i for i, p in enumerate(paths)}
    name_idx = {Path(str(p)).name     : i for i, p in enumerate(paths)}
    return full_idx, name_idx


# ─── DATASET BUILDERS ─────────────────────────────────────────────────────────

def build_dataset_B_clean(embs, labels, paths):
    """
    Dataset B: Pipeline-cleaned images.
    Matches master report paths to embedding indices
    using full path first, filename fallback second.
    Reports match rate so path issues are visible.
    """
    master     = pd.read_csv("reports/train_master_report.csv")
    clean_rows = master[master["verdict"] == "CLEAN"]

    full_idx, name_idx = build_path_index(paths)

    matched   = []
    unmatched = []

    for _, row in clean_rows.iterrows():
        p        = normalise_path(row["file_path"])
        filename = Path(row["file_path"]).name

        if p in full_idx:
            matched.append(full_idx[p])
        elif filename in name_idx:
            matched.append(name_idx[filename])
        else:
            unmatched.append(row["file_path"])

    print(f"\n  Path matching (Dataset B):")
    print(f"    Clean in report  : {len(clean_rows):,}")
    print(f"    Matched          : {len(matched):,}")
    print(f"    Unmatched        : {len(unmatched):,}")
    if unmatched:
        print(f"    Sample unmatched : {unmatched[:2]}")

    idx = np.array(matched)
    return embs[idx], labels[idx]


def build_dataset_A_random(embs, labels, n: int):
    """Dataset A: Random subset — null hypothesis baseline."""
    idx = np.arange(len(embs))
    np.random.shuffle(idx)
    return embs[idx[:n]], labels[idx[:n]]


def build_dataset_C_full(embs, labels):
    """Dataset C: Full original — upper bound."""
    return embs, labels


# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

def run_experiment(name        : str,
                   X_train     : np.ndarray,
                   y_train     : np.ndarray,
                   X_val       : np.ndarray,
                   y_val       : np.ndarray) -> dict:
    """
    Logistic Regression on L2-normalised embeddings.
    Equivalent to nn.Linear(2048, 5) + softmax:
      - No hidden layers
      - Direct linear mapping from features to classes
      - Same as frozen ResNet50 + single FC head
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Train: {len(X_train):,}    Val: {len(X_val):,}")
    print(f"{'='*60}")

    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {cls:12s}: {cnt:,}")

    # L2 normalise — same as using cosine similarity
    # consistent with how ResNet50 embeddings are used elsewhere
    from sklearn.preprocessing import normalize
    X_tr_norm = normalize(X_train, norm="l2")
    X_vl_norm = normalize(X_val,   norm="l2")

    clf = LogisticRegression(
        C           = 1.0,
        max_iter    = 1000,
        random_state= RANDOM_SEED,
        solver      = "lbfgs",
    )

    t0 = time.time()
    clf.fit(X_tr_norm, y_train)
    t1 = time.time()

    y_pred   = clf.predict(X_vl_norm)
    val_acc  = accuracy_score(y_val, y_pred) * 100

    per_class = {}
    for cls in CLASSES:
        mask = y_val == cls
        if mask.sum() == 0:
            continue
        per_class[cls] = round(
            accuracy_score(y_val[mask], y_pred[mask]) * 100, 2
        )

    print(f"\n  Val accuracy  : {val_acc:.2f}%")
    print(f"  Training time : {t1-t0:.1f}s")
    print(f"  Iterations    : {clf.n_iter_[0]}")
    print(f"  Per-class:")
    for cls, acc in per_class.items():
        print(f"    {cls:12s}: {acc:.1f}%")

    return {
        "name"     : name,
        "n_train"  : len(X_train),
        "val_acc"  : round(val_acc, 2),
        "per_class": per_class,
        "duration_s": round(t1 - t0, 1),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading cached embeddings...")
    X_tr, y_tr, p_tr = load_split("train")
    X_vl, y_vl, _    = load_split("val")
    print(f"Train: {X_tr.shape}    Val: {X_vl.shape}")

    X_B, y_B = build_dataset_B_clean(X_tr, y_tr, p_tr)
    n_clean   = len(X_B)

    X_A, y_A = build_dataset_A_random(X_tr, y_tr, n_clean)
    X_C, y_C = build_dataset_C_full(X_tr, y_tr)

    print(f"\n── Dataset sizes ──────────────────────────────────")
    print(f"  A (random, n={len(X_A):,}) ← null baseline")
    print(f"  B (clean,  n={len(X_B):,}) ← pipeline output")
    print(f"  C (full,   n={len(X_C):,}) ← upper bound")

    results = {}
    results["A_random"] = run_experiment(
        "A — Random subset (null baseline)",
        X_A, y_A, X_vl, y_vl
    )
    results["B_clean"] = run_experiment(
        "B — Clean subset (pipeline output)",
        X_B, y_B, X_vl, y_vl
    )
    results["C_full"] = run_experiment(
        "C — Full original dataset",
        X_C, y_C, X_vl, y_vl
    )

    # ── Summary ───────────────────────────────────────────────────
    acc_A = results["A_random"]["val_acc"]
    acc_B = results["B_clean"]["val_acc"]
    acc_C = results["C_full"]["val_acc"]
    diff  = round(acc_B - acc_A, 2)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Dataset':35s}  {'N':>7}  {'Val acc':>8}")
    print(f"  {'-'*55}")
    print(f"  {'A: Random subset (null baseline)':35s}  "
          f"{len(X_A):>7,}  {acc_A:>7.2f}%")
    print(f"  {'B: Clean subset  (pipeline)':35s}  "
          f"{len(X_B):>7,}  {acc_B:>7.2f}%  ← KEY")
    print(f"  {'C: Full original':35s}  "
          f"{len(X_C):>7,}  {acc_C:>7.2f}%")

    print(f"\n  KEY COMPARISON — B vs A (same size, linear classifier):")
    print(f"    A (random) : {acc_A:.2f}%")
    print(f"    B (clean)  : {acc_B:.2f}%")
    print(f"    Difference : {diff:+.2f}%")

    print(f"    Difference : {diff:+.2f} percentage points")
    #print(f"    Conclusion : {conclusion}")

    print(f"\n  Per-class (B vs A):")
    print(f"  {'class':12s}  {'A_random':>10}  {'B_clean':>10}  {'diff':>8}")
    print(f"  {'-'*47}")
    for cls in CLASSES:
        a  = results["A_random"]["per_class"].get(cls, 0)
        b  = results["B_clean"]["per_class"].get(cls, 0)
        d  = round(b - a, 2)
        fl = " ✅" if d > 0.5 else (" ❌" if d < -0.5 else "  —")
        print(f"  {cls:12s}  {a:>10.1f}%  {b:>10.1f}%  {d:>+7.2f}%{fl}")

    print(f"\n  Note: classifier = Logistic Regression on L2-normalised")
    print(f"  embeddings = equivalent to nn.Linear(2048,5) + softmax")
    print(f"  (frozen ResNet50 backbone, same as original experiment)")

    # ── Save ──────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "experiment_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Controlled experiment: cleaning vs random subset\n"
        f"Logistic Regression on ResNet50 embeddings "
        f"(≡ Linear FC head)  |  same n={n_clean:,}  |  same val set",
        fontsize=10
    )

    # Bar — overall
    names  = ["A: Random\n(null baseline)",
               "B: Clean\n(pipeline)",
               "C: Full\n(upper bound)"]
    accs   = [acc_A, acc_B, acc_C]
    colors = ["#B0BEC5", "#1565C0", "#2E7D32"]
    bars   = axes[0].bar(names, accs, color=colors, width=0.5)
    axes[0].set_ylim(min(accs) - 3, 101)
    axes[0].set_ylabel("Val accuracy (%)")
    axes[0].set_title("Overall val accuracy")
    axes[0].axhline(y=acc_A, color="#B0BEC5",
                    linestyle="--", alpha=0.5, label="baseline")
    for bar, acc in zip(bars, accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold"
        )
    # Annotate B vs A difference
    axes[0].annotate(
        f"Δ={diff:+.2f}%\nvs baseline",
        xy=(1, acc_B),
        xytext=(1.6, (acc_A + acc_B) / 2),
        arrowprops=dict(arrowstyle="->", color="#C62828"),
        fontsize=9, color="#C62828"
    )

    # Bar — per-class A vs B
    x     = np.arange(len(CLASSES))
    w     = 0.35
    a_acc = [results["A_random"]["per_class"].get(c, 0) for c in CLASSES]
    b_acc = [results["B_clean"]["per_class"].get(c, 0)  for c in CLASSES]
    axes[1].bar(x - w/2, a_acc, w, label="A: random", color="#B0BEC5")
    axes[1].bar(x + w/2, b_acc, w, label="B: clean",  color="#1565C0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES)
    axes[1].set_ylabel("Val accuracy (%)")
    axes[1].set_title("Per-class: A (random) vs B (clean)")
    axes[1].legend()
    axes[1].set_ylim(min(a_acc + b_acc) - 5, 101)

    plt.tight_layout()
    out_png = RESULTS_DIR / "experiment_results.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\n  Results → {out_json}")
    print(f"  Plot    → {out_png}")


if __name__ == "__main__":
    main()