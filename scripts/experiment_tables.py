# scripts/experiment_tables.py
"""5-seed comparison and per-module attribution for the README."""
# Standard library
import json
import warnings
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EMB, REP = ROOT / "embeddings/Animal", ROOT / "reports/Animal"
SEEDS = (42, 123, 999, 2024, 7)


def load(split):
    return (np.load(EMB / f"{split}_embeddings.npy"),
            np.load(EMB / f"{split}_labels.npy", allow_pickle=True),
            np.load(EMB / f"{split}_file_paths.npy", allow_pickle=True))


def accuracy(X, y, Xv, yv, seed):
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(normalize(X), y)
    return 100 * accuracy_score(yv, clf.predict(normalize(Xv)))


def col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n].astype(str).str.lower().eq("true")
    return pd.Series(False, index=df.index)


def main():
    X, y, paths = load("train")
    Xv, yv, _ = load("val")
    master = pd.read_csv(REP / "train_master_report.csv")
    master = master.set_index("file_path").reindex([str(p) for p in paths])
    assert master["verdict"].notna().all(), "report/embedding paths differ"

    clean = master["verdict"].eq("CLEAN").to_numpy()
    n = int(clean.sum())
    b = accuracy(X[clean], y[clean], Xv, yv, 42)

    rows = []
    for seed in SEEDS:
        idx = np.random.default_rng(seed).permutation(len(X))[:n]
        a = accuracy(X[idx], y[idx], Xv, yv, seed)
        rows.append({"seed": seed, "random": round(a, 2),
                     "clean": round(b, 2), "diff": round(b - a, 2)})
    diffs = [r["diff"] for r in rows]

    full = accuracy(X, y, Xv, yv, 42)
    removals = {
        "Duplicates": col(master, "Exact_duplicates") | col(master, "near_duplicate"),
        "High-confidence mislabels":
            col(master, "is_mislabeled") & master["mislabel_confidence"].eq("HIGH"),
        "Blur": col(master, "is_blurry"),
        "Outliers": col(master, "is_outlier"),
        "Full pipeline": pd.Series(~clean, index=master.index),
    }
    attribution = [{"removed": "None", "n": len(X), "acc": round(full, 2)}]
    for name, mask in removals.items():
        keep = ~mask.to_numpy()
        attribution.append({
            "removed": name, "n": int(keep.sum()),
            "acc": round(accuracy(X[keep], y[keep], Xv, yv, 42), 2)})

    keep = ~master["verdict"].str.startswith("REMOVE").to_numpy()
    m = int(keep.sum())
    default_acc = accuracy(X[keep], y[keep], Xv, yv, 42)
    random_accs = []
    for seed in SEEDS:
        idx = np.random.default_rng(seed).permutation(len(X))[:m]
        random_accs.append(accuracy(X[idx], y[idx], Xv, yv, seed))

    out = {"n_clean": n, "full": round(full, 2), "seeds": rows,
           "mean_random": round(float(np.mean([r["random"] for r in rows])), 2),
           "mean_diff": round(float(np.mean(diffs)), 2),
           "std_diff": round(float(np.std(diffs)), 2),
           "attribution": attribution,
           "default_workflow": {
           "n": m, "clean": round(default_acc, 2),
           "mean_random": round(float(np.mean(random_accs)), 2),
           "diff": round(default_acc - float(np.mean(random_accs)), 2)},}
    target = ROOT / "notebooks/experiment_results/readme_tables.json"
    target.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()