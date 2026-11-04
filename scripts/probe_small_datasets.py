# scripts/probe_small_datasets.py
# Standard library
import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Local
from src import mislabel, outliers  # noqa: E402

DIM = 2048
CLASS_COUNTS = (2, 5)
PER_CLASS = (3, 5, 10, 20, 40, 63, 64, 65, 70, 130)


def write_embeddings(embed_dir: Path, n_classes: int, per_class: int) -> int:
    """Well-separated synthetic classes with no mislabels by construction."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 1, (n_classes, DIM))
    embs, labels, paths = [], [], []
    for c in range(n_classes):
        embs.append(centers[c] + rng.normal(0, 0.5, (per_class, DIM)))
        labels += [f"class{c}"] * per_class
        paths += [f"ds/train/class{c}/{i}.jpg" for i in range(per_class)]
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(embed_dir / "train_embeddings.npy",
            np.vstack(embs).astype(np.float32))
    np.save(embed_dir / "train_labels.npy", np.array(labels))
    np.save(embed_dir / "train_file_paths.npy", np.array(paths))
    return len(labels)


def run(fn, report: Path, column: str, **kwargs) -> str:
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), \
                contextlib.redirect_stderr(sink), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn("train", **kwargs)
        flagged = int(pd.read_csv(report)[column].astype(bool).sum())
        return f"ok ({flagged} flagged)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {str(exc)[:60]}"


def main() -> None:
    print(f"{'cls':>3} {'per':>4} {'total':>5}  {'outliers':<38} mislabels")
    for n_classes in CLASS_COUNTS:
        for per_class in PER_CLASS:
            with tempfile.TemporaryDirectory() as tmp:
                embed = Path(tmp) / "emb"
                reports = Path(tmp) / "rep"
                reports.mkdir()
                total = write_embeddings(embed, n_classes, per_class)
                paths = dict(embed_dir=str(embed), reports_dir=str(reports))
                out = run(outliers.detect_outliers,
                          reports / "train_outlier_report.csv",
                          "is_outlier", **paths)
                mis = run(mislabel.detect_mislabels,
                          reports / "train_mislabel_report.csv",
                          "is_mislabeled", **paths)
            print(f"{n_classes:>3} {per_class:>4} {total:>5}  {out:<38} {mis}")


if __name__ == "__main__":
    main()