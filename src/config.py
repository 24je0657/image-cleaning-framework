# src/config.py
"""
Central configuration for the Image Cleaning Framework.
All modules import from here — no more hardcoded paths.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class PipelineConfig:
    # ── Paths ─────────────────────────────────────────────────────
    data_dir        : Path          # root of input dataset
    output_dir      : Path          # where clean images are written
    embeddings_dir  : Path = Path("embeddings")
    reports_dir     : Path = Path("reports")
    models_dir      : Path = Path("models")

    # ── Dataset ───────────────────────────────────────────────────
    splits          : List[str] = field(default_factory=list)
    classes         : List[str] = field(default_factory=list)

    # ── Module flags ──────────────────────────────────────────────
    run_duplicates  : bool  = True
    run_blur        : bool  = True
    run_noise       : bool  = True
    run_outliers    : bool  = True
    run_mislabels   : bool  = True

    # ── Thresholds ────────────────────────────────────────────────
    blur_percentile     : int   = 5        # p5 per class
    blur_contamination  : float = 0.30     # 30% fallback guard
    cosine_threshold    : float = 0.97
    hash_threshold      : int   = 4
    contamination       : float = 0.05
    noise_std_mult      : float = 2.0      # mean + N*std per class

    # ── Embedding ─────────────────────────────────────────────────
    embedding_dim       : int   = 2048
    pca_dims            : int   = 128
    batch_size          : int   = 32
    img_size            : int   = 224

    def __post_init__(self):
        self.data_dir   = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self._auto_discover()

    def _auto_discover(self):
        """
        Auto-discover splits and classes from dataset folder structure.

        Supports two layouts:
          Layout A (split-based):        Layout B (flat):
            data_dir/                      data_dir/
            ├── train/                     ├── classA/
            │   ├── classA/                ├── classB/
            │   └── classB/                └── classC/
            └── val/
                └── classA/

        Layout A → splits = ["train", "val"], classes from train/
        Layout B → splits = [""],             classes from data_dir/
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        # Detect layout: does data_dir contain class folders directly
        # or split folders (train/val/test)?
        KNOWN_SPLITS = {"train", "val", "test",
                        "training", "validation"}
        split_dirs   = [d for d in subdirs
                        if d.name.lower() in KNOWN_SPLITS]

        if split_dirs:
            # Layout A — split-based
            if not self.splits:
                self.splits = sorted([d.name for d in split_dirs])
            if not self.classes:
                # Discover classes from first split
                first_split = self.data_dir / self.splits[0]
                self.classes = sorted([
                    d.name for d in first_split.iterdir()
                    if d.is_dir()
                ])
        else:
            # Layout B — flat (classes directly in data_dir)
            if not self.splits:
                self.splits = [""]     # empty string = no split subdir
            if not self.classes:
                self.classes = sorted([
                    d.name for d in subdirs
                ])

    def split_path(self, split: str) -> Path:
        """Return full path to a split directory."""
        return self.data_dir / split if split else self.data_dir

    def output_split_path(self, split: str) -> Path:
        """Return full path to output split directory."""
        return self.output_dir / split if split else self.output_dir

    def embedding_path(self, split: str, kind: str) -> Path:
        """e.g. embeddings/train_embeddings.npy"""
        prefix = f"{split}_" if split else "flat_"
        return self.embeddings_dir / f"{prefix}{kind}.npy"

    def report_path(self, split: str, name: str) -> Path:
        """e.g. reports/train_blur_report.csv"""
        prefix = f"{split}_" if split else "flat_"
        return self.reports_dir / f"{prefix}{name}.csv"

    def summary(self) -> str:
        lines = [
            "── Dataset Configuration ──────────────────────────",
            f"  Data dir      : {self.data_dir}",
            f"  Output dir    : {self.output_dir}",
            f"  Splits        : {self.splits}",
            f"  Classes       : {self.classes}",
            f"  Num classes   : {len(self.classes)}",
        ]
        for split in self.splits:
            sp = self.split_path(split)
            total = sum(
                len(list((sp / cls).glob("*")))
                for cls in self.classes
                if (sp / cls).exists()
            )
            label = split if split else "all"
            lines.append(f"  {label:12s}  : {total:,} images")
        return "\n".join(lines)

    def to_json(self, path: Path):
        """Save config to JSON for reproducibility."""
        d = {
            "data_dir"       : str(self.data_dir),
            "output_dir"     : str(self.output_dir),
            "splits"         : self.splits,
            "classes"        : self.classes,
            "run_duplicates" : self.run_duplicates,
            "run_blur"       : self.run_blur,
            "run_noise"      : self.run_noise,
            "run_outliers"   : self.run_outliers,
            "run_mislabels"  : self.run_mislabels,
            "blur_percentile": self.blur_percentile,
            "cosine_threshold": self.cosine_threshold,
            "contamination"  : self.contamination,
        }
        path.write_text(json.dumps(d, indent=2))
        print(f"  Config saved → {path}")


def load_config(path: Path) -> PipelineConfig:
    """Load config from JSON — reproduce an earlier run exactly."""
    d = json.loads(path.read_text())
    return PipelineConfig(**d)