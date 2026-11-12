# scripts/readme_numbers.py
# Standard library
import sys
from pathlib import Path

# Third-party
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
dataset, splits = sys.argv[1], sys.argv[2:]
rows = [pd.read_csv(ROOT / f"reports/{dataset}/{s}_cleaning_summary.csv").iloc[0]
        for s in splits]

def pct(n, total):
    return f"{100 * n / total:.1f}%"

print("| Split | Images | Clean | Review | Remove |\n|---|---:|---:|---:|---:|")
for r in rows:
    t = int(r.total_images)
    print(f"| {r.split} | {t:,} | {int(r.clean):,} ({pct(r.clean, t)}) "
          f"| {int(r.to_review):,} ({pct(r.to_review, t)}) "
          f"| {int(r.to_remove):,} ({pct(r.to_remove, t)}) |")
t = sum(int(r.total_images) for r in rows)
print(f"| **Total** | **{t:,}** | **{sum(int(r.clean) for r in rows):,}** | "
      f"**{sum(int(r.to_review) for r in rows):,}** | "
      f"**{sum(int(r.to_remove) for r in rows):,}** |")
print("\n| Split | Exact dup | Near dup | Blurry | Noisy | Outlier | Mislabeled |"
      "\n|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r.split} | {int(r.exact_dup)} | {int(r.near_dup)} | {int(r.blurry)} "
          f"| {int(r.noisy)} | {int(r.outlier)} | {int(r.mislabeled)} |")