# notebooks/compute_validation_precision.py
import pandas as pd
from pathlib import Path

for split in ["train", "val"]:
    log = Path(f"notebooks/validation_outputs/{split}_validation_log.csv")
    if not log.exists():
        continue

    df = pd.read_csv(log)
    filled = df[df["human_verdict"].isin(["yes","no","UNSURE"])]

    if len(filled) == 0:
        print(f"{split}: no verdicts filled yet")
        continue

    true_pos  = (filled["human_verdict"] == "yes").sum()
    false_pos = (filled["human_verdict"] == "no").sum()
    unsure    = (filled["human_verdict"] == "UNSURE").sum()
    precision = true_pos / (true_pos + false_pos) if (true_pos+false_pos) > 0 else 0

    print(f"\n── {split.upper()} Mislabel Validation ──────────────")
    print(f"  Reviewed  : {len(filled)} / {len(df)}")
    print(f"  YES  (correctly flagged) : {true_pos}")
    print(f"  NO (false positives)   : {false_pos}")
    print(f"  UNSURE                    : {unsure}")
    print(f"  Precision                 : {precision:.1%}")

    # Per confusion pair
    print(f"\n  Confusion pairs reviewed:")
    pair_stats = filled.groupby(["declared","predicted"]).agg(
        count   = ("human_verdict","count"),
        correct = ("human_verdict", lambda x: (x=="yes").sum())
    ).reset_index()
    pair_stats["precision"] = (pair_stats["correct"] /
                                pair_stats["count"]).round(2)
    print(pair_stats.to_string(index=False))