"""
Decision engine service — merges module results into final verdict.
"""
import time
import numpy  as np
import pandas as pd
from pathlib import Path

WEIGHTS = {
    "exact_duplicate" : 10,
    "near_duplicate"  : 8,
    "is_blurry"       : 6,
    "is_noisy"        : 5,
    "is_outlier"      : 4,
    "is_mislabeled"   : 7,
}


class DecisionEngineService:

    def build_verdict(self, row: dict) -> tuple[str, int]:
        score = sum(
            WEIGHTS[k] for k in WEIGHTS
            if row.get(k, False)
        )
        if score == 0:
            return "CLEAN", score
        if row.get("exact_duplicate") or row.get("near_duplicate"):
            return "REMOVE — duplicate", score
        if row.get("is_mislabeled") and row.get("mislabel_confidence") == "HIGH":
            return "REMOVE — mislabeled", score
        if row.get("is_blurry") and row.get("blur_score", 999) < 50:
            return "REMOVE — blurry", score
        if score >= 10:
            return "REMOVE — high priority", score
        if score >= 5:
            return "REVIEW — medium priority", score
        return "REVIEW — low priority", score

    def merge_reports(self, split: str) -> dict:
        t0      = time.time()
        base    = f"reports/{split}"

        # Load all available module reports
        def safe_read(path, cols):
            p = Path(path)
            if not p.exists():
                return None
            df = pd.read_csv(p)
            return df if all(c in df.columns for c in cols) else None

        blur_df  = safe_read(f"{base}_blur_report.csv",
                             ["file_path", "is_blurry", "blur_score"])
        if blur_df is None:
            return {"error": "blur report not found — run pipeline first"}

        master = blur_df[["file_path","class","blur_score","is_blurry"]].copy()

        # Merge each module
        merges = [
            (f"{base}_duplicates_report.csv",
             ["file_path","exact_duplicate","near_duplicate",
              "duplicate_of","cosine_score"]),
            (f"{base}_noise_report.csv",
             ["file_path","is_noisy","reconstruction_error"]),
            (f"{base}_outliers_report.csv",
             ["file_path","is_outlier","global_score"]),
            (f"{base}_mislabel_report.csv",
             ["file_path","is_mislabeled","confidence","nn_predicted"]),
        ]
        for path, cols in merges:
            df = safe_read(path, cols[:2])
            if df is not None:
                master = master.merge(df[cols], on="file_path", how="left")

        # Fill NaN flags
        flag_cols = ["exact_duplicate","near_duplicate",
                     "is_noisy","is_outlier","is_mislabeled"]
        for col in flag_cols:
            if col in master.columns:
                master[col] = master[col].fillna(False)

        # Apply verdict
        verdicts = master.apply(
            lambda r: self.build_verdict(r.to_dict()), axis=1
        )
        master["verdict"]        = verdicts.apply(lambda x: x[0])
        master["priority_score"] = verdicts.apply(lambda x: x[1])
        master["total_flags"]    = master[
            [c for c in flag_cols if c in master.columns]
        ].astype(int).sum(axis=1)

        master.to_csv(f"{base}_master_report.csv", index=False)

        total  = len(master)
        clean  = int((master["verdict"] == "CLEAN").sum())
        remove = int(master["verdict"].str.startswith("REMOVE").sum())
        review = int(master["verdict"].str.startswith("REVIEW").sum())

        return {
            "split"               : split,
            "total_images"        : total,
            "clean"               : clean,
            "to_remove"           : remove,
            "to_review"           : review,
            "efficiency_pct"      : round(100 * clean / total, 2),
            "verdict_distribution": master["verdict"].value_counts().to_dict(),
            "duration_s"          : round(time.time() - t0, 2),
            "report_path"         : f"reports/{split}_master_report.csv"
        }