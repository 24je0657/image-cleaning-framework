import pandas as pd
import numpy as np
import os

# ----- Configuration -----
REPORTS_DIR = "reports"
BLUR_REMOVE_THRESHOLD = 50
REMOVE_PRIORITY = 10
REVIEW_PRIORITY = 5

CLASSES = ["cat","dog","elephant","horse", "lion"]


# Priority weights - How severe each issue is
# Higher = More Urgent to remove

WEIGHTS ={
    "Exact_duplicates" : 10 , # Always Remove
    "near_duplicate" : 8 , # Almost Always Remove
    "is_blurry" : 6 , # remove if severe
    "is_outlier" : 5 , # remove if severe
    "is_noisy" : 4 , # review first
    "is_mislabeled" : 7 # High Priority - Corrupts Training
}

os.makedirs(REPORTS_DIR, exist_ok = True)

# ----- Loaders -----

def load_report(path, required_cols):

    """ Load a CSV Report safely - return None if missing"""
    if not os.path.exists(path):
        print(f" Missing file :{path}")
        return None
    
    df = pd.read_csv(path)
    missing = [
        c 
        for c in required_cols
        if c not in df.columns
    ]
    if missing :
        print(f" {path} missing columns : {missing}")
        return None
    
    return df

# ----- Main Merge Function -----

def build_master_report(split = "train"):

    print(f"\n{'='*30}")
    print(f"DECISION ENGINE -- {split.upper()} SET")
    print(f"{'='*30}")

    dup_df   = load_report(
        f"{REPORTS_DIR}/{split}_duplicates_report.csv",
        ["file_path", "Exact_duplicates", "near_duplicate"]
    )
    blur_df  = load_report(
        f"{REPORTS_DIR}/{split}_blur_report.csv",
        ["file_path", "is_blurry", "blur_score"]
    )
    noise_df = load_report(
        f"{REPORTS_DIR}/{split}_noise_report.csv",
        ["file_path", "is_noisy", "reconstruction_error"]
    )
    out_df   = load_report(
        f"{REPORTS_DIR}/{split}_outlier_report.csv",
        ["file_path", "is_outlier"]
    )
    mis_df   = load_report(
        f"{REPORTS_DIR}/{split}_mislabel_report.csv",
        ["file_path", "is_mislabeled", "confidence", "declared_class"]
    )
    # Use blur report as base .. since contains every image
    master = blur_df[["file_path","class", "blur_score", "is_blurry"]].copy()

    # ----- Merge Duplicates -----
    if dup_df is not None :
        master = master.merge(
            dup_df[["file_path", "Exact_duplicates",
                    "near_duplicate", "duplicate_of",
                    "cosine_score"]],
            on="file_path", how="left"
        )
    else:
        master["Exact_duplicates"] = False
        master["near_duplicate"]  = False
        master["duplicate_of"]    = np.nan
        master["cosine_score"]    = np.nan

    master["Exact_duplicates"] = master["Exact_duplicates"].fillna(False)
    master["near_duplicate"]  = master["near_duplicate"].fillna(False)


    # ----- Merge Noise -----
    if noise_df is not None:
        master = master.merge(
            noise_df[["file_path", "is_noisy", "reconstruction_error"]],
            on="file_path", how="left"
        )
    else:
        master["is_noisy"]              = False
        master["reconstruction_error"]  = np.nan

    master["is_noisy"] = master["is_noisy"].fillna(False)

    # ----- Merge Outliers -----
    if out_df is not None:
        master = master.merge(
            out_df[["file_path", "is_outlier",
                    "global_score", "per_class_outlier"]],
            on="file_path", how="left"
        )
    else:
        master["is_outlier"]        = False
        master["global_score"]      = np.nan
        master["per_class_outlier"] = False

    master["is_outlier"] = master["is_outlier"].fillna(False)

    # ----- Merge Mislabels -----
    if mis_df is not None:
        master = master.merge(
            mis_df[["file_path", "is_mislabeled", "confidence",
                    "nn_predicted", "votes"]],
            on="file_path", how="left"
        )
        master = master.rename(columns={
            "confidence" : "mislabel_confidence",
            "votes"      : "mislabel_votes"
        })
    else:
        master["is_mislabeled"]       = False
        master["mislabel_confidence"] = "CLEAN"
        master["nn_predicted"]        = np.nan
        master["mislabel_votes"]      = 0

    master["is_mislabeled"] = master["is_mislabeled"].fillna(False)

    # Priority Score : Weighted Sum of all flags - higher = More Urgently removed

    master["priority_score"] = (
        master["Exact_duplicates"].astype(int) * WEIGHTS["Exact_duplicates"] +
        master["near_duplicate"].astype(int)  * WEIGHTS["near_duplicate"]  +
        master["is_blurry"].astype(int)       * WEIGHTS["is_blurry"]       +
        master["is_noisy"].astype(int)        * WEIGHTS["is_noisy"]        +
        master["is_outlier"].astype(int)      * WEIGHTS["is_outlier"]      +
        master["is_mislabeled"].astype(int)   * WEIGHTS["is_mislabeled"]
    )

    # ----- Final verdict -----
    
    def assign_verdict(row):
      if row["priority_score"] == 0:
        return "CLEAN"
      elif row["Exact_duplicates"] or row["near_duplicate"]:
        return "REMOVE — duplicate"
      elif row["is_mislabeled"] and row["mislabel_confidence"] == "HIGH":
        return "REMOVE — mislabeled"
      elif row["is_blurry"] and row["blur_score"] < BLUR_REMOVE_THRESHOLD:
        return "REMOVE — blurry"
      elif row["priority_score"] >= REMOVE_PRIORITY:
        return "REMOVE — high priority"
      elif row["priority_score"] >= REVIEW_PRIORITY:
        return "REVIEW — medium priority"
      else:
        return "REVIEW — low priority"

    master["verdict"] = master.apply(assign_verdict, axis=1)

    # ----- FLag Count Per Image -----
    flag_cols = ["Exact_duplicates", "near_duplicate",
                 "is_blurry", "is_noisy",
                 "is_outlier", "is_mislabeled"]
    master["total_flags"] = master[flag_cols].astype(int).sum(axis=1)
    
    # ----- Print Summary -----

    print(f"\n OVERALL SUMMARY -----------------------")
    print(f"  Total images     : {len(master)}")
    print(f"  Clean images     : {(master['verdict'] == 'CLEAN').sum()}")
    print(f"  To REMOVE        : {master['verdict'].str.startswith('REMOVE').sum()}")
    print(f"  To REVIEW        : {master['verdict'].str.startswith('REVIEW').sum()}")

    print(f"\n ISSUE BREAKDOWN -------------------------")
    for col, weight in WEIGHTS.items():
        if col in master.columns:
            count = master[col].astype(bool).sum()
            pct   = 100 * count / len(master)
            print(f"  {col:20s}: {count:5d}  ({pct:.2f}%)")
    
    print(f"\n Verdict BreakDown --------------------------")
    verdict_counts = master["verdict"].value_counts()
    for v, c in verdict_counts.items():
        print(f"  {v:35s}: {c}")

    print(f"\n Multi-Flag Images ----------------------------")
    for n in range(2, 7):
        count = (master["total_flags"] == n).sum()
        if count > 0:
            print(f"  {n} flags: {count} images")
    
    print(f"\n TOP 10 HIGHEST PRIORITY --------------------------")
    top = master[master["verdict"] != "CLEAN"].nlargest(10, "priority_score")
    print(top[["file_path", "class", "verdict",
               "priority_score", "total_flags"]].to_string(index=False))
    
    # ── 11. Per-class summary
    print(f"\n PER CLASS SUMMARY ------------------------")
    for cls in CLASSES:
      cls_df = master[master["class"] == cls]
      clean = (cls_df["verdict"] == "CLEAN").sum()
      remove = cls_df["verdict"].str.startswith("REMOVE").sum()
      review = cls_df["verdict"].str.startswith("REVIEW").sum()
      total = len(cls_df)

      print(f"  [{cls:10s}] total={total}  "
            f"clean={clean}  remove={remove}  review={review}")
    
    # ----- Save Master Report -----
    master_path = f"{REPORTS_DIR}/{split}_master_report.csv"
    master.to_csv(master_path, index=False)
    print(f" Master report saved ==> {master_path}")

    # ── 13. Save cleaning summary
    summary = {
        "split"           : split,
        "total_images"    : len(master),
        "clean"           : (master["verdict"] == "CLEAN").sum(),
        "to_remove"       : master["verdict"].str.startswith("REMOVE").sum(),
        "to_review"       : master["verdict"].str.startswith("REVIEW").sum(),
        "exact_dup"       : master["Exact_duplicates"].astype(bool).sum(),
        "near_dup"        : master["near_duplicate"].astype(bool).sum(),
        "blurry"          : master["is_blurry"].astype(bool).sum(),
        "noisy"           : master["is_noisy"].astype(bool).sum(),
        "outlier"         : master["is_outlier"].astype(bool).sum(),
        "mislabeled"      : master["is_mislabeled"].astype(bool).sum(),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = f"{REPORTS_DIR}/{split}_cleaning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f" Summary saved  ==>  {summary_path}")

    return master

if __name__ == "__main__" :
   for split in ["train", "val"]:
    df = build_master_report(split)

     




        





    

