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
        ["file_path", "Exact_duplicates", "near_duplicate","dup_confidence"]
    )
    blur_df  = load_report(
        f"{REPORTS_DIR}/{split}_blur_report.csv",
        ["file_path","class", "blur_score", "threshold_used", "is_blurry"]
    )
    noise_df = load_report(
        f"{REPORTS_DIR}/{split}_noise_report.csv",
        ["file_path", "is_noisy", "reconstruction_error","noise_verdict"]
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
    master = blur_df[["file_path","class", "blur_score", "threshold_used", "is_blurry"]].copy()

    # ----- Merge Duplicates -----
    if dup_df is not None :
        master = master.merge(
            dup_df[["file_path", "Exact_duplicates",
                    "near_duplicate", "duplicate_of",
                    "cosine_score","dup_confidence"]],
            on="file_path", how="left"
        )
    else:
        master["Exact_duplicates"] = False
        master["near_duplicate"]  = False
        master["duplicate_of"]    = np.nan
        master["cosine_score"]    = np.nan
        master["dup_confidence"]  = ""

    master["Exact_duplicates"] = master["Exact_duplicates"].fillna(False)
    master["near_duplicate"]  = master["near_duplicate"].fillna(False)
    master["dup_confidence"] = (
        master["dup_confidence"].fillna("")
    )


    # ----- Merge Noise -----
    if noise_df is not None:
        master = master.merge(
            noise_df[["file_path", "is_noisy", "reconstruction_error","noise_verdict"]],
            on="file_path", how="left"
        )
    else:
        master["is_noisy"]              = False
        master["reconstruction_error"]  = np.nan
        master["noise_verdict"] = "not_flagged"

    master["is_noisy"] = master["is_noisy"].fillna(False)
    master["noise_verdict"] = (
        master["noise_verdict"].fillna("not_flagged")
    )

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

    # ----- Priority Score -----

    def compute_priority_score(row: pd.Series) -> int:
        """
        Confidence-adjusted priority scoring.

        Exact duplicate       → 10
        Near-dup DEFINITE     → 10
        Near-dup LIKELY       → 7
        Near-dup POSSIBLE     → 4
        Mislabel HIGH         → 9
        Mislabel MEDIUM       → 6
        Mislabel LOW          → 3
        Blurry severe (<20)   → 8
        Blurry moderate       → 5
        Confirmed noise       → 6
        Ambiguous noise/OOD   → 3
        Downgraded OOD        → 0
        Outlier               → 4
        """

        score = 0

        # ----- Exact duplicate -----
        if row.get("Exact_duplicates", False):
            score += 10

        # ----- Near duplicate -----
        if row.get("near_duplicate", False):

            dup_conf = row.get("dup_confidence", "LIKELY")

            score += {
                "DEFINITE": 10,
                "LIKELY": 7,
                "POSSIBLE": 4
            }.get(dup_conf, 7)

        # ----- Mislabel -----
        if row.get("is_mislabeled", False):

            mis_conf = row.get(
                "mislabel_confidence",
                "MEDIUM"
            )

            score += {
                "HIGH": 9,
                "MEDIUM": 6,
                "LOW": 3
            }.get(mis_conf, 6)

        # ----- Blur -----
        if row.get("is_blurry", False):

            blur = row.get("blur_score", 50)

            if blur < 20:
                score += 8
            else:
                score += 5

        # ----- Noise -----
        if row.get("is_noisy", False):

            noise_verdict = row.get(
                "noise_verdict",
                ""
            )

            if noise_verdict == "confirmed_noisy":
                score += 6

            elif noise_verdict == "review_ood_or_noise":
                score += 3

            elif noise_verdict == "review_noise":
                score += 3

            elif noise_verdict == "review_possible_ood":
                score += 0

            elif noise_verdict == "downgraded_ood":
                score += 0

            else:
                # Conservative fallback
                score += 3

        # ----- Outlier -----
        if row.get("is_outlier", False):
            score += 4

        return score


    # Apply confidence-adjusted priority score
    master["priority_score"] = master.apply(
        compute_priority_score,
        axis=1
    )


    # ----- Final Verdict -----

    def assign_verdict(row):
        # Downgraded OOD is not clean — it needs review
        if row.get("noise_verdict") == "downgraded_ood":
            return "REVIEW — possible OOD"

        # No detected issue
        if row["priority_score"] == 0:
            return "CLEAN"

        # Exact duplicate
        elif row["Exact_duplicates"]:
            return "REMOVE — duplicate"

        # Definite near duplicate
        elif (
            row["near_duplicate"]
            and row.get("dup_confidence") == "DEFINITE"
        ):
            return "REMOVE — near duplicate"

        # High-confidence mislabel
        elif (
            row["is_mislabeled"]
            and row["mislabel_confidence"] == "HIGH"
        ):
            return "REMOVE — mislabeled"

        # Severe blur
        elif (
            row["is_blurry"]
            and row["blur_score"] < BLUR_REMOVE_THRESHOLD
        ):
            return "REMOVE — blurry"

        # Confirmed noise
        elif row.get("noise_verdict") == "confirmed_noisy":
            return "REMOVE — noisy"

        # High overall priority
        elif row["priority_score"] >= REMOVE_PRIORITY:
            return "REMOVE — high priority"

        # Ambiguous noise / OOD
        elif row.get("noise_verdict") == "review_ood_or_noise":
            return "REVIEW — OOD or noise"

        # Possible noise
        elif row.get("noise_verdict") == "review_noise":
            return "REVIEW — possible noise"

        # Possible OOD
        elif row.get("noise_verdict") in [
            "review_possible_ood",
            "downgraded_ood"
        ]:
            return "REVIEW — possible OOD"

        # Medium priority
        elif row["priority_score"] >= REVIEW_PRIORITY:
            return "REVIEW — medium priority"

        # Low priority
        else:
            return "REVIEW — low priority"


    master["verdict"] = master.apply(
        assign_verdict,
        axis=1
    )
    # ----- Flag Count Per Image -----

    flag_cols = [
        "Exact_duplicates",
        "near_duplicate",
        "is_blurry",
        "is_noisy",
        "is_outlier",
        "is_mislabeled"
    ]

    master["total_flags"] = (
        master[flag_cols]
        .astype(bool)
        .sum(axis=1)
    )


    # ----- Print Summary -----

    print(f"\nOVERALL SUMMARY -----------------------")
    print(f"  Total images     : {len(master)}")
    print(f"  Clean images     : {(master['verdict'] == 'CLEAN').sum()}")
    print(f"  To REMOVE        : {master['verdict'].str.startswith('REMOVE').sum()}")
    print(f"  To REVIEW        : {master['verdict'].str.startswith('REVIEW').sum()}")


    print(f"\nISSUE BREAKDOWN -------------------------")

    for col, weight in WEIGHTS.items():

        if col in master.columns:
 
            count = master[col].astype(bool).sum()
            pct = 100 * count / len(master)
  
            print(
                f"  {col:20s}: "
                f"{count:5d} ({pct:.2f}%)"
            )


    print(f"\nVERDICT BREAKDOWN --------------------------")

    verdict_counts = master["verdict"].value_counts()

    for v, c in verdict_counts.items():

        print(
            f"  {v:35s}: {c}"
        )


    print(f"\nMULTI-FLAG IMAGES ----------------------------")

    for n in range(2, 7):

        count = (master["total_flags"] == n).sum()

        if count > 0:

            print(
                f"  {n} flags: {count} images"
            )


    print(f"\nTOP 10 HIGHEST PRIORITY --------------------------")

    top = (
        master[
            master["verdict"] != "CLEAN"
        ]
    .    nlargest(10, "priority_score")
    )

    print(
        top[
            [
                "file_path",
                "class",
                "verdict",
                "priority_score",
                "total_flags"
            ]
        ].to_string(index=False)
    )


    # ----- Per Class Summary -----

    print(f"\nPER CLASS SUMMARY ------------------------")

    for cls in CLASSES:

        cls_df = master[
            master["class"] == cls
        ]

        clean = (
            cls_df["verdict"] == "CLEAN"
        ).sum()

        remove = (
            cls_df["verdict"]
            .str.startswith("REMOVE")
            .sum()
        )

        review = (
            cls_df["verdict"]
            .str.startswith("REVIEW")
            .sum()
        )

        total = len(cls_df)

        print(
            f"  [{cls:10s}] "
            f"total={total} "
            f"clean={clean} "
            f"remove={remove} "
            f"review={review}"
        )


    # ----- Save Master Report -----

    master_path = (
        f"{REPORTS_DIR}/{split}_master_report.csv"
    )

    master.to_csv(
        master_path,
        index=False
    )

    print(
        f"\nMaster report saved ==> {master_path}"
    )


    # ----- Save Cleaning Summary -----

    summary = {

        "split": split,

        "total_images": len(master),

        "clean": (
            master["verdict"] == "CLEAN"
        ).sum(),

        "to_remove": (
            master["verdict"]
            .str.startswith("REMOVE")
            .sum()
        ),

        "to_review": (
            master["verdict"]
            .str.startswith("REVIEW")
            .sum()
        ),

        "exact_dup": (
            master["Exact_duplicates"]
            .astype(bool)
            .sum()
        ),

        "near_dup": (
            master["near_duplicate"]
            .astype(bool)
            .sum()
        ),

        "blurry": (
            master["is_blurry"]
            .astype(bool)
            .sum()
        ),

        "noisy": (
            master["is_noisy"]
            .astype(bool)
            .sum()
        ),

        "outlier": (
            master["is_outlier"]
            .astype(bool)
            .sum()
        ),

        "mislabeled": (
            master["is_mislabeled"]
            .astype(bool)
            .sum()
        )
    }


    summary_df = pd.DataFrame([summary])

    summary_path = (
        f"{REPORTS_DIR}/{split}_cleaning_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Summary saved ==> {summary_path}"
    )

    return master

if __name__ == "__main__" :
   for split in ["train", "val"]:
       df = build_master_report(split)

     




        





    

