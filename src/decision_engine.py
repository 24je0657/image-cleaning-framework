# Standard library
import os
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd

# ----- Configuration -----
REPORTS_DIR = "reports"
BLUR_REMOVE_THRESHOLD = 50
REMOVE_PRIORITY = 10
REVIEW_PRIORITY = 5




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
DETECTOR_NAMES = (
    "duplicates",
    "blur",
    "noise",
    "outliers",
    "mislabels",
)


def _resolve_enabled_modules(enabled_modules):
    """None means all detectors are required; otherwise validate the names."""

    if enabled_modules is None:
        return frozenset(DETECTOR_NAMES)

    if isinstance(enabled_modules, str):
        raise TypeError(
            "enabled_modules must be a list of names, not a string."
        )

    enabled = frozenset(enabled_modules)

    unknown = enabled - set(DETECTOR_NAMES)

    if unknown:
        raise ValueError(
            f"Unknown detector names: {sorted(unknown)}"
        )

    if not enabled:
        raise ValueError("At least one detector must be enabled.")

    return enabled

# ----- Loaders -----

def load_report(path, required_cols):
    """Load a detector CSV safely; return None if unusable."""

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        print(f"Unable to read report {path}: {exc}")
        return None

    missing = [
        column
        for column in required_cols
        if column not in df.columns
    ]

    if missing:
        print(f"{path} missing columns: {missing}")
        return None

    if df.empty:
        print(f"Report contains no rows: {path}")
        return None

    if df["file_path"].isna().all():
        print(f"Report contains no usable file paths: {path}")
        return None

    return df


# ----- Main Merge Function -----
def build_master_report(
    split="train",
    reports_dir=REPORTS_DIR,
    embed_dir=None,
    enabled_modules=None,
):
    """
    Merge available detector reports into one master report.

    Image inventory priority:
      1. Embedding file_paths.npy
      2. Union of available detector reports
      3. Raise an error if no inventory is available
    """
    print(f"\n{'=' * 30}")
    print(f"DECISION ENGINE -- {split.upper()} SET")
    print(f"{'=' * 30}")

    enabled = _resolve_enabled_modules(enabled_modules)

    required_ran_columns = [
        f"_{name}_ran"
        for name in DETECTOR_NAMES
        if name in enabled
    ]

    print(f"  Detectors required for CLEAN: {sorted(enabled)}")

    # ----- Load detector reports -----
    dup_df = load_report(
        f"{reports_dir}/{split}_duplicates_report.csv",
        ["file_path", "Exact_duplicates", "near_duplicate", "dup_confidence"],
    )

    blur_df = load_report(
        f"{reports_dir}/{split}_blur_report.csv",
        ["file_path", "class", "blur_score", "threshold_used", "is_blurry"],
    )

    noise_df = load_report(
        f"{reports_dir}/{split}_noise_report.csv",
        [
            "file_path",
            "is_noisy",
            "is_noisy_refined",
            "reconstruction_error",
            "noise_verdict",
        ],
    )

    out_df = load_report(
        f"{reports_dir}/{split}_outlier_report.csv",
        ["file_path", "is_outlier"],
    )

    mis_df = load_report(
        f"{reports_dir}/{split}_mislabel_report.csv",
        ["file_path", "is_mislabeled", "confidence", "declared_class"],
    )

    # ----- Record which detector reports are available -----
    modules_available = {
        "duplicates": dup_df is not None,
        "blur": blur_df is not None,
        "noise": noise_df is not None,
        "outliers": out_df is not None,
        "mislabels": mis_df is not None,
    }

    print(
        "  Available detector reports:",
        [name for name, available in modules_available.items() if available],
    )

    unavailable = [
        name for name, available in modules_available.items() if not available
    ]

    if unavailable:
        print("  Missing detector reports:", unavailable)

    # ----- Build image inventory -----
    all_paths = []
    all_classes = {}

    if embed_dir is not None:
        paths_file = Path(embed_dir) / f"{split}_file_paths.npy"
        labels_file = Path(embed_dir) / f"{split}_labels.npy"

        if paths_file.exists():
            paths = np.load(paths_file, allow_pickle=True).tolist()

            if len(paths) != len(set(paths)):
                raise ValueError(
                    f"Duplicate image paths found in {paths_file}"
                )

            all_paths = paths

            if labels_file.exists():
                labels = np.load(
                    labels_file,
                    allow_pickle=True,
                ).tolist()

                if len(paths) != len(labels):
                    raise ValueError(
                        f"Path/label count mismatch: "
                        f"{len(paths)} paths and {len(labels)} labels "
                        f"in {embed_dir}"
                    )

                all_classes = dict(zip(paths, labels))

            print(
                f"  Image inventory: {len(all_paths)} paths "
                f"(embedding inventory)"
            )

    # Fallback: union of paths from available detector reports.
    if not all_paths:
        path_set = set()

        for report_df in (
            dup_df,
            blur_df,
            noise_df,
            out_df,
            mis_df,
        ):
            if report_df is None:
                continue

            path_set.update(report_df["file_path"].dropna().tolist())

            if "class" in report_df.columns:
                all_classes.update(
                    dict(
                        zip(
                            report_df["file_path"],
                            report_df["class"],
                        )
                    )
                )

        all_paths = sorted(path_set)

        if all_paths:
            print(
                f"  Image inventory: {len(all_paths)} paths "
                "(report union; inventory may be incomplete)"
            )
        else:
            raise RuntimeError(
                "Cannot build master report: no image inventory "
                "or usable detector report paths were found."
            )
    # ----- Create the master DataFrame -----
    # Derive class names from the parent directory containing each image.
    # Example: data/raw/Animal/train/cat/cat1.jpg -> cat

    def get_class_from_path(file_path):
        path_parts = Path(str(file_path)).parts

        if len(path_parts) >= 2:
            return path_parts[-2]

        return ""

    master = pd.DataFrame(
        {
            "file_path": all_paths,
            "class": [
                get_class_from_path(path) or all_classes.get(path, "")
                for path in all_paths
            ],
        }
    )

    # ----- Merge detector reports -----
    # Missing detector results remain unknown (NaN), not False.
    # Internal _ran columns record whether each detector evaluated
    # the particular image.

    detector_reports = {
        "duplicates": dup_df,
        "blur": blur_df,
        "noise": noise_df,
        "outliers": out_df,
        "mislabels": mis_df,
    }

    for detector_name, report_df in detector_reports.items():
        master[f"_{detector_name}_ran"] = False

        if report_df is not None:
            if detector_name == "mislabels":
                # An image counts as evaluated only if DBSCAN actually ran.
                if "dbscan_checked" in report_df.columns:
                    checked = report_df["dbscan_checked"].fillna(False).astype(bool)

                    checked_paths = set(
                        report_df.loc[checked, "file_path"].dropna().tolist()
                    )

                    master["_mislabels_ran"] = (
                        master["file_path"].isin(checked_paths)
                    )
            else:
                reported_paths = set(
                    report_df["file_path"].dropna().tolist()
                )

                master[f"_{detector_name}_ran"] = (
                    master["file_path"].isin(reported_paths)
                )

    # Duplicates
    if dup_df is not None:
        dup_cols = [
            "file_path",
            "Exact_duplicates",
            "near_duplicate",
            "dup_confidence",
        ]

        optional_dup_cols = [
            "duplicate_of",
            "cosine_score",
            "near_duplicate_of",
        ]

        dup_cols.extend(
            col for col in optional_dup_cols
            if col in dup_df.columns
        )

        master = master.merge(
            dup_df[dup_cols].drop_duplicates("file_path"),
            on="file_path",
            how="left",
            validate="one_to_one",
        )

    # Blur
    if blur_df is not None:
        blur_cols = [
            "file_path",
            "class",
            "blur_score",
            "threshold_used",
            "is_blurry",
        ]

        blur_merge = blur_df[blur_cols].drop_duplicates("file_path")

        master = master.merge(
            blur_merge,
            on="file_path",
            how="left",
            suffixes=("", "_blur"),
            validate="one_to_one",
        )

        if "class_blur" in master.columns:
            master["class"] = master["class"].where(
                master["class"].astype(str).str.len() > 0,
                master["class_blur"],
            )
            master.drop(columns=["class_blur"], inplace=True)

    # Noise
    if noise_df is not None:
        noise_cols = [
            "file_path",
            "is_noisy",
            "is_noisy_refined",
            "reconstruction_error",
            "noise_verdict",
        ]

        master = master.merge(
            noise_df[noise_cols].drop_duplicates("file_path"),
            on="file_path",
            how="left",
            validate="one_to_one",
        )

    # Outliers
    if out_df is not None:
        out_cols = ["file_path", "is_outlier"]

        optional_out_cols = [
            "global_score",
            "per_class_outlier",
        ]

        out_cols.extend(
            col for col in optional_out_cols
            if col in out_df.columns
        )

        master = master.merge(
            out_df[out_cols].drop_duplicates("file_path"),
            on="file_path",
            how="left",
            validate="one_to_one",
        )

    # Mislabels
    if mis_df is not None:
        mis_cols = [
            "file_path",
            "is_mislabeled",
            "confidence",
            "declared_class",
        ]

        optional_mis_cols = [
            "nn_predicted",
            "votes",
        ]

        mis_cols.extend(
            col for col in optional_mis_cols
            if col in mis_df.columns
        )

        master = master.merge(
            mis_df[mis_cols].drop_duplicates("file_path"),
            on="file_path",
            how="left",
            validate="one_to_one",
        )

        master.rename(
            columns={
                "confidence": "mislabel_confidence",
                "votes": "mislabel_votes",
            },
            inplace=True,
        )

    # Ensure downstream scoring and reporting columns exist.
    # Keep unknown detector flags as NaN until verdict assignment.
    required_detector_columns = {
        "Exact_duplicates": np.nan,
        "near_duplicate": np.nan,
        "dup_confidence": np.nan,
        "is_blurry": np.nan,
        "blur_score": np.nan,
        "threshold_used": np.nan,
        "is_noisy": np.nan,
        "is_noisy_refined": np.nan,
        "reconstruction_error": np.nan,
        "noise_verdict": np.nan,
        "is_outlier": np.nan,
        "global_score": np.nan,
        "per_class_outlier": np.nan,
        "is_mislabeled": np.nan,
        "mislabel_confidence": np.nan,
        "declared_class": np.nan,
        "nn_predicted": np.nan,
        "mislabel_votes": np.nan,
    }

    for column, default_value in required_detector_columns.items():
        if column not in master.columns:
            master[column] = default_value




    # ----- Priority Score -----

    def is_true(value) -> bool:
        """Return True only for an actual true Boolean value."""
        if pd.isna(value):
            return False

        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}

        return bool(value)

    def compute_priority_score(row: pd.Series) -> int:
        """
        Calculate the existing confidence-adjusted priority score.

        Missing detector results contribute zero to the score.
        Detector coverage is checked separately before assigning CLEAN.
        """

        score = 0

        # Exact duplicate
        if is_true(row.get("Exact_duplicates")):
            score += 10

        # Near duplicate
        if is_true(row.get("near_duplicate")):
            dup_conf = row.get("dup_confidence")

            if pd.isna(dup_conf):
                dup_conf = "LIKELY"

            score += {
                "DEFINITE": 10,
                "LIKELY": 7,
                "POSSIBLE": 4,
            }.get(str(dup_conf).upper(), 7)

        # Mislabel
        if is_true(row.get("is_mislabeled")):
            mis_conf = row.get("mislabel_confidence")

            if pd.isna(mis_conf):
                mis_conf = "MEDIUM"

            score += {
                "HIGH": 9,
                "MEDIUM": 6,
                "LOW": 3,
            }.get(str(mis_conf).upper(), 6)

        # Blur
        if is_true(row.get("is_blurry")):
            blur_score = row.get("blur_score")

            if pd.isna(blur_score):
                score += 5
            elif blur_score < 20:
                score += 8
            else:
                score += 5

        # Noise
        if is_true(row.get("is_noisy_refined")):
            noise_verdict = row.get("noise_verdict")

            if pd.isna(noise_verdict):
                noise_verdict = ""

            score += {
                "confirmed_noisy": 6,
                "review_ood_or_noise": 3,
                "review_noise": 3,
                "review_possible_ood": 0,
                "downgraded_ood": 0,
            }.get(str(noise_verdict), 3)

        # Outlier
        if is_true(row.get("is_outlier")):
            score += 4

        return score



    # Apply confidence-adjusted priority score
    master["priority_score"] = master.apply(
        compute_priority_score,
        axis=1
    )



    # ----- Final Verdict -----

    def assign_verdict(row):
        # Determine whether all five detectors evaluated this image.
        all_detectors_ran = all(
            is_true(row.get(column, False))
            for column in required_ran_columns
        )

        # Preserve existing issue-specific verdicts even if
        # another detector did not evaluate the image.
        if row.get("noise_verdict") == "downgraded_ood":
            return "REVIEW — possible OOD"

        # Exact duplicate
        if is_true(row.get("Exact_duplicates")):
            return "REMOVE — duplicate"

        # Definite near duplicate
        if (
            is_true(row.get("near_duplicate"))
            and row.get("dup_confidence") == "DEFINITE"
        ):
            return "REMOVE — near duplicate"

        # High-confidence mislabel
        if (
            is_true(row.get("is_mislabeled"))
            and row.get("mislabel_confidence") == "HIGH"
        ):
            return "REMOVE — mislabeled"

        # Severe blur
        blur_score = row.get("blur_score")
        if (
            is_true(row.get("is_blurry"))
            and pd.notna(blur_score)
            and blur_score < BLUR_REMOVE_THRESHOLD
        ):
            return "REMOVE — blurry"

        # Confirmed noise
        if row.get("noise_verdict") == "confirmed_noisy":
            return "REMOVE — noisy"

        # High overall priority
        if row["priority_score"] >= REMOVE_PRIORITY:
            return "REMOVE — high priority"

        # Ambiguous noise / OOD
        if row.get("noise_verdict") == "review_ood_or_noise":
            return "REVIEW — OOD or noise"

        # Possible noise
        if row.get("noise_verdict") == "review_noise":
            return "REVIEW — possible noise"

        # Possible OOD
        if row.get("noise_verdict") == "review_possible_ood":
            return "REVIEW — possible OOD"

        # Incomplete analysis must never be labelled CLEAN.
        if not all_detectors_ran:
            return "REVIEW — incomplete analysis"

        # Only fully analysed images with no detected issues are CLEAN.
        if row["priority_score"] == 0:
            return "CLEAN"

        # Medium priority
        if row["priority_score"] >= REVIEW_PRIORITY:
            return "REVIEW — medium priority"

        # Low priority
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
        "is_noisy_refined",
        "is_outlier",
        "is_mislabeled"
    ]

    master["total_flags"] = (
        master[flag_cols]
        .fillna(False)
        .map(is_true)
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
 
            count = master[col].fillna(False).map(is_true).sum()
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

    for cls in sorted(master["class"].dropna().unique()):

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

    reports_path =Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    master_path = (
        reports_path/f"{split}_master_report.csv"
    )

    # Exclude internal detector-coverage columns from the exported report

    internal_columns = [
        column
        for column in master.columns
        if column.startswith("_")
    ]
    output_master = master.drop(columns=internal_columns)

    output_master.to_csv(
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
            .map(is_true)
            .sum()
        ),

        "near_dup": (
            master["near_duplicate"]
            .map(is_true)
            .sum()
        ),

        "blurry": (
            master["is_blurry"]
            .map(is_true)
            .sum()
        ),

        "noisy": (
            master["is_noisy_refined"]
            .map(is_true)
            .sum()
        ),

        "outlier": (
            master["is_outlier"]
            .map(is_true)
            .sum()
        ),

        "mislabeled": (
            master["is_mislabeled"]
            .map(is_true)
            .sum()
        )
    }


    summary_df = pd.DataFrame([summary])

    summary_path = (
        reports_path/f"{split}_cleaning_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Summary saved ==> {summary_path}"
    )

    return output_master

if __name__ == "__main__" :
   for split in ["train", "val"]:
       df = build_master_report(split)

     




        





    

