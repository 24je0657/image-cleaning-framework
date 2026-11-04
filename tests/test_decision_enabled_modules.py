import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.decision_engine import DETECTOR_NAMES, build_master_report


PATHS = [
    f"ds/train/{c}/{i}.jpg"
    for c in ("cat", "dog")
    for i in range(3)
]

CLASSES = [p.split("/")[2] for p in PATHS]

ALL = list(DETECTOR_NAMES)

FILES = {
    "duplicates": "duplicates",
    "blur": "blur",
    "noise": "noise",
    "outliers": "outlier",
    "mislabels": "mislabel",
}


def frames():
    base = {"file_path": PATHS}

    return {
        "duplicates": pd.DataFrame({
            **base,
            "Exact_duplicates": False,
            "duplicate_of": "",
            "near_duplicate": False,
            "cosine_score": 0.0,
            "near_duplicate_of": "",
            "dup_confidence": "",
        }),

        "blur": pd.DataFrame({
            **base,
            "class": CLASSES,
            "blur_score": 500.0,
            "threshold_used": 100.0,
            "threshold_method": "adaptive",
            "is_blurry": False,
        }),

        "noise": pd.DataFrame({
            **base,
            "class": CLASSES,
            "reconstruction_error": 0.001,
            "noise_threshold": 0.002,
            "is_noisy": False,
            "blur_score": 500.0,
            "threshold_used": 100.0,
            "is_outlier": False,
            "blur_ratio": 5.0,
            "is_noisy_refined": False,
            "noise_verdict": "not_flagged",
        }),

        "outliers": pd.DataFrame({
            **base,
            "class": CLASSES,
            "global_outlier": False,
            "global_score": 0.1,
            "per_class_outlier": False,
            "per_class_score": 0.1,
            "is_outlier": False,
        }),

        "mislabels": pd.DataFrame({
            **base,
            "declared_class": CLASSES,
            "kmeans_predicted": CLASSES,
            "nn_predicted": CLASSES,
            "centroid_predicted": CLASSES,
            "kmeans_flag": False,
            "nn_flag": False,
            "centroid_flag": False,
            "votes": 0,
            "confidence": "CLEAN",
            "is_mislabeled": False,
            "dbscan_checked": True,
        }),
    }


def verdicts(skip=(), enabled=None, tweak=None):
    data = frames()

    if tweak:
        tweak(data)

    with tempfile.TemporaryDirectory() as tmp:
        for name, df in data.items():
            if name not in skip:
                df.to_csv(
                    Path(tmp) / f"train_{FILES[name]}_report.csv",
                    index=False,
                )

        result = build_master_report(
            "train",
            reports_dir=tmp,
            enabled_modules=enabled,
        )

    return dict(zip(result["file_path"], result["verdict"]))


def expect_exc(exc_types, **kwargs):
    try:
        build_master_report(
            "train",
            reports_dir="/nonexistent",
            **kwargs,
        )
    except exc_types:
        return

    raise AssertionError(f"Accepted invalid arguments: {kwargs}")


def test_default_all_enabled_is_clean():
    assert set(verdicts().values()) == {"CLEAN"}
    assert set(verdicts(enabled=ALL).values()) == {"CLEAN"}


def test_disabled_module_does_not_block_clean():
    others = [n for n in ALL if n != "blur"]

    assert set(
        verdicts(skip={"blur"}, enabled=others).values()
    ) == {"CLEAN"}


def test_enabled_but_missing_report_is_incomplete():
    expected = {"REVIEW — incomplete analysis"}

    assert set(verdicts(skip={"blur"}).values()) == expected

    assert set(
        verdicts(skip={"blur"}, enabled=ALL).values()
    ) == expected


def test_flagged_image_still_removed_when_module_disabled():
    def tweak(data):
        data["duplicates"].loc[0, "Exact_duplicates"] = True

    others = [n for n in ALL if n != "blur"]

    result = verdicts(
        skip={"blur"},
        enabled=others,
        tweak=tweak,
    )

    assert result[PATHS[0]] == "REMOVE — duplicate"

    assert all(
        result[p] == "CLEAN"
        for p in PATHS[1:]
    )


def test_partial_coverage_blocks_clean():
    def tweak(data):
        data["noise"] = data["noise"].iloc[:-1]

    result = verdicts(tweak=tweak)

    assert result[PATHS[-1]] == "REVIEW — incomplete analysis"

    assert all(
        result[p] == "CLEAN"
        for p in PATHS[:-1]
    )


def test_invalid_enabled_modules_rejected():
    expect_exc((ValueError,), enabled_modules=[])
    expect_exc((ValueError,), enabled_modules=["blurr"])
    expect_exc((TypeError,), enabled_modules="blur")

def test_skipped_dbscan_blocks_clean():
    def tweak(data):
        data["mislabels"]["dbscan_checked"] = False

    result = verdicts(tweak=tweak)

    assert set(result.values()) == {"REVIEW — incomplete analysis"}

def test_partial_dbscan_coverage_blocks_clean():
    def tweak(data):
        data["mislabels"].loc[data["mislabels"]["file_path"] == PATHS[0],"dbscan_checked"] = False

    result = verdicts(tweak=tweak)

    assert result[PATHS[0]] == "REVIEW — incomplete analysis"

    assert all(
        result[p] == "CLEAN"
        for p in PATHS[1:]
    )

def run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    for test in tests:
        test()

    print(
        f"Decision-engine enabled-module tests passed ({len(tests)})."
    )


if __name__ == "__main__":
    run()
