import numpy as np
import pandas as pd
import pytest

from src import outliers, mislabel


def create_embeddings(tmp_path, class_counts):
    """Create a small synthetic embedding dataset."""

    rng = np.random.default_rng(42)

    total = sum(class_counts.values())
    embeddings = rng.normal(size=(total, 16))

    labels = []
    paths = []

    for class_name, count in class_counts.items():
        labels.extend([class_name] * count)
        paths.extend(
            [f"/dataset/{class_name}/image_{i}.jpg"
             for i in range(count)]
        )

    np.save(tmp_path / "train_embeddings.npy", embeddings)
    np.save(tmp_path / "train_labels.npy", np.array(labels))
    np.save(tmp_path / "train_file_paths.npy", np.array(paths))

    return embeddings, np.array(labels), np.array(paths)


def test_outlier_detection_small_dataset(tmp_path):
    create_embeddings(
        tmp_path,
        {"cat": 3, "dog": 3},
    )

    report = outliers.detect_outliers(
        split="train",
        embed_dir=tmp_path,
        reports_dir=tmp_path / "outlier_reports",
    )

    assert len(report) == 6
    assert report["file_path"].nunique() == 6
    assert (tmp_path / "outlier_reports/train_outlier_report.csv").exists()


def test_mislabel_detection_small_dataset(tmp_path):
    create_embeddings(
        tmp_path,
        {"cat": 3, "dog": 3},
    )

    report = mislabel.detect_mislabels(
        split="train",
        embed_dir=tmp_path,
        reports_dir=tmp_path / "mislabel_reports",
    )

    assert len(report) == 6
    assert report["file_path"].nunique() == 6
    assert "dbscan_checked" in report.columns

    # Both classes have fewer than 10 images, so DBSCAN skips them.
    assert not report["dbscan_checked"].any()

    assert (
        tmp_path / "mislabel_reports/train_mislabel_report.csv"
    ).exists()


def test_outlier_pca_rejects_single_sample():
    embeddings = np.random.default_rng(42).normal(size=(1, 16))

    with pytest.raises(ValueError, match="at least 2 samples"):
        outliers.Reduce_Dimensions(embeddings)


def test_mislabel_pca_rejects_single_sample():
    embeddings = np.random.default_rng(42).normal(size=(1, 16))

    with pytest.raises(ValueError, match="at least 2 samples"):
        mislabel.reduce_dimensions(embeddings)


def test_dbscan_coverage_for_small_classes():
    rng = np.random.default_rng(42)

    # First class has 5 images; second class has 10.
    labels = np.array(["cat"] * 5 + ["dog"] * 10)
    reduced = rng.normal(size=(15, 8))

    flags, predictions, checked = mislabel.method1_dbscan(
        reduced,
        labels,
    )

    assert len(flags) == 15
    assert len(predictions) == 15
    assert len(checked) == 15

    # DBSCAN skips the first class.
    assert not checked[:5].any()

    # DBSCAN evaluates the second class.
    assert checked[5:].all()