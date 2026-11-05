# tests/test_upload_config.py
# Standard library
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api import upload_config as uc  # noqa: E402

MB = uc.MB


def expect_config_error(env: dict, fragment: str) -> None:
    try:
        uc.load_limits(env)
    except uc.UploadConfigError as exc:
        assert fragment in str(exc), f"{fragment!r} not in {exc}"
    else:
        raise AssertionError(f"accepted invalid env: {env}")


def test_defaults() -> None:
    lim = uc.load_limits({})
    assert lim.max_upload_bytes == 500 * MB
    assert lim.max_uncompressed_bytes == 2048 * MB
    assert lim.max_entries == 20_000
    assert lim.max_images == 5_000
    assert lim.max_compression_ratio == 100.0
    assert lim.min_classes == 2
    assert lim.min_images_per_class == 30
    assert lim.max_image_pixels == 50_000_000
    assert isinstance(uc.LIMITS, uc.UploadLimits)

def test_minimum_dataset_size_is_valid() -> None:
    lim = uc.load_limits({})
    assert lim.min_classes * lim.min_images_per_class == 60
    assert lim.max_images >= 60

def test_overrides_and_blank_values() -> None:
    lim = uc.load_limits({
        "IMAGE_CLEANING_MAX_UPLOAD_MB": "100",
        "IMAGE_CLEANING_MAX_UNCOMPRESSED_MB": "400",
        "IMAGE_CLEANING_MAX_IMAGES": "1000",
        "IMAGE_CLEANING_MAX_COMPRESSION_RATIO": "50.5",
        "IMAGE_CLEANING_MIN_IMAGES_PER_CLASS": "  25 ",
        "IMAGE_CLEANING_MAX_ZIP_ENTRIES": "",   # blank means default
    })
    assert lim.max_upload_bytes == 100 * MB
    assert lim.max_uncompressed_bytes == 400 * MB
    assert lim.max_images == 1000
    assert lim.max_compression_ratio == 50.5
    assert lim.min_images_per_class == 25
    assert lim.max_entries == 20_000


def test_invalid_values_are_rejected() -> None:
    p = "IMAGE_CLEANING_"
    expect_config_error({p + "MAX_UPLOAD_MB": "abc"}, "MAX_UPLOAD_MB")
    expect_config_error({p + "MAX_UPLOAD_MB": "0"}, "MAX_UPLOAD_MB")
    expect_config_error({p + "MAX_UPLOAD_MB": "-5"}, "MAX_UPLOAD_MB")
    expect_config_error({p + "MAX_UPLOAD_MB": "1.5"}, "MAX_UPLOAD_MB")
    expect_config_error({p + "MAX_COMPRESSION_RATIO": "nan"}, "finite")
    expect_config_error({p + "MAX_COMPRESSION_RATIO": "inf"}, "finite")
    expect_config_error({p + "MAX_COMPRESSION_RATIO": "0.5"}, "RATIO")
    expect_config_error({p + "MIN_CLASSES": "1"}, "MIN_CLASSES")


def test_inconsistent_combinations() -> None:
    p = "IMAGE_CLEANING_"
    expect_config_error(
        {p + "MAX_UPLOAD_MB": "3000", p + "MAX_UNCOMPRESSED_MB": "2000"},
        "must not exceed")
    expect_config_error(
        {p + "MAX_IMAGES": "30000", p + "MAX_ZIP_ENTRIES": "20000"},
        "MAX_IMAGES")
    expect_config_error(
        {p + "MAX_IMAGES": "50", p + "MIN_IMAGES_PER_CLASS": "30"},
        "no dataset could pass")
    # Direct construction is validated too.
    try:
        uc.UploadLimits(min_classes=1)
    except uc.UploadConfigError:
        pass
    else:
        raise AssertionError("UploadLimits(min_classes=1) accepted")


def test_limits_are_immutable() -> None:
    try:
        uc.LIMITS.max_images = 1
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("LIMITS is mutable")


def test_exceptions() -> None:
    classes = [
        uc.UploadValidationError, uc.UploadTooLargeError,
        uc.TooManyEntriesError, uc.CompressionRatioError,
        uc.UnsafePathError, uc.UnsupportedArchiveError,
        uc.InvalidLayoutError, uc.EmptyDatasetError,
        uc.InsufficientDataError,
    ]
    codes = [c.code for c in classes]
    assert len(set(codes)) == len(codes), "duplicate error codes"

    for cls in classes[1:]:
        assert issubclass(cls, uc.UploadValidationError)

    assert uc.UploadTooLargeError.http_status == 413
    assert uc.TooManyEntriesError.http_status == 413
    assert uc.UnsafePathError.http_status == 400

    err = uc.InvalidLayoutError("bad layout", details={"classes": ["a"]})
    assert err.to_dict() == {
        "code": "invalid_layout",
        "message": "bad layout",
        "details": {"classes": ["a"]},
    }
    assert str(err) == "bad layout"
    assert uc.UploadValidationError("x").details == {}

    # Config errors are a different family on purpose.
    assert not issubclass(uc.UploadConfigError, uc.UploadValidationError)


def test_class_name_pattern() -> None:
    good = ["cat", "Cat 2", "a", "x-1_y.z", "A" * 64, "1", "dog_"]
    bad = ["", ".hidden", "-x", " cat", "cat ", "cat.", "a/b", "a\\b",
           "A" * 65, "caf\u00e9", "cat\x00", "..", "a\nb"]
    for name in good:
        assert uc.CLASS_NAME_PATTERN.fullmatch(name), name
    for name in bad:
        assert not uc.CLASS_NAME_PATTERN.fullmatch(name), repr(name)


def test_extensions_match_pipeline() -> None:
    # Importing src.embeddings pulls in torch, so this takes a few seconds.
    from src.embeddings import VALID_EXTENSIONS
    assert uc.IMAGE_EXTENSIONS == frozenset(VALID_EXTENSIONS), (
        "upload and pipeline extension sets have drifted"
    )


def test_known_splits_exported_by_config() -> None:
    from src.config import KNOWN_SPLITS
    assert {"train", "val", "test"} <= set(KNOWN_SPLITS)

def test_direct_construction_validation() -> None:
    invalid_cases = [
        {"max_entries": 2.5},
        {"max_images": 0},
        {"max_compression_ratio": float("inf")},
        {"max_compression_ratio": float("nan")},
        {"min_images_per_class": -1},
        {"max_image_pixels": True},
        {"max_upload_bytes": 0},
        {"min_classes": 1},
    ]

    for kwargs in invalid_cases:
        try:
            uc.UploadLimits(**kwargs)
        except uc.UploadConfigError:
            pass
        else:
            raise AssertionError(
                f"Invalid configuration accepted: {kwargs}"
            )


def run() -> None:
    test_defaults()
    test_overrides_and_blank_values()
    test_invalid_values_are_rejected()
    test_inconsistent_combinations()
    test_limits_are_immutable()
    test_exceptions()
    test_class_name_pattern()
    test_extensions_match_pipeline()
    test_known_splits_exported_by_config()
    test_direct_construction_validation()
    print("✅ upload config tests passed.")


if __name__ == "__main__":
    run()