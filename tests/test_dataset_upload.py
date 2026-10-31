# tests/test_dataset_upload.py
# Standard library
import json
import stat
import struct
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api import dataset_upload as du  # noqa: E402
from api import upload_config as uc  # noqa: E402

IMG = b"fake-image-bytes"
LIM = uc.UploadLimits(min_images_per_class=2)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def imgs(prefix: str, n: int, ext: str = ".jpg") -> list:
    return [(f"{prefix}/{i}{ext}", IMG) for i in range(n)]


def make_zip(tmp, entries, compression=zipfile.ZIP_STORED,
             name="t.zip") -> Path:
    path = Path(tmp) / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(path, "w", compression) as zf:
            for first, data in entries:
                zf.writestr(first, data)
    return path


def inspect(entries, limits=LIM, compression=zipfile.ZIP_STORED):
    with tempfile.TemporaryDirectory() as tmp:
        return du.inspect_zip(
            make_zip(tmp, entries, compression), limits)


def expect(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got "
            f"{type(exc).__name__}: {exc}") from None
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


# ─── Valid layouts ────────────────────────────────────────────────────────────

def test_split_layout() -> None:
    plan = inspect(imgs("train/cat", 3) + imgs("train/dog", 3)
                   + imgs("val/cat", 2) + imgs("val/dog", 2))
    assert plan.layout == "split"
    assert plan.splits == ["train", "val"]
    assert plan.classes == ["cat", "dog"]
    assert plan.total_images == 10
    assert plan.counts == {"train": {"cat": 3, "dog": 3},
                           "val": {"cat": 2, "dog": 2}}
    assert plan.files[0].dest == "train/cat/0.jpg"
    assert plan.files[0].source == "train/cat/0.jpg"


def test_flat_layout_normalised_to_train() -> None:
    plan = inspect(imgs("cat", 3) + imgs("dog", 3))
    assert plan.layout == "flat"
    assert plan.splits == ["train"]
    assert all(f.dest.startswith("train/") for f in plan.files)
    assert plan.files[0].dest == "train/cat/0.jpg"


def test_wrapper_folders() -> None:
    plan = inspect(imgs("mydata/cat", 3) + imgs("mydata/dog", 3))
    assert plan.layout == "flat" and plan.wrapper_stripped == ["mydata"]
    assert plan.files[0].dest == "train/cat/0.jpg"

    plan = inspect(imgs("ds/train/cat", 3) + imgs("ds/train/dog", 3)
                   + imgs("ds/val/cat", 2) + imgs("ds/val/dog", 2))
    assert plan.layout == "split" and plan.wrapper_stripped == ["ds"]
    assert plan.splits == ["train", "val"]

    # Nested wrappers are stripped one level at a time.
    plan = inspect(imgs("a/b/cat", 3) + imgs("a/b/dog", 3))
    assert plan.wrapper_stripped == ["a", "b"]


def test_junk_is_ignored() -> None:
    plan = inspect(
        imgs("mydata/cat", 3) + imgs("mydata/dog", 3)
        + [("__MACOSX/mydata/cat/._0.jpg", b"x"),
           (".DS_Store", b""), ("mydata/.DS_Store", b"")])
    assert plan.ignored_by_reason == {"junk": 3}
    assert plan.wrapper_stripped == ["mydata"]
    assert plan.total_images == 6
    assert not any("MACOSX" in f.source for f in plan.files)


def test_split_name_normalisation() -> None:
    plan = inspect(imgs("Train/cat", 3) + imgs("Train/dog", 3))
    assert plan.layout == "split" and plan.splits == ["train"]
    assert plan.files[0].dest == "train/cat/0.jpg"
    # A single top-level split folder is not mistaken for a wrapper.
    assert plan.wrapper_stripped == []

    plan = inspect(imgs("training/cat", 3) + imgs("training/dog", 3))
    assert plan.splits == ["training"]


def test_ignored_categories() -> None:
    plan = inspect(
        imgs("cat", 3) + imgs("dog", 3)
        + [("cat/sub/x.jpg", IMG),     # nested_too_deep
           ("cat/notes.txt", b"hi"),   # not_an_image
           ("root.jpg", IMG)])         # root_level
    assert plan.ignored_by_reason == {
        "nested_too_deep": 1, "not_an_image": 1, "root_level": 1}
    assert plan.total_images == 6


def test_ignored_samples_are_capped() -> None:
    notes = [(f"docs/n{i}.txt", b"x") for i in range(100)]
    plan = inspect(imgs("cat", 3) + imgs("dog", 3) + notes)
    assert plan.ignored_total == 100
    assert len(plan.ignored_samples) == du.IGNORED_SAMPLE_LIMIT


def test_metadata_is_json_safe() -> None:
    plan = inspect(imgs("cat", 3) + imgs("dog", 3))
    meta = plan.to_metadata()
    json.dumps(meta)
    assert "files" not in meta and meta["layout"] == "flat"


# ─── Layout rejections ────────────────────────────────────────────────────────

def test_empty_and_ignored_only() -> None:
    expect(uc.EmptyDatasetError, inspect, [])
    expect(uc.EmptyDatasetError, inspect,
           [("README.txt", b"hi"), (".DS_Store", b"")])
    expect(uc.EmptyDatasetError, inspect, [("a.jpg", IMG)])


def test_single_class_rejected_not_flattened() -> None:
    err = expect(uc.InsufficientDataError, inspect, imgs("cat", 3))
    assert "classes" in err.message


def test_mixed_layout() -> None:
    expect(uc.InvalidLayoutError, inspect,
           imgs("train/cat", 3) + imgs("train/dog", 3) + imgs("bird", 3))


def test_ambiguous_split_names() -> None:
    expect(uc.InvalidLayoutError, inspect,
           imgs("Train/cat", 3) + imgs("Train/dog", 3)
           + imgs("train/cat", 3) + imgs("train/dog", 3))
    expect(uc.InvalidLayoutError, inspect,
           imgs("train/cat", 3) + imgs("train/dog", 3)
           + imgs("training/cat", 3) + imgs("training/dog", 3))


def test_class_mismatch_across_splits() -> None:
    err = expect(uc.InvalidLayoutError, inspect,
                 imgs("train/cat", 3) + imgs("train/dog", 3)
                 + imgs("val/cat", 3) + imgs("val/bird", 3))
    assert "missing_classes_by_split" in err.details


def test_bad_class_name() -> None:
    expect(uc.InvalidLayoutError, inspect,
           imgs("ca$t", 3) + imgs("dog", 3))


def test_case_collisions() -> None:
    expect(uc.InvalidLayoutError, inspect,
           imgs("Cat", 3) + imgs("cat", 3) + imgs("dog", 3))
    expect(uc.InvalidLayoutError, inspect,
           [("cat/a.jpg", IMG), ("cat/A.JPG", IMG)] + imgs("dog", 3))
    expect(uc.InvalidLayoutError, inspect,
           [("cat/a.jpg", IMG), ("cat/a.jpg", IMG)] + imgs("dog", 3))


def test_dataset_size_rules() -> None:
    expect(uc.InsufficientDataError, inspect,
           imgs("cat", 1) + imgs("dog", 3))
    small = uc.UploadLimits(max_entries=100, max_images=5,
                            min_images_per_class=1)
    expect(uc.UploadTooLargeError, inspect,
           imgs("cat", 3) + imgs("dog", 3), small)


# ─── Safety rejections ────────────────────────────────────────────────────────

def test_unsafe_paths() -> None:
    for bad in ["../evil.jpg", "cat/../../evil.jpg", "/abs/evil.jpg",
                "cat\\a.jpg", "C:/x.jpg", "cat/" + "a" * 300 + ".jpg"]:
        expect(uc.UnsafePathError, inspect,
               imgs("cat", 3) + imgs("dog", 3) + [(bad, IMG)])


def test_symlink_entry() -> None:
    link = zipfile.ZipInfo("cat/link.jpg")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    expect(uc.UnsafePathError, inspect,
           imgs("cat", 3) + imgs("dog", 3) + [(link, b"target")])


def test_member_checks_direct() -> None:
    enc = zipfile.ZipInfo("cat/a.jpg")
    enc.flag_bits = 0x1
    expect(uc.UnsupportedArchiveError, du._check_member, enc)

    odd = zipfile.ZipInfo("cat/a.jpg")
    odd.compress_type = 93
    expect(uc.UnsupportedArchiveError, du._check_member, odd)

    ok = zipfile.ZipInfo("cat/a.jpg")
    assert du._check_member(ok) == ["cat", "a.jpg"]


# ─── Archive-level rejections ─────────────────────────────────────────────────

def test_not_a_zip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        text = Path(tmp) / "fake.zip"
        text.write_text("this is not a zip")
        expect(uc.UnsupportedArchiveError, du.inspect_zip, text, LIM)
        empty = Path(tmp) / "empty.zip"
        empty.write_bytes(b"")
        expect(uc.UnsupportedArchiveError, du.inspect_zip, empty, LIM)


def test_size_limits() -> None:
    small = uc.UploadLimits(
        max_upload_bytes=10_000, max_uncompressed_bytes=10_000,
        max_entries=100, max_images=50, min_images_per_class=1)
    # Upload larger than the cap (stored, so the file itself is big).
    expect(uc.UploadTooLargeError, inspect,
           [("cat/a.jpg", b"x" * 20_000)], small)
    # Small upload that expands past the uncompressed cap.
    expect(uc.UploadTooLargeError, inspect,
           [("cat/a.jpg", b"\0" * 20_000)], small,
           zipfile.ZIP_DEFLATED)


def test_compression_bomb() -> None:
    expect(uc.CompressionRatioError, inspect,
           [("cat/a.jpg", b"\0" * 3_000_000)] + imgs("dog", 3),
           LIM, zipfile.ZIP_DEFLATED)


def test_entry_count_checked_before_opening() -> None:
    few = uc.UploadLimits(max_entries=5, max_images=5,
                          min_images_per_class=1)
    expect(uc.TooManyEntriesError, inspect, imgs("cat", 10), few)


def test_end_of_central_directory_guards() -> None:
    few = uc.UploadLimits(max_entries=5, max_images=5,
                          min_images_per_class=1)

    def eocd(n_total: int, cd_size: int) -> bytes:
        return struct.pack("<4s4H2LH", b"PK\x05\x06",
                           0, 0, n_total, n_total, cd_size, 0, 0)

    with tempfile.TemporaryDirectory() as tmp:
        huge_dir = Path(tmp) / "a.zip"
        huge_dir.write_bytes(eocd(1, 10_000_000))
        expect(uc.TooManyEntriesError, du.inspect_zip, huge_dir, few)

        zip64 = Path(tmp) / "b.zip"
        zip64.write_bytes(eocd(0xFFFF, 100))
        expect(uc.UnsupportedArchiveError, du.inspect_zip, zip64, few)

        # A valid ZIP comment may contain EOCD signature bytes.
        commented = make_zip(
            tmp,
            imgs("cat", 3) + imgs("dog", 3),
        )
        with zipfile.ZipFile(commented, "a") as zf:
            zf.comment = b"comment with PK\x05\x06 inside"

        # Our EOCD validator must locate the real EOCD,
        # even when the comment contains signature bytes.
        du._check_central_directory(commented, LIM)

        # Multi-disk ZIP archives must be rejected.
        multi_disk = Path(tmp) / "multi_disk.zip"
        multi_disk.write_bytes(
            struct.pack(
                "<4s4H2LH",
                b"PK\x05\x06",
                1, 1, 0, 0, 0, 0, 0,
            )
        )
        expect(
            uc.UnsupportedArchiveError,
            du._check_central_directory,
            multi_disk,
            LIM,
        )

        # Central directory must not extend beyond the EOCD.
        bad_offset = Path(tmp) / "bad_offset.zip"
        bad_offset.write_bytes(
            struct.pack(
                "<4s4H2LH",
                b"PK\x05\x06",
                0, 0, 0, 0, 10, 0, 0,
            )
        )
        expect(
            uc.UnsupportedArchiveError,
            du._check_central_directory,
            bad_offset,
            LIM,
        )


def test_inspection_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = make_zip(tmp, imgs("cat", 3) + imgs("dog", 3))
        before = sorted(p.name for p in Path(tmp).iterdir())
        du.inspect_zip(path, LIM)
        assert sorted(p.name for p in Path(tmp).iterdir()) == before


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ dataset upload inspection tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    run()