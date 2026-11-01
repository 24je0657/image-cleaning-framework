# tests/test_dataset_extract.py
# Standard library
import dataclasses
import hashlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api import dataset_extract as ex  # noqa: E402
from api import dataset_upload as du  # noqa: E402
from api import upload_config as uc  # noqa: E402

LIM = uc.UploadLimits(min_images_per_class=2)
PAYLOAD = b"fake-image-bytes"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def entries(prefix: str, n: int, data: bytes = PAYLOAD) -> list:
    return [(f"{prefix}/{i}.jpg", data) for i in range(n)]


def make_zip(tmp, items) -> Path:
    path = Path(tmp) / "t.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in items:
            zf.writestr(name, data)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file())


def leftovers(job: Path) -> list:
    return [p.name for p in job.iterdir() if p.name.startswith(".extract-")]


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


def setup(tmp, items, precreate=True):
    zip_path = make_zip(tmp, items)
    plan = du.inspect_zip(zip_path, LIM)
    job = Path(tmp) / "job"
    job.mkdir()
    dataset = job / "dataset"
    if precreate:
        dataset.mkdir()
    return zip_path, plan, job, dataset


# ─── Success ──────────────────────────────────────────────────────────────────

def test_clean_extraction_matches_plan() -> None:
    for precreate in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path, plan, job, dataset = setup(
                tmp, entries("cat", 3) + entries("dog", 3), precreate)
            before = sha(zip_path)

            result = ex.extract_plan(zip_path, plan, dataset, LIM)

            assert tree(dataset) == sorted(f.dest for f in plan.files)
            assert tree(dataset)[0] == "train/cat/0.jpg"  # flat -> train/
            assert result.files_written == 6
            assert result.bytes_written == 6 * len(PAYLOAD)
            assert (dataset / "train/cat/0.jpg").read_bytes() == PAYLOAD
            assert leftovers(job) == []
            assert sha(zip_path) == before, "ZIP was modified"


def test_split_layout_extraction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("train/cat", 3) + entries("train/dog", 3)
            + entries("val/cat", 2) + entries("val/dog", 2))
        ex.extract_plan(zip_path, plan, dataset, LIM)
        assert (dataset / "val/dog/1.jpg").is_file()


# ─── Destination rules ────────────────────────────────────────────────────────

def test_non_empty_destination_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3) + entries("dog", 3))
        (dataset / "existing.txt").write_text("keep me")
        expect(ex.DatasetExistsError, ex.extract_plan,
               zip_path, plan, dataset, LIM)
        assert tree(dataset) == ["existing.txt"]
        assert (dataset / "existing.txt").read_text() == "keep me"
        assert leftovers(job) == []


def test_destination_file_or_symlink_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3) + entries("dog", 3), precreate=False)
        dataset.write_text("a file, not a directory")
        expect(ex.DatasetExistsError, ex.extract_plan,
               zip_path, plan, dataset, LIM)
        dataset.unlink()

        target = Path(tmp) / "elsewhere"
        target.mkdir()
        dataset.symlink_to(target, target_is_directory=True)
        expect(ex.DatasetExistsError, ex.extract_plan,
               zip_path, plan, dataset, LIM)
        assert tree(target) == []

def test_dangling_symlink_destination_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp,
            entries("cat", 3) + entries("dog", 3),
            precreate=False,
        )

        missing_target = Path(tmp) / "missing_target"
        dataset.symlink_to(
            missing_target,
            target_is_directory=True,
        )

        expect(
            ex.DatasetExistsError,
            ex.extract_plan,
            zip_path,
            plan,
            dataset,
            LIM,
        )

        assert dataset.is_symlink()
        assert not missing_target.exists()
        assert leftovers(job) == []

# ─── Failure leaves nothing behind ────────────────────────────────────────────

def assert_untouched(job: Path, dataset: Path) -> None:
    assert dataset.is_dir() and list(dataset.iterdir()) == []
    assert leftovers(job) == []


def test_byte_budget_counts_actual_bytes() -> None:
    big = b"x" * 2000
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3, big) + entries("dog", 3, big))
        tight = uc.UploadLimits(
            max_upload_bytes=5000, max_uncompressed_bytes=5000,
            max_entries=100, max_images=10, min_images_per_class=2)
        expect(uc.UploadTooLargeError, ex.extract_plan,
               zip_path, plan, dataset, tight)
        assert_untouched(job, dataset)


def test_interrupted_extraction_leaves_no_partial_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3) + entries("dog", 3))
        original = ex._copy_member
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 4:
                raise RuntimeError("simulated crash mid-extraction")
            return original(*args, **kwargs)

        ex._copy_member = flaky
        try:
            expect(RuntimeError, ex.extract_plan,
                   zip_path, plan, dataset, LIM)
        finally:
            ex._copy_member = original
        assert calls["n"] == 4
        assert_untouched(job, dataset)


def test_corrupt_member_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = make_zip(tmp, entries("cat", 3) + entries("dog", 3))
        data = bytearray(zip_path.read_bytes())
        i = data.find(PAYLOAD)          # stored payload of the first file
        data[i] ^= 0xFF                 # CRC no longer matches
        zip_path.write_bytes(data)
        plan = du.inspect_zip(zip_path, LIM)   # inspection reads no data
        job = Path(tmp) / "job"
        dataset = job / "dataset"
        dataset.mkdir(parents=True)
        expect(uc.UnsupportedArchiveError, ex.extract_plan,
               zip_path, plan, dataset, LIM)
        assert_untouched(job, dataset)


def test_destination_collision_fails_loudly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3) + entries("dog", 3))
        plan.files.append(plan.files[0])   # same dest twice
        expect(FileExistsError, ex.extract_plan,
               zip_path, plan, dataset, LIM)
        assert_untouched(job, dataset)


def test_unsafe_planned_destination_rejected() -> None:
    for bad in [
    "../escape.jpg",
    "/abs.jpg",
    "a\\b.jpg",
    "train//x.jpg",
    "train/./x.jpg",
    "C:/escape.jpg",
    "train/cat/\x00.jpg",
    "",
]:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path, plan, job, dataset = setup(
                tmp, entries("cat", 3) + entries("dog", 3))
            plan.files[0] = dataclasses.replace(plan.files[0], dest=bad)
            expect(uc.UnsafePathError, ex.extract_plan,
                   zip_path, plan, dataset, LIM)
            assert_untouched(job, dataset)
            assert not (Path(tmp) / "escape.jpg").exists()


def test_member_missing_from_archive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3) + entries("dog", 3))
        plan.files[0] = dataclasses.replace(plan.files[0], source="nope.jpg")
        expect(uc.UnsupportedArchiveError, ex.extract_plan,
               zip_path, plan, dataset, LIM)
        assert_untouched(job, dataset)


def test_rename_failure_restores_empty_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path, plan, job, dataset = setup(
            tmp, entries("cat", 3) + entries("dog", 3))
        original = os.rename

        def boom(src, dst):
            raise OSError("simulated rename failure")

        os.rename = boom
        try:
            expect(OSError, ex.extract_plan, zip_path, plan, dataset, LIM)
        finally:
            os.rename = original
        assert_untouched(job, dataset)


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ dataset extraction tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    run()