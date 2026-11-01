# tests/test_dataset_validate.py
# Standard library
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from PIL import Image, ImageFile  # noqa: E402

# Local
from api import dataset_extract as ex  # noqa: E402
from api import dataset_upload as du  # noqa: E402
from api import dataset_validate as dv  # noqa: E402
from api import upload_config as uc  # noqa: E402

LIM = uc.UploadLimits(min_images_per_class=2)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def image_bytes(fmt="JPEG", size=(8, 8), noise=False) -> bytes:
    if noise:
        img = Image.frombytes("RGB", size,
                              os.urandom(size[0] * size[1] * 3))
    else:
        img = Image.new("RGB", size, (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


GOOD = image_bytes()


def entries(prefix, n, data=None):
    data = GOOD if data is None else data
    return [(f"{prefix}/{i}.jpg", data) for i in range(n)]


def build(tmp, items, limits=LIM):
    zip_path = Path(tmp) / "t.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in items:
            zf.writestr(name, data)
    plan = du.inspect_zip(zip_path, limits)
    dataset = Path(tmp) / "job" / "dataset"
    ex.extract_plan(zip_path, plan, dataset, limits)
    return plan, dataset


def tree(root: Path) -> list:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() or p.is_symlink())


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


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_all_valid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, entries("cat", 4) + entries("dog", 4))
        before = tree(dataset)
        result = dv.validate_images(dataset, plan, LIM)
        assert result.accepted == 8 and result.rejected == []
        assert result.counts == {"train": {"cat": 4, "dog": 4}}
        assert tree(dataset) == before


def test_corrupt_images_rejected_and_removed() -> None:
    items = (entries("cat", 4)
             + [("cat/bad.jpg", os.urandom(200)), ("cat/empty.jpg", b"")]
             + entries("dog", 4))
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, items)
        result = dv.validate_images(dataset, plan, LIM)
        assert result.accepted == 8
        assert result.rejected_by_reason == {"unreadable": 2}
        assert result.rejected_by_class == {"train/cat": 2}
        assert result.counts == {"train": {"cat": 4, "dog": 4}}
        assert not (dataset / "train/cat/bad.jpg").exists()
        assert not (dataset / "train/cat/empty.jpg").exists()
        assert (dataset / "train/cat/0.jpg").exists()


def test_truncated_jpeg_rejected() -> None:
    full = image_bytes(size=(64, 64), noise=True)
    items = (entries("cat", 3) + [("cat/cut.jpg", full[:len(full) // 2])]
             + entries("dog", 3))
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, items)
        result = dv.validate_images(dataset, plan, LIM)
        assert result.rejected_by_reason == {"unreadable": 1}
        assert not (dataset / "train/cat/cut.jpg").exists()


def test_format_mismatch_rejected() -> None:
    png = image_bytes("PNG")
    items = entries("cat", 3) + [("cat/fake.jpg", png)] + entries("dog", 3)
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, items)
        result = dv.validate_images(dataset, plan, LIM)
        assert result.rejected_by_reason == {"format_mismatch": 1}
    # The matching case is accepted: a real PNG named .png.
    items = entries("cat", 3) + [("cat/ok.png", png)] + entries("dog", 3)
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, items)
        assert dv.validate_images(dataset, plan, LIM).rejected == []


def test_pixel_limit() -> None:
    limits = uc.UploadLimits(min_images_per_class=2, max_image_pixels=100)
    big = image_bytes(size=(20, 20))   # 400 px > 100
    items = entries("cat", 3) + [("cat/big.jpg", big)] + entries("dog", 3)
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, items, limits)
        result = dv.validate_images(dataset, plan, limits)
        assert result.rejected_by_reason == {"too_many_pixels": 1}
        assert not (dataset / "train/cat/big.jpg").exists()


def test_minimum_recheck_fails_and_deletes_nothing() -> None:
    items = (entries("cat", 1) + [("cat/bad.jpg", b"junk")]
             + entries("dog", 3))
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, items)
        before = tree(dataset)
        err = expect(uc.InsufficientDataError,
                     dv.validate_images, dataset, plan, LIM)
        assert err.details["rejected_total"] == 1
        assert err.details["examples"][0]["class"] == "cat"
        assert tree(dataset) == before, "files deleted despite failure"
        assert str(tmp) not in json.dumps(err.to_dict())


def test_missing_file_and_symlink() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, entries("cat", 4) + entries("dog", 4))
        (dataset / "train/cat/0.jpg").unlink()
        link = dataset / "train/cat/1.jpg"
        link.unlink()
        link.symlink_to(dataset / "train/cat/2.jpg")
        result = dv.validate_images(dataset, plan, LIM)
        assert result.rejected_by_reason == {"missing": 1, "unreadable": 1}
        assert (dataset / "train/cat/2.jpg").is_file(), "link target deleted"
        assert not link.exists() and not link.is_symlink()


def test_progress_callback() -> None:
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp, entries("cat", 4) + entries("dog", 4))
        dv.validate_images(dataset, plan, LIM,
                           progress=lambda d, t: calls.append((d, t)))
    assert calls and calls[-1] == (8, 8)


def test_metadata_merge_is_json_safe_and_capped() -> None:
    bad = [(f"cat/bad{i}.jpg", b"junk") for i in range(150)]
    limits = uc.UploadLimits(min_images_per_class=2)
    with tempfile.TemporaryDirectory() as tmp:
        plan, dataset = build(tmp,
                              entries("cat", 4) + bad + entries("dog", 4),
                              limits)
        result = dv.validate_images(dataset, plan, limits)
        meta = dv.merge_metadata(plan, result)
        json.dumps(meta)
        assert meta["rejected_total"] == 150
        assert len(meta["rejected_samples"]) == dv.REJECTED_SAMPLE_LIMIT
        assert meta["total_images"] == 8
        assert meta["counts"] == {"train": {"cat": 4, "dog": 4}}
        assert meta["layout"] == "flat"


def test_pipeline_consistency() -> None:
    assert set(dv.EXTENSION_FORMATS) == set(uc.IMAGE_EXTENSIONS)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ image validation tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    run()