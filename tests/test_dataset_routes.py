# tests/test_dataset_routes.py
# Standard library
import io
import json
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

# Local
from api import main, worker  # noqa: E402
from api import upload_config as uc  # noqa: E402
from api.routes import datasets as routes  # noqa: E402

LIM = uc.UploadLimits(min_images_per_class=2)
HEADERS = {"content-type": "application/zip"}


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


GOOD = jpeg()


def entries(prefix, n):
    return [(f"{prefix}/{i}.jpg", GOOD) for i in range(n)]


def zip_bytes(items) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in items:
            zf.writestr(name, data)
    return buf.getvalue()


@contextmanager
def isolated(limits=LIM):
    saved = (worker.JOBS_ROOT, worker.jobs.copy(), routes.LIMITS)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jobs"
        root.mkdir()
        try:
            worker.JOBS_ROOT = root
            worker.jobs.clear()
            routes.LIMITS = limits
            yield TestClient(main.app), root
        finally:
            worker.jobs.clear()
            worker.jobs.update(saved[1])
            worker.JOBS_ROOT, routes.LIMITS = saved[0], saved[2]


def wait_for(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("ready", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


def assert_nothing_left(root: Path) -> None:
    assert list(root.iterdir()) == [], list(root.iterdir())
    assert worker.jobs == {}


def test_valid_upload_becomes_ready() -> None:
    items = entries("cat", 4) + entries("dog", 4)
    with isolated() as (client, root):
        r = client.post("/datasets/upload", content=zip_bytes(items),
                        headers=HEADERS)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["layout"] == "flat" and body["classes"] == ["cat", "dog"]

        job = wait_for(client, body["job_id"])
        assert job["status"] == "ready" and job["progress"] == 100
        assert job["result"]["total_images"] == 8

        ws = root / body["job_id"]
        assert (ws / "dataset/train/cat/0.jpg").is_file()
        assert not (ws / "input/uploaded_dataset.zip").exists()
        meta = json.loads((ws / "job.json").read_text())
        assert meta["status"] == "ready"
        assert meta["dataset"]["layout"] == "flat"

        # A READY job survives a restart untouched.
        worker.jobs.clear()
        worker.load_persisted_jobs()
        assert worker.jobs[body["job_id"]]["status"] == "ready"


def test_split_upload() -> None:
    items = (entries("train/cat", 3) + entries("train/dog", 3)
             + entries("val/cat", 2) + entries("val/dog", 2))
    with isolated() as (client, root):
        r = client.post("/datasets/upload", content=zip_bytes(items),
                        headers=HEADERS)
        assert r.status_code == 202, r.text
        assert wait_for(client, r.json()["job_id"])["status"] == "ready"


def test_invalid_layout_rejected_and_discarded() -> None:
    with isolated() as (client, root):
        r = client.post("/datasets/upload",
                        content=zip_bytes(entries("cat", 4)), headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "insufficient_data"
        assert str(root) not in r.text
        assert_nothing_left(root)


def test_not_a_zip_and_empty_body() -> None:
    with isolated() as (client, root):
        r = client.post("/datasets/upload", content=b"not a zip at all",
                        headers=HEADERS)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "unsupported_archive"
        assert_nothing_left(root)

        r = client.post("/datasets/upload", content=b"", headers=HEADERS)
        assert r.status_code == 400
        assert_nothing_left(root)


def test_oversize_rejected_by_content_length_and_by_stream() -> None:
    small = uc.UploadLimits(
        max_upload_bytes=10_000, max_uncompressed_bytes=10_000,
        max_entries=100, max_images=50, min_images_per_class=1)
    with isolated(small) as (client, root):
        # Declared size: rejected before any job is created.
        r = client.post("/datasets/upload", content=b"x" * 20_000,
                        headers=HEADERS)
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "upload_too_large"
        assert_nothing_left(root)

        # Chunked body with no Content-Length: rejected while streaming.
        def chunks():
            for _ in range(10):
                yield b"x" * 5_000

        r = client.post("/datasets/upload", content=chunks(),
                        headers=HEADERS)
        assert r.status_code == 413, r.text
        assert_nothing_left(root)


def test_background_failure_keeps_error_and_purges_data() -> None:
    items = (entries("cat", 1)
             + [("cat/bad1.jpg", b"junk"), ("cat/bad2.jpg", b"junk")]
             + entries("dog", 3))
    with isolated() as (client, root):
        r = client.post("/datasets/upload", content=zip_bytes(items),
                        headers=HEADERS)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        job = wait_for(client, job_id)
        assert job["status"] == "failed"
        assert job["result"]["error"]["code"] == "insufficient_data"
        assert str(root) not in json.dumps(job)

        ws = root / job_id
        assert not (ws / "dataset").exists()
        assert not (ws / "input/uploaded_dataset.zip").exists()
        assert not list(ws.glob(".extract-*"))
        meta = json.loads((ws / "job.json").read_text())
        assert meta["status"] == "failed" and meta["error"]


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ dataset route tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    run()