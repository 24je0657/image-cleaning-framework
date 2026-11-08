# tests/test_job_downloads.py
# Standard library
import io
import json
import os
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Local
from test_job_results import approve, isolated, make_job, sha_tree  # noqa: E402
from api import download_service as dl  # noqa: E402
from api.job_locks import job_lock  # noqa: E402

FAKE_ID = "0" * 32


def get(client, job_id):
    return client.get(f"/jobs/{job_id}/download")


def zip_of(response) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(response.content))


def test_requires_approval_and_valid_job() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        r = get(client, job_id)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "download_not_available"
        running, _ = make_job(status="running")
        assert get(client, running).status_code == 409
        for bad in (FAKE_ID, "not-a-job"):
            assert get(client, bad).status_code == 404


def test_zip_contents_flat_and_no_server_paths() -> None:
    with isolated() as (client, root, tmp):
        job_id, ws = make_job()
        assert approve(client, job_id,
                       apply_proposed_removals=True).status_code == 200
        r = get(client, job_id)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/zip"
        assert r.headers["x-missing-files"] == "0"
        assert "attachment" in r.headers["content-disposition"]

        z = zip_of(r)
        assert z.testzip() is None
        final = sha_tree(ws.final_dir)
        assert sorted(z.namelist()) == sorted(
            [f"dataset/{k}" for k in final]
            + ["approval_decisions.csv", "summary.json"])
        for rel in final:
            assert z.read(f"dataset/{rel}") == (ws.final_dir / rel).read_bytes()
        assert not any(n.startswith("dataset/train/") for n in z.namelist())

        summary = json.loads(z.read("summary.json"))
        assert summary["decisions"]["kept"] == 6
        assert summary["layout"] == "flat"
        assert summary["checks_enabled"] is None       # no run settings saved
        for member in ("summary.json", "approval_decisions.csv"):
            text = z.read(member).decode()
            assert str(tmp) not in text and ws.job_id not in text


def test_split_layout_and_enabled_checks() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job(layout="split")
        ws.update_meta(run={"run_duplicates": True, "run_blur": False,
                            "run_noise": True, "run_outliers": True,
                            "run_mislabels": True})
        approve(client, job_id, apply_proposed_removals=True)
        z = zip_of(get(client, job_id))
        names = z.namelist()
        assert any(n.startswith("dataset/train/") for n in names)
        assert any(n.startswith("dataset/val/") for n in names)
        summary = json.loads(z.read("summary.json"))
        assert summary["checks_enabled"] == [
            "duplicates", "noise", "outliers", "mislabels"]


def test_cache_reuse_and_rebuild_after_reapproval() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        approve(client, job_id, apply_proposed_removals=True)
        first = get(client, job_id)
        before = ws.download_zip.stat()
        again = get(client, job_id)
        after = ws.download_zip.stat()
        assert (before.st_ino, before.st_mtime_ns) == (
            after.st_ino, after.st_mtime_ns), "ZIP was rebuilt needlessly"
        assert again.content == first.content

        approve(client, job_id, apply_proposed_removals=False)
        rebuilt = zip_of(get(client, job_id))
        assert len(rebuilt.namelist()) == 8 + 2       # 8 images, 2 metadata
        assert not list(ws.root.glob(".download-*"))


def test_missing_files_header() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        approve(client, job_id, apply_proposed_removals=True)
        ws.update_meta(approval={**ws.read_meta()["approval"],
                                 "missing_files": 2})
        assert get(client, job_id).headers["x-missing-files"] == "2"


def test_stale_approval_after_final_removed() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        approve(client, job_id, apply_proposed_removals=True)
        shutil.rmtree(ws.final_dir)
        ws.final_dir.mkdir()
        r = get(client, job_id)
        assert r.status_code == 409
        assert "again" in r.json()["error"]["message"]


def test_build_waits_for_job_lock() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        approve(client, job_id, apply_proposed_removals=True)
        out = []
        lock = job_lock(job_id)
        lock.acquire()
        thread = threading.Thread(
            target=lambda: out.append(dl.build_download(ws)))
        thread.start()
        time.sleep(0.3)
        assert out == []                               # blocked by the lock
        lock.release()
        thread.join(10)
        assert len(out) == 1


def test_approval_waits_for_job_lock() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        out = []
        lock = job_lock(job_id)
        lock.acquire()
        thread = threading.Thread(target=lambda: out.append(
            approve(client, job_id, apply_proposed_removals=True)))
        thread.start()
        time.sleep(0.3)
        assert out == []
        lock.release()
        thread.join(10)
        assert out and out[0].status_code == 200


def test_sweep_deletes_only_stale_inactive_downloads() -> None:
    with isolated() as (client, root, _):
        old, old_ws = make_job()
        fresh, fresh_ws = make_job()
        for job_id in (old, fresh):
            approve(client, job_id, apply_proposed_removals=True)
            assert get(client, job_id).status_code == 200
        busy, busy_ws = make_job(status="running")
        busy_ws.download_zip.write_bytes(b"x")

        stale = time.time() - 48 * 3600
        for ws in (old_ws, busy_ws):
            os.utime(ws.download_zip, (stale, stale))

        assert dl.sweep_downloads(24, root) == [old]
        assert not old_ws.download_zip.exists()
        assert not (old_ws.root / "download.stamp").exists()
        assert fresh_ws.download_zip.exists()
        assert busy_ws.download_zip.exists()           # active job untouched
        assert get(client, old).status_code == 200     # rebuilds on demand


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ job download tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()