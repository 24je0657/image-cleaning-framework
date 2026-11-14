# tests/test_maintenance.py
# Standard library
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Local
from test_job_results import isolated, make_job  # noqa: E402
from api import maintenance as mt  # noqa: E402
from api import upload_config as uc  # noqa: E402
from api import worker  # noqa: E402

MB = uc.MB


def expect_config_error(env, fragment) -> None:
    try:
        mt.load_settings(env)
    except uc.UploadConfigError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted {env}")


def test_settings_defaults_overrides_and_validation() -> None:
    d = mt.load_settings({})
    assert (d.retention_hours, d.download_retention_hours,
            d.sweep_interval_minutes, d.min_free_bytes) == (
        24.0, 6.0, 30.0, 3072 * MB)
    s = mt.load_settings({"IMAGE_CLEANING_RETENTION_HOURS": "48",
                          "IMAGE_CLEANING_MIN_FREE_MB": "100",
                          "IMAGE_CLEANING_SWEEP_INTERVAL_MINUTES": ""})
    assert s.retention_hours == 48.0 and s.min_free_bytes == 100 * MB
    assert s.sweep_interval_minutes == 30.0
    p = "IMAGE_CLEANING_"
    expect_config_error({p + "RETENTION_HOURS": "abc"}, "RETENTION_HOURS")
    expect_config_error({p + "RETENTION_HOURS": "0"}, "RETENTION_HOURS")
    expect_config_error({p + "RETENTION_HOURS": "nan"}, "finite")
    expect_config_error({p + "MIN_FREE_MB": "-5"}, "MIN_FREE_MB")
    expect_config_error({p + "RETENTION_HOURS": "2",
                         p + "DOWNLOAD_RETENTION_HOURS": "5"}, "must not exceed")


def test_required_space_and_boundary() -> None:
    limits = uc.UploadLimits(min_images_per_class=2)
    settings = mt.MaintenanceSettings()
    assert mt.required_free_bytes(limits, settings) == 3072 * MB
    big = uc.UploadLimits(max_upload_bytes=2000 * MB,
                          max_uncompressed_bytes=4000 * MB,
                          min_images_per_class=2)
    assert mt.required_free_bytes(big, settings) == 6000 * MB

    need = mt.required_free_bytes(limits, settings)
    original = mt.free_bytes
    try:
        mt.free_bytes = lambda path: need
        mt.ensure_free_space("/anywhere", limits, settings)       # exact: ok
        mt.free_bytes = lambda path: need - 1
        try:
            mt.ensure_free_space("/anywhere", limits, settings)
        except mt.StorageLowError as exc:
            assert exc.http_status == 503 and exc.code == "server_storage_low"
            assert "/anywhere" not in exc.message
            assert not any(ch.isdigit() for ch in exc.message)
        else:
            raise AssertionError("low space accepted")
    finally:
        mt.free_bytes = original


def test_upload_refused_when_storage_low() -> None:
    with isolated() as (client, root, _):
        original = mt.free_bytes
        mt.free_bytes = lambda path: 0
        try:
            r = client.post("/datasets/upload", content=b"PK\x03\x04",
                            headers={"content-type": "application/zip"})
        finally:
            mt.free_bytes = original
        assert r.status_code == 503, r.text
        assert r.json()["error"]["code"] == "server_storage_low"
        assert list(root.iterdir()) == [] and worker.jobs == {}


def test_sweep_removes_only_stale_inactive_data() -> None:
    settings = mt.MaintenanceSettings(retention_hours=24,
                                      download_retention_hours=6)
    with isolated() as (client, root, _):
        old, old_ws = make_job()
        fresh, fresh_ws = make_job()
        running, running_ws = make_job(status="running")
        stale = time.time() - 48 * 3600
        for ws in (old_ws, running_ws):
            os.utime(ws.meta_path, (stale, stale))
        fresh_ws.download_zip.write_bytes(b"zip")
        ten_hours = time.time() - 10 * 3600
        os.utime(fresh_ws.download_zip, (ten_hours, ten_hours))

        result = mt.run_sweep(root, settings)

        assert result["expired_jobs"] == [old]
        assert not old_ws.root.exists() and old not in worker.jobs
        assert fresh_ws.root.is_dir() and fresh in worker.jobs
        assert running_ws.root.is_dir() and running in worker.jobs
        assert result["stale_downloads"] == [fresh]
        assert not fresh_ws.download_zip.exists()
        assert fresh_ws.root.is_dir()                  # job itself kept
        assert client.get(f"/jobs/{old}").status_code == 404
        assert mt.run_sweep(root, settings) == {
            "expired_jobs": [], "stale_downloads": []}  # idempotent


def test_sweeper_thread_runs_survives_errors_and_stops() -> None:
    calls = []

    def task():
        calls.append(time.time())
        if len(calls) == 1:
            raise RuntimeError("first run fails")

    sweeper = mt.Sweeper(0.05, task)
    sweeper.start()
    sweeper.start()                                    # idempotent
    deadline = time.time() + 5
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert len(calls) >= 3, "loop died after an exception"
    sweeper.stop()
    assert not sweeper.alive
    count = len(calls)
    time.sleep(0.2)
    assert len(calls) == count, "ran after stop"


def test_sweeper_can_be_disabled_by_env() -> None:
    os.environ["IMAGE_CLEANING_SWEEPER"] = "off"
    try:
        assert mt.start_sweeper() is False
    finally:
        del os.environ["IMAGE_CLEANING_SWEEPER"]


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ maintenance tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()