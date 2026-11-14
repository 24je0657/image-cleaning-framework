# tests/test_api_startup.py
# Standard library
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
from fastapi.testclient import TestClient  # noqa: E402

# Local
from api import main, maintenance, worker  # noqa: E402
from api.workspace import get_workspace  # noqa: E402


@contextmanager
def isolated_startup():
    """Temp jobs dir; no model loading; shared executor is never shut down."""
    saved = {
        "root": worker.JOBS_ROOT,
        "jobs": worker.jobs.copy(),
        "resnet": main.get_resnet,
        "autoencoder": main.get_autoencoder,
        "shutdown": main.shutdown,
        "sweeper_env": os.environ.get("IMAGE_CLEANING_SWEEPER"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        jobs_root = Path(tmp) / "jobs"
        jobs_root.mkdir()
        try:
            worker.JOBS_ROOT = jobs_root
            worker.jobs.clear()
            main.get_resnet = lambda: None
            main.get_autoencoder = lambda: None
            # TestClient shutdown would otherwise kill the global executor
            # and break every later test in the process.
            main.shutdown = lambda: None
            os.environ.pop("IMAGE_CLEANING_SWEEPER", None)
            maintenance.stop_sweeper()
            yield jobs_root
        finally:
            maintenance.stop_sweeper()
            worker.jobs.clear()
            worker.jobs.update(saved["jobs"])
            worker.JOBS_ROOT = saved["root"]
            main.get_resnet = saved["resnet"]
            main.get_autoencoder = saved["autoencoder"]
            main.shutdown = saved["shutdown"]
            if saved["sweeper_env"] is None:
                os.environ.pop("IMAGE_CLEANING_SWEEPER", None)
            else:
                os.environ["IMAGE_CLEANING_SWEEPER"] = saved["sweeper_env"]


def test_startup_recovers_jobs() -> None:
    with isolated_startup() as jobs_root:
        running_id = worker.create_persistent_job("Working")["job_id"]
        worker.update_job(running_id, status="running", progress=40)
        done_id = worker.create_persistent_job("Finished")["job_id"]
        worker.update_job(done_id, status="completed", progress=100,
                          result={"ok": True})

        worker.jobs.clear()                      # simulate a restart



def test_startup_starts_and_shutdown_stops_sweeper() -> None:
    with isolated_startup():
        with TestClient(main.app):
            assert maintenance._sweeper is not None
            assert maintenance._sweeper.alive
        assert not maintenance._sweeper.alive


def test_sweeper_disabled_by_env_is_not_started() -> None:
    with isolated_startup():
        os.environ["IMAGE_CLEANING_SWEEPER"] = "off"
        with TestClient(main.app):
            assert (maintenance._sweeper is None
                    or not maintenance._sweeper.alive)


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ API startup tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()