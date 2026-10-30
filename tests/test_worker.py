# tests/test_worker.py
# Standard library
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api import worker  # noqa: E402
from api.workspace import get_workspace  # noqa: E402


INTERRUPT_MESSAGE = "Job interrupted because the application restarted."
INTERRUPT_ERROR = "Application restarted while the job was active."


@contextmanager
def isolated_jobs_root():
    """
    Point the worker at a temporary jobs directory and restore the
    original JOBS_ROOT and in-memory jobs cache afterwards.
    """
    original_jobs_root = worker.JOBS_ROOT
    original_jobs = worker.jobs.copy()

    with tempfile.TemporaryDirectory() as temp_dir:
        jobs_root = Path(temp_dir) / "jobs"
        jobs_root.mkdir()

        try:
            worker.JOBS_ROOT = jobs_root
            worker.jobs.clear()
            yield jobs_root
        finally:
            worker.jobs.clear()
            worker.jobs.update(original_jobs)
            worker.JOBS_ROOT = original_jobs_root


def run_lifecycle_test() -> None:
    """Original test: create, update, restart, no-overwrite, complete."""
    with isolated_jobs_root() as jobs_root:
        # Workspace creation and in-memory registration.
        job = worker.create_persistent_job("Test job queued")
        job_id = job["job_id"]

        assert len(job_id) == 32
        assert job["status"] == "pending"
        assert job["message"] == "Test job queued"
        assert job_id in worker.jobs

        workspace = get_workspace(job_id, jobs_root)
        metadata = workspace.read_meta()

        assert metadata["job_id"] == job_id
        assert metadata["status"] == "pending"
        assert metadata["message"] == "Test job queued"

        # Persist progress and status changes.
        worker.update_job(
            job_id,
            status="running",
            progress=45,
            message="Processing images",
        )

        metadata = workspace.read_meta()

        assert metadata["status"] == "running"
        assert metadata["progress"] == 45
        assert metadata["message"] == "Processing images"
        assert metadata["updated_at"]

        # Simulate application restart by clearing the in-memory cache.
        worker.jobs.clear()

        loaded = worker.load_persisted_jobs(recover_active=True)

        assert loaded == 1
        assert job_id in worker.jobs

        restored = worker.jobs[job_id]

        assert restored["status"] == "interrupted"
        assert restored["progress"] == 45
        assert restored["message"] == INTERRUPT_MESSAGE
        assert restored["error"] == INTERRUPT_ERROR

        # The interruption must also be persisted to job.json.
        metadata = workspace.read_meta()

        assert metadata["status"] == "interrupted"
        assert metadata["progress"] == 45
        assert metadata["error"] == INTERRUPT_ERROR

        # Existing in-memory state must not be overwritten by stale disk data.
        restored["progress"] = 80
        restored["message"] = "Newer in-memory state"

        loaded_again = worker.load_persisted_jobs(recover_active=True)

        assert loaded_again == 0
        assert worker.jobs[job_id]["progress"] == 80
        assert worker.jobs[job_id]["message"] == "Newer in-memory state"

        # Move the persisted job to a completed state.
        worker.update_job(
            job_id,
            status="completed",
            progress=100,
            message="Test completed",
            result={"passed": True},
            error=None,
        )

        metadata = workspace.read_meta()

        assert metadata["status"] == "completed"
        assert metadata["progress"] == 100
        assert metadata["result"] == {"passed": True}
        assert metadata["error"] is None

    print("✅ Worker lifecycle test passed.")


def run_restart_scenarios_test() -> None:
    """
    Restart recovery for every job status.

    Active statuses (pending, running) become 'interrupted'.
    Terminal statuses (completed, failed, interrupted) must be
    preserved exactly, and their job.json must not be rewritten.
    """
    with isolated_jobs_root() as jobs_root:
        # ── Arrange: one job per scenario, in a known state ───────
        job_ids = {}

        # pending: status as created by create_persistent_job
        job_ids["pending"] = worker.create_persistent_job("Waiting")["job_id"]

        # queued: simulate a job waiting in the queue.
        job_ids["queued"] = worker.create_persistent_job("Starting")["job_id"]

        worker.update_job(
            job_ids["queued"],
            status="queued",
            message="Waiting in queue",
        )

        # running: mid-processing
        job_ids["running"] = worker.create_persistent_job("Starting")["job_id"]
        worker.update_job(
            job_ids["running"],
            status="running",
            progress=45,
            message="Processing images",
        )

        # completed: has a result and no error
        job_ids["completed"] = worker.create_persistent_job("Starting")["job_id"]
        worker.update_job(
            job_ids["completed"],
            status="completed",
            progress=100,
            message="Pipeline complete",
            result={"clean": 90, "remove": 5, "review": 5},
            error=None,
        )

        # failed: has its own error that must not be overwritten
        job_ids["failed"] = worker.create_persistent_job("Starting")["job_id"]
        worker.update_job(
            job_ids["failed"],
            status="failed",
            progress=30,
            message="Pipeline failed",
            error="Embedding extraction failed",
        )

        # interrupted: already interrupted, with its own custom text
        job_ids["interrupted"] = worker.create_persistent_job("Starting")["job_id"]
        worker.update_job(
            job_ids["interrupted"],
            status="interrupted",
            progress=60,
            message="Custom interrupted message",
            error="Custom interrupted error",
        )

        # Snapshot persisted metadata before the simulated restart.
        before = {
            name: get_workspace(job_id, jobs_root).read_meta()
            for name, job_id in job_ids.items()
        }

        # ── Act: simulate restart ─────────────────────────────────
        worker.jobs.clear()
        loaded = worker.load_persisted_jobs(recover_active=True)

        # ── Assert ────────────────────────────────────────────────
        assert loaded == 6, f"expected 6 jobs loaded, got {loaded}"
        for job_id in job_ids.values():
            assert job_id in worker.jobs

        expected_status = {
            "pending": "interrupted",
            "queued": "interrupted",
            "running": "interrupted",
            "completed": "completed",
            "failed": "failed",
            "interrupted": "interrupted",
        }

        for name, job_id in job_ids.items():
            restored = worker.jobs[job_id]
            persisted = get_workspace(job_id, jobs_root).read_meta()

            # Status mapping, in memory and on disk.
            assert restored["status"] == expected_status[name], name
            assert persisted["status"] == expected_status[name], name

            # In-memory timestamps are real datetimes.
            assert isinstance(restored["created_at"], datetime), name
            assert isinstance(restored["updated_at"], datetime), name

            # Progress is always preserved.
            assert restored["progress"] == before[name]["progress"], name

        # Active jobs get the standard interruption message and error.
        for name in ("pending","queued", "running"):
            restored = worker.jobs[job_ids[name]]
            assert restored["message"] == INTERRUPT_MESSAGE, name
            assert restored["error"] == INTERRUPT_ERROR, name

        # Running job keeps its mid-run progress.
        assert worker.jobs[job_ids["running"]]["progress"] == 45

        # Completed job: result and null error preserved.
        completed = worker.jobs[job_ids["completed"]]
        assert completed["result"] == {"clean": 90, "remove": 5, "review": 5}
        assert completed["error"] is None
        assert completed["message"] == "Pipeline complete"

        # Failed job: its own error is not replaced by the restart error.
        failed = worker.jobs[job_ids["failed"]]
        assert failed["error"] == "Embedding extraction failed"
        assert failed["message"] == "Pipeline failed"

        # Already-interrupted job: custom text is not overwritten.
        interrupted = worker.jobs[job_ids["interrupted"]]
        assert interrupted["message"] == "Custom interrupted message"
        assert interrupted["error"] == "Custom interrupted error"

        # Terminal jobs must not be rewritten on restart.
        for name in ("completed", "failed", "interrupted"):
            after = get_workspace(job_ids[name], jobs_root).read_meta()
            assert after == before[name], f"{name} job.json was rewritten"

        # A second restart is idempotent: nothing reloaded, nothing changed.
        snapshot = {jid: dict(job) for jid, job in worker.jobs.items()}
        assert worker.load_persisted_jobs(recover_active=True) == 0
        assert worker.jobs == snapshot

        # A third restart from an empty cache keeps the same final states.
        worker.jobs.clear()
        assert worker.load_persisted_jobs(recover_active=True) == 6
        for name, job_id in job_ids.items():
            assert worker.jobs[job_id]["status"] == expected_status[name], name

    print("✅ Worker restart-scenario tests passed (6 scenarios).")

def run_normal_loading_test():
    with isolated_jobs_root():
        # Create a new job
        job = worker.create_persistent_job("Normal load test")
        job_id = job["job_id"]

        # Make the job active
        worker.update_job(job_id, status="running", progress=40)

        # Simulate ordinary loading (NOT application restart)
        worker.jobs.clear()
        loaded = worker.load_persisted_jobs()

        assert loaded == 1
        assert worker.jobs[job_id]["status"] == "running"

        # Verify persisted metadata was not changed to interrupted
        metadata = get_workspace(job_id, worker.JOBS_ROOT).read_meta()
        assert metadata["status"] == "running"

        print("✅ Normal loading preserves active jobs.")


def run() -> None:
    run_lifecycle_test()
    run_restart_scenarios_test()
    print("Worker persistence tests passed.")


if __name__ == "__main__":
    run_lifecycle_test()
    run_restart_scenarios_test()
    run_normal_loading_test()
    print("Worker persistence tests passed.")