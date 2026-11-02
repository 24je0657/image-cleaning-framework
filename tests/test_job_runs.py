"""Tests for job-specific pipeline execution."""

import sys
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api import main, worker
from api import job_runner
from api.routes import runs


def isolated():
    """Create an isolated environment for job-run tests."""

    class TestEnvironment:
        def __enter__(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name) / "jobs"
            self.root.mkdir()

            self.saved = {
                "worker_root": worker.JOBS_ROOT,
                "runner_root": job_runner.JOBS_ROOT,
                "jobs": worker.jobs.copy(),
            }

            worker.JOBS_ROOT = self.root
            job_runner.JOBS_ROOT = self.root
            worker.jobs.clear()

            self.client = TestClient(main.app)
            return self.client, self.root

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.client.close()

            worker.jobs.clear()
            worker.jobs.update(self.saved["jobs"])

            worker.JOBS_ROOT = self.saved["worker_root"]
            job_runner.JOBS_ROOT = self.saved["runner_root"]

            self.tmp.cleanup()

    return TestEnvironment()




def create_ready_job(root: Path, job_id: str = "a" * 32) -> str:
    """Create a persisted READY job with a minimal valid dataset."""

    workspace = root / job_id

    # Create the required workspace directories.
    directories = [
        "input",
        "dataset/train/cat",
        "dataset/train/dog",
        "embeddings",
        "reports",
        "output",
        "final",
    ]

    for directory in directories:
        (workspace / directory).mkdir(parents=True, exist_ok=True)

    # Minimal files: _dataset_available() checks for files,
    # while pipeline execution will be mocked in these tests.
    (workspace / "dataset/train/cat/cat1.jpg").write_bytes(b"test")
    (workspace / "dataset/train/dog/dog1.jpg").write_bytes(b"test")

    metadata = {
        "job_id": job_id,
        "status": "ready",
        "progress": 100,
        "message": "Dataset ready",
        "created_at": "2026-11-02T10:00:00",
        "updated_at": "2026-11-02T10:00:00",
        "result": {
            "total_images": 2,
        },
        "error": None,
        "dataset": {
            "layout": "split",
            "splits": ["train"],
            "classes": ["cat", "dog"],
            "counts": {
                "train": {
                    "cat": 1,
                    "dog": 1,
                }
            },
            "total_images": 2,
        },
    }

    (workspace / "job.json").write_text(
        json.dumps(metadata, indent=2)
    )

    # Register the job in the worker's in-memory dictionary.
    worker.jobs[job_id] = {
        "job_id": job_id,
        "status": "ready",
        "progress": 100,
        "message": "Dataset ready",
        "created_at": metadata["created_at"],
        "updated_at": metadata["updated_at"],
        "result": metadata["result"],
        "error": None,
    }

    return job_id

def test_valid_job_submission():
    """A valid job should return 202 and become RUNNING."""

    original_submit = worker.executor.submit
    submitted = []

    def fake_submit(fn, *args, **kwargs):
        submitted.append((fn, args, kwargs))
        return None

    try:
        with isolated() as (client, root):
            job_id = create_ready_job(root)

            # Prevent actual pipeline execution.
            worker.executor.submit = fake_submit

            response = client.post(
                f"/jobs/{job_id}/run",
                json={},
            )

            assert response.status_code == 202, response.text

            body = response.json()

            assert body["job_id"] == job_id
            assert body["status"] == "running"
            assert len(submitted) == 1

            metadata = (
                root / job_id / "job.json"
            ).read_text()

            import json
            saved = json.loads(metadata)

            assert saved["status"] == "running"
            assert saved["progress"] == 5

            print("Valid submission: PASSED")

            # Restore the test job and release its claimed slot.
            worker.update_job(
                job_id,
                status="ready",
                progress=100,
                message="Dataset ready",
                error=None,
            )

            job_runner._RUN_SLOTS.release()

    finally:
        worker.executor.submit = original_submit

def test_unknown_job_returns_404():
    with isolated() as (client, root):
        response = client.post(
            f"/jobs/{'b' * 32}/run",
            json={},
        )

        assert response.status_code == 404
        print("Unknown job (404): PASSED")

def test_non_runnable_job_returns_409():
    with isolated() as (client, root):
        job_id = create_ready_job(root)

        # Change status to completed.
        worker.update_job(
            job_id,
            status="completed",
            progress=100,
            message="Pipeline completed",
        )

        response = client.post(
            f"/jobs/{job_id}/run",
            json={},
        )

        assert response.status_code == 409
        print("Non-runnable job (409): PASSED")

def test_invalid_request_returns_422():
    with isolated() as (client, root):
        job_id = create_ready_job(root)

        response = client.post(
            f"/jobs/{job_id}/run",
            json={
                "run_blur": False,
                "run_noise": False,
                "run_duplicates": False,
                "run_outliers": False,
                "run_mislabels": False,
            },
        )

        assert response.status_code == 422
        print("Invalid request (422): PASSED")

def test_busy_server_returns_429():
    with isolated() as (client, root):
        job_id = create_ready_job(root)

        # Occupy the only execution slot.
        acquired = job_runner._RUN_SLOTS.acquire(blocking=False)
        assert acquired

        try:
            response = client.post(
                f"/jobs/{job_id}/run",
                json={},
            )

            assert response.status_code == 429, response.text

            # The job must remain ready.
            metadata = (
                root / job_id / "job.json"
            ).read_text()

            import json
            saved = json.loads(metadata)

            assert saved["status"] == "ready"

            print("Busy server (429): PASSED")

        finally:
            job_runner._RUN_SLOTS.release()

def test_submission_failure_releases_slot():
    original_submit = worker.executor.submit

    try:
        with isolated() as (client, root):
            job_id = create_ready_job(root)

            def failing_submit(*args, **kwargs):
                raise RuntimeError("Simulated executor failure")

            worker.executor.submit = failing_submit

            response = client.post(
                f"/jobs/{job_id}/run",
                json={},
            )

            assert response.status_code == 500

            metadata = (
                root / job_id / "job.json"
            ).read_text()

            import json
            saved = json.loads(metadata)

            assert saved["status"] == "failed"
            assert saved["error"]

            # The slot should have been released.
            acquired = job_runner._RUN_SLOTS.acquire(
                blocking=False
            )
            assert acquired

            job_runner._RUN_SLOTS.release()

            print("Submission failure and slot release: PASSED")

    finally:
        worker.executor.submit = original_submit


def test_run_job_success():
    original_pipeline = job_runner.run_pipeline_config

    try:
        with isolated() as (client, root):
            job_id = create_ready_job(root)

            workspace, cfg = job_runner.start_run(
                job_id,
                job_runner.JobRunRequest(),
            )

            fake_result = {
                "success": True,
                "splits": ["train"],
                "summaries": {
                    "train": {
                        "success": True,
                        "steps": {
                            "duplicates": True,
                            "blur": True,
                            "noise": True,
                        },
                    }
                },
            }

            job_runner.run_pipeline_config = (
                lambda *args, **kwargs: fake_result
            )

            job_runner.run_job(job_id, cfg)

            metadata = workspace.read_meta()

            assert metadata["status"] == "completed"
            assert metadata["progress"] == 100
            assert metadata["pipeline_result"] == fake_result

            # Original upload result must remain unchanged.
            assert metadata["result"] == {"total_images": 2}

            # Uploaded dataset must still exist.
            assert workspace.dataset_dir.exists()

            print("Run job success: PASSED")

    finally:
        job_runner.run_pipeline_config = original_pipeline

def test_run_job_pipeline_failure():
    original_pipeline = job_runner.run_pipeline_config

    try:
        with isolated() as (client, root):
            job_id = create_ready_job(root)

            workspace, cfg = job_runner.start_run(
                job_id,
                job_runner.JobRunRequest(),
            )

            fake_result = {
                "success": False,
                "splits": ["train"],
                "summaries": {
                    "train": {
                        "success": False,
                        "steps": {
                            "duplicates": True,
                            "blur": False,
                            "noise": True,
                        },
                    }
                },
            }

            job_runner.run_pipeline_config = (
                lambda *args, **kwargs: fake_result
            )

            job_runner.run_job(job_id, cfg)

            metadata = workspace.read_meta()

            assert metadata["status"] == "failed"
            assert "train: blur" in metadata["error"]

            # Dataset and diagnostic reports must be preserved.
            assert workspace.dataset_dir.exists()
            assert workspace.reports_dir.exists()

            print("Run job pipeline failure: PASSED")

    finally:
        job_runner.run_pipeline_config = original_pipeline

def test_run_job_unexpected_exception():
    original_pipeline = job_runner.run_pipeline_config

    try:
        with isolated() as (client, root):
            job_id = create_ready_job(root)

            workspace, cfg = job_runner.start_run(
                job_id,
                job_runner.JobRunRequest(),
            )

            def failing_pipeline(*args, **kwargs):
                raise RuntimeError(
                    "Unexpected failure at /home/private/project"
                )

            job_runner.run_pipeline_config = failing_pipeline

            job_runner.run_job(job_id, cfg)

            metadata = workspace.read_meta()

            assert metadata["status"] == "failed"
            assert "RuntimeError" in metadata["error"]

            # Internal filesystem paths must not be exposed.
            assert "/home/private/project" not in metadata["error"]

            # Uploaded data must remain available.
            assert workspace.dataset_dir.exists()

            print("Run job unexpected exception: PASSED")

    finally:
        job_runner.run_pipeline_config = original_pipeline

def test_dataset_path_validation():
    # Safe directory names
    assert job_runner._is_safe_directory_name("train")
    assert job_runner._is_safe_directory_name("cat")
    assert job_runner._is_safe_directory_name("", allow_empty=True)

    # Unsafe directory names
    assert not job_runner._is_safe_directory_name("../outside")
    assert not job_runner._is_safe_directory_name("..")
    assert not job_runner._is_safe_directory_name(".")
    assert not job_runner._is_safe_directory_name("train/cat")
    assert not job_runner._is_safe_directory_name("train\\cat")

    # Empty class names must always be rejected.
    assert not job_runner._is_safe_directory_name("")

    print("Dataset path validation: PASSED")

def test_dataset_available_rejects_unsafe_metadata():
    with isolated() as (client, root):
        job_id = create_ready_job(root)

        workspace = job_runner.get_workspace(job_id, root)
        metadata = workspace.read_meta()

        # Malicious split name
        metadata["dataset"]["splits"] = ["../outside"]
        assert not job_runner._dataset_available(workspace, metadata)

        # Restore split and test malicious class name.
        metadata["dataset"]["splits"] = ["train"]
        metadata["dataset"]["classes"] = ["../outside"]
        assert not job_runner._dataset_available(workspace, metadata)

        print("Unsafe dataset metadata rejection: PASSED")

def test_dataset_available_supports_both_layouts():
    with isolated() as (client, root):
        # Split layout
        job_id = create_ready_job(root)
        workspace = job_runner.get_workspace(job_id, root)
        metadata = workspace.read_meta()

        assert job_runner._dataset_available(workspace, metadata)

        # Flat layout: dataset/class/image
        flat_job_id = "b" * 32
        flat_workspace = root / flat_job_id

        (flat_workspace / "dataset/cat").mkdir(
            parents=True, exist_ok=True
        )
        (flat_workspace / "dataset/dog").mkdir(
            parents=True, exist_ok=True
        )

        (flat_workspace / "dataset/cat/cat1.jpg").write_bytes(b"test")
        (flat_workspace / "dataset/dog/dog1.jpg").write_bytes(b"test")

        flat_metadata = {
            "dataset": {
                "layout": "flat",
                "splits": [""],
                "classes": ["cat", "dog"],
            }
        }

        flat_workspace_obj = job_runner.get_workspace(
            flat_job_id, root
        ) if (flat_workspace / "job.json").exists() else None

        # Create the minimal job metadata required by get_workspace.
        if flat_workspace_obj is None:
            import json

            (flat_workspace / "job.json").write_text(
                json.dumps({
                    "job_id": flat_job_id,
                    "status": "ready",
                    **flat_metadata,
                })
            )

            flat_workspace_obj = job_runner.get_workspace(
                flat_job_id, root
            )

        assert job_runner._dataset_available(
            flat_workspace_obj, flat_metadata
        )

        print("Split and flat layouts: PASSED")

def test_start_run_releases_slot_on_cleanup_failure():
    original_reset = job_runner._reset_outputs

    try:
        with isolated() as (client, root):
            job_id = create_ready_job(root)

            def failing_reset(workspace):
                raise OSError("Simulated cleanup failure")

            job_runner._reset_outputs = failing_reset

            try:
                job_runner.start_run(
                    job_id,
                    job_runner.JobRunRequest(),
                )
                raise AssertionError("Expected cleanup failure")
            except OSError:
                pass

            metadata = (
                root / job_id / "job.json"
            ).read_text()

            import json
            saved = json.loads(metadata)

            assert saved["status"] == "ready"

            # Verify the execution slot was released.
            acquired = job_runner._RUN_SLOTS.acquire(
                blocking=False
            )
            assert acquired

            job_runner._RUN_SLOTS.release()

            print("Cleanup failure releases slot: PASSED")

    finally:
        job_runner._reset_outputs = original_reset

def test_failed_job_can_be_retried():
    with isolated() as (client, root):
        job_id = create_ready_job(root)

        workspace = job_runner.get_workspace(job_id, root)

        # Simulate a previous failed run.
        worker.update_job(
            job_id,
            status="failed",
            progress=5,
            message="Pipeline failed",
            error="Previous pipeline failure",
        )

        # Create stale generated artifacts.
        (workspace.embeddings_dir / "old.npy").write_bytes(b"old")
        (workspace.reports_dir / "old.csv").write_text("old report")
        (workspace.output_dir / "old.json").write_text("{}")

        # Preserve original dataset and upload result.
        original_image = workspace.dataset_dir / "train/cat/cat1.jpg"
        original_result = workspace.read_meta()["result"]

        cfg_workspace, cfg = job_runner.start_run(
            job_id,
            job_runner.JobRunRequest(),
        )

        metadata = workspace.read_meta()

        assert metadata["status"] == "running"
        assert metadata["error"] is None

        # Old generated artifacts should be removed.
        assert not (workspace.embeddings_dir / "old.npy").exists()
        assert not (workspace.reports_dir / "old.csv").exists()
        assert not (workspace.output_dir / "old.json").exists()

        # Uploaded dataset and original result must remain.
        assert original_image.exists()
        assert metadata["result"] == original_result

        # Release the slot because this test only starts the run.
        job_runner._RUN_SLOTS.release()

        print("Failed job retry: PASSED")
def test_interrupted_job_can_be_restarted():
    with isolated() as (client, root):
        job_id = create_ready_job(root)

        worker.update_job(
            job_id,
            status="interrupted",
            progress=5,
            message="Interrupted after restart",
            error="Previous process stopped",
        )

        workspace, cfg = job_runner.start_run(
            job_id,
            job_runner.JobRunRequest(),
        )

        metadata = workspace.read_meta()

        assert metadata["status"] == "running"
        assert metadata["error"] is None
        assert workspace.dataset_dir.exists()

        # This test starts but does not execute the pipeline.
        job_runner._RUN_SLOTS.release()

        print("Interrupted job restart: PASSED")

def run() -> None:
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]

    for test in tests:
        test()

    print(f"✅ job run tests passed ({len(tests)} tests).")

if __name__ == "__main__":
    run()