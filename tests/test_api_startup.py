import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api import main, worker
from api.workspace import get_workspace


def run():
    original_root = worker.JOBS_ROOT
    original_jobs = worker.jobs.copy()

    original_resnet = main.get_resnet
    original_autoencoder = main.get_autoencoder
    original_shutdown = main.shutdown

    with tempfile.TemporaryDirectory() as tmp:
        jobs_root = Path(tmp) / "jobs"
        jobs_root.mkdir()

        try:
            worker.JOBS_ROOT = jobs_root
            worker.jobs.clear()

            # Avoid loading actual deep learning models during this test.
            main.get_resnet = lambda: None
            main.get_autoencoder = lambda: None

            # Prevent TestClient shutdown from permanently shutting down
            # the global ThreadPoolExecutor.
            main.shutdown = lambda: None

            # Create a running job.
            running = worker.create_persistent_job("Working")
            running_id = running["job_id"]
            worker.update_job(
                running_id,
                status="running",
                progress=40,
            )

            # Create a completed job.
            done = worker.create_persistent_job("Finished")
            done_id = done["job_id"]
            worker.update_job(
                done_id,
                status="completed",
                progress=100,
                result={"ok": True},
            )

            # Simulate application restart.
            worker.jobs.clear()

            with TestClient(main.app) as client:

                # Running job should become interrupted.
                response = client.get(f"/jobs/{running_id}")
                assert response.status_code == 200, response.text

                body = response.json()
                assert body["status"] == "interrupted", body
                assert body["progress"] == 40

                # Completed job should remain completed.
                response = client.get(f"/jobs/{done_id}")
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "completed"

                # Verify persisted metadata.
                metadata = get_workspace(
                    running_id, jobs_root
                ).read_meta()

                assert metadata["status"] == "interrupted"

                # Verify both jobs appear in the listing.
                response = client.get("/jobs")
                assert response.status_code == 200

                listed = {
                    job["job_id"]: job["status"]
                    for job in response.json()
                }

                assert listed[running_id] == "interrupted"
                assert listed[done_id] == "completed"

            print("✅ API startup recovery test passed.")

        finally:
            worker.jobs.clear()
            worker.jobs.update(original_jobs)
            worker.JOBS_ROOT = original_root

            main.get_resnet = original_resnet
            main.get_autoencoder = original_autoencoder
            main.shutdown = original_shutdown


if __name__ == "__main__":
    run()