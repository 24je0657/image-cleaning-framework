"""Tests for per-job workspace management."""

import os
import sys
import tempfile
import time

from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.workspace import (
    ActiveJobError,
    InvalidJobIdError,
    WorkspaceMetadataError,
    WorkspaceNotFoundError,
    cleanup_expired,
    create_workspace,
    delete_workspace,
    get_workspace,
)


def run() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        jobs_root = Path(temp_dir) / "jobs"

        # Creation and directory isolation.
        first = create_workspace(jobs_root)
        second = create_workspace(jobs_root)

        assert first.job_id != second.job_id
        assert len(first.job_id) == 32
        assert first.root != second.root

        for directory in (
            first.input_dir,
            first.dataset_dir,
            first.embeddings_dir,
            first.reports_dir,
            first.output_dir,
            first.final_dir,
        ):
            assert directory.is_dir(), directory

        assert first.read_meta()["status"] == "pending"

        # Lookup and persistent metadata.
        retrieved = get_workspace(first.job_id, jobs_root)
        assert retrieved.root == first.root

        retrieved.update_meta(status="running", progress=25)
        saved = get_workspace(first.job_id, jobs_root).read_meta()
        assert saved["status"] == "running"
        assert saved["progress"] == 25
        assert "updated_at" in saved

        # Reject malformed and hostile job IDs.
        invalid_ids = [
            "../etc",
            "..",
            "abc",
            first.job_id.upper(),
            f"{first.job_id}/../{second.job_id}",
            "",
            "a" * 31,
            None,
        ]

        for bad_id in invalid_ids:
            try:
                get_workspace(bad_id, jobs_root)
            except InvalidJobIdError:
                pass
            else:
                raise AssertionError(
                    f"Accepted invalid job ID: {bad_id!r}"
                )

        # Valid format but nonexistent job.
        try:
            get_workspace("0" * 32, jobs_root)
        except WorkspaceNotFoundError:
            pass
        else:
            raise AssertionError("Unknown job ID was accepted")

        # Explicit deletion must protect active jobs.
        try:
            delete_workspace(first.job_id, jobs_root)
        except ActiveJobError:
            pass
        else:
            raise AssertionError("Active job was deleted")

        # Corrupt metadata must be rejected.
        corrupt = jobs_root / ("c" * 32)
        corrupt.mkdir()
        (corrupt / "job.json").write_text(
            "{invalid json",
            encoding="utf-8",
        )

        try:
            get_workspace("c" * 32, jobs_root)
        except WorkspaceMetadataError:
            pass
        else:
            raise AssertionError("Corrupt metadata was accepted")

        # Expiry cleanup: retain running jobs and remove old completed jobs.
        old_timestamp = time.time() - 48 * 3600

        first.update_meta(status="running")
        os.utime(first.meta_path, (old_timestamp, old_timestamp))

        second.update_meta(status="completed")
        os.utime(second.meta_path, (old_timestamp, old_timestamp))

        deleted = cleanup_expired(24, jobs_root)

        assert second.job_id in deleted, deleted
        assert first.root.is_dir(), "Running job was deleted"
        assert not second.root.exists(), "Expired job was retained"
        assert corrupt.exists(), "Corrupt workspace should be preserved"

        # Invalid metadata cannot be written.
        try:
            first.write_meta(
                {
                    "job_id": "f" * 32,
                    "status": "completed",
                }
            )
        except WorkspaceMetadataError:
            pass
        else:
            raise AssertionError("Mismatched metadata ID was accepted")

        # Explicit deletion of a completed job.
        first.update_meta(status="completed")
        delete_workspace(first.job_id, jobs_root)
        assert not first.root.exists()

    print("Workspace tests passed.")


if __name__ == "__main__":
    run()
