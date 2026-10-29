"""
Per-job workspace management for the Image Cleaning Framework.

Each uploaded dataset receives an isolated workspace:

    jobs/<job_id>/
    ├── input/
    ├── dataset/
    ├── embeddings/
    ├── reports/
    ├── output/
    ├── final/
    └── job.json

The original uploaded ZIP and extracted dataset are kept separate.
This module manages paths and metadata; it does not run the pipeline.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

# Resolve the default root relative to the process working directory.
# Can be overridden for deployment or testing.
JOBS_ROOT = Path(
    os.environ.get("IMAGE_CLEANING_JOBS_ROOT", "jobs")
).resolve()

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# Jobs in these states must not be automatically deleted.
ACTIVE_STATUSES = {"pending", "queued", "running"}

# Initial implementation: protects metadata operations within this process.
# Multi-process deployments need a cross-process locking strategy.
_METADATA_LOCKS: dict[str, threading.RLock] = {}
_METADATA_LOCKS_GUARD = threading.Lock()


class InvalidJobIdError(ValueError):
    """Raised when a job ID is malformed or unsafe."""


class WorkspaceNotFoundError(FileNotFoundError):
    """Raised when a valid job ID has no workspace on disk."""


class WorkspaceMetadataError(RuntimeError):
    """Raised when workspace metadata is missing or invalid."""


class ActiveJobError(RuntimeError):
    """Raised when an active job cannot safely be deleted."""


def _now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _metadata_lock(job_id: str) -> threading.RLock:
    """Return a stable metadata lock for a job within this process."""
    with _METADATA_LOCKS_GUARD:
        if job_id not in _METADATA_LOCKS:
            _METADATA_LOCKS[job_id] = threading.RLock()
        return _METADATA_LOCKS[job_id]


@dataclass(frozen=True)
class JobWorkspace:
    """Paths and metadata operations for one isolated job."""

    job_id: str
    root: Path

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def zip_path(self) -> Path:
        return self.input_dir / "uploaded_dataset.zip"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def download_zip(self) -> Path:
        return self.root / "download.zip"

    @property
    def meta_path(self) -> Path:
        return self.root / "job.json"

    def ensure_dirs(self) -> None:
        """Create the required directories without deleting existing data."""
        for directory in (
            self.input_dir,
            self.dataset_dir,
            self.embeddings_dir,
            self.reports_dir,
            self.output_dir,
            self.final_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def read_meta(self) -> dict:
        """Read and validate persisted metadata."""
        with _metadata_lock(self.job_id):
            if not self.meta_path.is_file():
                raise WorkspaceMetadataError(
                    f"Job metadata missing for {self.job_id}."
                )

            try:
                meta = json.loads(
                    self.meta_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise WorkspaceMetadataError(
                    f"Unable to read metadata for {self.job_id}."
                ) from exc

            if not isinstance(meta, dict):
                raise WorkspaceMetadataError(
                    "Job metadata must be a JSON object."
                )

            if meta.get("job_id") != self.job_id:
                raise WorkspaceMetadataError(
                    "Metadata job_id does not match workspace."
                )

            if not isinstance(meta.get("status"), str):
                raise WorkspaceMetadataError(
                    "Metadata must contain a string status."
                )

            return meta

    def write_meta(self, meta: dict) -> None:
        """
        Atomically replace job metadata.

        A unique temporary file prevents concurrent writers from sharing
        the same temporary filename. The lock prevents concurrent metadata
        operations within this Python process.
        """
        if not isinstance(meta, dict):
            raise TypeError("meta must be a dictionary")

        if meta.get("job_id") != self.job_id:
            raise WorkspaceMetadataError(
                "Metadata job_id does not match workspace."
            )

        if not isinstance(meta.get("status"), str):
            raise WorkspaceMetadataError(
                "Metadata must contain a string status."
            )

        updated = {**meta, "updated_at": _now()}
        self.root.mkdir(parents=True, exist_ok=True)

        with _metadata_lock(self.job_id):
            fd, temp_name = tempfile.mkstemp(
                prefix=".job-",
                suffix=".json.tmp",
                dir=self.root,
            )
            temp_path = Path(temp_name)

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                    json.dump(updated, temp_file, indent=2)
                    temp_file.write("\n")
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                os.replace(temp_path, self.meta_path)

            finally:
                # Remove the temporary file if writing/replacement failed.
                temp_path.unlink(missing_ok=True)

    def update_meta(self, **fields) -> dict:
        """Update metadata fields and persist the result."""
        with _metadata_lock(self.job_id):
            meta = self.read_meta()
            meta.update(fields)
            self.write_meta(meta)

            logger.debug(
                "Updated job %s metadata fields: %s",
                self.job_id,
                sorted(fields),
            )
            return meta


def validate_job_id(job_id: str) -> str:
    """Accept only lowercase, 32-character hexadecimal job IDs."""
    if not isinstance(job_id, str):
        raise InvalidJobIdError("Invalid job ID.")

    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise InvalidJobIdError("Invalid job ID.")

    return job_id


def _resolve_jobs_root(jobs_root: Optional[Path] = None) -> Path:
    """Resolve the configured root or an explicitly supplied root."""
    if jobs_root is None:
        return JOBS_ROOT

    return Path(jobs_root).resolve()


def create_workspace(
    jobs_root: Optional[Path] = None,
) -> JobWorkspace:
    """Create a new workspace with a server-generated job ID."""
    root = _resolve_jobs_root(jobs_root)
    root.mkdir(parents=True, exist_ok=True)

    # mkdir(exist_ok=False) avoids silently reusing an existing workspace.
    for _ in range(5):
        job_id = uuid.uuid4().hex
        workspace_root = root / job_id

        try:
            workspace_root.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Unable to allocate a unique job ID.")

    workspace = JobWorkspace(job_id=job_id, root=workspace_root)

    try:
        workspace.ensure_dirs()
        workspace.write_meta(
            {
                "job_id": job_id,
                "status": "pending",
                "progress": 0,
                "message": "Job created",
                "created_at": _now(),
                "result": None,
                "error": None,
            }
        )
    except Exception:
        shutil.rmtree(workspace.root, ignore_errors=True)
        raise

    logger.info("Created workspace %s", job_id)
    return workspace


def get_workspace(
    job_id: str,
    jobs_root: Optional[Path] = None,
) -> JobWorkspace:
    """Validate an ID and retrieve its workspace."""
    validate_job_id(job_id)

    root = _resolve_jobs_root(jobs_root)
    workspace_root = root / job_id

    # Resolve the path to protect against unexpected symlinks.
    if not workspace_root.resolve().is_relative_to(root):
        raise InvalidJobIdError("Invalid job ID.")

    if workspace_root.is_symlink():
        raise InvalidJobIdError("Invalid job workspace.")

    if not workspace_root.is_dir():
        raise WorkspaceNotFoundError(
            f"No workspace exists for job {job_id}."
        )

    workspace = JobWorkspace(job_id=job_id, root=workspace_root)

    # Validate metadata before treating the directory as a valid job.
    workspace.read_meta()

    return workspace


def delete_workspace(
    job_id: str,
    jobs_root: Optional[Path] = None,
) -> None:
    """Delete a non-active workspace."""
    workspace = get_workspace(job_id, jobs_root)

    with _metadata_lock(job_id):
        meta = workspace.read_meta()

        if meta.get("status") in ACTIVE_STATUSES:
            raise ActiveJobError(
                f"Cannot delete active job {job_id}."
            )

        shutil.rmtree(workspace.root)

    logger.info("Deleted workspace %s", job_id)


def cleanup_expired(
    max_age_hours: float,
    jobs_root: Optional[Path] = None,
) -> list[str]:
    """
    Remove expired, inactive workspaces.

    Age is based on job.json modification time. Active jobs are skipped.
    Returns the IDs of removed workspaces.
    """
    if max_age_hours < 0:
        raise ValueError("max_age_hours must be non-negative.")

    root = _resolve_jobs_root(jobs_root)

    if not root.is_dir():
        return []

    cutoff = time.time() - max_age_hours * 3600
    deleted: list[str] = []

    for entry in root.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue

        if not JOB_ID_PATTERN.fullmatch(entry.name):
            continue

        job_id = entry.name
        workspace = JobWorkspace(job_id=job_id, root=entry)

        with _metadata_lock(job_id):
            try:
                meta = workspace.read_meta()
            except WorkspaceMetadataError:
                # Do not automatically delete malformed or unreadable jobs.
                logger.warning(
                    "Skipping workspace with invalid metadata: %s",
                    job_id,
                )
                continue

            if meta["status"] in ACTIVE_STATUSES:
                continue

            try:
                last_touched = workspace.meta_path.stat().st_mtime
            except OSError:
                continue

            if last_touched >= cutoff:
                continue

            # Recheck metadata immediately before deleting.
            latest_meta = workspace.read_meta()
            if latest_meta["status"] in ACTIVE_STATUSES:
                continue

            shutil.rmtree(workspace.root)
            deleted.append(job_id)

            logger.info("Expired workspace removed: %s", job_id)

    return deleted
