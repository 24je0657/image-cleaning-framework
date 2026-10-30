"""
Async job management and persistent job-state helpers.

Owns the ThreadPoolExecutor and coordinates in-memory job state
with per-job workspace metadata.
"""

import asyncio
import logging

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from api.workspace import (
    create_workspace,
    get_workspace,
    JOBS_ROOT,
    InvalidJobIdError,
    WorkspaceMetadataError,
    WorkspaceNotFoundError,
)


logger = logging.getLogger(__name__)

executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)

# In-memory cache. Persistent metadata is stored in each job's job.json.
jobs: dict[str, dict] = {}


def _now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def make_job(job_id: str) -> dict:
    """Create an in-memory job record."""
    now = _now()

    return {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }


def create_persistent_job(message: str = "Queued") -> dict:
    """
    Create a workspace and register its job in memory.

    Returns the in-memory job record.
    """
    workspace = create_workspace(JOBS_ROOT)

    job = make_job(workspace.job_id)
    job["message"] = message

    try:
        workspace.update_meta(
            status=job["status"],
            progress=job["progress"],
            message=message,
            result=None,
            error=None,
        )
    except Exception:
        logger.exception(
            "Failed to initialize metadata for job %s",
            workspace.job_id,
        )
        raise

    jobs[workspace.job_id] = job

    logger.info("Registered persistent job %s", workspace.job_id)

    return job


def update_job(job_id: str, **fields) -> dict:
    """
    Update a job in memory and persist the changes to job.json.

    Persist metadata before updating the in-memory cache so a failed
    write does not leave the cache ahead of persistent state.
    """
    if job_id not in jobs:
        raise KeyError(f"Job {job_id} is not registered in memory.")

    current_job = jobs[job_id]
    updated_job = current_job.copy()
    updated_job.update(fields)

    now = _now()
    updated_job["updated_at"] = now

    workspace = get_workspace(job_id,JOBS_ROOT)

    persisted = {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in updated_job.items()
    }

    workspace.update_meta(**persisted)

    # Update the in-memory cache only after persistence succeeds.
    current_job.update(updated_job)

    return current_job



def load_persisted_jobs(recover_active: bool = False) -> int:
    """
    Load valid job metadata from disk into the in-memory cache.

    Active jobs from a previous application session are marked
    as interrupted because they cannot safely resume.

    Returns the number of jobs loaded.
    """
    if not JOBS_ROOT.is_dir():
        return 0

    loaded = 0

    for entry in JOBS_ROOT.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue

        try:
            workspace = get_workspace(entry.name, JOBS_ROOT)
            metadata = workspace.read_meta()

            created_at = datetime.fromisoformat(metadata["created_at"])
            updated_at = datetime.fromisoformat(
                metadata.get("updated_at", metadata["created_at"])
            )

            status = metadata["status"]

            # Jobs that were active before restart cannot safely resume.
            if recover_active and status in ("pending", "queued", "running"):
                status = "interrupted"

                workspace.update_meta(
                    status=status,
                    message="Job interrupted because the application restarted.",
                    error="Application restarted while the job was active.",
                )

                # Reload metadata to get the updated timestamp and values.
                metadata = workspace.read_meta()

                updated_at = datetime.fromisoformat(
                    metadata["updated_at"]
                )

            job = {
                "job_id": metadata["job_id"],
                "status": status,
                "progress": metadata.get("progress", 0),
                "message": metadata.get("message", ""),
                "created_at": created_at,
                "updated_at": updated_at,
                "result": metadata.get("result"),
                "error": metadata.get("error"),
            }

            # Do not overwrite an existing in-memory job.
            if job["job_id"] not in jobs:
                jobs[job["job_id"]] = job
                loaded += 1

        except (
            InvalidJobIdError,
            WorkspaceMetadataError,
            WorkspaceNotFoundError,
            KeyError,
            TypeError,
            ValueError,
        ):
            logger.warning(
                "Skipping invalid persisted job workspace: %s",
                entry.name,
                exc_info=True,
            )

    logger.info("Loaded %d persisted jobs", loaded)

    return loaded


async def submit(fn, *args):
    """Submit a blocking function to the thread pool."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(executor, fn, *args)


def shutdown():
    """Shut down the thread pool without waiting for pending work."""
    executor.shutdown(wait=False)
