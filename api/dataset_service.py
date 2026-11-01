"""
Background processing of an uploaded dataset (Step 4f).

Processing flow:
    extract -> validate images -> persist metadata -> READY

On failure, dataset data is purged but job.json is retained
so the client can retrieve the error through GET /jobs/{id}.
"""

# Standard library
import logging
import shutil
from typing import Optional

# Local
from api import worker
from api.dataset_extract import extract_plan
from api.dataset_upload import DatasetPlan
from api.dataset_validate import merge_metadata, validate_images
from api.upload_config import LIMITS, UploadLimits, UploadValidationError
from api.workspace import (
    InvalidJobIdError,
    JobWorkspace,
    WorkspaceNotFoundError,
    delete_workspace,
    get_workspace,
)

logger = logging.getLogger(__name__)

SUMMARY_KEYS = (
    "layout",
    "splits",
    "classes",
    "counts",
    "total_images",
    "rejected_total",
    "rejected_by_reason",
    "ignored_total",
    "ignored_by_reason",
)


def _purge_data(workspace: JobWorkspace) -> None:
    """Delete extracted data, uploaded ZIP and temporary extraction files."""

    shutil.rmtree(workspace.dataset_dir, ignore_errors=True)

    for leftover in workspace.root.glob(".extract-*"):
        shutil.rmtree(leftover, ignore_errors=True)

    workspace.zip_path.unlink(missing_ok=True)
    workspace.zip_path.with_suffix(".part").unlink(missing_ok=True)


def _fail(job_id: str, workspace: JobWorkspace, payload: dict) -> None:
    """Purge dataset data and persist the processing failure."""

    _purge_data(workspace)

    worker.update_job(
        job_id,
        status="failed",
        message=payload["message"],
        error=payload["message"],
        result={"error": payload},
    )


def _safe_fail(job_id: str, workspace: JobWorkspace, payload: dict) -> None:
    """Attempt to persist a failure without crashing the worker."""

    try:
        _fail(job_id, workspace, payload)
    except Exception:
        logger.exception(
            "Could not record failure for job %s",
            job_id,
        )


def process_dataset(
    job_id: str,
    plan: DatasetPlan,
    limits: Optional[UploadLimits] = None,
) -> None:
    """
    Extract and validate an inspected dataset.

    This function runs in the worker executor and does not raise
    processing exceptions to the API request handler.
    """

    limits = limits or LIMITS

    try:
        workspace = get_workspace(job_id, worker.JOBS_ROOT)

    except (InvalidJobIdError, WorkspaceNotFoundError,WorkspaceMetadataError):
        logger.exception(
            "Workspace missing or invalid for job %s",
            job_id,
        )
        return

    try:
        worker.update_job(
            job_id,
            progress=10,
            message="Extracting dataset",
        )

        extract_plan(
            workspace.zip_path,
            plan,
            workspace.dataset_dir,
            limits,
        )

        worker.update_job(
            job_id,
            progress=40,
            message="Validating images",
        )

        def on_progress(done: int, total: int) -> None:
            progress = 40 + int(55 * done / max(total, 1))

            worker.update_job(
                job_id,
                progress=progress,
                message=f"Validating images ({done}/{total})",
            )

        result = validate_images(
            workspace.dataset_dir,
            plan,
            limits,
            progress=on_progress,
        )

        meta = merge_metadata(plan, result)

        workspace.update_meta(dataset=meta)

        # Keep the validated dataset, not the original ZIP.
        workspace.zip_path.unlink(missing_ok=True)

        worker.update_job(
            job_id,
            status="ready",
            progress=100,
            message="Dataset ready",
            result={
                key: meta[key]
                for key in SUMMARY_KEYS
                if key in meta
            },
            error=None,
        )

    except UploadValidationError as exc:
        _safe_fail(job_id, workspace, exc.to_dict())

    except Exception:
        logger.exception(
            "Dataset processing failed for job %s",
            job_id,
        )

        _safe_fail(
            job_id,
            workspace,
            {
                "code": "internal_error",
                "message": "Dataset processing failed unexpectedly.",
                "details": {},
            },
        )

def discard_job(job_id: str) -> None:
    """Remove a job created for an upload that failed before processing."""

    try:
        workspace = get_workspace(job_id, worker.JOBS_ROOT)

    except (InvalidJobIdError, WorkspaceNotFoundError):
        worker.jobs.pop(job_id, None)
        return

    try:
        # Mark the job inactive before deleting its workspace.
        worker.update_job(
            job_id,
            status="failed",
            message="Upload discarded",
            error="Upload was not accepted.",
        )

        delete_workspace(job_id, worker.JOBS_ROOT)

    except Exception:
        logger.exception(
            "Could not cleanly discard job %s",
            job_id,
        )
        shutil.rmtree(workspace.root, ignore_errors=True)

    finally:
        worker.jobs.pop(job_id, None)