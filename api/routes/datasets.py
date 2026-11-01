"""
POST /datasets/upload: raw-body ZIP upload.
"""

# Standard library
import asyncio
import logging
import os
from pathlib import Path
import shutil

# Third-party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

# Local
from api import worker
from api.dataset_service import discard_job, process_dataset
from api.dataset_upload import inspect_zip
from api.upload_config import (
    LIMITS,
    MB,
    UnsupportedArchiveError,
    UploadTooLargeError,
    UploadValidationError,
)
from api.workspace import (
    InvalidJobIdError,
    JobWorkspace,
    WorkspaceNotFoundError,
    delete_workspace,
    get_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Datasets"])

def _error(status: int, payload: dict) -> JSONResponse:
    """Return a consistent upload error response."""
    return JSONResponse(
        status_code=status,
        content={"error": payload},
    )

def _too_large(limit_bytes: int) -> UploadTooLargeError:
    """Create a standard upload-too-large exception."""
    return UploadTooLargeError(
        f"Upload exceeds the {limit_bytes // MB:,} MB limit.",
        details={"limit_mb": limit_bytes // MB},
    )

async def _stream_to_disk(
    request: Request,
    dest: Path,
    max_bytes: int,
) -> int:
    """
    Stream the raw request body to disk in chunks.

    Stops immediately when the upload exceeds max_bytes.
    Does not load the complete ZIP into memory.
    """

    part = dest.with_suffix(".part")
    written = 0

    try:
        with open(part, "wb") as out:

            async for chunk in request.stream():

                if not chunk:
                    continue

                written += len(chunk)

                if written > max_bytes:
                    raise _too_large(max_bytes)

                await run_in_threadpool(out.write, chunk)

        if written == 0:
            raise UnsupportedArchiveError("The upload was empty.")

        os.replace(part, dest)

    except BaseException:
        part.unlink(missing_ok=True)
        raise

    return written

def discard_job(job_id: str) -> None:
    """Remove a job created for an upload that failed before processing."""

    try:
        workspace = get_workspace(job_id, worker.JOBS_ROOT)
    except (InvalidJobIdError, WorkspaceNotFoundError):
        worker.jobs.pop(job_id, None)
        return

    try:
        # Mark it inactive first because delete_workspace blocks active jobs.
        worker.update_job(
            job_id,
            status="failed",
            message="Upload discarded",
            error="Upload was not accepted.",
        )

        delete_workspace(job_id, worker.JOBS_ROOT)

    except Exception:
        logger.exception("Could not cleanly discard job %s", job_id)
        shutil.rmtree(workspace.root, ignore_errors=True)

    finally:
        worker.jobs.pop(job_id, None)

def _log_future_error(future) -> None:
    """Log unexpected background-task failures."""
    if future.cancelled():
        return

    exc = future.exception()
    if exc is not None:
        logger.error(
            "Background dataset task crashed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


@router.post("/datasets/upload", status_code=202)
async def upload_dataset(request: Request):
    """
    Upload a ZIP dataset using the raw request body.

    The endpoint streams and inspects the ZIP, then schedules
    extraction and image validation in the background.
    """
    limits = LIMITS

    # Reject oversized uploads early when Content-Length is available.
    declared = request.headers.get("content-length")

    if declared and declared.isdigit():
        if int(declared) > limits.max_upload_bytes:
            exc = _too_large(limits.max_upload_bytes)
            return _error(exc.http_status, exc.to_dict())

    loop = asyncio.get_running_loop()

    # Create a persistent job and its isolated workspace.
    job = worker.create_persistent_job("Receiving upload")
    job_id = job["job_id"]

    try:
        workspace = get_workspace(job_id, worker.JOBS_ROOT)

        worker.update_job(
            job_id,
            status="running",
            progress=1,
            message="Receiving upload",
        )

        # Stream the request body to the job's ZIP file.
        await _stream_to_disk(
            request,
            workspace.zip_path,
            limits.max_upload_bytes,
        )

        worker.update_job(
            job_id,
            progress=3,
            message="Inspecting archive",
        )

        # Inspect the ZIP without extracting it.
        plan = await loop.run_in_executor(
            worker.executor,
            inspect_zip,
            workspace.zip_path,
            limits,
        )

    except UploadValidationError as exc:
        discard_job(job_id)
        return _error(exc.http_status, exc.to_dict())

    except ClientDisconnect:
        discard_job(job_id)
        return _error(
            400,
            {
                "code": "client_disconnected",
                "message": "The upload was interrupted.",
                "details": {},
            },
        )

    except asyncio.CancelledError:
        discard_job(job_id)
        raise

    except Exception:
        logger.exception("Upload failed for job %s", job_id)
        discard_job(job_id)

        return _error(
            500,
            {
                "code": "internal_error",
                "message": "The upload could not be processed.",
                "details": {},
            },
        )

    # ZIP inspection succeeded. Schedule extraction and validation.
    worker.update_job(
        job_id,
        progress=5,
        message="Queued for extraction",
    )

    future = loop.run_in_executor(
        worker.executor,
        process_dataset,
        job_id,
        plan,
        limits,
    )
    future.add_done_callback(_log_future_error)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "running",
            "layout": plan.layout,
            "splits": plan.splits,
            "classes": plan.classes,
            "images_planned": plan.total_images,
            "ignored_total": plan.ignored_total,
            "status_url": f"/jobs/{job_id}",
        },
    )