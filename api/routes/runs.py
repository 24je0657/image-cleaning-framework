"""API routes for running image-cleaning jobs."""

import logging

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool

from api.schemas import JobRunRequest
from api.job_runner import (
    start_run,
    run_job,
    JobNotRunnableError,
    ServerBusyError,
)
from api.workspace import (
    InvalidJobIdError,
    WorkspaceNotFoundError,
    WorkspaceMetadataError,
)
from api.worker import executor


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Job Runs"])


@router.post(
    "/jobs/{job_id}/run",
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_uploaded_job(
    job_id: str,
    request: JobRunRequest,
):
    """Start the cleaning pipeline for an uploaded dataset."""

    try:
        workspace, cfg = await run_in_threadpool(
            start_run,
            job_id,
            request,
        )

    except InvalidJobIdError:
        raise HTTPException(
            status_code=400,
            detail="Invalid job ID.",
        )

    except WorkspaceNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    except WorkspaceMetadataError:
        raise HTTPException(
            status_code=500,
            detail="Could not read job metadata.",
        )

    except JobNotRunnableError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        )

    except ServerBusyError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        )

    try:
        executor.submit(run_job, job_id, cfg)

    except Exception:
        logger.exception("Could not submit pipeline for job %s", job_id)

        # The job was marked running, but execution never started.
        from api.job_runner import _RUN_SLOTS, _mark_failed

        _mark_failed(job_id, "Could not schedule pipeline execution.")
        _RUN_SLOTS.release()

        raise HTTPException(
            status_code=500,
            detail="Could not schedule pipeline execution.",
        )

    return {
        "job_id": job_id,
        "status": "running",
        "message": "Pipeline execution started.",
    }