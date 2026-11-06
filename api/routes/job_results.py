# api/routes/job_results.py
"""Job results review and approval endpoints (Step 6)."""
# Standard library
import logging
from pathlib import Path
from typing import Literal, Optional

# Third-party
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

# Local
from api import results_service as rs
from api import worker
from api.workspace import (
    InvalidJobIdError,
    WorkspaceMetadataError,
    WorkspaceNotFoundError,
    get_workspace,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Results"])

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png"}


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Required on purpose: removals are never applied by omission.
    apply_proposed_removals: bool
    review_default: Literal["keep", "remove"] = "keep"
    keep: list[str] = Field(default_factory=list, max_length=10_000)
    remove: list[str] = Field(default_factory=list, max_length=10_000)


def _error(status: int, code: str, message: str, details=None):
    return JSONResponse(status_code=status, content={
        "error": {"code": code, "message": message,
                  "details": details or {}}})


def _load(job_id: str):
    ws = get_workspace(job_id, worker.JOBS_ROOT)
    return ws, rs.get_results(ws)


async def _guarded(job_id: str, work):
    """Run blocking work off the event loop and map errors to responses."""
    try:
        return await run_in_threadpool(work)
    except (InvalidJobIdError, WorkspaceNotFoundError):
        return _error(404, "job_not_found", "Job not found.")
    except rs.ResultsError as exc:
        return JSONResponse(status_code=exc.http_status,
                            content={"error": exc.to_dict()})
    except WorkspaceMetadataError:
        return _error(500, "internal_error", "Could not read job data.")
    except Exception:  # noqa: BLE001
        logger.exception("results request failed for job %s", job_id)
        return _error(500, "internal_error", "The request failed.")


@router.get("/jobs/{job_id}/results")
async def job_results(job_id: str):
    def work():
        _, result = _load(job_id)
        return rs.summarize(result)
    return await _guarded(job_id, work)


@router.get("/jobs/{job_id}/images")
async def job_images(
    job_id: str,
    verdict: Optional[Literal["CLEAN", "REVIEW", "REMOVE"]] = None,
    split: Optional[str] = Query(None, max_length=64),
    class_name: Optional[str] = Query(None, alias="class", max_length=64),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    def work():
        _, result = _load(job_id)
        return rs.list_images(result, group=verdict, split=split,
                              class_name=class_name, limit=limit,
                              offset=offset)
    return await _guarded(job_id, work)


@router.get("/jobs/{job_id}/images/file")
async def job_image_file(job_id: str, path: str = Query(..., max_length=300)):
    def work():
        _, result = _load(job_id)
        rec = rs.get_record(result, path)
        suffix = Path(rec.path).suffix.lower()
        if (suffix not in MEDIA_TYPES or not rec.source.is_file()
                or rec.source.is_symlink()):
            raise rs.ImageNotFoundError("Image not found in this job.")
        return FileResponse(
            rec.source, media_type=MEDIA_TYPES[suffix],
            headers={"Cache-Control": "private, max-age=300",
                     "X-Content-Type-Options": "nosniff"})
    return await _guarded(job_id, work)


@router.post("/jobs/{job_id}/approve")
async def approve_job(job_id: str, body: ApprovalRequest):
    def work():
        ws, result = _load(job_id)
        stats = rs.approve(
            ws, result, keep=body.keep, remove=body.remove,
            review_default=body.review_default,
            apply_proposed_removals=body.apply_proposed_removals)
        return {"job_id": job_id, "status": "approved", **stats}
    return await _guarded(job_id, work)