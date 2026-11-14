import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.schemas import PipelineRequest, JobResponse, JobStatus
from api.worker import (
    jobs,
    submit,
    create_persistent_job,
    load_persisted_jobs,
    update_job,
)
from api.workspace import (
    delete_workspace,
    InvalidJobIdError,
    WorkspaceNotFoundError,
    WorkspaceMetadataError,
)
from api.services.pipeline import PipelineService
from api.dataset_service import discard_job
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["Jobs"])


@router.post("/pipeline/run", response_model=JobResponse)
async def run_pipeline(request: PipelineRequest):
    """
    Submit the pipeline as a background job.
    Returns a persistent job ID immediately.
    """
    job = create_persistent_job(message="Pipeline queued")
    job_id = job["job_id"]

    try:
        config = request.model_dump()
        svc = PipelineService(config)

        await submit(svc.run, job_id, jobs, request.split)

    except Exception as exc:
        # Record submission failures instead of leaving a pending job.

        update_job(
            job_id,
            status="failed",
            message="Failed to submit pipeline",
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to submit pipeline",
        ) from exc

    j = jobs[job_id]

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0,
        message="Pipeline queued",
        created_at=j["created_at"],
        updated_at=j["updated_at"],
    )



@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    if job_id not in jobs:
        try:
            load_persisted_jobs()
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Failed to load persisted jobs",
            )

    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    j = jobs[job_id]
    return JobResponse(**j)

@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] in ("pending", "queued", "running"):
        raise HTTPException(400, "Cannot delete a running job")
    await run_in_threadpool(discard_job, job_id)
    return {"message": "Job deleted"}


@router.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """Real-time progress — push updates every second."""
    await websocket.accept()
    if job_id not in jobs:
        load_persisted_jobs()

    if job_id not in jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return
    try:
        while True:
            j = jobs.get(job_id, {})
            await websocket.send_json({
                "job_id"  : job_id,
                "status"  : j.get("status"),
                "progress": j.get("progress", 0),
                "message" : j.get("message", ""),
            })
            if j.get("status") in ("completed", "failed","interrupted"):
                await websocket.send_json(
                    {"job_id": job_id,
                     "status": j["status"],
                     "result": j.get("result"),
                     "error" : j.get("error")}
                )
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass