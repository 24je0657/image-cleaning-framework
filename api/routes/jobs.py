import uuid
import asyncio
from fastapi          import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from api.schemas      import PipelineRequest, JobResponse, JobStatus
from api.worker       import jobs, executor, make_job, submit
from api.services.pipeline import PipelineService

router = APIRouter(tags=["Jobs"])


@router.post("/pipeline/run", response_model=JobResponse)
async def run_pipeline(request: PipelineRequest):
    """
    Submit full pipeline as async background job.
    Returns job_id immediately — poll /jobs/{id} for progress.
    """
    job_id   = str(uuid.uuid4())
    jobs[job_id] = make_job(job_id)

    config = request.model_dump()
    svc    = PipelineService(config)

    await submit(svc.run, job_id, jobs, request.split)

    j = jobs[job_id]
    return JobResponse(
        job_id     = job_id,
        status     = JobStatus.PENDING,
        progress   = 0,
        message    = "Pipeline queued",
        created_at = j["created_at"],
        updated_at = j["updated_at"],
    )


@router.get("/jobs", tags=["Jobs"])
async def list_jobs():
    return [
        {"job_id": jid, "status": j["status"],
         "progress": j["progress"], "message": j["message"]}
        for jid, j in jobs.items()
    ]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    j = jobs[job_id]
    return JobResponse(**j)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] == "running":
        raise HTTPException(400, "Cannot delete a running job")
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


@router.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """Real-time progress — push updates every second."""
    await websocket.accept()
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
            if j.get("status") in ("completed", "failed"):
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