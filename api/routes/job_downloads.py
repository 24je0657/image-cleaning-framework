# api/routes/job_downloads.py
"""GET /jobs/{job_id}/download (Step 7)."""
# Third-party
from fastapi import APIRouter
from fastapi.responses import FileResponse

# Local
from api import download_service as dl
from api import worker
from api.routes.job_results import _guarded
from api.workspace import get_workspace

router = APIRouter(tags=["Results"])


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str):
    def work():
        ws = get_workspace(job_id, worker.JOBS_ROOT)
        path, meta = dl.build_download(ws)
        missing = int((meta.get("approval") or {}).get("missing_files") or 0)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"cleaned_dataset_{job_id[:8]}.zip",
            headers={"X-Missing-Files": str(missing),
                     "Cache-Control": "private, no-store"})
    return await _guarded(job_id, work)