# api/main.py
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib            import Path
from typing             import Optional
import pandas           as pd
import numpy            as np

from fastapi                    import (FastAPI, BackgroundTasks,
                                        UploadFile, File,
                                        HTTPException, WebSocket,
                                        WebSocketDisconnect)
from fastapi.middleware.cors    import CORSMiddleware
from fastapi.responses          import FileResponse, JSONResponse

from api.schemas      import (PipelineRequest, SingleImageRequest,
                               ExportRequest, JobResponse,
                               HealthResponse, JobStatus)
from api.dependencies import get_resnet, get_autoencoder
from api.worker       import analyze_single_image, run_full_pipeline

import os, shutil

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Image Cleaning Framework API",
    description = "Production API for automated image dataset cleaning "
                  "using deep learning — ResNet50, Autoencoder, "
                  "Isolation Forest, and ensemble mislabel detection.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
# In production: replace with Redis
jobs: dict = {}

# Thread pool for CPU-bound tasks (model inference, OpenCV, sklearn)
# Never run these with await directly — they block the event loop
executor = ThreadPoolExecutor(max_workers=4)


# ─── STARTUP / SHUTDOWN ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """
    Pre-load models at startup.
    This means the FIRST request is fast,
    not slow from cold-loading ResNet50.
    """
    print("🚀 API starting — pre-loading models...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, get_resnet)
    await loop.run_in_executor(executor, get_autoencoder)
    print("✅ Models ready")


@app.on_event("shutdown")
async def shutdown():
    executor.shutdown(wait=False)
    print("API shut down cleanly")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint.
    Call this to verify API + models are ready before sending requests.
    """
    return HealthResponse(
        status        = "ok",
        version       = "1.0.0",
        models_loaded = Path("models/autoencoder.pth").exists(),
        reports_ready = {
            "train_master" : Path("reports/train_master_report.csv").exists(),
            "val_master"   : Path("reports/val_master_report.csv").exists(),
            "train_embeddings": Path("embeddings/train_embeddings.npy").exists(),
        },
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/stats/{split}", tags=["Reports"])
async def get_stats(split: str = "train"):
    """
    Returns cleaning summary stats for a given split.
    Reads from precomputed master report — fast, no model inference.
    """
    path = f"reports/{split}_master_report.csv"
    if not Path(path).exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"No master report found for '{split}'. "
                          f"Run the full pipeline first."
        )

    loop   = asyncio.get_event_loop()
    # Read CSV in executor so we don't block event loop on large files
    df     = await loop.run_in_executor(
        executor, pd.read_csv, path
    )

    def compute_stats(df):
        total  = len(df)
        return {
            "split"         : split,
            "total_images"  : total,
            "clean"         : int((df["verdict"] == "CLEAN").sum()),
            "to_remove"     : int(df["verdict"].str.startswith("REMOVE").sum()),
            "to_review"     : int(df["verdict"].str.startswith("REVIEW").sum()),
            "issue_breakdown": {
                "exact_duplicates": int(df.get("exact_duplicate",
                    pd.Series([False]*total)).astype(bool).sum()),
                "near_duplicates" : int(df.get("near_duplicate",
                    pd.Series([False]*total)).astype(bool).sum()),
                "blurry"          : int(df.get("is_blurry",
                    pd.Series([False]*total)).astype(bool).sum()),
                "noisy"           : int(df.get("is_noisy",
                    pd.Series([False]*total)).astype(bool).sum()),
                "outliers"        : int(df.get("is_outlier",
                    pd.Series([False]*total)).astype(bool).sum()),
                "mislabeled"      : int(df.get("is_mislabeled",
                    pd.Series([False]*total)).astype(bool).sum()),
            },
            "per_class": {
                cls: {
                    "total"  : int(len(df[df["class"]==cls])),
                    "clean"  : int((df[df["class"]==cls]["verdict"]=="CLEAN").sum()),
                    "remove" : int(df[df["class"]==cls]["verdict"]
                                   .str.startswith("REMOVE").sum()),
                    "review" : int(df[df["class"]==cls]["verdict"]
                                   .str.startswith("REVIEW").sum()),
                }
                for cls in ["cat","dog","elephant","horse","lion"]
                if cls in df["class"].values
            },
            "verdict_distribution": df["verdict"].value_counts().to_dict()
        }

    stats = await loop.run_in_executor(executor, compute_stats, df)
    return JSONResponse(content=stats)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analyze/image", tags=["Analysis"])
async def analyze_image(request: SingleImageRequest):
    """
    Analyze a single image synchronously.
    Fast enough for single images — returns result immediately.
    Use /analyze/batch for multiple images.
    """
    if not Path(request.image_path).exists():
        raise HTTPException(
            status_code = 404,
            detail      = f"Image not found: {request.image_path}"
        )

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        analyze_single_image,
        request.image_path
    )
    return JSONResponse(content=result)


@app.post("/analyze/upload", tags=["Analysis"])
async def analyze_uploaded_image(file: UploadFile = File(...)):
    """
    Upload an image directly and analyze it immediately.
    Useful for demo/testing without needing local file paths.
    """
    # Save temporarily
    tmp_path = Path(f"/tmp/{uuid.uuid4()}_{file.filename}")
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            analyze_single_image,
            str(tmp_path)
        )
        result["original_filename"] = file.filename
        return JSONResponse(content=result)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()   # always clean up temp file


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE — ASYNC JOB
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/pipeline/run", response_model=JobResponse, tags=["Pipeline"])
async def run_pipeline(
    request          : PipelineRequest,
    background_tasks : BackgroundTasks
):
    """
    Submit a full pipeline run as a background job.

    Returns a job_id immediately.
    Poll GET /jobs/{job_id} to check progress.
    Connect to WS /ws/{job_id} for real-time updates.

    This is the KEY async pattern:
    - Request returns in <1ms with a job_id
    - Heavy work (ResNet50 inference on 13k images) runs in background
    - Client polls or uses WebSocket for live progress
    """
    job_id = str(uuid.uuid4())
    now    = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    jobs[job_id] = {
        "job_id"     : job_id,
        "status"     : JobStatus.PENDING,
        "progress"   : 0,
        "message"    : "Job queued",
        "created_at" : now,
        "updated_at" : now,
        "result"     : None,
        "error"      : None,
    }

    config = {
        "run_duplicates"   : request.run_duplicates,
        "run_blur"         : request.run_blur,
        "run_noise"        : request.run_noise,
        "run_outliers"     : request.run_outliers,
        "run_mislabels"    : request.run_mislabels,
        "blur_threshold"   : request.blur_threshold,
        "cosine_threshold" : request.cosine_threshold,
        "contamination"    : request.contamination,
    }

    # ← THIS IS THE KEY PATTERN
    # run_full_pipeline is CPU-bound (PyTorch, OpenCV, sklearn)
    # We run it in ThreadPoolExecutor so it never blocks the event loop
    # FastAPI can still serve other requests while this runs
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_full_pipeline,
        job_id, jobs, request.split, config
    )

    return JobResponse(
        job_id     = job_id,
        status     = JobStatus.PENDING,
        progress   = 0,
        message    = "Pipeline job queued successfully",
        created_at = now,
        updated_at = now
    )


# ─────────────────────────────────────────────────────────────────────────────
# JOB STATUS POLLING
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(job_id: str):
    """
    Poll job status and progress.
    Call every 2–5 seconds until status = 'completed' or 'failed'.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404,
                            detail=f"Job {job_id} not found")
    j = jobs[job_id]
    return JobResponse(
        job_id     = j["job_id"],
        status     = j["status"],
        progress   = j["progress"],
        message    = j["message"],
        created_at = j["created_at"],
        updated_at = j["updated_at"],
        result     = j.get("result"),
        error      = j.get("error")
    )


@app.get("/jobs", tags=["Jobs"])
async def list_jobs():
    """List all jobs and their current status."""
    return [
        {
            "job_id"    : jid,
            "status"    : j["status"],
            "progress"  : j["progress"],
            "message"   : j["message"],
            "created_at": j["created_at"],
        }
        for jid, j in jobs.items()
    ]


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Remove a completed job from the store."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] == "running":
        raise HTTPException(status_code=400,
                            detail="Cannot delete a running job")
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — REAL-TIME PROGRESS
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """
    Real-time job progress via WebSocket.
    Connect immediately after submitting a pipeline job.
    Receives JSON updates every second until job completes.

    Client usage:
        const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);
        ws.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    await websocket.accept()

    if job_id not in jobs:
        await websocket.send_json({"error": f"Job {job_id} not found"})
        await websocket.close()
        return

    try:
        while True:
            j = jobs.get(job_id, {})
            await websocket.send_json({
                "job_id"   : job_id,
                "status"   : j.get("status"),
                "progress" : j.get("progress", 0),
                "message"  : j.get("message", ""),
            })

            # Close WS when job finishes
            if j.get("status") in ("completed", "failed"):
                if j.get("result"):
                    await websocket.send_json({
                        "job_id" : job_id,
                        "status" : "completed",
                        "result" : j["result"]
                    })
                break

            await asyncio.sleep(1)   # push updates every second

    except WebSocketDisconnect:
        pass   # client disconnected — that's fine


# ─────────────────────────────────────────────────────────────────────────────
# REPORTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/reports/{split}/{report_type}", tags=["Reports"])
async def download_report(split: str, report_type: str):
    """
    Download any generated CSV report directly.
    report_type: master | duplicates | blur | noise | outliers | mislabel
    """
    path = Path(f"reports/{split}_{report_type}_report.csv")
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Report not found: {path}")
    return FileResponse(
        path         = str(path),
        filename     = path.name,
        media_type   = "text/csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/export", tags=["Export"])
async def export_clean_dataset(
    request          : ExportRequest,
    background_tasks : BackgroundTasks
):
    """
    Export cleaned dataset based on filter config.
    Runs as background job — returns job_id.
    """
    path = f"reports/{request.split}_master_report.csv"
    if not Path(path).exists():
        raise HTTPException(status_code=404,
                            detail="Run pipeline first")

    job_id = str(uuid.uuid4())
    now    = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    jobs[job_id] = {
        "job_id"     : job_id,
        "status"     : "pending",
        "progress"   : 0,
        "message"    : "Export queued",
        "created_at" : now,
        "updated_at" : now,
        "result"     : None,
        "error"      : None,
    }

    def do_export():
        import shutil
        try:
            df      = pd.read_csv(path)
            mask    = pd.Series(False, index=df.index)

            if request.remove_duplicates:
                mask |= df.get("exact_duplicate", False).astype(bool)
                mask |= df.get("near_duplicate",  False).astype(bool)
            if request.remove_blurry:
                mask |= df.get("is_blurry", False).astype(bool)
            if request.remove_noisy:
                mask |= df.get("is_noisy",  False).astype(bool)
            if request.remove_outliers:
                mask |= df.get("is_outlier",False).astype(bool)
            if request.remove_mislabeled:
                mask |= (
                    df.get("is_mislabeled", False).astype(bool) &
                    (df.get("mislabel_confidence","") == "HIGH")
                )
            mask |= df["priority_score"] >= request.min_priority_score

            keep_df    = df[~mask]
            clean_root = Path("data_clean") / request.split
            copied = 0

            for i, (_, row) in enumerate(keep_df.iterrows()):
                src = Path(row["file_path"])
                dst = clean_root / row["class"] / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except:
                    pass
                jobs[job_id]["progress"] = int(100 * i / len(keep_df))

            jobs[job_id].update({
                "status"  : "completed",
                "progress": 100,
                "message" : f"Exported {copied} images",
                "result"  : {
                    "copied"       : copied,
                    "removed"      : int(mask.sum()),
                    "output_path"  : f"data_clean/{request.split}/"
                }
            })
        except Exception as e:
            jobs[job_id].update({
                "status" : "failed",
                "error"  : str(e)
            })

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, do_export)

    return {"job_id": job_id, "message": "Export job started"}