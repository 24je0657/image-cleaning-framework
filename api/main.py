"""
FastAPI application factory.
Registers all routers. Handles startup/shutdown.
This file stays small — no business logic here.
"""
import asyncio
from fastapi                 import FastAPI,Depends
from fastapi.middleware.cors import CORSMiddleware

from api.routes   import health, upload, jobs, results, datasets, runs, job_results, job_downloads
from api.dependencies import get_resnet, get_autoencoder
from api.worker import executor, shutdown, load_persisted_jobs
from api import maintenance, access

_docs = access.SETTINGS.public_docs or not access.SETTINGS.enabled
app = FastAPI(
    title="Image Cleaning Framework API",
    description="Production API for automated deep learning "
                "image dataset cleaning.",
    version="1.0.0",
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(access.SETTINGS.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

guard = [Depends(access.require_access)]
app.include_router(health.router)
for _module in (upload, jobs, results, datasets, runs, job_results,
                job_downloads):
    app.include_router(_module.router, dependencies=guard)



# ── Startup: pre-load models once ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    if not access.SETTINGS.enabled:
        print("⚠ ACCESS CONTROL DISABLED: set IMAGE_CLEANING_ACCESS_CODES "
              "before exposing this service.")
    print("Starting Image Cleaning API...")

    loop = asyncio.get_running_loop()

    # Recover persisted jobs before accepting requests.
    restored = await loop.run_in_executor(
        executor,
        load_persisted_jobs,
        True,
    )

    print(f"Restored {restored} persisted job(s)")
    maintenance.start_sweeper()

    # Load models.
    await loop.run_in_executor(executor, get_resnet)
    await loop.run_in_executor(executor, get_autoencoder)

    print("API ready ✅  →  http://localhost:8000/docs")


@app.on_event("shutdown")
async def on_shutdown():
    shutdown()
    maintenance.stop_sweeper()
    print("API shutdown complete")
