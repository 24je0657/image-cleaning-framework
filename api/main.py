"""
FastAPI application factory.
Registers all routers. Handles startup/shutdown.
This file stays small — no business logic here.
"""
import asyncio
from fastapi                 import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes   import health, upload, jobs, results
from api.worker   import executor, shutdown
from api.dependencies import get_resnet, get_autoencoder

app = FastAPI(
    title       = "Image Cleaning Framework API",
    description = "Production API for automated deep learning "
                  "image dataset cleaning.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(results.router)


# ── Startup: pre-load models once ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("Starting Image Cleaning API...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, get_resnet)
    await loop.run_in_executor(executor, get_autoencoder)
    print("API ready ✅  →  http://localhost:8000/docs")


@app.on_event("shutdown")
async def on_shutdown():
    shutdown()
    print("API shutdown complete")