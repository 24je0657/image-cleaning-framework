# run_api.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host        = "0.0.0.0",
        port        = 8000,
        reload      = True,    # auto-reload on code changes (dev only)
        workers     = 1,       # 1 worker — models are loaded once per worker
        log_level   = "info"
    )