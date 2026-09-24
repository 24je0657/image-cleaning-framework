"""
Async job management.
Owns the ThreadPoolExecutor — one place only.
"""
import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

executor : ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
jobs     : dict               = {}   # In production: replace with Redis


def make_job(job_id: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "job_id"     : job_id,
        "status"     : "pending",
        "progress"   : 0,
        "message"    : "Queued",
        "created_at" : now,
        "updated_at" : now,
        "result"     : None,
        "error"      : None,
    }


async def submit(fn, *args):
    """Submit a blocking function to the thread pool."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, fn, *args)


def shutdown():
    executor.shutdown(wait=False)