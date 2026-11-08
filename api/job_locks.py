# api/job_locks.py
"""Per-job locks shared by approval and download (Step 7)."""
# Standard library
import threading

_LOCKS: dict = {}
_GUARD = threading.Lock()


def job_lock(job_id: str) -> threading.Lock:
    """One lock per job id. Entries are tiny and live for the process."""
    with _GUARD:
        return _LOCKS.setdefault(job_id, threading.Lock())