# api/access.py
"""Shared access codes, job ownership and per-code quotas (Step 9b)."""
# Standard library
import hashlib
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

# Third-party
from fastapi import HTTPException, Request

# Local
from api import worker
from api.upload_config import (
    UploadConfigError,
    UploadValidationError,
    _read_number,
)
from api.workspace import get_workspace

HEADER = "x-access-code"
MIN_CODE_LENGTH = 16
MAX_HEADER_LENGTH = 256
LEGACY_PREFIXES = ("/pipeline", "/stats", "/reports", "/export", "/analyze")


class QuotaExceededError(UploadValidationError):
    code = "quota_exceeded"
    http_status = 429


@dataclass(frozen=True)
class AccessSettings:
    codes: tuple = ()
    max_live_jobs: int = 3
    uploads_per_hour: int = 5
    legacy_routes: bool = False
    public_docs: bool = False
    cors_origins: tuple = ()

    def __post_init__(self):
        if len(set(self.codes)) != len(self.codes):
            raise UploadConfigError("ACCESS_CODES contains duplicates.")
        for code in self.codes:
            if len(code) < MIN_CODE_LENGTH:
                raise UploadConfigError(
                    f"Every access code needs at least {MIN_CODE_LENGTH} "
                    "characters.")
        for name in ("max_live_jobs", "uploads_per_hour"):
            if getattr(self, name) < 1:
                raise UploadConfigError(f"{name} must be at least 1.")

    @property
    def enabled(self) -> bool:
        return bool(self.codes)


def _flag(env, name: str) -> bool:
    raw = env.get("IMAGE_CLEANING_" + name, "")
    return raw.strip().lower() in ("on", "1", "true", "yes")


def _csv(env, name: str) -> tuple:
    raw = env.get("IMAGE_CLEANING_" + name, "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def load_settings(environ=None) -> AccessSettings:
    env = os.environ if environ is None else environ
    d = AccessSettings()
    return AccessSettings(
        codes=_csv(env, "ACCESS_CODES"),
        max_live_jobs=_read_number(
            env, "MAX_LIVE_JOBS", d.max_live_jobs, int, 1),
        uploads_per_hour=_read_number(
            env, "UPLOADS_PER_HOUR", d.uploads_per_hour, int, 1),
        legacy_routes=_flag(env, "LEGACY_ROUTES"),
        public_docs=_flag(env, "PUBLIC_DOCS"),
        cors_origins=_csv(env, "CORS_ORIGINS"),
    )


SETTINGS = load_settings()


def owner_id(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:32]


def match_code(provided: str, settings: Optional[AccessSettings] = None):
    """Owner id for a valid code, else None. Checks every code, no early exit."""
    s = settings or SETTINGS
    found = None
    for code in s.codes:
        if hmac.compare_digest(provided.encode(), code.encode()):
            found = owner_id(code)
    return found


def _owns(job_id: str, owner: str) -> bool:
    try:
        meta = get_workspace(job_id, worker.JOBS_ROOT).read_meta()
    except Exception:  # noqa: BLE001 - unknown, malformed or unreadable
        return False
    stored = meta.get("owner")
    return isinstance(stored, str) and hmac.compare_digest(stored, owner)


async def require_access(request: Request) -> None:
    s = SETTINGS
    request.state.owner = None
    if not s.enabled:
        return
    provided = request.headers.get(HEADER, "")
    owner = (match_code(provided, s)
             if provided and len(provided) <= MAX_HEADER_LENGTH else None)
    if owner is None:
        raise HTTPException(status_code=401, detail="Access code required.")
    request.state.owner = owner

    path = request.url.path
    if path.startswith(LEGACY_PREFIXES) and not s.legacy_routes:
        raise HTTPException(status_code=404, detail="Not found.")
    job_id = request.path_params.get("job_id")
    if job_id is not None and not _owns(job_id, owner):
        raise HTTPException(status_code=404, detail="Job not found.")


# ─── Quotas ───────────────────────────────────────────────────────────────────

_LOCK = threading.Lock()
_UPLOADS: dict = defaultdict(deque)


def reset_state() -> None:
    with _LOCK:
        _UPLOADS.clear()


def live_jobs(owner: str) -> int:
    count = 0
    for job_id in list(worker.jobs):
        if _owns(job_id, owner):
            count += 1
    return count


def enforce_upload_quota(request, now: Optional[float] = None,
                         settings: Optional[AccessSettings] = None):
    """Raise QuotaExceededError if over limits; return the owner id (or None)."""
    s = settings or SETTINGS
    owner = getattr(request.state, "owner", None)
    if not s.enabled or owner is None:
        return None
    now = time.time() if now is None else now
    with _LOCK:
        window = _UPLOADS[owner]
        while window and window[0] <= now - 3600:
            window.popleft()
        if len(window) >= s.uploads_per_hour:
            raise QuotaExceededError(
                "Upload limit reached. Please try again later.")
        if live_jobs(owner) >= s.max_live_jobs:
            raise QuotaExceededError(
                "Too many stored datasets. Delete one or wait for it to "
                "expire.")
        window.append(now)
    return owner