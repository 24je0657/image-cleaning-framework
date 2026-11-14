# api/maintenance.py
"""Disk lifecycle for the job store (Step 9a). No FastAPI."""
# Standard library
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Local
from api import download_service, worker
from api.upload_config import (
    LIMITS,
    MB,
    UploadConfigError,
    UploadLimits,
    UploadValidationError,
    _read_number,
)
from api.workspace import cleanup_expired

logger = logging.getLogger(__name__)


class StorageLowError(UploadValidationError):
    code = "server_storage_low"
    http_status = 503


@dataclass(frozen=True)
class MaintenanceSettings:
    retention_hours: float = 24.0
    download_retention_hours: float = 6.0
    sweep_interval_minutes: float = 30.0
    min_free_bytes: int = 3072 * MB

    def __post_init__(self):
        for name in ("retention_hours", "download_retention_hours",
                     "sweep_interval_minutes", "min_free_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0 or value != value:
                raise UploadConfigError(f"{name} must be a positive number.")
        if self.download_retention_hours > self.retention_hours:
            raise UploadConfigError(
                "DOWNLOAD_RETENTION_HOURS must not exceed RETENTION_HOURS.")


def load_settings(environ=None) -> MaintenanceSettings:
    env = os.environ if environ is None else environ
    d = MaintenanceSettings()
    return MaintenanceSettings(
        retention_hours=_read_number(
            env, "RETENTION_HOURS", d.retention_hours, float, 0.01),
        download_retention_hours=_read_number(
            env, "DOWNLOAD_RETENTION_HOURS", d.download_retention_hours,
            float, 0.01),
        sweep_interval_minutes=_read_number(
            env, "SWEEP_INTERVAL_MINUTES", d.sweep_interval_minutes,
            float, 0.01),
        min_free_bytes=MB * _read_number(
            env, "MIN_FREE_MB", d.min_free_bytes // MB, int, 1),
    )


SETTINGS = load_settings()


# ─── Free-space guard ─────────────────────────────────────────────────────────

def free_bytes(path) -> int:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def required_free_bytes(limits: UploadLimits = LIMITS,
                        settings: MaintenanceSettings = SETTINGS) -> int:
    return max(settings.min_free_bytes,
               limits.max_upload_bytes + limits.max_uncompressed_bytes)


def ensure_free_space(jobs_root=None, limits: UploadLimits = LIMITS,
                      settings: MaintenanceSettings = SETTINGS) -> None:
    root = Path(jobs_root) if jobs_root else worker.JOBS_ROOT
    if free_bytes(root) < required_free_bytes(limits, settings):
        raise StorageLowError(
            "The service is low on storage. Please try again later.")


# ─── Sweeping ─────────────────────────────────────────────────────────────────

def run_sweep(jobs_root=None,
              settings: MaintenanceSettings = SETTINGS) -> dict:
    root = Path(jobs_root) if jobs_root else worker.JOBS_ROOT
    expired = cleanup_expired(settings.retention_hours, root)
    for job_id in expired:
        worker.jobs.pop(job_id, None)
    downloads = download_service.sweep_downloads(
        settings.download_retention_hours, root)
    if expired or downloads:
        logger.info("sweep removed %d jobs, %d downloads",
                    len(expired), len(downloads))
    return {"expired_jobs": expired, "stale_downloads": downloads}


class Sweeper:
    """Runs task now and then every interval until stopped."""

    def __init__(self, interval_seconds: float, task: Callable[[], object]):
        self._interval = interval_seconds
        self._task = task
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="job-sweeper", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        while True:
            try:
                self._task()
            except Exception:  # noqa: BLE001 - the loop must survive
                logger.exception("sweep failed")
            if self._stop.wait(self._interval):
                return


_sweeper: Optional[Sweeper] = None


def start_sweeper(settings: MaintenanceSettings = SETTINGS) -> bool:
    """Start the background sweeper. Returns False if disabled by env."""
    global _sweeper
    if os.environ.get("IMAGE_CLEANING_SWEEPER", "").strip().lower() == "off":
        logger.info("sweeper disabled by IMAGE_CLEANING_SWEEPER=off")
        return False
    if _sweeper is None:
        _sweeper = Sweeper(settings.sweep_interval_minutes * 60,
                           lambda: run_sweep(settings=settings))
    _sweeper.start()
    return True


def stop_sweeper() -> None:
    if _sweeper is not None:
        _sweeper.stop()