# api/download_service.py
"""
Cleaned-dataset download packaging (Step 7). No FastAPI.

Builds <job>/download.zip from final/ plus the decision CSV and a summary,
under the per-job lock shared with approval. The ZIP is written to a temp
file and swapped in atomically. Raw master reports are never shipped: they
contain server-side paths.
"""
# Standard library
import json
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

# Local
from api import results_service as rs
from api.job_locks import job_lock
from api.workspace import (
    ACTIVE_STATUSES,
    JOB_ID_PATTERN,
    JobWorkspace,
    WorkspaceMetadataError,
)

logger = logging.getLogger(__name__)

CHECKS = (
    ("duplicates", "run_duplicates"),
    ("blur", "run_blur"),
    ("noise", "run_noise"),
    ("outliers", "run_outliers"),
    ("mislabels", "run_mislabels"),
)
CHECKS_NOTE = (
    "CLEAN means no issue was found by the checks that were enabled. "
    "It is not a guarantee that the image is free of every problem."
)


class DownloadNotAvailableError(rs.ResultsError):
    code = "download_not_available"
    http_status = 409


def _stamp_path(ws: JobWorkspace) -> Path:
    return ws.root / "download.stamp"


def _final_files(ws: JobWorkspace) -> list:
    if ws.final_dir.is_symlink() or not ws.final_dir.is_dir():
        return []
    root = ws.final_dir.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            files.append((path, path.resolve().relative_to(root).as_posix()))
        except ValueError:
            continue
    return files


def _summary(meta: dict, file_count: int) -> dict:
    dataset = meta.get("dataset") or {}
    approval = meta["approval"]
    run = meta.get("run")
    enabled = None
    if isinstance(run, dict):
        enabled = [name for name, key in CHECKS if run.get(key) is True]
    return {
        "layout": dataset.get("layout"),
        "classes": dataset.get("classes"),
        "splits": dataset.get("splits"),
        "uploaded": {
            "images": dataset.get("total_images"),
            "rejected_invalid": dataset.get("rejected_total"),
            "ignored_files": dataset.get("ignored_total"),
        },
        "checks_enabled": enabled,
        "checks_note": CHECKS_NOTE,
        "decisions": {
            key: approval.get(key) for key in (
                "kept", "removed", "rescued_from_remove",
                "removed_from_review", "apply_proposed_removals",
                "review_default", "explicit_keep", "explicit_remove")
        },
        "files_in_download": file_count,
        "missing_files": approval.get("missing_files", 0),
        "warnings": approval.get("warnings", []),
        "approved_at": approval.get("approved_at"),
    }


def _write_zip(ws: JobWorkspace, meta: dict, files: list) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=".download-", suffix=".zip.tmp", dir=ws.root)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            for src, rel in files:
                # Images are already compressed; store them as-is.
                zf.write(src, f"dataset/{rel}",
                         compress_type=zipfile.ZIP_STORED)
            decisions = ws.reports_dir / "approval_decisions.csv"
            if decisions.is_file():
                zf.write(decisions, "approval_decisions.csv",
                         compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("summary.json",
                        json.dumps(_summary(meta, len(files)), indent=2),
                        compress_type=zipfile.ZIP_DEFLATED)
        _stamp_path(ws).unlink(missing_ok=True)
        os.replace(tmp, ws.download_zip)
        _stamp_path(ws).write_text(
            meta["approval"]["approved_at"], encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)


def build_download(ws: JobWorkspace) -> tuple:
    """Return (zip_path, meta), building the ZIP if it is missing or stale."""
    with job_lock(ws.job_id):
        meta = ws.read_meta()
        if meta.get("status") != "completed":
            raise DownloadNotAvailableError(
                "The job has no completed results to download.",
                details={"status": meta.get("status")})
        approval = meta.get("approval")
        stamp: Optional[str] = (
            approval.get("approved_at") if isinstance(approval, dict) else None)
        if not stamp:
            raise DownloadNotAvailableError(
                "Approve the cleaned dataset before downloading it.")
        files = _final_files(ws)
        if not files:
            raise DownloadNotAvailableError(
                "The approved dataset is no longer available. "
                "Approve the results again.")
        try:
            cached = (ws.download_zip.is_file()
                      and _stamp_path(ws).read_text(encoding="utf-8") == stamp)
        except OSError:
            cached = False
        if not cached:
            _write_zip(ws, meta, files)
        return ws.download_zip, meta


def sweep_downloads(max_age_hours: float, jobs_root) -> list:
    """Delete stale download ZIPs of inactive jobs. Returns job ids swept."""
    root = Path(jobs_root)
    if not root.is_dir():
        return []
    cutoff = time.time() - max_age_hours * 3600
    swept = []
    for entry in root.iterdir():
        if (entry.is_symlink() or not entry.is_dir()
                or not JOB_ID_PATTERN.fullmatch(entry.name)):
            continue
        ws = JobWorkspace(job_id=entry.name, root=entry)
        try:
            if (not ws.download_zip.is_file()
                    or ws.download_zip.stat().st_mtime >= cutoff):
                continue
            with job_lock(entry.name):
                if ws.read_meta().get("status") in ACTIVE_STATUSES:
                    continue
                ws.download_zip.unlink(missing_ok=True)
                _stamp_path(ws).unlink(missing_ok=True)
            swept.append(entry.name)
        except (OSError, WorkspaceMetadataError):
            logger.warning("could not sweep download for %s", entry.name)
    return swept