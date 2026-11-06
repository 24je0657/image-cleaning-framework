"""
Job-specific pipeline execution.

Runs the existing image-cleaning pipeline inside an isolated
per-job workspace without modifying the dashboard pipeline.
"""

import logging
import os
import shutil
import threading

from pathlib import Path

from api.schemas import JobRunRequest
from api.workspace import (
    get_workspace,
    JOBS_ROOT,
    InvalidJobIdError,
    WorkspaceNotFoundError,
    WorkspaceMetadataError,
)
from api.worker import jobs, update_job, executor

from src.config import PipelineConfig
from run_pipeline import run_pipeline_config


logger = logging.getLogger(__name__)


# ── Run configuration ────────────────────────────────────────────

MAX_CONCURRENT_RUNS = max(
    1,
    int(os.environ.get("IMAGE_CLEANING_MAX_CONCURRENT_RUNS", "1"))
)

_RUN_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)

# Protects the status-check + slot-claim + RUNNING transition.
_START_LOCK = threading.Lock()

RUNNABLE_STATUSES = {
    "ready",
    "failed",
    "interrupted",
}


# ── Exceptions ───────────────────────────────────────────────────

class JobNotRunnableError(RuntimeError):
    """Raised when a job cannot be started in its current state."""


class ServerBusyError(RuntimeError):
    """Raised when all pipeline execution slots are occupied."""

def _is_safe_directory_name(value: str, allow_empty: bool = False) -> bool:
    """Check that a value represents one safe directory name."""

    if not isinstance(value, str):
        return False

    if allow_empty and value == "":
        return True

    if not value or value in {".", ".."}:
        return False

    if "/" in value or "\\" in value:
        return False

    if Path(value).name != value:
        return False

    return True


def _dataset_available(workspace, metadata: dict) -> bool:
    """Check that uploaded dataset metadata and image directories exist."""

    dataset = metadata.get("dataset")

    if not isinstance(dataset, dict):
        return False

    splits = dataset.get("splits")
    classes = dataset.get("classes")

    if not isinstance(splits, list) or not splits:
        return False

    if not isinstance(classes, list) or not classes:
        return False

    if not all(
        _is_safe_directory_name(split, allow_empty=True)
        for split in splits
    ):
        return False

    if not all(
        _is_safe_directory_name(class_name)
        for class_name in classes
    ):
        return False

    # Verify every expected split/class directory exists.
    for split in splits:
        split_dir = workspace.dataset_dir / split if split else workspace.dataset_dir

        if not split_dir.is_dir():
            return False

        for class_name in classes:
            class_dir = split_dir / class_name

            if not class_dir.is_dir():
                return False

            if not any(path.is_file() for path in class_dir.iterdir()):
                return False

    return True


def build_config(workspace, metadata: dict, request: JobRunRequest) -> PipelineConfig:
    """Build an isolated PipelineConfig for one uploaded job."""

    dataset = metadata["dataset"]

    return PipelineConfig(
        data_dir=workspace.dataset_dir,
        output_dir=workspace.output_dir,
        embeddings_dir=workspace.embeddings_dir,
        reports_dir=workspace.reports_dir,
        models_dir=workspace.root / "models",

        splits=dataset["splits"],
        classes=dataset["classes"],

        run_duplicates=request.run_duplicates,
        run_blur=request.run_blur,
        run_noise=request.run_noise,
        run_outliers=request.run_outliers,
        run_mislabels=request.run_mislabels,

        blur_percentile=request.blur_percentile,
    )

def _reset_outputs(workspace) -> None:
    """Remove generated artifacts before a fresh pipeline run."""

    directories = (
        workspace.embeddings_dir,
        workspace.reports_dir,
        workspace.output_dir,
        workspace.final_dir,
    )

    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(parents=True, exist_ok=True)

    # Remove previous job-specific model artifacts before retrying.
    models_dir = workspace.root / "models"

    if models_dir.exists():
        shutil.rmtree(models_dir)

    models_dir.mkdir(parents=True, exist_ok=True)

def _ensure_job_registered(job_id: str, metadata: dict) -> None:
    """Register a persisted job in memory if it is not already registered."""

    if job_id not in jobs:
        jobs[job_id] = {
            "job_id": job_id,
            "status": metadata.get("status", "unknown"),
            "progress": metadata.get("progress", 0),
            "message": metadata.get("message", ""),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "result": metadata.get("result"),
            "error": metadata.get("error"),
        }


def start_run(job_id: str, request: JobRunRequest) -> tuple:
    """Validate and atomically start a pipeline run for one job."""

    with _START_LOCK:
        workspace = get_workspace(job_id, JOBS_ROOT)
        metadata = workspace.read_meta()

        _ensure_job_registered(job_id, metadata)

        status = metadata.get("status")

        if status not in RUNNABLE_STATUSES:
            raise JobNotRunnableError(
                f"Job cannot be started from status '{status}'."
            )

        if not _dataset_available(workspace, metadata):
            raise JobNotRunnableError(
                "The uploaded dataset is missing or incomplete."
            )

        if not _RUN_SLOTS.acquire(blocking=False):
            raise ServerBusyError("All pipeline execution slots are occupied.")

        try:
            cfg = build_config(workspace, metadata, request)
            _reset_outputs(workspace)

            update_job(
                job_id,
                status="running",
                progress=5,
                message="Pipeline started",
                error=None,
            )

        except Exception:
            _RUN_SLOTS.release()
            raise

        return workspace, cfg

def _summarise_failures(result: dict) -> list[str]:
    """Extract failed split and pipeline-step names from the result."""

    failures = []

    summaries = result.get("summaries", {})

    for split, summary in summaries.items():
        if not isinstance(summary, dict):
            failures.append(f"{split}: invalid summary")
            continue

        steps = summary.get("steps", {})

        if isinstance(steps, dict):
            for step, succeeded in steps.items():
                if succeeded is False:
                    failures.append(f"{split}: {step}")

        if summary.get("success") is False and not any(
            item.startswith(f"{split}:") for item in failures
        ):
            failures.append(f"{split}: pipeline failed")

    if result.get("success") is False and not failures:
        failures.append("Pipeline reported failure")

    return failures

def _make_progress_callback(job_id: str):
    """Map pipeline progress (0..1) onto the job's 5..95 range, never backwards."""
    last = {"value": 5}

    def callback(fraction: float, message: str) -> None:
        value = max(last["value"], min(95, 5 + round(90 * fraction)))
        last["value"] = value
        update_job(job_id, progress=value, message=message)

    return callback


def run_job(job_id: str, cfg: PipelineConfig) -> None:
    """Execute the pipeline for a job and persist its final status."""

    try:
        result = run_pipeline_config(
            cfg,
            skip_embeddings=False,
            skip_preprocessing=True,
            progress=_make_progress_callback(job_id),
        )

        failures = _summarise_failures(result)

        if result.get("success") is True and not failures:
            workspace = get_workspace(job_id, JOBS_ROOT)

            # Keep the original upload result intact.
            workspace.update_meta(pipeline_result=result)

            update_job(
                job_id,
                status="completed",
                progress=100,
                message="Pipeline completed successfully",
                error=None,
            )

        else:
            error_message = (
                "Pipeline failed: " + "; ".join(failures)
            )

            _mark_failed(job_id, error_message)

    except Exception as exc:
        logger.exception("Pipeline execution failed for job %s", job_id)

        # Avoid exposing local filesystem paths in the public error.
        error_message = (
            f"Pipeline execution failed ({type(exc).__name__})."
        )

        _mark_failed(job_id, error_message)

    finally:
        _RUN_SLOTS.release()

def _mark_failed(job_id: str, message: str) -> None:
    """Persist a failed job while preserving its uploaded dataset."""

    try:
        update_job(
            job_id,
            status="failed",
            progress=5,
            message="Pipeline failed",
            error=message,
        )
    except Exception:
        logger.exception("Could not update failed status for job %s", job_id)