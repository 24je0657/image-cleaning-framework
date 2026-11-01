# api/dataset_extract.py
"""
Safe extraction of a validated DatasetPlan (Step 4c).

Extracts into a temporary directory next to dataset_dir, counts the bytes
actually written, and renames the result into place only after every file
has been written. On any failure the temporary directory is removed and
dataset_dir is left exactly as it was found.

The ZIP is opened read-only and never modified.
"""
# Standard library
import lzma
import os
import shutil
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path ,PureWindowsPath

# Local
from api.dataset_upload import DatasetPlan
from api.upload_config import (
    LIMITS,
    MB,
    UnsafePathError,
    UnsupportedArchiveError,
    UploadLimits,
    UploadTooLargeError,
)

CHUNK_BYTES = 64 * 1024
TEMP_PREFIX = ".extract-"


class DatasetExistsError(RuntimeError):
    """dataset_dir already holds data; refusing to mix two uploads."""


@dataclass(frozen=True)
class ExtractionResult:
    files_written: int
    bytes_written: int


class _ByteBudget:
    """Counts bytes actually written and enforces the total limit."""

    def __init__(self, limit_bytes: int):
        self.limit = limit_bytes
        self.used = 0

    def consume(self, n: int) -> None:
        self.used += n
        if self.used > self.limit:
            raise UploadTooLargeError(
                "Extracted dataset exceeds the size limit of "
                f"{self.limit // MB:,} MB.",
                details={"limit_mb": self.limit // MB},
            )


def _check_destination(dataset_dir: Path) -> None:
    if dataset_dir.is_symlink():
        raise DatasetExistsError("Dataset directory is a symbolic link.")
    if not dataset_dir.exists():
        return
    if not dataset_dir.is_dir() or any(dataset_dir.iterdir()):
        raise DatasetExistsError(
            "Dataset directory is not empty; refusing to extract into it."
        )


def _safe_target(root: Path, dest: str) -> Path:
    """Re-validate a planned destination and keep it inside root."""

    windows_path = PureWindowsPath(dest)
    parts = dest.split("/")

    if (
        not dest
        or "\x00" in dest
        or "\\" in dest
        or dest.startswith("/")
        or windows_path.drive
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise UnsafePathError(
            "Plan contains an unsafe destination path."
        )

    target = root.joinpath(*parts)

    if not target.resolve().is_relative_to(root):
        raise UnsafePathError(
            "Plan destination escapes the dataset root."
        )

    return target


def _copy_member(zf, info, target: Path, budget: _ByteBudget) -> int:
    """Copy one member in chunks, counting the bytes actually written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zf.open(info) as src, open(target, "xb") as out:
        while True:
            chunk = src.read(CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > info.file_size:
                raise UnsupportedArchiveError(
                    "An archive entry is larger than its declared size.")
            budget.consume(len(chunk))
            out.write(chunk)
    return written


def _promote(tmp: Path, dataset_dir: Path) -> None:
    """Move the finished tree into place; restore an empty dir on failure."""

    if dataset_dir.is_symlink():
        raise DatasetExistsError(
            "Dataset directory is a symbolic link."
        )

    if dataset_dir.exists():
        try:
            dataset_dir.rmdir()
        except OSError:
            raise DatasetExistsError(
                "Dataset directory changed during extraction."
            ) from None

    try:
        os.rename(tmp, dataset_dir)
    except OSError:
        if not dataset_dir.exists() and not dataset_dir.is_symlink():
            dataset_dir.mkdir()
        raise


def extract_plan(
    zip_path,
    plan: DatasetPlan,
    dataset_dir,
    limits: UploadLimits = LIMITS,
) -> ExtractionResult:
    """Extract exactly the files in plan into dataset_dir, or nothing."""
    zip_path = Path(zip_path)
    dataset_dir = Path(dataset_dir)

    _check_destination(dataset_dir)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=dataset_dir.parent))

    try:
        root = tmp.resolve()
        budget = _ByteBudget(limits.max_uncompressed_bytes)
        files_written = 0

        try:
            with zipfile.ZipFile(zip_path) as zf:
                for planned in plan.files:
                    info = zf.getinfo(planned.source)
                    target = _safe_target(root, planned.dest)
                    _copy_member(zf, info, target, budget)
                    files_written += 1
        except (zipfile.BadZipFile, KeyError, EOFError,
                NotImplementedError, zlib.error, lzma.LZMAError):
            raise UnsupportedArchiveError(
                "The archive is corrupt or does not match its "
                "inspection plan.") from None

        _promote(tmp, dataset_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return ExtractionResult(files_written, budget.used)