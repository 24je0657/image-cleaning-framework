# api/dataset_validate.py
"""
Image validation for an extracted dataset (Step 4d).

Every planned image is opened with Pillow, checked for format/extension
agreement and pixel count, and fully decoded to catch truncation.
Invalid images are removed only after the per-class minimums have been
re-checked; if too many are invalid, InsufficientDataError is raised and
nothing is deleted.

Blocking and CPU-bound: call from a worker thread, not the event loop.
"""
# Standard library
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Third-party
from PIL import Image

# Local
from api.dataset_upload import DatasetPlan, PlannedFile
from api.upload_config import (
    IMAGE_EXTENSIONS,
    LIMITS,
    InsufficientDataError,
    UploadLimits,
)

logger = logging.getLogger(__name__)

# Pillow's name for the real format, keyed by file extension.
EXTENSION_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}

PROGRESS_EVERY = 100
REJECTED_SAMPLE_LIMIT = 100

REASON_UNREADABLE = "unreadable"
REASON_FORMAT = "format_mismatch"
REASON_PIXELS = "too_many_pixels"
REASON_MISSING = "missing"


@dataclass(frozen=True)
class RejectedImage:
    path: str          # dest path relative to dataset/, from the plan
    reason: str
    split: str
    class_name: str


@dataclass
class ValidationResult:
    checked: int
    accepted: int
    rejected: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)   # {split: {class: n}}
    rejected_by_reason: dict = field(default_factory=dict)
    rejected_by_class: dict = field(default_factory=dict)  # "split/class"

    def to_metadata(self) -> dict:
        """JSON-safe rejection summary (capped sample list)."""
        return {
            "validated": self.checked,
            "rejected_total": len(self.rejected),
            "rejected_by_reason": self.rejected_by_reason,
            "rejected_by_class": self.rejected_by_class,
            "rejected_samples": [
                {"path": r.path[:100], "reason": r.reason}
                for r in self.rejected[:REJECTED_SAMPLE_LIMIT]
            ],
        }


def merge_metadata(plan: DatasetPlan, result: ValidationResult) -> dict:
    """Plan metadata with post-rejection counts and rejection details."""
    meta = plan.to_metadata()
    meta["counts"] = result.counts
    meta["total_images"] = result.accepted
    meta.update(result.to_metadata())
    return meta


def _dest_path(root: Path, dest: str) -> Path:
    parts = dest.split("/")

    if (
        not dest
        or "\\" in dest
        or "\x00" in dest
        or dest.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("Unsafe planned image destination.")

    target = root.joinpath(*parts)

    if not target.resolve().is_relative_to(root):
        raise ValueError("Image destination escapes dataset directory.")

    return target


def _check_image(
    path: Path, expected_format: str, max_pixels: int
) -> Optional[str]:
    """Return a rejection reason, or None if the image is valid."""
    try:
        with Image.open(path) as img:
            if img.format != expected_format:
                return REASON_FORMAT
            width, height = img.size
            if width < 1 or height < 1:
                return REASON_UNREADABLE
            if width * height > max_pixels:
                return REASON_PIXELS
            img.load()  # full decode; raises on truncated data
    except Image.DecompressionBombError:
        return REASON_PIXELS
    except Exception:  # noqa: BLE001 - untrusted input; Pillow raises many types
        logger.debug("image failed validation", exc_info=True)
        return REASON_UNREADABLE
    return None


def _check_file(
    root: Path, planned: PlannedFile, limits: UploadLimits
) -> Optional[str]:
    path = _dest_path(root, planned.dest)
    try:
        if path.is_symlink():
            return REASON_UNREADABLE
        if not path.is_file():
            return REASON_MISSING
        if not path.resolve().is_relative_to(root):
            return REASON_UNREADABLE
    except OSError:
        return REASON_UNREADABLE
    expected = EXTENSION_FORMATS[Path(planned.dest).suffix.lower()]
    return _check_image(path, expected, limits.max_image_pixels)


def validate_images(
    dataset_dir,
    plan: DatasetPlan,
    limits: UploadLimits = LIMITS,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ValidationResult:
    """
    Validate every planned image, then remove the invalid ones.

    Raises InsufficientDataError (deleting nothing) if rejections leave
    any class below limits.min_images_per_class in any split.
    """
    root = Path(dataset_dir).resolve()
    total = len(plan.files)
    rejected = []
    accepted = defaultdict(Counter)

    for done, planned in enumerate(plan.files, start=1):
        reason = _check_file(root, planned, limits)
        if reason is None:
            accepted[planned.split][planned.class_name] += 1
        else:
            rejected.append(RejectedImage(
                planned.dest, reason, planned.split, planned.class_name))
        if progress and (done % PROGRESS_EVERY == 0 or done == total):
            progress(done, total)

    counts = {
        split: {cls: accepted[split][cls] for cls in plan.classes}
        for split in plan.splits
    }
    rejected_per = Counter((r.split, r.class_name) for r in rejected)

    short = [
        {"split": s, "class": c, "valid_images": counts[s][c],
         "rejected": rejected_per[(s, c)]}
        for s in plan.splits for c in plan.classes
        if counts[s][c] < limits.min_images_per_class
    ]
    if short:
        raise InsufficientDataError(
            "Too many images were invalid. Each class needs at least "
            f"{limits.min_images_per_class} valid images in every split.",
            details={
                "examples": short[:10],
                "rejected_total": len(rejected),
                "min_images_per_class": limits.min_images_per_class,
            },
        )

    # Checks passed: only now remove the rejected files.
    for r in rejected:
        path = _dest_path(root, r.path)
        path.unlink(missing_ok=True)

    return ValidationResult(
        checked=total,
        accepted=total - len(rejected),
        rejected=rejected,
        counts=counts,
        rejected_by_reason=dict(Counter(r.reason for r in rejected)),
        rejected_by_class={
            f"{s}/{c}": n for (s, c), n in sorted(rejected_per.items())
        },
    )