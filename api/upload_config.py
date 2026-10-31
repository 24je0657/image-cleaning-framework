# api/upload_config.py
"""
Upload limits and validation exceptions for dataset ZIP uploads.

No FastAPI dependency. Limits are read from IMAGE_CLEANING_* environment
variables and validated at import time, so a bad value stops the app at
startup instead of failing on the first upload.

Every exception carries a machine-readable ``code`` and an
``http_status`` so the route layer can map it without parsing strings.
Messages are user-safe: they never contain server paths.
"""
# Standard library
import math
import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

ENV_PREFIX = "IMAGE_CLEANING_"
MB = 1024 * 1024

# Must match VALID_EXTENSIONS in src/embeddings.py, src/quality.py and
# src/format_converter.py. tests/test_upload_config.py checks this.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

# 1-64 chars; starts with a letter/digit, ends with a letter/digit/_/-.
# No trailing space or dot (they collide on some filesystems), no leading dot.
CLASS_NAME_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9 _.-]{0,62}[A-Za-z0-9_-])?$"
)


# ─── Exceptions ───────────────────────────────────────────────────────────────

class UploadConfigError(ValueError):
    """An upload limit is missing, malformed or inconsistent (startup)."""


class UploadValidationError(Exception):
    """Base class for user-facing upload rejections."""

    code = "invalid_upload"
    http_status = 400

    def __init__(self, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class UploadTooLargeError(UploadValidationError):
    code = "upload_too_large"
    http_status = 413


class TooManyEntriesError(UploadValidationError):
    code = "too_many_entries"
    http_status = 413


class CompressionRatioError(UploadValidationError):
    code = "compression_ratio_exceeded"
    http_status = 400


class UnsafePathError(UploadValidationError):
    code = "unsafe_path"
    http_status = 400


class UnsupportedArchiveError(UploadValidationError):
    code = "unsupported_archive"
    http_status = 400


class InvalidLayoutError(UploadValidationError):
    code = "invalid_layout"
    http_status = 400


class EmptyDatasetError(UploadValidationError):
    code = "empty_dataset"
    http_status = 400


class InsufficientDataError(UploadValidationError):
    code = "insufficient_data"
    http_status = 400


# ─── Limits ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UploadLimits:
    max_upload_bytes: int = 500 * MB
    max_uncompressed_bytes: int = 2048 * MB
    max_entries: int = 20_000
    max_images: int = 5_000
    max_compression_ratio: float = 100.0
    min_classes: int = 2
    # PROVISIONAL: Step 5 sets this from a real run on a small dataset.
    min_images_per_class: int = 10
    max_image_pixels: int = 50_000_000

    def __post_init__(self):
        # All count and byte limits must be positive integers.
        integer_fields = (
            "max_upload_bytes",
            "max_uncompressed_bytes",
            "max_entries",
            "max_images",
            "min_classes",
            "min_images_per_class",
            "max_image_pixels",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise UploadConfigError(
                    f"{field_name} must be an integer."
                )

            if value <= 0:
                raise UploadConfigError(
                    f"{field_name} must be greater than zero."
                )

        # Dataset must contain at least two classes.
        if self.min_classes < 2:
            raise UploadConfigError(
                "MIN_CLASSES must be at least 2; mislabel detection "
                "needs more than one class."
            )

        # Compression ratio must be finite and >= 1.
        if (
            isinstance(self.max_compression_ratio, bool)
            or not isinstance(self.max_compression_ratio, (int, float))
            or not math.isfinite(self.max_compression_ratio)
            or self.max_compression_ratio < 1
        ):
            raise UploadConfigError(
                "MAX_COMPRESSION_RATIO must be finite and >= 1."
            )

        if self.max_upload_bytes > self.max_uncompressed_bytes:
            raise UploadConfigError(
                "MAX_UPLOAD_MB must not exceed MAX_UNCOMPRESSED_MB."
            )

        if self.max_images > self.max_entries:
            raise UploadConfigError(
                "MAX_IMAGES must not exceed MAX_ZIP_ENTRIES."
            )

        if self.min_classes * self.min_images_per_class > self.max_images:
            raise UploadConfigError(
                "MIN_CLASSES x MIN_IMAGES_PER_CLASS exceeds MAX_IMAGES; "
                "no dataset could pass."
            )


def _read_number(env: Mapping[str, str], name: str, default, kind, minimum):
    """Read IMAGE_CLEANING_<name>; empty or unset means use the default."""
    var = ENV_PREFIX + name
    raw = env.get(var)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = kind(raw.strip())
    except ValueError:
        raise UploadConfigError(
            f"{var} must be of type {kind.__name__}, got {raw!r}."
        ) from None
    if kind is float and not math.isfinite(value):
        raise UploadConfigError(f"{var} must be finite, got {raw!r}.")
    if value < minimum:
        raise UploadConfigError(f"{var} must be >= {minimum}, got {raw!r}.")
    return value


def load_limits(environ: Optional[Mapping[str, str]] = None) -> UploadLimits:
    """Build validated limits from the environment (or a given mapping)."""
    env = os.environ if environ is None else environ
    d = UploadLimits()  # defaults live in one place
    return UploadLimits(
        max_upload_bytes=MB * _read_number(
            env, "MAX_UPLOAD_MB", d.max_upload_bytes // MB, int, 1),
        max_uncompressed_bytes=MB * _read_number(
            env, "MAX_UNCOMPRESSED_MB", d.max_uncompressed_bytes // MB,
            int, 1),
        max_entries=_read_number(
            env, "MAX_ZIP_ENTRIES", d.max_entries, int, 1),
        max_images=_read_number(
            env, "MAX_IMAGES", d.max_images, int, 1),
        max_compression_ratio=_read_number(
            env, "MAX_COMPRESSION_RATIO", d.max_compression_ratio,
            float, 1.0),
        min_classes=_read_number(
            env, "MIN_CLASSES", d.min_classes, int, 2),
        min_images_per_class=_read_number(
            env, "MIN_IMAGES_PER_CLASS", d.min_images_per_class, int, 1),
        max_image_pixels=_read_number(
            env, "MAX_IMAGE_PIXELS", d.max_image_pixels, int, 1),
    )


# Evaluated at import: an invalid environment fails at application startup.
LIMITS = load_limits()