# api/dataset_upload.py
"""
ZIP inspection and dataset-layout detection (Step 4b).

inspect_zip() reads the archive's metadata only. It never extracts and
never writes to disk. It returns a DatasetPlan describing exactly what
extraction (Step 4c) should write, or raises an UploadValidationError.

Supported layouts (after stripping any wrapper folder):

    split:  <split>/<class>/<image>      (train/val/test/training/validation)
    flat:   <class>/<image>

A flat dataset is normalised to dest paths under "train/" so the pipeline
only ever sees a split-based dataset. The original layout is recorded in
the plan so the final export can restore it.

Declared sizes in a ZIP header can be forged. The limits applied here are
a first screen only; extraction must count the bytes it actually writes.
"""
# Standard library
import stat
import struct
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Local
from api.upload_config import (
    CLASS_NAME_PATTERN,
    IMAGE_EXTENSIONS,
    LIMITS,
    MB,
    CompressionRatioError,
    EmptyDatasetError,
    InsufficientDataError,
    InvalidLayoutError,
    TooManyEntriesError,
    UnsafePathError,
    UnsupportedArchiveError,
    UploadLimits,
    UploadTooLargeError,
)
from src.config import KNOWN_SPLITS

JUNK_FILE_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
JUNK_DIR_NAMES = frozenset({"__macosx"})
SPLIT_ALIAS_GROUPS = (
    frozenset({"train", "training"}),
    frozenset({"val", "validation"}),
)
SUPPORTED_COMPRESSION = frozenset({
    zipfile.ZIP_STORED,
    zipfile.ZIP_DEFLATED,
    zipfile.ZIP_BZIP2,
    zipfile.ZIP_LZMA,
})

MAX_PATH_DEPTH = 32
MAX_SEGMENT_BYTES = 255
MAX_WRAPPER_DEPTH = 3
IGNORED_SAMPLE_LIMIT = 50
CENTRAL_DIR_BYTES_PER_ENTRY = 1024
EOCD_SIZE = 22
EOCD_MAX_TAIL = EOCD_SIZE + 65_535
ZIP64_16 = 0xFFFF
ZIP64_32 = 0xFFFFFFFF


# ─── Plan ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlannedFile:
    source: str        # exact member name inside the ZIP
    dest: str          # posix path relative to dataset/, e.g. train/cat/a.jpg
    split: str
    class_name: str
    size: int          # declared uncompressed size (untrusted)
    compress_size: int


@dataclass
class DatasetPlan:
    layout: str                              # "split" or "flat"
    splits: list
    classes: list
    counts: dict                             # {split: {class: n_images}}
    files: list = field(default_factory=list)
    wrapper_stripped: list = field(default_factory=list)
    total_images: int = 0
    total_uncompressed_bytes: int = 0
    ignored_total: int = 0
    ignored_by_reason: dict = field(default_factory=dict)
    ignored_samples: list = field(default_factory=list)

    def to_metadata(self) -> dict:
        """JSON-safe summary for job.json (excludes the file list)."""
        return {
            "layout": self.layout,
            "splits": self.splits,
            "classes": self.classes,
            "counts": self.counts,
            "total_images": self.total_images,
            "wrapper_stripped": self.wrapper_stripped,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "ignored_total": self.ignored_total,
            "ignored_by_reason": self.ignored_by_reason,
            "ignored_samples": self.ignored_samples,
        }


# ─── Path and member checks ───────────────────────────────────────────────────

def _display(name: str) -> str:
    return name[:100]


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _is_junk(parts: list) -> bool:
    if parts[-1].lower() in JUNK_FILE_NAMES:
        return True
    return any(p.lower() in JUNK_DIR_NAMES or p.startswith(".") for p in parts)


def _safe_parts(name: str) -> list:
    """Split a member name into safe path segments or raise."""
    unsafe = UnsafePathError(
        "Archive contains an unsafe file path.",
        details={"entry": _display(name)},
    )
    if not name or "\\" in name or name.startswith("/"):
        raise unsafe
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        raise unsafe  # drive letter

    parts = [p for p in name.split("/") if p not in ("", ".")]
    if len(parts) > MAX_PATH_DEPTH:
        raise unsafe
    for part in parts:
        if part == "..":
            raise unsafe
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in part):
            raise unsafe
        if len(part.encode("utf-8", "surrogatepass")) > MAX_SEGMENT_BYTES:
            raise unsafe
    return parts


def _check_member(info: zipfile.ZipInfo) -> list:
    """Reject unsafe or unsupported entries; return the safe path parts."""
    if info.flag_bits & 0x1:
        raise UnsupportedArchiveError(
            "Encrypted ZIP entries are not supported.")
    if info.compress_type not in SUPPORTED_COMPRESSION:
        raise UnsupportedArchiveError(
            "Archive uses an unsupported compression method.")
    mode = info.external_attr >> 16
    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise UnsafePathError(
            "Archive contains a symbolic link or special file.",
            details={"entry": _display(info.filename)},
        )
    return _safe_parts(info.filename)


def _check_central_directory(path: Path, limits: UploadLimits) -> None:
    """
    Validate the ZIP End of Central Directory (EOCD) record before
    zipfile loads the central directory into memory.
    """
    size = path.stat().st_size

    if size < EOCD_SIZE:
        return  # zipfile will report an invalid ZIP.

    tail_len = min(size, EOCD_MAX_TAIL)
    with open(path, "rb") as fh:
        fh.seek(size - tail_len)
        tail = fh.read(tail_len)

    # Search backwards for an EOCD whose comment ends exactly at EOF.
    idx = -1

    search_end = len(tail)

    while True:
        candidate = tail.rfind(b"PK\x05\x06", 0, search_end)

        if candidate < 0:
            break

        if len(tail) - candidate >= EOCD_SIZE:
            fields = struct.unpack(
                "<4s4H2LH",
                tail[candidate:candidate + EOCD_SIZE],
            )

            comment_length = fields[7]

            if candidate + EOCD_SIZE + comment_length == len(tail):
                idx = candidate
                break

        search_end = candidate

    if idx < 0:
        return  # zipfile will report an invalid ZIP.

    (
        _signature,
        disk_number,
        cd_start_disk,
        entries_on_disk,
        total_entries,
        cd_size,
        cd_offset,
        _comment_length,
    ) = struct.unpack(
        "<4s4H2LH",
        tail[idx:idx + EOCD_SIZE],
    )

    # Multi-disk ZIP archives are not supported.
    if (
        disk_number != 0
        or cd_start_disk != 0
        or entries_on_disk != total_entries
    ):
        raise UnsupportedArchiveError(
            "Multi-disk ZIP archives are not supported."
        )

    # ZIP64 archives are not supported.
    if (
        total_entries == ZIP64_16
        or cd_size == ZIP64_32
        or cd_offset == ZIP64_32
        or entries_on_disk == ZIP64_16
    ):
        raise UnsupportedArchiveError(
            "ZIP64 archives are not supported."
        )

    if total_entries > limits.max_entries:
        raise TooManyEntriesError(
            f"Archive has {total_entries:,} entries; the limit is "
            f"{limits.max_entries:,}.",
            details={
                "entries": total_entries,
                "limit": limits.max_entries,
            },
        )

    if cd_size > limits.max_entries * CENTRAL_DIR_BYTES_PER_ENTRY:
        raise TooManyEntriesError(
            "Archive directory is too large.",
            details={"limit_entries": limits.max_entries},
        )

    # Validate central-directory boundaries.
    absolute_eocd_offset = size - tail_len + idx

    if cd_offset + cd_size > absolute_eocd_offset:
        raise UnsupportedArchiveError(
            "Invalid ZIP central-directory boundaries."
        )


# ─── Wrapper detection ────────────────────────────────────────────────────────

def _wrapper_name(real: list, _depth: int = 0) -> Optional[str]:
    """
    Return the name of a single wrapper folder, or None.

    A top-level folder is a wrapper if it is not itself a split name
    and either contains a split folder, or contains two or more
    image-bearing class folders.

    Nested wrapper folders are detected recursively.
    """
    if _depth >= MAX_WRAPPER_DEPTH:
        return None

    top_dirs = {p[0] for _, p in real if len(p) >= 2}

    if len(top_dirs) != 1:
        return None

    top = next(iter(top_dirs))

    if top.lower() in KNOWN_SPLITS:
        return None

    # Do not flatten mixed root-level content.
    if any(len(p) == 1 and _is_image(p[0]) for _, p in real):
        return None

    # Do not flatten a folder containing images directly.
    if any(
        len(p) == 2 and p[0] == top and _is_image(p[1])
        for _, p in real
    ):
        return None

    children = {
        p[1]
        for _, p in real
        if len(p) >= 3 and p[0] == top
    }

    # Directly contains split folders.
    if any(c.lower() in KNOWN_SPLITS for c in children):
        return top

    # Directly contains at least two image-bearing class folders.
    image_children = {
        p[1]
        for _, p in real
        if len(p) >= 3 and p[0] == top and _is_image(p[-1])
    }

    if len(image_children) >= 2:
        return top

    # Check whether the only remaining structure contains
    # another valid wrapper.
    nested = [
        (info, parts[1:])
        for info, parts in real
        if len(parts) >= 2 and parts[0] == top
    ]

    if nested and len(nested) == len(real):
        if _wrapper_name(nested, _depth + 1) is not None:
            return top

    return None


# ─── Planning ─────────────────────────────────────────────────────────────────

def _first(items, n=10) -> list:
    return sorted(items)[:n]


def _plan_from_infos(infos: list, limits: UploadLimits) -> DatasetPlan:
    if len(infos) > limits.max_entries:
        raise TooManyEntriesError(
            f"Archive has {len(infos):,} entries; the limit is "
            f"{limits.max_entries:,}.",
            details={"entries": len(infos), "limit": limits.max_entries},
        )

    ignored = []          # (filename, reason)
    real = []             # (ZipInfo, parts)
    total_uncompressed = 0
    total_compressed = 0

    for info in infos:
        parts = _check_member(info)
        if info.is_dir() or not parts:
            continue

        total_uncompressed += info.file_size
        total_compressed += info.compress_size

        if (info.file_size >= MB and info.file_size
                > limits.max_compression_ratio * max(info.compress_size, 1)):
            raise CompressionRatioError(
                "Archive compression ratio is too high.",
                details={"entry": _display(info.filename)},
            )
        if _is_junk(parts):
            ignored.append((info.filename, "junk"))
        else:
            real.append((info, parts))

    if total_uncompressed > limits.max_uncompressed_bytes:
        raise UploadTooLargeError(
            "Extracted dataset would exceed the size limit of "
            f"{limits.max_uncompressed_bytes // MB:,} MB.",
            details={"limit_mb": limits.max_uncompressed_bytes // MB},
        )
    if (total_uncompressed >= MB and total_uncompressed
            > limits.max_compression_ratio * max(total_compressed, 1)):
        raise CompressionRatioError("Archive compression ratio is too high.")

    # ── Strip wrapper folders ─────────────────────────────────────
    stripped = []
    for _ in range(MAX_WRAPPER_DEPTH):
        top = _wrapper_name(real)
        if top is None:
            break
        stripped.append(top)
        kept = []
        for info, parts in real:
            if len(parts) >= 2 and parts[0] == top:
                kept.append((info, parts[1:]))
            else:
                ignored.append((info.filename, "outside_wrapper"))
        real = kept

    # ── Detect layout from image-bearing top-level folders ────────
    tops = {p[0] for _, p in real if len(p) >= 2 and _is_image(p[-1])}
    if not tops:
        raise EmptyDatasetError(
            "No usable images were found. Expected class folders "
            "containing .jpg, .jpeg or .png files.",
            details={"ignored_files": len(ignored) + len(real)},
        )

    split_tops = {t for t in tops if t.lower() in KNOWN_SPLITS}
    other_tops = tops - split_tops
    if split_tops and other_tops:
        raise InvalidLayoutError(
            "Archive mixes split folders (train/val/test) with class "
            "folders at the top level.",
            details={"split_folders": _first(split_tops),
                     "other_folders": _first(other_tops)},
        )

    layout = "split" if split_tops else "flat"
    normalised = {}
    if layout == "split":
        groups = defaultdict(list)
        for t in split_tops:
            groups[t.lower()].append(t)
        clashes = {n: sorted(v) for n, v in groups.items() if len(v) > 1}
        if clashes:
            raise InvalidLayoutError(
                "Split folder names are ambiguous (differ only by case).",
                details={"folders": clashes},
            )
        present = set(groups)
        for alias_group in SPLIT_ALIAS_GROUPS:
            if len(alias_group & present) > 1:
                raise InvalidLayoutError(
                    "Split folder names are ambiguous "
                    f"({' and '.join(sorted(alias_group & present))}).",
                    details={"folders": sorted(alias_group & present)},
                )
        normalised = {t: t.lower() for t in split_tops}

    # ── Classify every remaining file ─────────────────────────────
    depth_ok = 3 if layout == "split" else 2
    accepted = []   # (info, parts)
    for info, parts in real:
        if not _is_image(parts[-1]):
            ignored.append((info.filename, "not_an_image"))
        elif len(parts) == 1:
            ignored.append((info.filename, "root_level"))
        elif layout == "split" and len(parts) == 2:
            ignored.append((info.filename, "not_in_class_folder"))
        elif len(parts) > depth_ok:
            ignored.append((info.filename, "nested_too_deep"))
        else:
            accepted.append((info, parts))

    # ── Build planned files and validate classes ──────────────────
    files = []
    classes_by_split = defaultdict(set)
    counts = defaultdict(Counter)
    for info, parts in accepted:
        if layout == "split":
            split, class_name = normalised[parts[0]], parts[1]
        else:
            split, class_name = "train", parts[0]
        classes_by_split[split].add(class_name)
        counts[split][class_name] += 1
        files.append(PlannedFile(
            source=info.filename,
            dest=f"{split}/{class_name}/{parts[-1]}",
            split=split,
            class_name=class_name,
            size=info.file_size,
            compress_size=info.compress_size,
        ))

    all_classes = set().union(*classes_by_split.values())

    bad_names = [c for c in all_classes if not CLASS_NAME_PATTERN.fullmatch(c)]
    if bad_names:
        raise InvalidLayoutError(
            "Class folder names may contain only letters, digits, spaces, "
            "'_', '-' and '.', and must be 1-64 characters.",
            details={"invalid_names": _first(bad_names, 5)},
        )

    for split, class_set in classes_by_split.items():
        folded = defaultdict(list)
        for c in class_set:
            folded[c.casefold()].append(c)
        clashes = {k: sorted(v) for k, v in folded.items() if len(v) > 1}
        if clashes:
            raise InvalidLayoutError(
                "Class folder names differ only by case.",
                details={"split": split, "classes": clashes},
            )

    mismatch = {
        s: sorted(all_classes - cs)
        for s, cs in classes_by_split.items() if cs != all_classes
    }
    if mismatch:
        raise InvalidLayoutError(
            "Every split must contain the same classes.",
            details={"missing_classes_by_split": mismatch},
        )

    seen = {}
    collisions = []
    for f in files:
        key = unicodedata.normalize("NFC", f.dest).casefold()
        if key in seen:
            collisions.append((seen[key], f.dest))
        else:
            seen[key] = f.dest
    if collisions:
        raise InvalidLayoutError(
            "Archive contains file names that collide (duplicates or "
            "names differing only by case).",
            details={"examples": [list(c) for c in collisions[:5]]},
        )

    # ── Dataset-size rules ────────────────────────────────────────
    total_images = len(files)
    if total_images > limits.max_images:
        raise UploadTooLargeError(
            f"Dataset has {total_images:,} images; the limit is "
            f"{limits.max_images:,}.",
            details={"images": total_images, "limit": limits.max_images},
        )
    if len(all_classes) < limits.min_classes:
        raise InsufficientDataError(
            f"At least {limits.min_classes} classes are required; "
            f"found {len(all_classes)}.",
            details={"classes": sorted(all_classes)},
        )
    # PROVISIONAL: applied per (split, class). Step 5 sets the real value.
    too_few = [
        {"split": s, "class": c, "images": counts[s][c]}
        for s in sorted(classes_by_split)
        for c in sorted(all_classes)
        if counts[s][c] < limits.min_images_per_class
    ]
    if too_few:
        raise InsufficientDataError(
            f"Each class needs at least {limits.min_images_per_class} "
            "images in every split.",
            details={"examples": too_few[:10],
                     "min_images_per_class": limits.min_images_per_class},
        )

    files.sort(key=lambda f: f.dest)
    reasons = Counter(reason for _, reason in ignored)
    return DatasetPlan(
        layout=layout,
        splits=sorted(classes_by_split),
        classes=sorted(all_classes),
        counts={s: dict(sorted(counts[s].items()))
                for s in sorted(counts)},
        files=files,
        wrapper_stripped=stripped,
        total_images=total_images,
        total_uncompressed_bytes=total_uncompressed,
        ignored_total=len(ignored),
        ignored_by_reason=dict(reasons),
        ignored_samples=[
            {"path": _display(name), "reason": reason}
            for name, reason in ignored[:IGNORED_SAMPLE_LIMIT]
        ],
    )


def inspect_zip(
    zip_path, limits: UploadLimits = LIMITS
) -> DatasetPlan:
    """Validate a ZIP and return an extraction plan. Writes nothing."""
    path = Path(zip_path)
    if path.stat().st_size > limits.max_upload_bytes:
        raise UploadTooLargeError(
            f"Upload exceeds the {limits.max_upload_bytes // MB:,} MB limit.",
            details={"limit_mb": limits.max_upload_bytes // MB},
        )
    _check_central_directory(path, limits)

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, NotImplementedError, EOFError,
            ValueError, struct.error):
        raise UnsupportedArchiveError(
            "The uploaded file is not a valid ZIP archive.") from None

    return _plan_from_infos(infos, limits)