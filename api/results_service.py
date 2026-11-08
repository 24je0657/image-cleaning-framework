# api/results_service.py
"""
Results review and approval for a completed job (Step 6). No FastAPI.

Reads the job's master reports, lists flagged images with reasons, and
builds final/ from dataset/ after explicit approval. dataset/ is never
modified. Every client-supplied path is validated against the reports.
"""
# Standard library
import csv
import logging
import os
import shutil
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Third-party
import pandas as pd

# Local
from api.workspace import JobWorkspace
from api.job_locks import job_lock

logger = logging.getLogger(__name__)

GROUPS = ("CLEAN", "REVIEW", "REMOVE")
MAX_LISTED_UNKNOWN = 5


# ─── Exceptions ───────────────────────────────────────────────────────────────

class ResultsError(Exception):
    code = "results_error"
    http_status = 400

    def __init__(self, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "details": self.details}


class ResultsNotAvailableError(ResultsError):
    code = "results_not_available"
    http_status = 409


class InvalidSelectionError(ResultsError):
    code = "invalid_selection"
    http_status = 422


class ImageNotFoundError(ResultsError):
    code = "image_not_found"
    http_status = 404


class ApprovalInProgressError(ResultsError):
    code = "approval_in_progress"
    http_status = 409


# ─── Records ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImageRecord:
    path: str            # dataset-relative, e.g. train/cat/0.jpg
    split: str
    class_name: str
    verdict: str
    group: str           # CLEAN / REVIEW / REMOVE
    priority: float
    reasons: tuple
    source: Path         # resolved path inside the job's dataset/


@dataclass
class ResultSet:
    records: dict
    skipped_rows: int
    layout: str
    splits: list
    classes: list


def verdict_group(verdict: str) -> str:
    upper = verdict.strip().upper()
    if upper.startswith("REMOVE"):
        return "REMOVE"
    if upper == "CLEAN":
        return "CLEAN"
    return "REVIEW"      # REVIEW-*, and anything unrecognised, needs a human


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "1.0", "yes")


def _num(value) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _rel_in_dataset(value, root: Path) -> Optional[str]:
    if not value:
        return None
    try:
        return Path(value).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return None


def _reasons(row: dict, root: Path) -> tuple:
    def add(code, label, **detail):
        out.append({"code": code, "label": label,
                    "detail": {k: v for k, v in detail.items()
                               if v not in (None, "")}})

    out: list = []
    if _truthy(row.get("Exact_duplicates")):
        add("exact_duplicate", "Exact duplicate",
            duplicate_of=_rel_in_dataset(row.get("duplicate_of"), root))
    if _truthy(row.get("near_duplicate")):
        add("near_duplicate", "Near duplicate",
            confidence=row.get("dup_confidence"),
            cosine_score=_num(row.get("cosine_score")),
            similar_to=_rel_in_dataset(row.get("near_duplicate_of"), root))
    if _truthy(row.get("is_blurry")):
        add("blurry", "Low sharpness",
            blur_score=_num(row.get("blur_score")),
            threshold=_num(row.get("threshold_used")))
    noise = str(row.get("noise_verdict", "")).strip()
    if noise and noise != "not_flagged":
        add("noise_or_ood", "Possible noise or unusual content",
            verdict=noise,
            reconstruction_error=_num(row.get("reconstruction_error")))
    if _truthy(row.get("is_outlier")):
        add("outlier", "Statistical outlier",
            score=_num(row.get("global_score")))
    if _truthy(row.get("is_mislabeled")):
        add("mislabeled", "Possibly mislabeled",
            confidence=row.get("mislabel_confidence"),
            predicted_class=row.get("nn_predicted"))
    verdict = str(row.get("verdict", "")).lower()
    if "incomplete analysis" in verdict or "not analysed" in verdict:
        add("incomplete_analysis", "Not every enabled check covered this image")
    return tuple(out)


def _load(ws: JobWorkspace, meta: dict, paths: list) -> ResultSet:
    dataset = meta["dataset"]
    splits = list(dataset["splits"])
    root = ws.dataset_dir.resolve()
    records: dict = {}
    skipped = 0

    for split, report in zip(splits, paths):
        df = pd.read_csv(report, dtype=str, keep_default_na=False)
        missing = {"file_path", "verdict"} - set(df.columns)
        if missing:
            raise ResultsNotAvailableError(
                "Master report is missing required columns.",
                details={"split": split, "columns": sorted(missing)})
        for row in df.to_dict("records"):
            rel = _rel_in_dataset(row["file_path"], root)
            parts = rel.split("/") if rel else []
            if (not row["verdict"].strip() or len(parts) != 3
                    or parts[0] != split or rel in records):
                skipped += 1
                continue
            verdict = row["verdict"].strip()
            records[rel] = ImageRecord(
                path=rel, split=split, class_name=parts[1],
                verdict=verdict, group=verdict_group(verdict),
                priority=_num(row.get("priority_score")) or 0.0,
                reasons=_reasons(row, root),
                source=root.joinpath(*parts))

    if not records:
        raise ResultsNotAvailableError(
            "The master report contains no images for this dataset.")
    return ResultSet(records, skipped, dataset.get("layout", "split"),
                     splits, list(dataset.get("classes", [])))


_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 8


def get_results(ws: JobWorkspace) -> ResultSet:
    meta = ws.read_meta()
    if meta.get("status") != "completed":
        raise ResultsNotAvailableError(
            "Results are available once the pipeline run has completed.",
            details={"status": meta.get("status")})
    dataset = meta.get("dataset")
    splits = dataset.get("splits") if isinstance(dataset, dict) else None
    if not isinstance(splits, list) or not splits:
        raise ResultsNotAvailableError("Job has no dataset information.")

    reports = [ws.reports_dir / f"{s}_master_report.csv" for s in splits]
    try:
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size)
                          for p in reports)
    except OSError:
        raise ResultsNotAvailableError("Master report is missing.") from None

    with _CACHE_LOCK:
        hit = _CACHE.get(ws.job_id)
        if hit and hit[0] == signature:
            return hit[1]
    result = _load(ws, meta, reports)
    with _CACHE_LOCK:
        _CACHE.pop(ws.job_id, None)
        _CACHE[ws.job_id] = (signature, result)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
    return result


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


# ─── Queries ──────────────────────────────────────────────────────────────────

def display_path(result: ResultSet, path: str) -> str:
    """Path as the user will see it in the downloaded dataset."""
    parts = path.split("/")
    return "/".join(parts[1:] if result.layout == "flat" else parts)


def _counts(records) -> dict:
    by_group = Counter(r.group for r in records)
    return {"images": len(records), "clean": by_group["CLEAN"],
            "review": by_group["REVIEW"], "remove": by_group["REMOVE"]}


def summarize(result: ResultSet) -> dict:
    records = list(result.records.values())
    return {
        "layout": result.layout,
        "totals": _counts(records),
        "by_verdict": dict(sorted(Counter(r.verdict for r in records).items())),
        "by_split": {s: _counts([r for r in records if r.split == s])
                     for s in result.splits},
        "by_class": {c: _counts([r for r in records if r.class_name == c])
                     for c in sorted({r.class_name for r in records})},
        "reasons": dict(Counter(x["code"] for r in records
                                for x in r.reasons)),
        "unmatched_report_rows": result.skipped_rows,
    }


def _item(result: ResultSet, rec: ImageRecord) -> dict:
    return {"path": rec.path,
            "display_path": display_path(result, rec.path),
            "split": rec.split, "class": rec.class_name,
            "verdict": rec.verdict, "group": rec.group,
            "priority_score": rec.priority,
            "reasons": [dict(r) for r in rec.reasons]}


def list_images(result: ResultSet, *, group=None, split=None,
                class_name=None, limit=50, offset=0) -> dict:
    selected = [r for r in result.records.values()
                if (group is None or r.group == group)
                and (split is None or r.split == split)
                and (class_name is None or r.class_name == class_name)]
    selected.sort(key=lambda r: (-r.priority, r.path))
    page = selected[offset:offset + limit]
    return {"total": len(selected), "limit": limit, "offset": offset,
            "items": [_item(result, r) for r in page]}


def get_record(result: ResultSet, path: str) -> ImageRecord:
    rec = result.records.get(path)
    if rec is None:
        raise ImageNotFoundError("Image not found in this job.")
    return rec


# ─── Approval ─────────────────────────────────────────────────────────────────

def plan_decisions(result: ResultSet, *, keep, remove, review_default: str,
                   apply_proposed_removals: bool) -> dict:
    keep_set, remove_set = set(keep), set(remove)
    unknown = sorted((keep_set | remove_set) - result.records.keys())
    if unknown:
        raise InvalidSelectionError(
            "Some submitted paths do not belong to this job.",
            details={"unknown_count": len(unknown),
                     "examples": [u[:100] for u in unknown[:MAX_LISTED_UNKNOWN]]})
    overlap = keep_set & remove_set
    if overlap:
        raise InvalidSelectionError(
            "Some paths were submitted in both keep and remove.",
            details={"conflict_count": len(overlap)})

    decisions = {}
    for path, rec in result.records.items():
        if path in remove_set:
            decision = "remove"
        elif path in keep_set:
            decision = "keep"
        elif rec.group == "REMOVE":
            decision = "remove" if apply_proposed_removals else "keep"
        elif rec.group == "REVIEW":
            decision = review_default
        else:
            decision = "keep"
        decisions[path] = decision
    if "keep" not in decisions.values():
        raise InvalidSelectionError("These choices would remove every image.")
    return decisions


_BUILDING: set = set()
_BUILD_GUARD = threading.Lock()


def approve(ws: JobWorkspace, result: ResultSet, *, keep, remove,
            review_default: str, apply_proposed_removals: bool) -> dict:
    decisions = plan_decisions(
        result, keep=keep, remove=remove, review_default=review_default,
        apply_proposed_removals=apply_proposed_removals)

    with _BUILD_GUARD:
        if ws.job_id in _BUILDING:
            raise ApprovalInProgressError(
                "An approval is already being built for this job.")
        _BUILDING.add(ws.job_id)
    try:
        with job_lock(ws.job_id):
            return _build_final(ws, result, decisions, keep, remove,
                                review_default, apply_proposed_removals)
    finally:
        with _BUILD_GUARD:
            _BUILDING.discard(ws.job_id)


def _build_final(
    ws,
    result,
    decisions,
    keep,
    remove,
    review_default,
    apply_proposed_removals,
) -> dict:

    dataset_root = ws.dataset_dir.resolve()

    ws.reports_dir.mkdir(parents=True, exist_ok=True)

    # Temporary output on the same filesystem as final/.
    tmp_final = Path(tempfile.mkdtemp(
        prefix=".final-",
        dir=ws.root,
    ))

    # Temporary CSV on the same filesystem as approval_decisions.csv.
    fd, csv_name = tempfile.mkstemp(
        prefix=".approval-",
        suffix=".tmp",
        dir=ws.reports_dir,
    )
    os.close(fd)
    tmp_csv = Path(csv_name)

    final_backup = None
    csv_backup = None
    csv_backup_dir = None

    final_installed = False
    csv_installed = False

    kept_by_class: Counter = Counter()
    missing = 0

    try:
        # --------------------------------------------------
        # 1. Build the new final directory.
        # --------------------------------------------------
        for path in sorted(result.records):
            if decisions[path] != "keep":
                continue

            rec = result.records[path]
            src = rec.source

            try:
                ok = (
                    src.is_file()
                    and not src.is_symlink()
                    and src.resolve().is_relative_to(dataset_root)
                )
            except (OSError, ValueError):
                ok = False

            if not ok:
                missing += 1
                continue

            relative = display_path(result, path)
            target = tmp_final.joinpath(*relative.split("/"))

            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)

            kept_by_class[rec.class_name] += 1

        # --------------------------------------------------
        # 2. Prepare the decision CSV before replacing anything.
        # --------------------------------------------------
        _write_decisions(
            ws,
            result,
            decisions,
            target=tmp_csv,
        )

        kept = sum(kept_by_class.values())
        removed = sum(
            1 for decision in decisions.values()
            if decision == "remove"
        )

        stats = {
            "kept": kept,
            "removed": removed,
            "rescued_from_remove": sum(
                1 for path, decision in decisions.items()
                if decision == "keep"
                and result.records[path].group == "REMOVE"
            ),
            "removed_from_review": sum(
                1 for path, decision in decisions.items()
                if decision == "remove"
                and result.records[path].group == "REVIEW"
            ),
            "missing_files": missing,
            "warnings": (
                [
                    {"code": "class_empty", "class": cls}
                    for cls in sorted(
                        {
                            rec.class_name
                            for rec in result.records.values()
                        } - set(kept_by_class)
                    )
                ]
                + (
                    [{"code": "missing_files", "count": missing}]
                    if missing else []
                )
            ),
        }

        # --------------------------------------------------
        # 3. Reserve backup locations.
        # --------------------------------------------------
        final_backup = Path(tempfile.mkdtemp(
            prefix=".final-backup-",
            dir=ws.root,
        ))
        final_backup.rmdir()

        csv_backup_dir = Path(tempfile.mkdtemp(
            prefix=".approval-backup-",
            dir=ws.reports_dir,
        ))
        csv_backup = csv_backup_dir / "approval_decisions.csv"

        # --------------------------------------------------
        # 4. Move old outputs aside.
        # --------------------------------------------------
        if ws.final_dir.exists():
            os.replace(ws.final_dir, final_backup)

        old_csv = ws.reports_dir / "approval_decisions.csv"

        if old_csv.exists():
            os.replace(old_csv, csv_backup)

        # --------------------------------------------------
        # 5. Install the new outputs.
        # --------------------------------------------------
        os.replace(tmp_final, ws.final_dir)
        final_installed = True

        os.replace(tmp_csv, old_csv)
        csv_installed = True

        # --------------------------------------------------
        # 6. Update metadata last.
        # JobWorkspace already writes metadata atomically.
        # --------------------------------------------------
        ws.update_meta(
            approval={
                **stats,
                "apply_proposed_removals": apply_proposed_removals,
                "review_default": review_default,
                "explicit_keep": len(set(keep)),
                "explicit_remove": len(set(remove)),
                "approved_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

        # --------------------------------------------------
        # 7. Success: old backups are no longer needed.
        # Cleanup errors must not invalidate a successful approval.
        # --------------------------------------------------
        if final_backup.exists():
            shutil.rmtree(final_backup, ignore_errors=True)

        if csv_backup_dir is not None and csv_backup_dir.exists():
            shutil.rmtree(csv_backup_dir, ignore_errors=True)

        return stats


    except BaseException:
        # --------------------------------------------------
        # Roll back the new final directory.
        # Preserve backups if restoration fails.
        # --------------------------------------------------
        try:
            if final_installed and ws.final_dir.exists():
                shutil.rmtree(ws.final_dir)

            if final_backup is not None and final_backup.exists():
                os.replace(final_backup, ws.final_dir)

        except BaseException:
            logger.exception(
                "Could not restore previous final directory for job %s. "
                "Backup preserved at %s",
                ws.job_id,
                final_backup,
            )

        # --------------------------------------------------
        # Roll back the approval CSV independently.
        # --------------------------------------------------
        current_csv = ws.reports_dir / "approval_decisions.csv"

        try:
            if csv_installed and current_csv.exists():
                current_csv.unlink()

            if csv_backup is not None and csv_backup.exists():
                os.replace(csv_backup, current_csv)

        except BaseException:
            logger.exception(
                "Could not restore previous approval CSV for job %s. "
                "Backup preserved at %s",
                ws.job_id,
                csv_backup,
            )

        # --------------------------------------------------
        # Remove the CSV backup directory only if its
        # backup file has been successfully restored.
        # --------------------------------------------------
        if (
            csv_backup_dir is not None
            and csv_backup_dir.exists()
            and (csv_backup is None or not csv_backup.exists())
        ):
            try:
                csv_backup_dir.rmdir()
            except OSError:
                logger.warning(
                    "Could not remove empty approval backup directory: %s",
                    csv_backup_dir,
                )

        logger.exception(
            "Approval failed for job %s; rollback attempted",
            ws.job_id,
        )
        raise


    finally:
        # Clean temporary, uninstalled output only.
        if tmp_final.exists():
            shutil.rmtree(tmp_final, ignore_errors=True)

        if tmp_csv.exists():
            tmp_csv.unlink(missing_ok=True)

        # Do not delete backup directories here.
        # A failed rollback may still need them.


def _write_decisions(
    ws: JobWorkspace,
    result: ResultSet,
    decisions: dict,
    target: Optional[Path] = None,
) -> None:
    """Write approval decisions to the specified path."""
    if target is None:
        target = ws.reports_dir / "approval_decisions.csv"

    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "path", "split", "class", "verdict", "decision", "reasons"
        ])

        for path in sorted(result.records):
            rec = result.records[path]
            writer.writerow([
                display_path(result, path),
                rec.split,
                rec.class_name,
                rec.verdict,
                decisions[path],
                ";".join(r["code"] for r in rec.reasons),
            ])