# cleaner_client.py
"""
HTTP client and pure helpers for the Streamlit cleaning UI (Step 8).

No Streamlit import, so everything here is unit-testable. A custom
httpx.Client can be injected (tests pass the FastAPI TestClient).
"""
# Standard library
import os
import re
from typing import Optional

# Third-party
import httpx

DEFAULT_API_URL = "http://localhost:8000"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_OVERRIDES = 10_000   # server limit per list (keep / remove)

MODULES = (
    ("run_duplicates", "Duplicate detection"),
    ("run_blur", "Blur detection"),
    ("run_noise", "Noise detection"),
    ("run_outliers", "Outlier detection"),
    ("run_mislabels", "Mislabel detection"),
)
CHECKS_NOTE = (
    "CLEAN means no issue was found by the checks you enabled. "
    "It is not a guarantee that an image is free of every problem."
)


class ApiError(Exception):
    """A failed API call, with a message that is safe to show a user."""

    def __init__(self, status: int, code: str, message: str,
                 details: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = dict(details or {})


def _parse_error(resp: httpx.Response) -> ApiError:
    code = "http_error"
    message = f"The request failed (HTTP {resp.status_code})."
    details: dict = {}
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        detail = body.get("detail")
        if isinstance(error, dict):
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
            details = error.get("details") or {}
        elif isinstance(detail, str) and resp.status_code < 500:
            message = detail
        elif isinstance(detail, list):
            code, message = "invalid_request", "The request was not valid."
    return ApiError(resp.status_code, code, message, details)


# ─── Client ───────────────────────────────────────────────────────────────────

class CleanerClient:
    def __init__(self, base_url: Optional[str] = None,
                 http: Optional[httpx.Client] = None,
                 access_code: Optional[str] = None):
        self._http = http or httpx.Client(
            base_url=base_url or os.environ.get(
                "IMAGE_CLEANING_API_URL", DEFAULT_API_URL),
            timeout=httpx.Timeout(30.0, read=600.0, write=600.0),
        )
        if access_code:
            self._http.headers["X-Access-Code"] = access_code

    def delete_job(self, job_id: str) -> dict:
        return self._send("DELETE", self._job_path(job_id)).json()

    def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            resp = self._http.request(method, url, **kwargs)
        except httpx.RequestError:
            raise ApiError(0, "unreachable",
                           "Could not reach the cleaning service. "
                           "Try again in a moment.") from None
        if resp.status_code >= 400:
            raise _parse_error(resp)
        return resp

    @staticmethod
    def _job_path(job_id, suffix: str = "") -> str:
        if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
            raise ApiError(400, "invalid_job_id", "That job link is not valid.")
        return f"/jobs/{job_id}{suffix}"

    def upload_zip(self, data: bytes) -> dict:
        return self._send("POST", "/datasets/upload", content=data,
                          headers={"content-type": "application/zip"}).json()

    def job(self, job_id: str) -> dict:
        return self._send("GET", self._job_path(job_id)).json()

    def start_run(self, job_id: str, options: dict) -> dict:
        return self._send("POST", self._job_path(job_id, "/run"),
                          json=options).json()

    def results(self, job_id: str) -> dict:
        return self._send("GET", self._job_path(job_id, "/results")).json()

    def images(self, job_id: str, verdict: Optional[str] = None,
               class_name: Optional[str] = None, limit: int = 12,
               offset: int = 0) -> dict:
        params = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        if class_name:
            params["class"] = class_name
        return self._send("GET", self._job_path(job_id, "/images"),
                          params=params).json()

    def image_bytes(self, job_id: str, path: str) -> bytes:
        return self._send("GET", self._job_path(job_id, "/images/file"),
                          params={"path": path}).content

    def approve(self, job_id: str, payload: dict) -> dict:
        return self._send("POST", self._job_path(job_id, "/approve"),
                          json=payload).json()

    def download(self, job_id: str) -> tuple:
        resp = self._send("GET", self._job_path(job_id, "/download"))
        try:
            missing = int(resp.headers.get("x-missing-files", "0"))
        except ValueError:
            missing = 0
        return resp.content, missing


# ─── Pure helpers ─────────────────────────────────────────────────────────────

_STAGES = {
    "pending": "working", "queued": "working", "running": "working",
    "ready": "configure", "completed": "review",
    "failed": "failed", "interrupted": "interrupted",
}


def stage_for(status: str) -> str:
    return _STAGES.get(status, "unknown")


def job_error_text(job: dict) -> str:
    result = job.get("result")
    if isinstance(result, dict) and isinstance(result.get("error"), dict):
        message = result["error"].get("message")
        if message:
            return str(message)
    return str(job.get("error") or job.get("message") or "The job failed.")


def default_decision(group: str, apply_removals: bool,
                     review_default: str) -> str:
    if group == "REMOVE":
        return "remove" if apply_removals else "keep"
    if group == "REVIEW":
        return review_default
    return "keep"


def build_approval_payload(apply_removals: bool, review_default: str,
                           overrides: dict) -> dict:
    keep = sorted(p for p, (d, _g) in overrides.items() if d == "keep")
    remove = sorted(p for p, (d, _g) in overrides.items() if d == "remove")
    if len(keep) > MAX_OVERRIDES or len(remove) > MAX_OVERRIDES:
        raise ValueError(
            f"Too many individual choices (limit {MAX_OVERRIDES:,} per list).")
    return {"apply_proposed_removals": apply_removals,
            "review_default": review_default,
            "keep": keep, "remove": remove}


def preview_counts(totals: dict, apply_removals: bool, review_default: str,
                   overrides: dict) -> dict:
    kept = totals.get("clean", 0)
    if review_default == "keep":
        kept += totals.get("review", 0)
    if not apply_removals:
        kept += totals.get("remove", 0)
    for decision, group in overrides.values():
        if decision != default_decision(group, apply_removals, review_default):
            kept += 1 if decision == "keep" else -1
    total = sum(totals.get(k, 0) for k in ("clean", "review", "remove"))
    return {"keep": kept, "remove": total - kept}


def describe_reason(reason: dict) -> str:
    label = str(reason.get("label") or reason.get("code") or "Issue")
    detail = reason.get("detail") or {}
    bits = [f"{key.replace('_', ' ')}: {value}"
            for key, value in detail.items()
            if isinstance(value, (int, float, str)) and value != ""]
    return f"{label} ({', '.join(bits[:3])})" if bits else label

HIGH_REMOVAL_PERCENT = 30   # judgement call; Animal was 4.65%, Intel ~3%


def override_class(path: str) -> Optional[str]:
    """Class name from a dataset-relative path split/class/file, else None."""
    parts = path.split("/")
    return parts[1] if len(parts) == 3 else None


def class_preview(by_class: dict, apply_removals: bool, review_default: str,
                  overrides: dict) -> dict:
    """Images kept per class under the current choices."""
    kept = {}
    for name, counts in by_class.items():
        n = counts.get("clean", 0)
        if review_default == "keep":
            n += counts.get("review", 0)
        if not apply_removals:
            n += counts.get("remove", 0)
        kept[name] = n
    for path, (decision, group) in overrides.items():
        name = override_class(path)
        if name not in kept:
            continue
        if decision != default_decision(group, apply_removals, review_default):
            kept[name] += 1 if decision == "keep" else -1
    return kept


def empty_classes(kept_by_class: dict) -> list:
    return sorted(name for name, n in kept_by_class.items() if n <= 0)


def removal_warning(totals: dict) -> Optional[dict]:
    """Return details when proposed removals reach HIGH_REMOVAL_PERCENT."""
    images = totals.get("images", 0)
    remove = totals.get("remove", 0)
    if images <= 0 or remove * 100 < images * HIGH_REMOVAL_PERCENT:
        return None
    return {"remove": remove, "images": images,
            "share_percent": round(100 * remove / images, 1)}