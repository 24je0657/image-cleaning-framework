# tests/test_cleaner_client.py
# Standard library
import io
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import httpx  # noqa: E402

# Local
from cleaner_client import (  # noqa: E402
    ApiError,
    CleanerClient,
    build_approval_payload,
    default_decision,
    describe_reason,
    job_error_text,
    preview_counts,
    stage_for,
)
from test_job_results import isolated, make_job  # noqa: E402
from api import job_runner  # noqa: E402
from api import upload_config as uc  # noqa: E402
from api.routes import datasets as dataset_routes  # noqa: E402

JOB = "a" * 32


def mock_client(handler) -> CleanerClient:
    return CleanerClient(http=httpx.Client(
        base_url="http://api", transport=httpx.MockTransport(handler)))


def expect_api_error(fn, *args, **kwargs) -> ApiError:
    try:
        fn(*args, **kwargs)
    except ApiError as exc:
        return exc
    raise AssertionError("ApiError not raised")


def test_error_parsing() -> None:
    cases = [
        (409, {"error": {"code": "results_not_available", "message": "m",
                         "details": {"a": 1}}}, "results_not_available", "m"),
        (429, {"detail": "busy"}, "http_error", "busy"),
        (422, {"detail": [{"msg": "x"}]}, "invalid_request",
         "The request was not valid."),
    ]
    for status, body, code, message in cases:
        client = mock_client(lambda r, s=status, b=body: httpx.Response(s, json=b))
        exc = expect_api_error(client.job, JOB)
        assert (exc.status, exc.code, exc.message) == (status, code, message)
    assert expect_api_error(
        mock_client(lambda r: httpx.Response(
            409, json={"error": {"code": "c", "message": "m",
                                 "details": {"a": 1}}})).job, JOB
    ).details == {"a": 1}

    leak = mock_client(lambda r: httpx.Response(
        500, text="Traceback /home/secret/x.py"))
    exc = expect_api_error(leak.job, JOB)
    assert exc.status == 500 and "secret" not in exc.message
    leak = mock_client(lambda r: httpx.Response(500, json={"detail": "/srv/secret"}))
    assert "secret" not in expect_api_error(leak.job, JOB).message

    def refuse(request):
        raise httpx.ConnectError("down")

    exc = expect_api_error(mock_client(refuse).job, JOB)
    assert (exc.status, exc.code) == (0, "unreachable")


def test_job_id_is_validated_before_any_request() -> None:
    calls = []
    client = mock_client(lambda r: calls.append(r) or httpx.Response(200, json={}))
    for bad in ("../x", "short", "A" * 32, "", None, JOB + "/x"):
        for fn in (client.job, client.results, client.images, client.download,
                   lambda j: client.start_run(j, {}),
                   lambda j: client.approve(j, {}),
                   lambda j: client.image_bytes(j, "p")):
            assert expect_api_error(fn, bad).code == "invalid_job_id"
    assert calls == []


def test_request_shapes() -> None:
    seen = []
    client = mock_client(lambda r: seen.append(r) or httpx.Response(
        200, json={}, headers={"x-missing-files": "3"}, ))
    client.images(JOB, verdict="REMOVE", class_name="cat", limit=5, offset=10)
    client.images(JOB)
    client.upload_zip(b"PK-data")
    client.image_bytes(JOB, "train/cat/0.jpg")
    first, second, upload, image = seen
    assert dict(first.url.params) == {
        "limit": "5", "offset": "10", "verdict": "REMOVE", "class": "cat"}
    assert dict(second.url.params) == {"limit": "12", "offset": "0"}
    assert upload.headers["content-type"] == "application/zip"
    assert upload.content == b"PK-data"
    assert image.url.params["path"] == "train/cat/0.jpg"
    assert client.download(JOB) == (b"{}", 3)
    odd = mock_client(lambda r: httpx.Response(
        200, content=b"z", headers={"x-missing-files": "oops"}))
    assert odd.download(JOB) == (b"z", 0)


def test_pure_helpers() -> None:
    assert [stage_for(s) for s in (
        "pending", "queued", "running", "ready", "completed", "failed",
        "interrupted", "weird")] == [
        "working", "working", "working", "configure", "review", "failed",
        "interrupted", "unknown"]
    assert job_error_text({"result": {"error": {"message": "bad zip"}},
                           "error": "x"}) == "bad zip"
    assert job_error_text({"error": "Pipeline failed: train: blur"}) == (
        "Pipeline failed: train: blur")
    assert job_error_text({}) == "The job failed."

    assert default_decision("REMOVE", True, "keep") == "remove"
    assert default_decision("REMOVE", False, "keep") == "keep"
    assert default_decision("REVIEW", False, "remove") == "remove"
    assert default_decision("CLEAN", True, "remove") == "keep"

    reason = {"label": "Low sharpness",
              "detail": {"blur_score": 12.5, "threshold": 60, "x": "", "y": None}}
    assert describe_reason(reason) == (
        "Low sharpness (blur score: 12.5, threshold: 60)")
    assert describe_reason({"code": "outlier"}) == "outlier"


def test_payload_and_preview() -> None:
    overrides = {"b": ("keep", "REMOVE"), "a": ("remove", "REVIEW"),
                 "c": ("remove", "CLEAN")}
    payload = build_approval_payload(True, "keep", overrides)
    assert payload == {"apply_proposed_removals": True,
                       "review_default": "keep",
                       "keep": ["b"], "remove": ["a", "c"]}
    big = {f"p{i}": ("keep", "CLEAN") for i in range(10_001)}
    try:
        build_approval_payload(True, "keep", big)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized override list accepted")

    totals = {"clean": 4, "review": 2, "remove": 2}
    assert preview_counts(totals, True, "keep", {}) == {"keep": 6, "remove": 2}
    assert preview_counts(totals, False, "keep", {}) == {"keep": 8, "remove": 0}
    assert preview_counts(totals, True, "remove", {}) == {"keep": 4, "remove": 4}
    assert preview_counts(totals, True, "keep", overrides) == {
        "keep": 5, "remove": 3}      # +1 rescued, -1 review, -1 clean


def test_review_flow_against_real_api() -> None:
    with isolated() as (http, root, _):
        client = CleanerClient(http=http)
        job_id, _ws = make_job()
        summary = client.results(job_id)
        assert summary["totals"]["remove"] == 2
        page = client.images(job_id, verdict="REMOVE", limit=1)
        assert page["total"] == 2 and len(page["items"]) == 1
        item = page["items"][0]
        assert client.image_bytes(job_id, item["path"])[:2] == b"\xff\xd8"

        overrides = {item["path"]: ("keep", item["group"])}
        stats = client.approve(
            job_id, build_approval_payload(True, "keep", overrides))
        assert stats["kept"] == 7 and stats["rescued_from_remove"] == 1
        assert preview_counts(summary["totals"], True, "keep", overrides) == {
            "keep": 7, "remove": 1}

        data, missing = client.download(job_id)
        assert missing == 0
        archive = zipfile.ZipFile(io.BytesIO(data))
        assert archive.testzip() is None
        assert len([n for n in archive.namelist()
                    if n.startswith("dataset/")]) == 7

        exc = expect_api_error(client.results, "0" * 32)
        assert (exc.status, exc.code) == (404, "job_not_found")


def jpeg() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def wait(client, job_id, wanted, timeout=30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.job(job_id)
        if job["status"] in wanted:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job stuck: {job}")


def test_upload_and_run_flow_against_real_api() -> None:
    with isolated() as (http, root, _):
        saved = (dataset_routes.LIMITS, job_runner.run_pipeline_config,
                 job_runner.JOBS_ROOT)
        dataset_routes.LIMITS = uc.UploadLimits(min_images_per_class=2)
        job_runner.run_pipeline_config = (
            lambda cfg, **kw: {"success": True, "summaries": {}})
        job_runner.JOBS_ROOT = root
        try:
            client = CleanerClient(http=http)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for cls in ("cat", "dog"):
                    for i in range(4):
                        zf.writestr(f"{cls}/{i}.jpg", jpeg())

            for bad in (b"", b"not a zip"):
                assert expect_api_error(client.upload_zip, bad).status == 400

            job_id = client.upload_zip(buf.getvalue())["job_id"]
            job = wait(client, job_id, {"ready", "failed"})
            assert job["status"] == "ready"
            assert stage_for(job["status"]) == "configure"
            assert job["result"]["classes"] == ["cat", "dog"]

            client.start_run(job_id, {
                "run_duplicates": True, "run_blur": False,
                "run_noise": True, "run_outliers": True,
                "run_mislabels": True, "blur_percentile": 5})
            job = wait(client, job_id, {"completed", "failed"})
            assert job["status"] == "completed"

            exc = expect_api_error(client.results, job_id)   # fake: no reports
            assert (exc.status, exc.code) == (409, "results_not_available")
            exc = expect_api_error(client.start_run, job_id, {})
            assert exc.status == 409                         # already completed
        finally:
            (dataset_routes.LIMITS, job_runner.run_pipeline_config,
             job_runner.JOBS_ROOT) = saved


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ cleaner client tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()