# tests/test_access.py
# Standard library
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Local
from test_dataset_routes import HEADERS, entries, zip_bytes  # noqa: E402
from test_job_results import isolated, make_job  # noqa: E402
from api import access, worker  # noqa: E402
from api import upload_config as uc  # noqa: E402
from api.routes import datasets as dataset_routes  # noqa: E402
from api.workspace import get_workspace  # noqa: E402

A = "alpha-code-0123456789"
B = "bravo-code-0123456789"
FAKE = "0" * 32


def auth(code):
    return {"X-Access-Code": code}


def cfg(**kw):
    return access.AccessSettings(**{"codes": (A, B), **kw})


@contextmanager
def configured(settings):
    saved = access.SETTINGS
    access.SETTINGS = settings
    access.reset_state()
    try:
        yield
    finally:
        access.SETTINGS = saved
        access.reset_state()


def owned_job(code, status="completed"):
    job_id, ws = make_job(status=status)
    ws.update_meta(owner=access.owner_id(code))
    return job_id, ws


def expect_config_error(fn) -> None:
    try:
        fn()
    except uc.UploadConfigError:
        return
    raise AssertionError("invalid settings accepted")


def test_settings_parsing_and_validation() -> None:
    s = access.load_settings({
        "IMAGE_CLEANING_ACCESS_CODES": f" {A} , {B},, ",
        "IMAGE_CLEANING_MAX_LIVE_JOBS": "7",
        "IMAGE_CLEANING_LEGACY_ROUTES": "ON",
        "IMAGE_CLEANING_CORS_ORIGINS": "https://a.example"})
    assert s.codes == (A, B) and s.enabled and s.max_live_jobs == 7
    assert s.legacy_routes and not s.public_docs
    assert s.cors_origins == ("https://a.example",)
    assert not access.load_settings({}).enabled
    expect_config_error(lambda: access.AccessSettings(codes=("short",)))
    expect_config_error(lambda: access.AccessSettings(codes=(A, A)))
    expect_config_error(lambda: access.load_settings({
        "IMAGE_CLEANING_UPLOADS_PER_HOUR": "0"}))


def test_match_code() -> None:
    s = cfg()
    assert access.match_code(A, s) == access.owner_id(A)
    assert access.owner_id(A) != access.owner_id(B)
    for bad in ("", "x" * 10, A + "x", A[:-1], A.upper()):
        assert access.match_code(bad, s) is None


def test_disabled_mode_allows_everything() -> None:
    with isolated() as (client, root, _), configured(access.AccessSettings()):
        job_id, _ws = make_job()
        assert client.get(f"/jobs/{job_id}").status_code == 200


def test_missing_and_wrong_code() -> None:
    with isolated() as (client, root, _), configured(cfg()):
        job_id, _ws = owned_job(A)
        url = f"/jobs/{job_id}"
        assert client.get(url).status_code == 401
        assert client.get(url, headers=auth("wrong")).status_code == 401
        assert client.get(url, headers=auth("x" * 300)).status_code == 401
        assert client.get("/health").status_code == 200
        assert client.get(url, headers=auth(A)).status_code == 200


def test_ownership_isolation_without_existence_oracle() -> None:
    with isolated() as (client, root, _), configured(cfg()):
        job_id, _ws = owned_job(A)
        missing = client.get(f"/jobs/{FAKE}", headers=auth(B))
        assert missing.status_code == 404
        calls = [
            ("get", f"/jobs/{job_id}", None),
            ("get", f"/jobs/{job_id}/results", None),
            ("get", f"/jobs/{job_id}/images", None),
            ("get", f"/jobs/{job_id}/images/file?path=train/cat/0.jpg", None),
            ("get", f"/jobs/{job_id}/download", None),
            ("post", f"/jobs/{job_id}/approve",
             {"apply_proposed_removals": True}),
            ("post", f"/jobs/{job_id}/run", {}),
            ("delete", f"/jobs/{job_id}", None),
        ]
        for method, url, body in calls:
            kwargs = {"json": body} if body is not None else {}
            r = getattr(client, method)(url, headers=auth(B), **kwargs)
            assert r.status_code == 404, (method, url, r.status_code)
            assert r.json() == missing.json(), (method, url)
        assert client.get(f"/jobs/{job_id}/results",
                          headers=auth(A)).status_code == 200


def test_unowned_job_fails_closed() -> None:
    with isolated() as (client, root, _), configured(cfg()):
        job_id, _ws = make_job()                      # no owner recorded
        assert client.get(f"/jobs/{job_id}", headers=auth(A)).status_code == 404


def test_legacy_routes_hidden_unless_enabled() -> None:
    with isolated() as (client, root, _):
        with configured(cfg()):
            assert client.get("/stats/train").status_code == 401
            r = client.get("/stats/train", headers=auth(A))
            assert (r.status_code, r.json()["detail"]) == (404, "Not found.")
            assert client.post("/export", json={},
                               headers=auth(A)).status_code == 404
        with configured(cfg(legacy_routes=True)):
            r = client.get("/stats/train", headers=auth(A))
            assert r.json().get("detail") != "Not found."


def test_delete_frees_job_and_files() -> None:
    with isolated() as (client, root, _), configured(cfg()):
        job_id, ws = owned_job(A)
        assert client.delete(f"/jobs/{job_id}", headers=auth(A)).status_code == 200
        assert not ws.root.exists() and job_id not in worker.jobs
        running, _ = owned_job(A, status="running")
        assert client.delete(f"/jobs/{running}",
                             headers=auth(A)).status_code == 400


def test_quota_window_with_fake_clock() -> None:
    with isolated() as (client, root, _):
        s = cfg(uploads_per_hour=2)
        req = SimpleNamespace(state=SimpleNamespace(owner="o"))
        with configured(s):
            assert access.enforce_upload_quota(req, now=0, settings=s) == "o"
            access.enforce_upload_quota(req, now=10, settings=s)
            try:
                access.enforce_upload_quota(req, now=20, settings=s)
            except access.QuotaExceededError as exc:
                assert exc.http_status == 429
            else:
                raise AssertionError("limit not enforced")
            access.enforce_upload_quota(req, now=3601, settings=s)   # slot freed
        none = SimpleNamespace(state=SimpleNamespace(owner=None))
        assert access.enforce_upload_quota(none, settings=s) is None


def test_upload_rate_limit_is_per_code() -> None:
    with isolated() as (client, root, _), configured(cfg(uploads_per_hour=2)):
        def post(code):
            return client.post("/datasets/upload", content=b"not a zip",
                               headers={**HEADERS, **auth(code)})
        assert [post(A).status_code for _ in range(2)] == [400, 400]
        r = post(A)
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "quota_exceeded"
        assert post(B).status_code == 400
        assert list(root.iterdir()) == []             # nothing left behind


def test_live_job_cap() -> None:
    with isolated() as (client, root, _), configured(cfg(max_live_jobs=2)):
        owned_job(A)
        owned_job(A)
        r = client.post("/datasets/upload", content=b"not a zip",
                        headers={**HEADERS, **auth(A)})
        assert r.status_code == 429
        r = client.post("/datasets/upload", content=b"not a zip",
                        headers={**HEADERS, **auth(B)})
        assert r.status_code == 400                   # other code unaffected


def test_upload_records_owner_hash_only() -> None:
    items = entries("cat", 4) + entries("dog", 4)
    with isolated() as (client, root, _), configured(cfg()):
        saved = dataset_routes.LIMITS
        dataset_routes.LIMITS = uc.UploadLimits(min_images_per_class=2)
        try:
            r = client.post("/datasets/upload", content=zip_bytes(items),
                            headers={**HEADERS, **auth(A)})
            assert r.status_code == 202, r.text
            job_id = r.json()["job_id"]
            deadline = time.time() + 30
            status = ""
            while time.time() < deadline and status not in ("ready", "failed"):
                time.sleep(0.05)
                status = client.get(f"/jobs/{job_id}",
                                    headers=auth(A)).json()["status"]
            assert status == "ready"
            meta = get_workspace(job_id, worker.JOBS_ROOT).read_meta()
            assert meta["owner"] == access.owner_id(A)
            assert A not in json.dumps(meta)
            assert client.get(f"/jobs/{job_id}",
                              headers=auth(B)).status_code == 404
        finally:
            dataset_routes.LIMITS = saved


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ access control tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()