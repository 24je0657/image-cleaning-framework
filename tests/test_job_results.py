# tests/test_job_results.py
# Standard library
import hashlib
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

# Local
from api import main, worker  # noqa: E402
from api import results_service as rs  # noqa: E402
from api.workspace import get_workspace, JobWorkspace  # noqa: E402
from unittest import mock
from unittest.mock import patch

FAKE_ID = "0" * 32
CLASSES = ("cat", "dog")

# (class, index) -> overrides on the base row
FLAGS = {
    ("cat", 0): {"verdict": "REMOVE — duplicate", "Exact_duplicates": "True",
                 "duplicate_of": "{dup}", "priority_score": "10"},
    ("cat", 1): {"verdict": "REVIEW — low priority", "is_outlier": "True",
                 "priority_score": "4"},
    ("dog", 0): {"verdict": "REMOVE — blurry", "is_blurry": "True",
                 "blur_score": "12.5", "threshold_used": "60",
                 "priority_score": "8"},
    ("dog", 1): {"verdict": "REVIEW — possible OOD",
                 "noise_verdict": "review_possible_ood"},
}


def jpeg(shade: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def sha_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@contextmanager
def isolated():
    saved = (worker.JOBS_ROOT, worker.jobs.copy())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "jobs"
        root.mkdir()
        try:
            worker.JOBS_ROOT = root
            worker.jobs.clear()
            rs.clear_cache()
            yield TestClient(main.app), root, Path(tmp)
        finally:
            worker.jobs.clear()
            worker.jobs.update(saved[1])
            worker.JOBS_ROOT = saved[0]
            rs.clear_cache()


def make_job(layout="flat", status="completed", relative_to=None,
             columns=None, n=4):
    job_id = worker.create_persistent_job("x")["job_id"]
    ws = get_workspace(job_id, worker.JOBS_ROOT)
    splits = ["train"] if layout == "flat" else ["train", "val"]
    for split in splits:
        rows = []
        for cls in CLASSES:
            for i in range(n):
                target = ws.dataset_dir / split / cls / f"{i}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(jpeg(40 + i * 20 + (cls == "dog") * 5))
                shown = (os.path.relpath(target, relative_to)
                         if relative_to else str(target))
                row = {"file_path": shown, "class": cls, "verdict": "CLEAN",
                       "priority_score": "0", "Exact_duplicates": "False",
                       "duplicate_of": "", "near_duplicate": "False",
                       "dup_confidence": "", "cosine_score": "",
                       "is_blurry": "False", "blur_score": "500",
                       "threshold_used": "60", "noise_verdict": "not_flagged",
                       "reconstruction_error": "0.001", "is_outlier": "False",
                       "global_score": "0.1", "is_mislabeled": "False",
                       "mislabel_confidence": "CLEAN", "nn_predicted": cls}
                for key, value in FLAGS.get((cls, i), {}).items():
                    row[key] = (str(ws.dataset_dir / split / "cat" / "1.jpg")
                                if value == "{dup}" else value)
                rows.append(row)
        df = pd.DataFrame(rows)
        if columns:
            df = df[columns]
        df.to_csv(ws.reports_dir / f"{split}_master_report.csv", index=False)
    ws.update_meta(dataset={"layout": layout, "splits": splits,
                            "classes": list(CLASSES)})
    worker.update_job(job_id, status=status, progress=100)
    return job_id, ws


def approve(client, job_id, **body):
    return client.post(f"/jobs/{job_id}/approve", json=body)


def test_summary_and_listing() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        r = client.get(f"/jobs/{job_id}/results")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["totals"] == {"images": 8, "clean": 4,
                                  "review": 2, "remove": 2}
        assert body["by_class"]["cat"]["remove"] == 1
        assert body["reasons"]["exact_duplicate"] == 1
        assert body["reasons"]["noise_or_ood"] == 1
        assert body["unmatched_report_rows"] == 0

        r = client.get(f"/jobs/{job_id}/images", params={"verdict": "REMOVE"})
        data = r.json()
        assert data["total"] == 2
        first = data["items"][0]                      # highest priority
        assert first["path"] == "train/cat/0.jpg"
        assert first["display_path"] == "cat/0.jpg"   # flat: no train/
        assert first["reasons"][0]["code"] == "exact_duplicate"
        assert (first["reasons"][0]["detail"]["duplicate_of"]
                == "train/cat/1.jpg")

        paged = client.get(f"/jobs/{job_id}/images",
                           params={"limit": 3, "offset": 6}).json()
        assert paged["total"] == 8 and len(paged["items"]) == 2
        only_dog = client.get(f"/jobs/{job_id}/images",
                              params={"class": "dog"}).json()
        assert only_dog["total"] == 4
        assert client.get(f"/jobs/{job_id}/images",
                          params={"verdict": "BAD"}).status_code == 422
        assert client.get(f"/jobs/{job_id}/images",
                          params={"limit": 1000}).status_code == 422


def test_no_server_paths_in_responses() -> None:
    with isolated() as (client, root, tmp):
        job_id, ws = make_job()
        texts = [
            client.get(f"/jobs/{job_id}/results").text,
            client.get(f"/jobs/{job_id}/images").text,
            approve(client, job_id, apply_proposed_removals=True).text,
            approve(client, job_id, apply_proposed_removals=True,
                    keep=["/etc/passwd"]).text,
        ]
        for text in texts:
            assert str(tmp) not in text and "jobs/" not in text.replace(
                "jobs/{", "")


def test_image_file_endpoint() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        url = f"/jobs/{job_id}/images/file"
        r = client.get(url, params={"path": "train/cat/2.jpg"})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content == (ws.dataset_dir / "train/cat/2.jpg").read_bytes()
        for bad in ("../../job.json", "/etc/passwd", "train/cat/9.jpg",
                    "train/cat/../dog/0.jpg", "", "job.json"):
            r = client.get(url, params={"path": bad})
            assert r.status_code in (404, 422), (bad, r.status_code)


def test_approve_builds_final_flat() -> None:
    with isolated() as (client, root, tmp):
        job_id, ws = make_job()
        before = sha_tree(ws.dataset_dir)
        r = approve(client, job_id, apply_proposed_removals=True,
                    keep=["train/cat/0.jpg"])          # rescue one
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["kept"] == 7 and body["removed"] == 1
        assert body["rescued_from_remove"] == 1
        assert body["missing_files"] == 0

        final = sha_tree(ws.final_dir)
        assert "cat/0.jpg" in final and "dog/0.jpg" not in final
        assert not any(k.startswith("train/") for k in final)    # flat
        assert len(final) == 7
        assert sha_tree(ws.dataset_dir) == before                # untouched
        assert not list(ws.root.glob(".final-*"))
        assert (ws.reports_dir / "approval_decisions.csv").is_file()
        meta = ws.read_meta()
        assert meta["approval"]["kept"] == 7
        assert meta["status"] == "completed"


def test_apply_choice_and_review_default() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        r = approve(client, job_id, apply_proposed_removals=False)
        assert r.json()["kept"] == 8                  # nothing removed
        r = approve(client, job_id, apply_proposed_removals=True,
                    review_default="remove")
        assert r.json()["kept"] == 4                  # clean images only
        assert r.json()["removed_from_review"] == 2
        r = approve(client, job_id, apply_proposed_removals=False,
                    remove=["train/dog/3.jpg"])
        assert r.json()["kept"] == 7                  # explicit remove wins
        assert len(sha_tree(ws.final_dir)) == 7       # final replaced, not merged


def test_split_layout_keeps_splits() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job(layout="split")
        r = approve(client, job_id, apply_proposed_removals=True)
        assert r.status_code == 200, r.text
        final = sha_tree(ws.final_dir)
        assert "val/cat/2.jpg" in final and "train/dog/0.jpg" not in final
        assert r.json()["kept"] == 12


def test_invalid_selections_change_nothing() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        approve(client, job_id, apply_proposed_removals=True)
        good = sha_tree(ws.final_dir)

        r = approve(client, job_id, apply_proposed_removals=True,
                    keep=["train/cat/0.jpg", "nope.jpg"])
        assert r.status_code == 422
        assert r.json()["error"]["details"]["unknown_count"] == 1
        r = approve(client, job_id, apply_proposed_removals=True,
                    keep=["train/cat/0.jpg"], remove=["train/cat/0.jpg"])
        assert r.status_code == 422
        everything = [f"train/{c}/{i}.jpg" for c in CLASSES for i in range(4)]
        r = approve(client, job_id, apply_proposed_removals=True,
                    remove=everything)
        assert r.status_code == 422
        assert sha_tree(ws.final_dir) == good


def test_request_validation() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        url = f"/jobs/{job_id}/approve"
        assert client.post(url, json={}).status_code == 422   # no default
        assert client.post(url, json={
            "apply_proposed_removals": True, "oops": 1}).status_code == 422
        assert client.post(url, json={
            "apply_proposed_removals": True,
            "review_default": "maybe"}).status_code == 422


def test_job_state_errors() -> None:
    with isolated() as (client, root, _):
        running, _ = make_job(status="running")
        for url in (f"/jobs/{running}/results", f"/jobs/{running}/images"):
            r = client.get(url)
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "results_not_available"
        assert approve(client, running,
                       apply_proposed_removals=True).status_code == 409
        for bad in (FAKE_ID, "not-a-job"):
            assert client.get(f"/jobs/{bad}/results").status_code == 404
            assert approve(client, bad,
                           apply_proposed_removals=True).status_code == 404

        done, ws = make_job()
        (ws.reports_dir / "train_master_report.csv").unlink()
        assert client.get(f"/jobs/{done}/results").status_code == 409


def test_missing_optional_columns_and_unmatched_rows() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job(columns=["file_path", "verdict"])
        r = client.get(f"/jobs/{job_id}/results")
        assert r.status_code == 200
        assert r.json()["totals"]["images"] == 8
        report = ws.reports_dir / "train_master_report.csv"
        df = pd.read_csv(report)
        df.loc[0, "file_path"] = "/somewhere/else/x.jpg"   # outside dataset
        df.to_csv(report, index=False)
        body = client.get(f"/jobs/{job_id}/results").json()
        assert body["totals"]["images"] == 7
        assert body["unmatched_report_rows"] == 1          # cache invalidated


def test_relative_report_paths() -> None:
    with isolated() as (client, root, tmp):
        job_id, ws = make_job(relative_to=tmp)

        report = pd.read_csv(
            ws.reports_dir / "train_master_report.csv",
            dtype=str,
            keep_default_na=False,
        )

        # The primary file_path values should be relative.
        assert not Path(report.iloc[0]["file_path"]).is_absolute()

        original = os.getcwd()
        os.chdir(tmp)

        try:
            response = client.get(f"/jobs/{job_id}/results")
        finally:
            os.chdir(original)

        assert response.status_code == 200, response.text

        body = response.json()
        assert body["totals"]["images"] == 8
        assert body["unmatched_report_rows"] == 0

def _leftovers(ws) -> list:
    return (list(ws.root.glob(".final-*"))
            + list(ws.reports_dir.glob(".approval-*")))


def test_failed_approval_restores_previous_output() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()
        assert approve(client, job_id,
                       apply_proposed_removals=False).status_code == 200
        final_before = sha_tree(ws.final_dir)
        csv_path = ws.reports_dir / "approval_decisions.csv"
        csv_before = csv_path.read_bytes()
        approval_before = ws.read_meta()["approval"]

        def assert_restored(response) -> None:
            assert response.status_code == 500
            assert "boom" not in response.text
            assert sha_tree(ws.final_dir) == final_before
            assert csv_path.read_bytes() == csv_before
            assert ws.read_meta()["approval"] == approval_before
            assert _leftovers(ws) == []

        # Failure after both outputs are installed (metadata write fails).
        with mock.patch.object(JobWorkspace, "update_meta",
                               side_effect=RuntimeError("boom")):
            assert_restored(approve(client, job_id,
                                    apply_proposed_removals=True))

        # Failure while installing the CSV, after final/ was already swapped.
        real_replace = os.replace

        def flaky(src, dst):
            if (Path(dst).name == "approval_decisions.csv"
                    and Path(src).name.startswith(".approval-")):
                raise OSError("boom")
            return real_replace(src, dst)

        with mock.patch("os.replace", flaky):
            assert_restored(approve(client, job_id,
                                    apply_proposed_removals=True))

        # A later approval still works.
        r = approve(client, job_id, apply_proposed_removals=True)
        assert r.status_code == 200 and r.json()["kept"] == 6


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ job results tests passed ({len(tests)}).")

def test_approval_rolls_back_if_final_install_fails() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()

        # Create a previously approved version.
        response = approve(client, job_id, apply_proposed_removals=True)
        assert response.status_code == 200

        old_final = sha_tree(ws.final_dir)
        old_csv = (ws.reports_dir / "approval_decisions.csv").read_bytes()
        old_meta = ws.read_meta()

        result = rs.get_results(ws)
        original_replace = os.replace

        def failing_replace(src, dst):
            src = Path(src)
            dst = Path(dst)

            if (
                dst == ws.final_dir
                and src.name.startswith(".final-")
                and not src.name.startswith(".final-backup-")
            ):
                raise OSError("Simulated final installation failure")

            return original_replace(src, dst)

        with patch.object(rs.os, "replace", side_effect=failing_replace):
            try:
                rs.approve(
                    ws, result,
                    keep=[], remove=[],
                    review_default="keep",
                    apply_proposed_removals=True,
                )
                assert False, "Expected approval to fail"
            except OSError as exc:
                assert "Simulated final installation failure" in str(exc)

        assert sha_tree(ws.final_dir) == old_final
        assert (ws.reports_dir / "approval_decisions.csv").read_bytes() == old_csv
        assert ws.read_meta() == old_meta

def test_approval_rolls_back_if_csv_install_fails() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()

        response = approve(client, job_id, apply_proposed_removals=True)
        assert response.status_code == 200

        old_final = sha_tree(ws.final_dir)
        old_csv = (ws.reports_dir / "approval_decisions.csv").read_bytes()
        old_meta = ws.read_meta()

        result = rs.get_results(ws)
        original_replace = os.replace
        decisions_path = ws.reports_dir / "approval_decisions.csv"

        def failing_replace(src, dst):
            src = Path(src)
            dst = Path(dst)

            if (
                dst == decisions_path
                and src.parent == ws.reports_dir
                and src.name.startswith(".approval-")
            ):
                raise OSError("Simulated CSV installation failure")

            return original_replace(src, dst)

        with patch.object(rs.os, "replace", side_effect=failing_replace):
            try:
                rs.approve(
                    ws, result,
                    keep=[], remove=[],
                    review_default="keep",
                    apply_proposed_removals=True,
                )
                assert False, "Expected approval to fail"
            except OSError as exc:
                assert "Simulated CSV installation failure" in str(exc)

        assert sha_tree(ws.final_dir) == old_final
        assert decisions_path.read_bytes() == old_csv
        assert ws.read_meta() == old_meta

def test_approval_rolls_back_if_metadata_update_fails() -> None:
    with isolated() as (client, root, _):
        job_id, ws = make_job()

        response = approve(client, job_id, apply_proposed_removals=True)
        assert response.status_code == 200

        old_final = sha_tree(ws.final_dir)
        old_csv = (ws.reports_dir / "approval_decisions.csv").read_bytes()
        old_meta = ws.read_meta()

        result = rs.get_results(ws)

        with patch.object(
            type(ws),
            "update_meta",
            side_effect=OSError("Simulated metadata failure"),
        ):
            try:
                rs.approve(
                    ws, result,
                    keep=[], remove=[],
                    review_default="keep",
                    apply_proposed_removals=True,
                )
                assert False, "Expected approval to fail"
            except OSError as exc:
                assert "Simulated metadata failure" in str(exc)

        assert sha_tree(ws.final_dir) == old_final
        assert (ws.reports_dir / "approval_decisions.csv").read_bytes() == old_csv
        assert ws.read_meta() == old_meta


if __name__ == "__main__":
    run()