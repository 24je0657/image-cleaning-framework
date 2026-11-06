# tests/test_step_progress.py
# Standard library
import contextlib
import io
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Local
import run_pipeline as rp  # noqa: E402
from api import job_runner  # noqa: E402
from src.config import PipelineConfig  # noqa: E402


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def cfg_for(tmp, **flags):
    data = Path(tmp) / "ds"
    for split in ("train", "val"):
        for cls in ("cat", "dog"):
            (data / split / cls).mkdir(parents=True)
    return PipelineConfig(
        data_dir=data, output_dir=Path(tmp) / "out",
        embeddings_dir=Path(tmp) / "emb",
        reports_dir=Path(tmp) / "rep", **flags)


ALL = ["3", "4", "5", "6", "6b", "7", "8", "9", "10"]


def test_run_step_unchanged_without_tracker() -> None:
    assert rp._STEP_TRACKER.get() is None
    assert quiet(rp.run_step, "Step 3 — x",
                 lambda a, b=1: a + b, 1, b=2) == (3, True)

    def boom():
        raise RuntimeError("x")

    assert quiet(rp.run_step, "Step 3 — x", boom) == (None, False)


def test_tracker_reports_monotonic_progress_and_failure_counts() -> None:
    events = []
    tracker = rp.StepTracker(ALL, lambda f, m: events.append((f, m)))
    token = rp._STEP_TRACKER.set(tracker)
    try:
        for sid in ALL:
            def work(sid=sid):
                if sid == "5":
                    raise RuntimeError("blur failed")
            quiet(rp.run_step, f"Step {sid} — test (train)", work)
    finally:
        rp._STEP_TRACKER.reset(token)
    fractions = [f for f, _ in events]
    assert fractions[0] == 0.0
    assert fractions == sorted(fractions)
    assert tracker.fraction() == 1.0       # failed step still counted


def test_callback_exception_never_breaks_a_step() -> None:
    def bad(fraction, message):
        raise RuntimeError("callback bug")

    tracker = rp.StepTracker(ALL, bad)
    token = rp._STEP_TRACKER.set(tracker)
    try:
        assert quiet(rp.run_step, "Step 3 — t", lambda: 7) == (7, True)
    finally:
        rp._STEP_TRACKER.reset(token)


def test_complete_counts_steps_that_never_ran() -> None:
    events = []
    tracker = rp.StepTracker(ALL, lambda f, m: events.append((f, m)))
    token = rp._STEP_TRACKER.set(tracker)
    try:
        quiet(rp.run_step, "Step 3 — t", lambda: None)
    finally:
        rp._STEP_TRACKER.reset(token)
    assert tracker.fraction() < 1.0        # steps 4..10 never ran
    tracker.complete()
    assert tracker.fraction() == 1.0 and events[-1][0] == 1.0


def test_split_offset() -> None:
    events = []
    tracker = rp.StepTracker(ALL, lambda f, m: events.append(f), 1, 2)
    tracker.begin("Step 3 — t")
    assert events[0] == 0.5


def test_planned_steps_follow_flags() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = cfg_for(tmp, run_duplicates=False, run_blur=False,
                      run_outliers=False, run_mislabels=False)
        assert rp.planned_steps(cfg, True) == ["3", "6", "6b", "9", "10"]
        assert rp.planned_steps(cfg, False)[:3] == ["1", "2", "3"]


def test_config_drives_tracker_per_split() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = cfg_for(tmp, run_noise=False)
        events = []

        def fake(cfg_, split, skip_embeddings=False,
                 skip_preprocessing=False):
            for sid in rp.planned_steps(cfg_, skip_preprocessing):
                rp.run_step(f"Step {sid} — fake ({split})", lambda: None)
            return {"success": True, "steps": {}}

        original = rp.run_pipeline
        rp.run_pipeline = fake
        try:
            out = quiet(rp.run_pipeline_config, cfg,
                        skip_preprocessing=True,
                        progress=lambda f, m: events.append((f, m)))
        finally:
            rp.run_pipeline = original
        fractions = [f for f, _ in events]
        assert out["success"] is True
        assert fractions == sorted(fractions) and fractions[-1] == 1.0
        assert any(0.5 <= f < 1.0 for f in fractions)   # second split ran
        assert not any(m.startswith("Step 6") for _, m in events)
        assert rp._STEP_TRACKER.get() is None            # reset afterwards


def test_step_labels_match_weights() -> None:
    source = (PROJECT_ROOT / "run_pipeline.py").read_text(encoding="utf-8")
    for step_id in rp.STEP_WEIGHTS:
        assert f"Step {step_id} —" in source, step_id


def test_job_callback_is_monotonic_and_capped() -> None:
    calls = []
    original = job_runner.update_job
    job_runner.update_job = lambda job_id, **fields: calls.append(fields)
    try:
        callback = job_runner._make_progress_callback("a" * 32)
        for fraction in (0.0, 0.5, 0.3, 1.0, 1.7):
            callback(fraction, "m")
    finally:
        job_runner.update_job = original
    assert [c["progress"] for c in calls] == [5, 50, 50, 95, 95]


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ step progress tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()