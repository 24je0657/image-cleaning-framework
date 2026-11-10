# tests/test_cleaner_safeguards.py
# Standard library
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Local
from cleaner_client import (  # noqa: E402
    HIGH_REMOVAL_PERCENT,
    class_preview,
    empty_classes,
    override_class,
    preview_counts,
    removal_warning,
)

BY_CLASS = {
    "cat": {"images": 10, "clean": 6, "review": 2, "remove": 2},
    "dog": {"images": 4, "clean": 0, "review": 1, "remove": 3},
}
TOTALS = {"images": 14, "clean": 6, "review": 3, "remove": 5}


def test_warning_boundary() -> None:
    assert HIGH_REMOVAL_PERCENT == 30
    assert removal_warning({"images": 100, "remove": 29}) is None
    hit = removal_warning({"images": 100, "remove": 30})
    assert hit == {"remove": 30, "images": 100, "share_percent": 30.0}
    assert removal_warning({"images": 100, "remove": 31})["remove"] == 31
    assert removal_warning({"images": 10, "remove": 3}) is not None
    assert removal_warning({"images": 10, "remove": 2}) is None
    assert removal_warning({"images": 3, "remove": 1}) is not None   # 33%
    # 1*100 >= 3*30 is true, so a 3-image job with 1 removal does warn.


def test_warning_degenerate_inputs() -> None:
    assert removal_warning({}) is None
    assert removal_warning({"images": 0, "remove": 0}) is None
    assert removal_warning({"images": 5, "remove": 0}) is None
    assert removal_warning({"images": 5, "remove": 5})["share_percent"] == 100.0


def test_class_preview_defaults() -> None:
    assert class_preview(BY_CLASS, True, "keep", {}) == {"cat": 8, "dog": 1}
    assert class_preview(BY_CLASS, False, "keep", {}) == {"cat": 10, "dog": 4}
    assert class_preview(BY_CLASS, True, "remove", {}) == {"cat": 6, "dog": 0}
    assert class_preview(BY_CLASS, False, "remove", {}) == {"cat": 8, "dog": 3}


def test_empty_class_boundaries() -> None:
    assert empty_classes({"cat": 8, "dog": 1}) == []          # one is enough
    assert empty_classes({"cat": 8, "dog": 0}) == ["dog"]
    assert empty_classes({"b": 0, "a": -1, "c": 2}) == ["a", "b"]
    assert empty_classes({}) == []


def test_overrides_change_emptiness() -> None:
    kept = class_preview(BY_CLASS, True, "remove", {})
    assert empty_classes(kept) == ["dog"]
    rescue = {"train/dog/1.jpg": ("keep", "REMOVE")}
    assert empty_classes(class_preview(BY_CLASS, True, "remove", rescue)) == []
    # Removing the only kept dog empties the class again.
    only = {"train/dog/1.jpg": ("remove", "REVIEW")}
    assert class_preview(BY_CLASS, True, "keep", only)["dog"] == 0
    # An override equal to the default changes nothing.
    same = {"train/dog/1.jpg": ("remove", "REMOVE")}
    assert class_preview(BY_CLASS, True, "keep", same) == {"cat": 8, "dog": 1}


def test_malformed_and_unknown_overrides_are_ignored() -> None:
    base = class_preview(BY_CLASS, True, "keep", {})
    junk = {"nope.jpg": ("keep", "REMOVE"),
            "train/bird/1.jpg": ("keep", "REMOVE"),
            "a/b/c/d.jpg": ("remove", "CLEAN")}
    assert class_preview(BY_CLASS, True, "keep", junk) == base
    assert override_class("train/cat/0.jpg") == "cat"
    assert override_class("cat/0.jpg") is None
    assert override_class("") is None


def test_class_preview_agrees_with_overall_preview() -> None:
    overrides = {"train/cat/1.jpg": ("remove", "CLEAN"),
                 "train/dog/2.jpg": ("keep", "REMOVE"),
                 "train/cat/3.jpg": ("keep", "REVIEW")}
    for apply_removals in (True, False):
        for review_default in ("keep", "remove"):
            per_class = class_preview(BY_CLASS, apply_removals,
                                      review_default, overrides)
            overall = preview_counts(TOTALS, apply_removals, review_default,
                                     overrides)
            assert sum(per_class.values()) == overall["keep"], (
                apply_removals, review_default)


def run() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"✅ UI safeguard tests passed ({len(tests)}).")


if __name__ == "__main__":
    run()