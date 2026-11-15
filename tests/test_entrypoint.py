# tests/test_entrypoint.py
# Standard library
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def start(codes):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("IMAGE_CLEANING_ACCESS_CODES", None)
    if codes is not None:
        env["IMAGE_CLEANING_ACCESS_CODES"] = codes
    return subprocess.run(["bash", str(ROOT / "scripts/start.sh")], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=60)


def test_refuses_without_valid_codes() -> None:
    for codes in (None, "", "short"):
        r = start(codes)
        assert r.returncode != 0, codes
        assert "ACCESS_CODES" in r.stderr or "16" in r.stderr, r.stderr


def run() -> None:
    test_refuses_without_valid_codes()
    print("✅ entrypoint tests passed.")


if __name__ == "__main__":
    run()