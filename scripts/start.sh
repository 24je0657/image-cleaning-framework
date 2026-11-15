#!/usr/bin/env bash
set -euo pipefail

# Refuse to start unprotected or with invalid settings.
python - <<'PY' || { echo "FATAL: set IMAGE_CLEANING_ACCESS_CODES (comma-separated, 16+ chars each)." >&2; exit 1; }
import sys
from api import access
sys.exit(0 if access.SETTINGS.enabled else 1)
PY

mkdir -p "${IMAGE_CLEANING_JOBS_ROOT:-jobs}"

# API: loopback only, single worker, no reload.
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

for _ in $(seq 1 120); do
  if python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=2).status_code==200 else 1)" 2>/dev/null; then
    break
  fi
  kill -0 "$API_PID" 2>/dev/null || { echo "FATAL: API exited during startup." >&2; exit 1; }
  sleep 1
done

streamlit run cleaner_app.py \
  --server.port 7860 --server.address 0.0.0.0 \
  --server.maxUploadSize 200 --server.headless true \
  --browser.gatherUsageStats false &
UI_PID=$!

# Exit if either process dies, so the platform restarts the container.
wait -n "$API_PID" "$UI_PID"
exit 1