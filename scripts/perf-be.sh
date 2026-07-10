#!/usr/bin/env bash
set -euo pipefail
mkdir -p perf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PYINSTRUMENT_BIN="${ROOT_DIR}/.venv/bin/pyinstrument"
LOCUST_BIN="${ROOT_DIR}/.venv/bin/locust"
HOST="${PERF_BE_HOST:-http://127.0.0.1:7860}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! "$PYTHON_BIN" -m pytest tests/test_benchmark_smoke.py --benchmark-json=perf/be-bench.json; then
  "$PYTHON_BIN" -m pytest tests/test_benchmark_smoke.py -q
  printf '{"status":"pytest-benchmark unavailable","tests":"test_benchmark_smoke.py"}\n' > perf/be-bench.json
fi

if [[ -x "$PYINSTRUMENT_BIN" ]]; then
  "$PYINSTRUMENT_BIN" -r json -o perf/be-profile.json scripts/profile_be_smoke.py
else
  printf '{"status":"pyinstrument unavailable"}\n' > perf/be-profile.json
fi

"$PYTHON_BIN" scripts/benchmark_pipeline_status.py > perf/be-pipeline-status.json

if [[ -x "$LOCUST_BIN" ]]; then
  if ! curl -fsS "$HOST/api/live" >/dev/null 2>&1; then
    LOAD_MODELS=false "$PYTHON_BIN" -m uvicorn api.main:app \
      --host 127.0.0.1 --port 7860 > perf/be-server.log 2>&1 &
    SERVER_PID=$!
    for _ in {1..60}; do
      curl -fsS "$HOST/api/live" >/dev/null 2>&1 && break
      sleep 1
    done
  fi

  curl -fsS "$HOST/api/live" >/dev/null
  "$LOCUST_BIN" -f tests/locustfile.py --headless -u 10 -r 2 --run-time 60s \
    --host "$HOST" --csv=perf/be-load --html=perf/be-load.html
else
  printf 'status,detail\nskipped,locust unavailable\n' > perf/be-load_stats.csv
fi
