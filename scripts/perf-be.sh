#!/usr/bin/env bash
set -euo pipefail
mkdir -p perf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VIRTUAL_ENV:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
PYINSTRUMENT_BIN="${VENV_DIR}/bin/pyinstrument"
LOCUST_BIN="${VENV_DIR}/bin/locust"
UVICORN_BIN="${VENV_DIR}/bin/uvicorn"
HOST="${PERF_BE_HOST:-http://127.0.0.1:7860}"
SERVER_START_TIMEOUT_SEC="${PERF_BE_START_TIMEOUT_SEC:-120}"
LOAD_RUN_TIME="${PERF_BE_LOAD_RUN_TIME:-60s}"
SERVER_LOG="perf/be-server.log"
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
    LLM_BASE_URL="${LLM_BASE_URL:-https://llm.invalid/v1}" \
      LLM_MODEL="${LLM_MODEL:-perf-smoke-model}" \
      LOAD_MODELS=false "$UVICORN_BIN" api.main:app \
      --host 127.0.0.1 --port 7860 > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    server_ready=false
    for ((attempt = 0; attempt < SERVER_START_TIMEOUT_SEC; attempt += 1)); do
      if curl -fsS "$HOST/api/live" >/dev/null 2>&1; then
        server_ready=true
        break
      fi
      if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        printf 'Performance server exited before becoming live.\n' >&2
        sed -n '1,240p' "$SERVER_LOG" >&2
        exit 1
      fi
      sleep 1
    done
    if [[ "$server_ready" != true ]]; then
      printf 'Performance server was not live after %s seconds.\n' \
        "$SERVER_START_TIMEOUT_SEC" >&2
      sed -n '1,240p' "$SERVER_LOG" >&2
      exit 1
    fi
  fi

  curl -fsS "$HOST/api/live" >/dev/null
  "$LOCUST_BIN" -f tests/locustfile.py --headless -u 10 -r 2 --run-time "$LOAD_RUN_TIME" \
    --host "$HOST" --csv=perf/be-load --html=perf/be-load.html
else
  printf 'status,detail\nskipped,locust unavailable\n' > perf/be-load_stats.csv
fi
