#!/usr/bin/env bash
set -euo pipefail
mkdir -p perf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
PYINSTRUMENT_BIN="${ROOT_DIR}/.venv/bin/pyinstrument"
LOCUST_BIN="${ROOT_DIR}/.venv/bin/locust"

if ! "$PYTHON_BIN" -m pytest tests/test_benchmark_smoke.py --benchmark-json=perf/be-bench.json; then
  "$PYTHON_BIN" -m pytest tests/test_benchmark_smoke.py -q
  printf '{"status":"pytest-benchmark unavailable","tests":"test_benchmark_smoke.py"}\n' > perf/be-bench.json
fi

if [[ -x "$PYINSTRUMENT_BIN" ]]; then
  "$PYINSTRUMENT_BIN" -r json -o perf/be-profile.json scripts/profile_be_smoke.py
else
  printf '{"status":"pyinstrument unavailable"}\n' > perf/be-profile.json
fi

if [[ -x "$LOCUST_BIN" ]]; then
  "$LOCUST_BIN" -f tests/locustfile.py --headless -u 2 -r 1 --run-time 5s --host http://127.0.0.1:7860 --csv=perf/be-load --html=perf/be-load.html >/dev/null 2>&1 || true
else
  printf 'status,detail\nskipped,locust unavailable\n' > perf/be-load_stats.csv
fi
