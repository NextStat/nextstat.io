#!/usr/bin/env bash
set -euo pipefail

# Stable-surface verification gate for the promoted ads + weekly timeseries subset.
#
# Scope:
#   - ns_inference::{BetaBinomialModel, DelayCorrectionModel, hill, adstock_geometric}
#   - weekly Kalman convenience builders / CLI aliases
#   - matching Python wrappers in nextstat.ads and nextstat.timeseries
#
# Optional env vars:
#   - ATS_STABLE_PY: Python executable (default: ./.venv/bin/python, else python3)
#   - ATS_STABLE_MATURIN: maturin executable (default: ./.venv/bin/maturin, else maturin)
#   - ATS_STABLE_PYTHONPATH: pythonpath for local bindings (default: bindings/ns-py/python)
#   - ATS_STABLE_CARGO_TARGET_DIR: isolated cargo target dir
#   - ATS_STABLE_OUT_DIR: directory for emitted benchmark-smoke artifacts
#   - ATS_STABLE_NEXTSTAT_BIN: optional path to a prebuilt nextstat CLI binary
#   - ATS_STABLE_SKIP_MATURIN: set to 1 to skip `maturin develop --release`
#   - ATS_STABLE_SKIP_DOC_CHECKS: set to 1 to skip release-governance doc checks

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

skip_maturin="${ATS_STABLE_SKIP_MATURIN:-0}"
skip_doc_checks="${ATS_STABLE_SKIP_DOC_CHECKS:-0}"
py_path="${ATS_STABLE_PYTHONPATH:-bindings/ns-py/python}"
cargo_target_dir="${ATS_STABLE_CARGO_TARGET_DIR:-${repo_root}/tmp/cargo_target_ads_timeseries_stable}"
out_dir="${ATS_STABLE_OUT_DIR:-${repo_root}/tmp/ads-timeseries-stable-surface}"
bench_json="${out_dir}/ads_timeseries_benchmark.json"
bench_md="${out_dir}/ads_timeseries_benchmark.md"
bench_work_root="${out_dir}/work"
nextstat_bin="${ATS_STABLE_NEXTSTAT_BIN:-}"

if [[ -n "${ATS_STABLE_PY:-}" ]]; then
  py="${ATS_STABLE_PY}"
elif [[ -e "./.venv/bin/python" && -x "./.venv/bin/python" ]] && "./.venv/bin/python" -V >/dev/null 2>&1; then
  py="${repo_root}/.venv/bin/python"
else
  py="python3"
fi

if [[ -n "${ATS_STABLE_MATURIN:-}" ]]; then
  maturin_cmd=("${ATS_STABLE_MATURIN}")
elif "${py}" -m maturin --version >/dev/null 2>&1; then
  maturin_cmd=("${py}" "-m" "maturin")
elif [[ -x "${repo_root}/.venv/bin/maturin" ]]; then
  maturin_cmd=("${repo_root}/.venv/bin/maturin")
elif command -v maturin >/dev/null 2>&1; then
  maturin_cmd=("maturin")
else
  echo "maturin not found. Install via: pip install maturin" >&2
  exit 7
fi

run_maturin() {
  "${maturin_cmd[@]}" "$@"
}

require_exec() {
  local value="$1"
  if [[ "${value}" == */* ]]; then
    [[ -x "${value}" ]] || {
      echo "Missing required executable: ${value}" >&2
      exit 6
    }
  else
    command -v "${value}" >/dev/null 2>&1 || {
      echo "Missing required command: ${value}" >&2
      exit 6
    }
  fi
}

require_exec cargo
require_exec "${py}"
if [[ "${skip_maturin}" != "1" ]]; then
  run_maturin --version >/dev/null 2>&1 || { echo "maturin not working (tried: ${maturin_cmd[*]})" >&2; exit 7; }
fi

mkdir -p "${cargo_target_dir}" "${out_dir}"

if [[ "${skip_doc_checks}" != "1" ]]; then
  required_files=(
    "docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08.md"
    "docs/benchmarks/ads-timeseries-support-matrix-2026-03-08.md"
    "docs/benchmarks/ads-timeseries-release-notes-2026-03-08.md"
    "docs/benchmarks/ads-timeseries-release-pr-checklist-2026-03-08.md"
    "docs/benchmarks/ads-timeseries-runtime-gate.md"
    "docs/benchmarks/ads-timeseries-promotion-runbook-2026-03-08.md"
    "docs/schemas/benchmarks/ads_timeseries_benchmark_result_v1.schema.json"
    "docs/schemas/benchmarks/ads_timeseries_benchmark_compare_report_v1.schema.json"
    "docs/schemas/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json"
    "docs/schemas/benchmarks/ads_timeseries_benchmark_gate_report_v1.schema.json"
    "docs/specs/benchmarks/ads_timeseries_benchmark_result_v1.example.json"
    "docs/specs/benchmarks/ads_timeseries_benchmark_compare_report_v1.example.json"
    "docs/specs/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.example.json"
    "docs/specs/benchmarks/ads_timeseries_benchmark_gate_report_v1.example.json"
    "benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json"
    ".github/workflows/ads-timeseries-stable-surface.yml"
    "scripts/benchmarks/bench_ads_timeseries_surface.py"
    "scripts/benchmarks/bench_ads_timeseries_surface_remote.sh"
    "scripts/benchmarks/compare_ads_timeseries_benchmark.py"
    "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py"
    "scripts/benchmarks/run_ads_timeseries_benchmark_gate.py"
    "scripts/benchmarks/ads_timeseries_stable_surface_gate.sh"
    "tests/python/test_ads_timeseries_benchmark_smoke.py"
  )
  for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || {
      echo "Missing ads-timeseries stable-surface evidence file: ${file}" >&2
      exit 8
    }
  done

  grep -qF "nextstat.ads" docs/references/python-api.md
  grep -qF "local_level_weekly_model" docs/references/python-api.md
  grep -qF "local_linear_trend_weekly_model" docs/references/python-api.md
  grep -qF "Ads-native Observation / Response / Experimentation Helpers" docs/references/rust-api.md
  grep -qF "local_level_weekly" docs/references/cli.md
  grep -qF "ads-timeseries-stable-surface-acceptance-2026-03-08" docs/README.md
  grep -qF "ads-timeseries-support-matrix-2026-03-08" docs/README.md
  grep -qF "ads-timeseries-runtime-gate.md" docs/README.md
  grep -qF "ads-timeseries-release-pr-checklist-2026-03-08" docs/README.md
  grep -qF "ads-timeseries-stable-surface-acceptance-2026-03-08" docs/benchmarks.md
  grep -qF "ads-timeseries-support-matrix-2026-03-08" docs/benchmarks.md
  grep -qF "ads-timeseries-runtime-gate" docs/benchmarks.md
  grep -qF "ads-timeseries-promotion-runbook-2026-03-08" docs/benchmarks.md
fi

if [[ "${skip_maturin}" != "1" ]]; then
  echo "Building nextstat wheel for ads-timeseries stable-surface gate..."
  ats_wheels="${repo_root}/tmp/ats_stable_wheels"
  rm -rf "${ats_wheels}"
  mkdir -p "${ats_wheels}"
  (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release -o "${ats_wheels}")
  "${py}" -m pip install --no-deps --force-reinstall "${ats_wheels}"/*.whl
  # Installed package in site-packages has _core.so; clear py_path to avoid shadowing.
  py_path=""
  echo
fi

echo "Running ads Rust gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-inference ads::tests -- --test-threads=1
echo

echo "Running weekly Kalman builder Rust gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-inference local_level_weekly_matches_period_7_seasonal_builder -- --nocapture
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-inference local_linear_trend_weekly_matches_period_7_seasonal_builder -- --nocapture
echo

echo "Running weekly CLI contract gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_timeseries weekly_contract
echo

echo "Running Python ads / weekly gate..."
PYTHONPATH="${py_path}" "${py}" -m pytest -q \
  tests/python/test_ads.py \
  tests/python/test_timeseries_kalman.py \
  tests/python/test_ads_timeseries_benchmark_smoke.py
echo

if [[ -z "${nextstat_bin}" ]]; then
  echo "Building CLI benchmark binary..."
  CARGO_TARGET_DIR="${cargo_target_dir}" cargo build -p ns-cli
  nextstat_bin="${cargo_target_dir}/debug/nextstat"
fi

echo "Running ads-timeseries benchmark smoke..."
PYTHONPATH="${py_path}" "${py}" scripts/benchmarks/bench_ads_timeseries_surface.py \
  --nextstat-bin "${nextstat_bin}" \
  --smoke \
  --deterministic \
  --out "${bench_json}" \
  --markdown-out "${bench_md}" \
  --work-root "${bench_work_root}"
echo

echo "Validating ads-timeseries benchmark artifact..."
"${py}" - "${bench_json}" "${repo_root}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])

try:
    import jsonschema  # type: ignore
except Exception as exc:  # pragma: no cover - operational dependency
    raise SystemExit(f"jsonschema is required for the stable-surface gate: {exc}") from exc

schema_path = repo_root / "docs" / "schemas" / "benchmarks" / "ads_timeseries_benchmark_result_v1.schema.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
schema = json.loads(schema_path.read_text(encoding="utf-8"))
jsonschema.validate(instance=report, schema=schema)

assert report["schema_version"] == "nextstat.ads_timeseries_benchmark_result.v1"
assert report["suite"] == "ads_timeseries_surface"
assert report["derived"]["all_cases_ok"] is True
assert report["derived"]["case_count"] == 9
assert report["meta"]["smoke"] is True
assert report["protocol"]["runs"] == 1
assert report["protocol"]["warmups"] == 0

print(
    "validated",
    f"cases={report['derived']['case_count']}",
    f"slowest={report['derived']['slowest_case_id']}",
)
PY
echo

echo "Checking formatting..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo fmt --all --check
echo

echo "OK. Ads-timeseries stable-surface gate passed."
echo "Artifact: ${bench_json}"
