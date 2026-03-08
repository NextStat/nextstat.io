#!/usr/bin/env bash
set -euo pipefail

# Stable-surface verification gate for promoted ads variance-reduction helpers.
#
# Scope:
#   - ns_inference::{cuped_adjust, cure_adjust}
#   - nextstat.ads.{cuped_adjust, cure_adjust}
#   - nextstat.tools.{nextstat_ads_cuped_adjust, nextstat_ads_cure_adjust}
#   - nextstat-server server-safe tool-runtime exposure for the same helpers
#
# Optional env vars:
#   - AVR_STABLE_PY: Python executable (default: ./.venv/bin/python, else python3)
#   - AVR_STABLE_MATURIN: maturin executable (default: ./.venv/bin/maturin, else maturin)
#   - AVR_STABLE_PYTHONPATH: pythonpath for local bindings (default: bindings/ns-py/python)
#   - AVR_STABLE_CARGO_TARGET_DIR: isolated cargo target dir
#   - AVR_STABLE_OUT_DIR: directory for emitted benchmark-smoke artifacts
#   - AVR_STABLE_NEXTSTAT_BIN: optional path to a prebuilt nextstat CLI binary
#   - AVR_STABLE_SKIP_MATURIN: set to 1 to skip `maturin develop --release`
#   - AVR_STABLE_SKIP_DOC_CHECKS: set to 1 to skip docs / schema / golden sync checks

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

skip_maturin="${AVR_STABLE_SKIP_MATURIN:-0}"
skip_doc_checks="${AVR_STABLE_SKIP_DOC_CHECKS:-0}"
py_path="${AVR_STABLE_PYTHONPATH:-bindings/ns-py/python}"
cargo_target_dir="${AVR_STABLE_CARGO_TARGET_DIR:-${repo_root}/tmp/cargo_target_ads_variance_reduction_stable}"
out_dir="${AVR_STABLE_OUT_DIR:-${repo_root}/tmp/ads-variance-reduction-stable-surface}"
bench_json="${out_dir}/ads_variance_reduction_benchmark.json"
bench_md="${out_dir}/ads_variance_reduction_benchmark.md"
bench_work_root="${out_dir}/work"
nextstat_bin="${AVR_STABLE_NEXTSTAT_BIN:-}"

if [[ -n "${AVR_STABLE_PY:-}" ]]; then
  py="${AVR_STABLE_PY}"
elif [[ -e "./.venv/bin/python" && -x "./.venv/bin/python" ]] && "./.venv/bin/python" -V >/dev/null 2>&1; then
  py="${repo_root}/.venv/bin/python"
else
  py="python3"
fi

if [[ -n "${AVR_STABLE_MATURIN:-}" ]]; then
  maturin_cmd=("${AVR_STABLE_MATURIN}")
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
    "docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md"
    "docs/benchmarks/ads-variance-reduction-runtime-gate.md"
    "docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md"
    "docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md"
    "docs/schemas/benchmarks/ads_variance_reduction_benchmark_result_v1.schema.json"
    "docs/schemas/benchmarks/ads_variance_reduction_benchmark_compare_report_v1.schema.json"
    "docs/schemas/benchmarks/ads_variance_reduction_benchmark_gate_report_v1.schema.json"
    "docs/schemas/benchmarks/ads_variance_reduction_benchmark_baseline_promotion_report_v1.schema.json"
    "docs/specs/benchmarks/ads_variance_reduction_benchmark_result_v1.example.json"
    "docs/specs/benchmarks/ads_variance_reduction_benchmark_compare_report_v1.example.json"
    "docs/specs/benchmarks/ads_variance_reduction_benchmark_gate_report_v1.example.json"
    "docs/specs/benchmarks/ads_variance_reduction_benchmark_baseline_promotion_report_v1.example.json"
    "docs/schemas/tools/nextstat_tool_result_strict_v1.schema.json"
    "docs/schemas/tools/nextstat_tool_result_server_strict_v1.schema.json"
    "docs/specs/nextstat_tool_schema_local_v1.example.json"
    "docs/specs/nextstat_tool_schema_server_v1.example.json"
    "benchmarks/artifacts/ads_variance_reduction_baselines/nextstat-bench/accepted.json"
    ".github/workflows/ads-variance-reduction-stable-surface.yml"
    "scripts/benchmarks/bench_ads_variance_reduction_matrix.py"
    "scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh"
    "scripts/benchmarks/compare_ads_variance_reduction_benchmark.py"
    "scripts/benchmarks/promote_ads_variance_reduction_benchmark_baseline.py"
    "scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py"
    "scripts/benchmarks/ads_variance_reduction_stable_surface_gate.sh"
    "tests/python/test_ads.py"
    "tests/python/test_ads_variance_reduction_fixtures.py"
    "tests/python/test_ads_variance_reduction_benchmark_smoke.py"
    "tests/python/test_tools_ads_variance_reduction.py"
    "tests/fixtures/variance_reduction_benchmark/scenario_matrix.json"
  )
  for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || {
      echo "Missing ads-variance-reduction stable-surface evidence file: ${file}" >&2
      exit 8
    }
  done

  grep -qF "nextstat.ads.cuped_adjust" docs/references/python-api.md
  grep -qF "nextstat_ads_cuped_adjust" docs/references/python-api.md
  grep -qF "nextstat.ads.cure_adjust" docs/references/python-api.md
  grep -qF "nextstat_ads_cure_adjust" docs/references/python-api.md
  grep -qF "shared Rust API surface" docs/references/rust-api.md
  grep -qF "nextstat.tools" docs/references/tool-api.md
  grep -qF "nextstat_ads_cuped_adjust" docs/references/tool-api.md
  grep -qF "nextstat_ads_cure_adjust" docs/references/tool-api.md
  grep -qF "server-safe subset" docs/references/server-api.md
  grep -qF "nextstat_ads_cuped_adjust" docs/references/server-api.md
  grep -qF "nextstat_ads_cure_adjust" docs/references/server-api.md
  grep -qF "ads-variance-reduction-stable-surface-acceptance-2026-03-09" docs/README.md
  grep -qF "ads-variance-reduction-runtime-gate.md" docs/README.md
  grep -qF "ads-variance-reduction-stable-surface-acceptance-2026-03-09" docs/benchmarks.md
  grep -qF "ads-variance-reduction-runtime-gate" docs/benchmarks.md
fi

if [[ "${skip_maturin}" != "1" ]]; then
  echo "Building nextstat wheel for ads-variance-reduction stable-surface gate..."
  avr_wheels="${repo_root}/tmp/avr_stable_wheels"
  rm -rf "${avr_wheels}"
  mkdir -p "${avr_wheels}"
  (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release -o "${avr_wheels}")
  "${py}" -m pip install --no-deps --force-reinstall "${avr_wheels}"/*.whl
  # Installed package in site-packages has _core.so; clear py_path to avoid shadowing.
  py_path=""
  echo
fi

if [[ "${skip_doc_checks}" != "1" ]]; then
  echo "Checking tool manifest / generated tool contracts..."
  PYTHONPATH="${py_path}" "${py}" scripts/validate_tool_manifest.py
  PYTHONPATH="${py_path}" "${py}" scripts/generate_tool_contract_schemas.py --check
  PYTHONPATH="${py_path}" "${py}" scripts/generate_tool_schema_examples.py --check
  PYTHONPATH="${py_path}" "${py}" scripts/generate_tool_reference_docs.py --check
  PYTHONPATH="${py_path}" "${py}" -m scripts.generate_agent_bootstrap_packs --check
  PYTHONPATH="${py_path}" "${py}" scripts/generate_tool_goldens.py --check
  echo
fi

echo "Running ads variance-reduction Rust gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-inference variance_reduction::tests --lib -- --test-threads=1
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-inference --test variance_reduction_fixtures -- --test-threads=1
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-inference --test calculator_tests -- --test-threads=1
echo

echo "Running ads variance-reduction server-tool Rust gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-server tool_execute_ads_variance_reduction_supports_server_safe_modes -- --nocapture
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-server server_tools_match_local_tool_goldens_on_ads_cuped_adjust_small_deterministic -- --nocapture
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-server server_tools_match_local_tool_goldens_on_ads_cure_adjust_small_deterministic -- --nocapture
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-server server_tool_schema_matches_server_strict_schema -- --nocapture
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-server server_tool_schema_exposes_capability_policy_metadata -- --nocapture
echo

echo "Running Python ads / tool gate..."
PYTHONPATH="${py_path}" "${py}" -m pytest -q \
  tests/python/test_ads.py \
  tests/python/test_ads_variance_reduction_fixtures.py \
  tests/python/test_ads_variance_reduction_benchmark_smoke.py \
  tests/python/test_tools_ads_variance_reduction.py
echo

if [[ -z "${nextstat_bin}" ]]; then
  echo "Building CLI benchmark binary..."
  CARGO_TARGET_DIR="${cargo_target_dir}" cargo build -p ns-cli
  nextstat_bin="${cargo_target_dir}/debug/nextstat"
fi

echo "Running ads variance-reduction benchmark smoke..."
PYTHONPATH="${py_path}" "${py}" scripts/benchmarks/bench_ads_variance_reduction_matrix.py \
  --nextstat-bin "${nextstat_bin}" \
  --smoke \
  --deterministic \
  --out "${bench_json}" \
  --markdown-out "${bench_md}" \
  --work-root "${bench_work_root}"
echo

echo "Validating ads variance-reduction benchmark artifact..."
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

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "ads_variance_reduction_benchmark_result_v1.schema.json"
)
report = json.loads(report_path.read_text(encoding="utf-8"))
schema = json.loads(schema_path.read_text(encoding="utf-8"))
jsonschema.validate(instance=report, schema=schema)

assert report["schema_version"] == "nextstat.ads_variance_reduction_benchmark_result.v1"
assert report["suite"] == "ads_variance_reduction_matrix"
assert report["derived"]["all_cases_ok"] is True
assert report["derived"]["case_count"] == 12
assert report["derived"]["scenario_count"] == 4
assert report["derived"]["method_count"] == 3
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

echo "OK. Ads-variance-reduction stable-surface gate passed."
echo "Artifact: ${bench_json}"
