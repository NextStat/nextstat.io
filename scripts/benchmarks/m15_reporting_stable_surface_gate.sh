#!/usr/bin/env bash
set -euo pipefail

# Stable-surface promotion gate for the M15 reporting runtime surface.
#
# This gate is release-grade evidence only. It must run on the canonical
# nextstat-bench host, build a fresh release ns-cli binary from the current
# snapshot, emit the benchmark artifact, and compare it against the accepted
# release baseline.
#
# Optional env vars:
#   - M15_STABLE_PY: Python executable (default: python3)
#   - M15_STABLE_HOSTNAME: required hostname (default: nextstat-bench)
#   - M15_STABLE_CARGO_TARGET_DIR: isolated cargo target dir
#   - M15_STABLE_OUT_DIR: output directory for benchmark artifacts
#   - M15_STABLE_RUNS: benchmark runs (default: 5)
#   - M15_STABLE_WARMUPS: benchmark warmups (default: 1)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

required_hostname="${M15_STABLE_HOSTNAME:-nextstat-bench}"
actual_hostname="$(hostname -s 2>/dev/null || hostname)"
actual_hostname_full="$(hostname)"
py="${M15_STABLE_PY:-python3}"
cargo_target_dir="${M15_STABLE_CARGO_TARGET_DIR:-${repo_root}/tmp/cargo_target_m15_reporting_stable_surface}"
out_dir="${M15_STABLE_OUT_DIR:-${repo_root}/tmp/m15_reporting_stable_surface}"
runs="${M15_STABLE_RUNS:-5}"
warmups="${M15_STABLE_WARMUPS:-1}"
nextstat_bin="${cargo_target_dir}/release/nextstat"
benchmark_json="${out_dir}/m15_reporting_benchmark.json"
benchmark_md="${out_dir}/m15_reporting_benchmark.md"
compare_json="${out_dir}/m15_reporting_compare.json"
baseline_json="${repo_root}/benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json"

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

if [[ "${actual_hostname}" != "${required_hostname}" && "${actual_hostname_full}" != "${required_hostname}" ]]; then
  echo "M15 stable-surface gate must run on ${required_hostname}; found ${actual_hostname_full}" >&2
  exit 7
fi

required_files=(
  "scripts/benchmarks/bench_m15_reporting.py"
  "scripts/benchmarks/compare_m15_reporting_benchmark.py"
  "docs/benchmarks/m15-reporting-runtime-gate.md"
  "docs/references/m15-reporting.md"
  "docs/schemas/benchmarks/m15_reporting_benchmark_result_v1.schema.json"
  "docs/schemas/benchmarks/m15_reporting_benchmark_compare_report_v1.schema.json"
  "docs/specs/pharma/m15_reporting_benchmark_result_v1.example.json"
  "docs/specs/pharma/m15_reporting_benchmark_compare_report_v1.example.json"
  "benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json"
  ".github/workflows/m15-reporting-stable-surface.yml"
)
for file in "${required_files[@]}"; do
  [[ -f "${file}" ]] || {
    echo "Missing M15 stable-surface evidence file: ${file}" >&2
    exit 8
  }
done

grep -qF "make m15-reporting-stable-surface-gate" docs/benchmarks/m15-reporting-runtime-gate.md
grep -qF ".github/workflows/m15-reporting-stable-surface.yml" docs/benchmarks/m15-reporting-runtime-gate.md
grep -qF "make m15-reporting-stable-surface-gate" docs/references/m15-reporting.md
grep -qF "m15-reporting-stable-surface.yml" docs/references/m15-reporting.md

mkdir -p "${cargo_target_dir}" "${out_dir}"

git_commit="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"

echo "Building release ns-cli for M15 stable-surface gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo build --release -p ns-cli
echo

echo "Running M15 reporting benchmark..."
NEXTSTAT_BENCH_GIT_COMMIT="${git_commit}" "${py}" scripts/benchmarks/bench_m15_reporting.py \
  --nextstat-bin "${nextstat_bin}" \
  --out "${benchmark_json}" \
  --markdown-out "${benchmark_md}" \
  --work-root "${out_dir}/work" \
  --runs "${runs}" \
  --warmups "${warmups}" \
  --deterministic
echo

echo "Comparing M15 benchmark against accepted nextstat-bench baseline..."
"${py}" scripts/benchmarks/compare_m15_reporting_benchmark.py \
  --baseline "${baseline_json}" \
  --current "${benchmark_json}" \
  --out "${compare_json}" \
  --fail-on-review
echo

echo "OK. M15 reporting stable-surface gate passed."
echo "Benchmark artifact: ${benchmark_json}"
echo "Markdown artifact: ${benchmark_md}"
echo "Compare artifact: ${compare_json}"
