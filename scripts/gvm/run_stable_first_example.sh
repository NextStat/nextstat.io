#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

manifest="docs/examples/gvm-stable-first/manifest.yaml"
out_dir="tmp/gvm-stable-first-example"
solver="auto"
ci_level="0.68"
n_toys="32"
seeds="42,43"
threads="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      manifest="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --solver)
      solver="$2"
      shift 2
      ;;
    --ci-level)
      ci_level="$2"
      shift 2
      ;;
    --n-toys)
      n_toys="$2"
      shift 2
      ;;
    --seeds)
      seeds="$2"
      shift 2
      ;;
    --threads)
      threads="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/gvm/run_stable_first_example.sh [options]

Options:
  --manifest PATH   Manifest to build from (default: docs/examples/gvm-stable-first/manifest.yaml)
  --out-dir DIR     Output directory for generated artifacts (default: tmp/gvm-stable-first-example)
  --solver NAME     Solver for fit/calibration/study (default: auto)
  --ci-level FLOAT  Confidence level for fit/calibration/study (default: 0.68)
  --n-toys INT      Number of toys for calibration and study (default: 32)
  --seeds CSV       Seeds for calibration study (default: 42,43)
  --threads INT     Thread count for CLI commands (default: 1)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${manifest}" ]]; then
  echo "Manifest not found: ${manifest}" >&2
  exit 3
fi

mkdir -p "${out_dir}"

spec_out="${out_dir}/spec.json"
fit_out="${out_dir}/result.json"
calibration_out="${out_dir}/calibration.json"
study_out="${out_dir}/calibration_study.json"

run_nextstat() {
  if command -v nextstat >/dev/null 2>&1; then
    nextstat "$@"
  else
    local cargo_target_dir
    cargo_target_dir="${CARGO_TARGET_DIR:-${repo_root}/target/gvm-stable-first-example}"
    CARGO_TARGET_DIR="${cargo_target_dir}" cargo run -q -p ns-cli -- "$@"
  fi
}

echo "Building stable-first GVM spec..."
run_nextstat combine-measurements-build-spec \
  --manifest "${manifest}" \
  --output "${spec_out}"

echo "Running stable-first GVM fit..."
run_nextstat combine-measurements \
  --input "${spec_out}" \
  --output "${fit_out}" \
  --solver "${solver}" \
  --ci-level "${ci_level}" \
  --threads "${threads}"

echo "Running stable-first GVM calibration..."
run_nextstat combine-measurements-calibrate \
  --input "${spec_out}" \
  --output "${calibration_out}" \
  --solver "${solver}" \
  --ci-level "${ci_level}" \
  --n-toys "${n_toys}" \
  --seed 42 \
  --threads "${threads}"

echo "Running stable-first GVM calibration study..."
run_nextstat combine-measurements-calibrate-study \
  --input "${spec_out}" \
  --output "${study_out}" \
  --solver "${solver}" \
  --ci-level "${ci_level}" \
  --n-toys "${n_toys}" \
  --seeds "${seeds}" \
  --threads "${threads}"

echo "Stable-first GVM example completed."
echo "Artifacts written to ${out_dir}"
