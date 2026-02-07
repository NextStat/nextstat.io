#!/usr/bin/env bash
set -euo pipefail

# Pre-release gate for Apex2 baselines (no cluster).
#
# This is intended to be run on the same "reference" machine where baselines were recorded,
# so that performance numbers are comparable.
#
# Usage:
#   make apex2-pre-release-gate
#
# Optional env vars:
#   - APEX2_COMPARE_ARGS: extra args for compare runner (default: "--require-same-host")
#   - APEX2_PY: python executable (default: ./.venv/bin/python)
#   - APEX2_PYTHONPATH: pythonpath for nextstat bindings (default: bindings/ns-py/python)
#   - APEX2_ALLOW_DIRTY: set to 1 to skip git-clean check
#   - APEX2_SKIP_CARGO: set to 1 to skip cargo build/test
#   - APEX2_SKIP_PYTEST: set to 1 to skip pytest
#   - APEX2_SKIP_MATURIN: set to 1 to skip `maturin develop --release`
#   - APEX2_PYTEST_MARKER: pytest -m expression (default: "not slow")
#   - APEX2_PYTEST_PATHS: space-separated paths (default: "tests/python")
#   - APEX2_PYTEST_EXTRA_ARGS: extra pytest args (default: empty)
#   - APEX2_CARGO_BUILD_ARGS: override cargo build args (default: "--workspace --release")
#   - APEX2_CARGO_TEST_ARGS: override cargo test args (default: "--workspace --all-features")
#   - APEX2_SKIP_TREX_SPEC: set to 1 to skip TREx analysis-spec baseline compare
#   - APEX2_TREX_COMPARE_ARGS: extra args for trex compare (default: "--require-same-host")

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

py="${APEX2_PY:-./.venv/bin/python}"
py_path="${APEX2_PYTHONPATH:-bindings/ns-py/python}"
compare_args="${APEX2_COMPARE_ARGS:---require-same-host --p6-attempts 2}"
allow_dirty="${APEX2_ALLOW_DIRTY:-0}"
skip_cargo="${APEX2_SKIP_CARGO:-0}"
skip_pytest="${APEX2_SKIP_PYTEST:-0}"
skip_maturin="${APEX2_SKIP_MATURIN:-0}"
pytest_marker="${APEX2_PYTEST_MARKER:-not slow}"
pytest_paths="${APEX2_PYTEST_PATHS:-tests/python}"
pytest_extra_args="${APEX2_PYTEST_EXTRA_ARGS:-}"
cargo_build_args="${APEX2_CARGO_BUILD_ARGS:---workspace --release}"
cargo_test_args="${APEX2_CARGO_TEST_ARGS:---workspace --all-features}"
skip_trex="${APEX2_SKIP_TREX_SPEC:-0}"
trex_compare_args="${APEX2_TREX_COMPARE_ARGS:---require-same-host}"

manifest="tmp/baselines/latest_manifest.json"
report="tmp/baseline_compare_report.json"
trex_manifest="tmp/baselines/latest_trex_analysis_spec_manifest.json"
trex_report="tmp/trex_analysis_spec_compare_report.json"

if [[ "${allow_dirty}" != "1" ]]; then
  if command -v git >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "Git working tree is dirty. Commit/stash changes before release gating," >&2
      echo "or set APEX2_ALLOW_DIRTY=1 to override." >&2
      exit 5
    fi
  fi
fi

if [[ "${skip_cargo}" != "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "Missing cargo in PATH; set APEX2_SKIP_CARGO=1 to skip." >&2
    exit 6
  fi
  echo "Running cargo build (${cargo_build_args})..."
  cargo build ${cargo_build_args}
  echo
  echo "Running cargo test (${cargo_test_args})..."
  cargo test ${cargo_test_args}
  echo
fi

if [[ "${skip_maturin}" != "1" ]]; then
  if [[ ! -x "./.venv/bin/maturin" ]]; then
    echo "Missing ./.venv/bin/maturin; set APEX2_SKIP_MATURIN=1 to skip." >&2
    exit 7
  fi
  echo "Rebuilding Python bindings (maturin develop --release)..."
  (cd bindings/ns-py && ../../.venv/bin/maturin develop --release)
  echo
fi

if [[ "${skip_pytest}" != "1" ]]; then
  if [[ ! -x "./.venv/bin/pytest" ]]; then
    echo "Missing ./.venv/bin/pytest; set APEX2_SKIP_PYTEST=1 to skip." >&2
    exit 8
  fi
  echo "Running pytest (-m \"${pytest_marker}\")..."
  pytest_argv=(-q -m "${pytest_marker}")
  if [[ -n "${pytest_extra_args}" ]]; then
    read -r -a extra_argv <<<"${pytest_extra_args}"
    pytest_argv+=("${extra_argv[@]}")
  fi
  read -r -a paths_argv <<<"${pytest_paths}"
  pytest_argv+=("${paths_argv[@]}")
  PYTHONPATH="${py_path}" "${py}" -m pytest "${pytest_argv[@]}"
  echo
fi

if [[ ! -f "${manifest}" ]]; then
  echo "Missing baseline manifest: ${manifest}" >&2
  echo "Record baselines first:" >&2
  echo "  make apex2-baseline-record" >&2
  exit 3
fi

echo "Running Apex2 pre-release gate..."
echo "  manifest: ${manifest}"
echo "  report:   ${report}"
echo

set +e
PYTHONPATH="${py_path}" "${py}" tests/compare_with_latest_baseline.py ${compare_args}
rc=$?
set -e

if [[ "${rc}" -eq 2 ]]; then
  echo
  echo "Compare failed (rc=2). Retrying once to reduce perf flakiness..."
  echo
  set +e
  PYTHONPATH="${py_path}" "${py}" tests/compare_with_latest_baseline.py ${compare_args}
  rc=$?
  set -e
fi

if [[ "${rc}" -ne 0 ]]; then
  exit "${rc}"
fi

echo
echo "OK. Report: ${report}"

if [[ "${skip_trex}" != "1" && -f "${trex_manifest}" ]]; then
  echo
  echo "Running TREx analysis-spec baseline compare..."
  echo "  manifest: ${trex_manifest}"
  echo "  report:   ${trex_report}"
  echo
  PYTHONPATH="${py_path}" "${py}" tests/compare_trex_analysis_spec_with_latest_baseline.py \
    --manifest "${trex_manifest}" \
    --out "${trex_report}" \
    ${trex_compare_args}
  echo
  echo "OK. TREx report: ${trex_report}"
fi
