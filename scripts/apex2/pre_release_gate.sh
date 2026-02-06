#!/usr/bin/env bash
set -euo pipefail

# Pre-release gate for Apex2 baselines (no cluster).
#
# This is intended to be run on the same "reference" machine where baselines were recorded,
# so that performance numbers are comparable.
#
# Usage:
#   bash scripts/apex2/pre_release_gate.sh
#
# Optional env vars:
#   - APEX2_COMPARE_ARGS: extra args for compare runner (default: "--require-same-host")
#   - APEX2_PY: python executable (default: ./.venv/bin/python)
#   - APEX2_PYTHONPATH: pythonpath for nextstat bindings (default: bindings/ns-py/python)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

py="${APEX2_PY:-./.venv/bin/python}"
py_path="${APEX2_PYTHONPATH:-bindings/ns-py/python}"
compare_args="${APEX2_COMPARE_ARGS:---require-same-host}"

manifest="tmp/baselines/latest_manifest.json"
report="tmp/baseline_compare_report.json"

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

PYTHONPATH="${py_path}" "${py}" tests/compare_with_latest_baseline.py ${compare_args}

echo
echo "OK. Report: ${report}"

