#!/usr/bin/env bash
set -euo pipefail

# Wrapper for HTCondor job arrays: run one ROOT parity case selected by index.
#
# Usage: run_apex2_root_suite_case.sh <case_index>
#
# Required env vars:
# - APEX2_ROOT_CASES_JSON: path to cases JSON (relative to repo or absolute)
# - APEX2_RESULTS_DIR: output directory (relative or absolute); created if missing
#
# Optional env vars:
# - APEX2_ROOT_SETUP: command to source ROOT environment (e.g. "source /cvmfs/.../setup.sh")
# - APEX2_PYTHON: python executable (default: python3)
# - APEX2_ROOT_DQ_ATOL: dq tolerance (default: runner default)
# - APEX2_ROOT_MU_HAT_ATOL: mu_hat tolerance (default: runner default)

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <case_index>" >&2
  exit 2
fi

case_index="$1"
repo_root="${PWD}"
python_bin="${APEX2_PYTHON:-python3}"
cases_json="${APEX2_ROOT_CASES_JSON:?missing APEX2_ROOT_CASES_JSON}"
results_dir="${APEX2_RESULTS_DIR:?missing APEX2_RESULTS_DIR}"

if [[ -n "${APEX2_ROOT_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  eval "${APEX2_ROOT_SETUP}"
fi

mkdir -p "${results_dir}"

host="$(hostname -s || hostname)"
out_json="${results_dir}/apex2_root_case_${case_index}_${host}.json"
workdir="${results_dir}/workdir_case_${case_index}_${host}"

export PYTHONPATH="${repo_root}/bindings/ns-py/python${PYTHONPATH:+:${PYTHONPATH}}"

extra_args=()
if [[ -n "${APEX2_ROOT_DQ_ATOL:-}" ]]; then
  extra_args+=(--dq-atol "${APEX2_ROOT_DQ_ATOL}")
fi
if [[ -n "${APEX2_ROOT_MU_HAT_ATOL:-}" ]]; then
  extra_args+=(--mu-hat-atol "${APEX2_ROOT_MU_HAT_ATOL}")
fi

"${python_bin}" tests/apex2_root_suite_report.py \
  --cases "${cases_json}" \
  --case-index "${case_index}" \
  --workdir "${workdir}" \
  --out "${out_json}" \
  "${extra_args[@]}"

echo "Wrote: ${out_json}"
