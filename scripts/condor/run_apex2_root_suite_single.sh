#!/usr/bin/env bash
set -euo pipefail

# Wrapper for HTCondor: run the full ROOT/HistFactory parity suite (all cases).
#
# Required env vars:
# - APEX2_ROOT_CASES_JSON: path to cases JSON (relative to repo or absolute)
# - APEX2_RESULTS_DIR: output directory (relative or absolute); created if missing
#
# Optional env vars:
# - APEX2_ROOT_SETUP: command to source ROOT environment (e.g. "source /cvmfs/.../setup.sh")
# - APEX2_PYTHON: python executable (default: python3)

repo_root="${PWD}"
python_bin="${APEX2_PYTHON:-python3}"
cases_json="${APEX2_ROOT_CASES_JSON:?missing APEX2_ROOT_CASES_JSON}"
results_dir="${APEX2_RESULTS_DIR:?missing APEX2_RESULTS_DIR}"

if [[ -n "${APEX2_ROOT_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  eval "${APEX2_ROOT_SETUP}"
fi

mkdir -p "${results_dir}"

ts="$(date -u +%Y%m%d_%H%M%S)"
host="$(hostname -s || hostname)"
out_json="${results_dir}/apex2_root_suite_${host}_${ts}.json"
workdir="${results_dir}/workdir_${host}_${ts}"

export PYTHONPATH="${repo_root}/bindings/ns-py/python${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" tests/apex2_root_suite_report.py \
  --cases "${cases_json}" \
  --keep-going \
  --workdir "${workdir}" \
  --out "${out_json}"

echo "Wrote: ${out_json}"

