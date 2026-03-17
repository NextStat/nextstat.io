#!/usr/bin/env bash
set -euo pipefail

# Stable-first verification gate for the promoted GVM subset.
#
# Scope:
#   - nextstat combine-measurements
#   - nextstat combine-measurements-calibrate
#   - nextstat combine-measurements-calibrate-study
#   - matching Python wrappers in nextstat.hep
#
# Optional env vars:
#   - GVM_STABLE_PY: Python executable (default: ./.venv/bin/python, else python3, else python)
#   - GVM_STABLE_MATURIN: maturin executable (default: maturin)
#   - GVM_STABLE_PYTHONPATH: pythonpath for local bindings (default: bindings/ns-py/python)
#   - GVM_STABLE_ALLOW_DIRTY: set to 1 to skip git-clean check
#   - GVM_STABLE_SKIP_MATURIN: set to 1 to skip `maturin develop`
#   - GVM_STABLE_SKIP_DOC_CHECKS: set to 1 to skip documentation/support-artifact checks

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

allow_dirty="${GVM_STABLE_ALLOW_DIRTY:-0}"
skip_maturin="${GVM_STABLE_SKIP_MATURIN:-0}"
skip_doc_checks="${GVM_STABLE_SKIP_DOC_CHECKS:-0}"
py_path="${GVM_STABLE_PYTHONPATH:-bindings/ns-py/python}"

if [[ -n "${GVM_STABLE_PY:-}" ]]; then
  py="${GVM_STABLE_PY}"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  py="${repo_root}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="python3"
else
  py="python"
fi

if [[ -n "${GVM_STABLE_MATURIN:-}" ]]; then
  maturin_cmd=("${GVM_STABLE_MATURIN}")
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

if [[ "${allow_dirty}" != "1" ]] && command -v git >/dev/null 2>&1; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Git working tree is dirty. Commit or stash changes before running the stable-first gate," >&2
    echo "or set GVM_STABLE_ALLOW_DIRTY=1 to override." >&2
    exit 5
  fi
fi

for cmd in cargo "${py}"; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 6
  fi
done

if [[ "${skip_maturin}" != "1" ]] && ! run_maturin --version >/dev/null 2>&1; then
  echo "Missing maturin executable (tried: ${maturin_cmd[*]})" >&2
  echo "Install maturin or set GVM_STABLE_SKIP_MATURIN=1." >&2
  exit 7
fi

if [[ "${skip_doc_checks}" != "1" ]]; then
  required_files=(
    "docs/benchmarks/gvm-measurement-combine-snapshot-2026-03-07.md"
    "docs/benchmarks/gvm-numerical-paper-robustness-snapshot-2026-03-07.md"
    "docs/benchmarks/gvm-stable-surface-readiness-2026-03-07.md"
    "docs/benchmarks/gvm-stable-surface-support-policy-2026-03-07.md"
    "docs/benchmarks/gvm-stable-first-decision-2026-03-07.md"
    "docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md"
    "docs/benchmarks/gvm-stable-first-release-notes-2026-03-07.md"
    "docs/benchmarks/gvm-stable-first-release-candidate-v0.10.0-2026-03-08.md"
    "docs/benchmarks/gvm-stable-first-release-pr-checklist-2026-03-07.md"
    "docs/benchmarks/gvm-stable-first-launch-checklist-2026-03-07.md"
    "docs/guides/gvm-external-validation-kit.md"
    "docs/guides/gvm-external-validator-outreach-pack.md"
    "docs/guides/gvm-external-validation-tracker-template.md"
    "docs/examples/gvm-stable-first/README.md"
    "docs/examples/gvm-stable-first/external-validator-invite-template.md"
    "docs/examples/gvm-stable-first/external-validation-report-template.md"
    "docs/examples/gvm-stable-first/manifest.yaml"
    "docs/examples/gvm-stable-first/measurements.csv"
    "docs/examples/gvm-stable-first/stat_covariance.csv"
    "docs/examples/gvm-stable-first/systematics.csv"
    "docs/examples/gvm-stable-first/correlations.csv"
    "docs/examples/gvm-stable-first/spec.json"
    "scripts/gvm/run_stable_first_example.sh"
  )
  for file in "${required_files[@]}"; do
    if [[ ! -f "${file}" ]]; then
      echo "Missing stable-first evidence file: ${file}" >&2
      exit 8
    fi
  done

  grep -qF "stable-first CLI entry point" docs/references/cli.md
  grep -qF "stable-first tabular ingress" docs/references/cli.md
  grep -qF "stable-first manifest ingress helper" docs/references/cli.md
  grep -qF "stable-first toy-calibration companion command" docs/references/cli.md
  grep -qF "stable-first repeated-seed companion command" docs/references/cli.md
  grep -qF "stable-first tabular ingress helper" docs/references/python-api.md
  grep -qF "stable-first manifest ingress helper" docs/references/python-api.md
  grep -qF "stable-first scalar measurement combination wrapper" docs/references/python-api.md
  grep -qF "stable-first toy-calibration wrapper" docs/references/python-api.md
  grep -qF "stable-first repeated-seed calibration wrapper" docs/references/python-api.md
fi

echo "Running stable-first GVM Rust core gate..."
cargo test -p ns-inference --lib measurement_combine -- --test-threads=1
echo

echo "Running stable-first GVM CLI gate..."
cargo test -q -p ns-cli --test cli_measurement_combine
echo

echo "Running stable-first GVM example bundle smoke..."
example_out="$(mktemp -d "${TMPDIR:-/tmp}/gvm-stable-first-gate.XXXXXX")"
trap 'rm -rf "${example_out}"' EXIT
bash scripts/gvm/run_stable_first_example.sh \
  --out-dir "${example_out}" \
  --n-toys 4 \
  --seeds 42,43 \
  --threads 1
test -f "${example_out}/spec.json"
test -f "${example_out}/result.json"
test -f "${example_out}/calibration.json"
test -f "${example_out}/calibration_study.json"
echo

if [[ "${skip_maturin}" != "1" ]]; then
  echo "Building local wheelhouse for stable-first GVM gate..."
  wheelhouse="${repo_root}/tmp/gvm_wheels"
  rm -rf "${wheelhouse}"
  mkdir -p "${wheelhouse}"
  (cd bindings/ns-cli-py && run_maturin build --release --interpreter "${py}" -o "${wheelhouse}")
  (cd bindings/ns-py && run_maturin build --release --interpreter "${py}" -o "${wheelhouse}")
  "${py}" -m pip install --force-reinstall --no-deps \
    "${wheelhouse}"/nextstat_cli-*.whl \
    "${wheelhouse}"/nextstat-*.whl
  # Installed package in site-packages has _core.so; clear py_path to avoid shadowing.
  py_path=""
  NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="" "${py}" - <<'PY'
import nextstat
import nextstat._core as core

assert callable(nextstat.set_threads), nextstat.set_threads
print(f"nextstat={nextstat.__file__}")
print(f"_core={core.__file__}")
PY
  echo
fi

echo "Running stable-first GVM Python gate..."
NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="${py_path}" "${py}" -m pytest -q tests/python/test_hep_module_api.py
echo

echo "Checking formatting..."
cargo fmt --all --check
echo

echo "OK. Stable-first GVM gate passed."
