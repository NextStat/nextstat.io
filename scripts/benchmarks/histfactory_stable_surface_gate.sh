#!/usr/bin/env bash
set -euo pipefail

# Stable-surface verification gate for the promoted HistFactory core subset.
#
# Scope:
#   - pyhf JSON workspace input contract
#   - deterministic CPU parity: fit, hypotest, upper-limit, scan, audit
#   - CLI, Python, tool, and server HistFactory surfaces
#
# Optional env vars:
#   - HF_STABLE_PY: Python executable (default: ./.venv/bin/python, else python3, else python)
#   - HF_STABLE_MATURIN: maturin executable (default: python -m maturin, else .venv/bin/maturin, else maturin)
#   - HF_STABLE_PYTHONPATH: pythonpath for local bindings (default: bindings/ns-py/python)
#   - HF_STABLE_CARGO_TARGET_DIR: isolated cargo target dir
#   - HF_STABLE_SKIP_MATURIN: set to 1 to skip local wheelhouse build/install
#   - HF_STABLE_SKIP_DOC_CHECKS: set to 1 to skip release-governance doc checks

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

skip_doc_checks="${HF_STABLE_SKIP_DOC_CHECKS:-0}"
skip_maturin="${HF_STABLE_SKIP_MATURIN:-0}"
py_path="${HF_STABLE_PYTHONPATH:-bindings/ns-py/python}"
cargo_target_dir="${HF_STABLE_CARGO_TARGET_DIR:-${repo_root}/tmp/cargo_target_hf_stable_surface}"

if [[ -n "${HF_STABLE_PY:-}" ]]; then
  py="${HF_STABLE_PY}"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  py="${repo_root}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="python3"
else
  py="python"
fi

if [[ -n "${HF_STABLE_MATURIN:-}" ]]; then
  maturin_cmd=("${HF_STABLE_MATURIN}")
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
if [[ "${skip_maturin}" != "1" ]] && ! run_maturin --version >/dev/null 2>&1; then
  echo "Missing maturin executable (tried: ${maturin_cmd[*]})" >&2
  exit 6
fi

mkdir -p "${cargo_target_dir}"

if [[ "${skip_doc_checks}" != "1" ]]; then
  required_files=(
    "docs/benchmarks/histfactory-support-matrix-2026-03-17.md"
    "docs/benchmarks/histfactory-stable-surface-acceptance-2026-03-17.md"
  )
  for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || {
      echo "Missing HistFactory stable-surface evidence file: ${file}" >&2
      exit 8
    }
  done
fi

if [[ "${skip_maturin}" != "1" ]]; then
  echo "Building local wheelhouse for HistFactory stable-surface gate..."
  wheelhouse="${repo_root}/tmp/histfactory_stable_wheels"
  rm -rf "${wheelhouse}"
  mkdir -p "${wheelhouse}"
  (cd bindings/ns-cli-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release --interpreter "${py}" -o "${wheelhouse}")
  (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release --interpreter "${py}" -o "${wheelhouse}")
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

echo "Running HistFactory translation gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-translate histfactory --lib -- --test-threads=1
echo

echo "Running HistFactory CLI smoke gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_run_histfactory_smoke
echo

echo "Running HistFactory CLI import gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_import_histfactory
echo

echo "Running HistFactory CLI export gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_export_histfactory
echo

echo "Running HistFactory CLI validate gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_validate
echo

echo "Running HistFactory Python parity gate..."
NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="${py_path}" "${py}" -m pytest -q \
  tests/python/test_pyhf_generated_workspaces.py \
  tests/python/test_bindings_api.py \
  tests/python/test_hypotest_cls.py \
  tests/python/test_histfactory_bin_edges_contract.py \
  tests/python/test_histfactory_provenance_contract.py \
  tests/python/test_histfactory_xml_import_contract.py \
  tests/python/test_hist_mode_root_hists.py \
  -k "histfactory or pyhf or workspace_audit or fit or hypotest or upper_limit or scan or hist_mode"
echo

echo "Checking formatting..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo fmt --all --check
echo

echo "OK. HistFactory stable-surface gate passed."
