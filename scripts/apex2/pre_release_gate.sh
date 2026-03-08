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
#   - APEX2_SKIP_MATURIN: set to 1 to skip local wheelhouse build+install (dev/debug only;
#       NOT a valid canonical prerelease path — compiled nextstat must come from wheelhouse)
#   - APEX2_PYTEST_MARKER: pytest -m expression (default: "not slow")
#   - APEX2_PYTEST_PATHS: space-separated paths (default: "tests/python")
#   - APEX2_PYTEST_EXTRA_ARGS: extra pytest args (default: empty)
#   - APEX2_CARGO_BUILD_ARGS: override cargo build args (default: "--workspace --release")
#   - APEX2_CARGO_TEST_ARGS: override cargo test args (default: "--workspace")
#   - APEX2_SKIP_TREX_SPEC: set to 1 to skip TREx analysis-spec baseline compare
#   - APEX2_TREX_COMPARE_ARGS: extra args for trex compare (default: "--require-same-host")
#   - APEX2_SKIP_ROOT_SUITE: set to 1 to skip ROOT suite baseline compare
#   - APEX2_ROOT_CASES: path to ROOT suite cases JSON (default: tests/fixtures/trex_parity_pack/cases_minimal.json)

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
# `--all-features` pulls in optional GPU backends (e.g. CUDA via cudarc) which require
# toolchains like `nvcc` to be present. Keep the default runnable on a plain dev machine.
cargo_test_args="${APEX2_CARGO_TEST_ARGS:---workspace}"
skip_trex="${APEX2_SKIP_TREX_SPEC:-0}"
trex_compare_args="${APEX2_TREX_COMPARE_ARGS:---require-same-host}"
skip_root_suite="${APEX2_SKIP_ROOT_SUITE:-0}"
root_cases="${APEX2_ROOT_CASES:-tests/fixtures/trex_parity_pack/cases_minimal.json}"

manifest="tmp/baselines/latest_manifest.json"
report="tmp/baseline_compare_report.json"
trex_manifest="tmp/baselines/latest_trex_analysis_spec_manifest.json"
trex_report="tmp/trex_analysis_spec_compare_report.json"
root_manifest="tmp/baselines/latest_root_manifest.json"
root_report="tmp/root_suite_compare_report.json"
surface_report="tmp/release_surface_matrix_report.json"
surface_report_md="tmp/release_surface_matrix_report.md"
release_manifest="tmp/release_manifest.json"
release_manifest_md="tmp/release_manifest.md"
release_candidate_bundle_dir="tmp/release_candidate_bundle"
release_dry_run_dir="tmp/release_full_fidelity_simulation"
release_dry_run_report="tmp/release_full_fidelity_simulation_report.json"
release_dry_run_report_md="tmp/release_full_fidelity_simulation_report.md"

if [[ "${allow_dirty}" != "1" ]]; then
  if command -v git >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "Git working tree is dirty. Commit/stash changes before release gating," >&2
      echo "or set APEX2_ALLOW_DIRTY=1 to override." >&2
      exit 5
    fi
  fi
fi

echo "Validating release surface matrix..."
PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_surface_matrix \
  --check \
  --out-json "${surface_report}" \
  --out-md "${surface_report_md}"
echo "OK. Release surface report: ${surface_report}"
echo "OK. Release surface summary: ${surface_report_md}"
echo

version="$(grep '^version' Cargo.toml | head -1 | sed 's/.*\"\(.*\)\"/\1/')"
release_tag="v${version}"
echo "Rendering release manifest..."
PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_manifest \
  --release-tag "${release_tag}" \
  --mode prepare \
  --out-json "${release_manifest}" \
  --out-md "${release_manifest_md}"
echo "OK. Release manifest: ${release_manifest}"
echo "OK. Release manifest summary: ${release_manifest_md}"
echo

echo "Running local full-fidelity release simulation..."
PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_full_fidelity_simulation \
  --release-tag "${release_tag}" \
  --mode prepare \
  --out-dir "${release_dry_run_dir}" \
  --out-json "${release_dry_run_report}" \
  --out-md "${release_dry_run_report_md}"
echo "OK. Release dry-run report: ${release_dry_run_report}"
echo "OK. Release dry-run summary: ${release_dry_run_report_md}"
echo

if [[ "${skip_cargo}" != "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "Missing cargo in PATH; set APEX2_SKIP_CARGO=1 to skip." >&2
    exit 6
  fi

  # ── CI-parity: cargo fmt (catches formatting drift before CI) ──
  echo "Running cargo fmt --all --check..."
  cargo fmt --all --check
  echo

  # ── CI-parity: clippy -D warnings (exact same flags as release-candidate.yml) ──
  echo "Running cargo clippy --workspace --all-targets -- -D warnings..."
  cargo clippy --workspace --all-targets -- -D warnings
  echo

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

  # ── Local wheelhouse install (no PyPI dependency) ────────────────────────
  # Pre-release runs BEFORE packages are published to PyPI, so we cannot
  # rely on `maturin develop` resolving `nextstat-cli==X.Y.Z` from the index.
  # Instead: build both wheels locally → install from a local wheelhouse
  # with --no-index.  This keeps pyproject.toml metadata honest while making
  # the gate fully offline-capable.
  wheelhouse="${repo_root}/tmp/wheelhouse"
  rm -rf "${wheelhouse}"
  mkdir -p "${wheelhouse}"

  echo "Building nextstat-cli wheel (bindings/ns-cli-py)..."
  (cd bindings/ns-cli-py && \
    CARGO_TARGET_DIR="${repo_root}/tmp/cargo_target_maturin" \
    ../../.venv/bin/maturin build --release -o "${wheelhouse}")
  echo

  echo "Building nextstat wheel (bindings/ns-py)..."
  (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${repo_root}/tmp/cargo_target_maturin" \
    ../../.venv/bin/maturin build --release -o "${wheelhouse}")
  echo

  echo "Installing from local wheelhouse (--no-index, no PyPI)..."
  "${py}" -m pip install --force-reinstall --no-deps --no-index \
    --find-links "${wheelhouse}" \
    "nextstat-cli==${version}" "nextstat==${version}"
  echo

  # After local install, nextstat + _core.so live in venv site-packages.
  # Clear py_path so PYTHONPATH does NOT shadow the installed package with
  # the source-only wrapper at bindings/ns-py/python/.
  py_path=""
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

if [[ "${skip_trex}" != "1" ]]; then
  if [[ ! -f "${trex_manifest}" ]]; then
    echo "FAIL: TREx analysis-spec baseline missing: ${trex_manifest}" >&2
    echo "Record baselines first:  make apex2-baseline-record" >&2
    exit 9
  fi
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

# ── ROOT suite baseline ──────────────────────────────────────────────────
if [[ "${skip_root_suite}" != "1" ]]; then
  if [[ ! -f "${root_manifest}" ]]; then
    echo "FAIL: ROOT suite baseline missing: ${root_manifest}" >&2
    echo "Record baselines first:" >&2
    echo "  PYTHONPATH=${py_path} ${py} tests/apex2_root_suite_report.py \\" >&2
    echo "    --cases ${root_cases} --keep-going --out tmp/apex2_root_suite_report.json" >&2
    echo "  PYTHONPATH=${py_path} ${py} tests/record_baseline.py --only root \\" >&2
    echo "    --root-suite-existing tmp/apex2_root_suite_report.json \\" >&2
    echo "    --root-cases-existing ${root_cases}" >&2
    exit 9
  fi
  echo
  echo "Running ROOT suite baseline compare (allows expected ROOT divergences)..."
  echo "  manifest: ${root_manifest}"
  echo "  cases:    ${root_cases}"
  echo "  report:   ${root_report}"
  echo

  baseline_root_cases="$("${py}" -c "import json; print((json.load(open('${root_manifest}')).get('baselines',{}).get('root_cases',{}) or {}).get('path',''))")"
  baseline_root_suite="$("${py}" -c "import json; print((json.load(open('${root_manifest}')).get('baselines',{}).get('root_suite',{}) or {}).get('path',''))")"
  if [[ -z "${baseline_root_cases}" || -z "${baseline_root_suite}" ]]; then
    echo "FAIL: invalid ROOT manifest (missing root_cases/root_suite): ${root_manifest}" >&2
    exit 9
  fi
  root_cases_run="${baseline_root_cases}"
  if [[ "${baseline_root_cases}" != "${root_cases}" ]]; then
    echo "NOTE: Using baseline ROOT cases file (recorded copy) for determinism." >&2
    echo "  baseline root_cases: ${baseline_root_cases}" >&2
    echo "  gate root_cases:     ${root_cases}" >&2
    echo "To change cases, re-record the ROOT baseline with APEX2_ROOT_CASES=<path>." >&2
  fi

  # 1) Produce current suite report (may contain known/expected failing cases).
  set +e
  PYTHONPATH="${py_path}" "${py}" tests/apex2_root_suite_report.py \
    --cases "${root_cases_run}" \
    --keep-going \
    --out "${root_report}"
  root_rc=$?
  set -e
  if [[ "${root_rc}" -ne 0 ]]; then
    echo "NOTE: ROOT suite runner returned rc=${root_rc} (expected if some cases fail vs ROOT)." >&2
  fi

  # 2) Compare perf + regressions against recorded baseline.
  root_perf_report="tmp/root_suite_perf_compare.json"
  PYTHONPATH="${py_path}" "${py}" tests/compare_apex2_root_suite_to_baseline.py \
    --baseline "${baseline_root_suite}" \
    --current "${root_report}" \
    --out "${root_perf_report}"
  echo
  echo "OK. ROOT suite report: ${root_report}"
  echo "OK. ROOT suite perf:   ${root_perf_report}"
fi

echo
echo "Building release candidate bundle..."
PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_candidate_bundle \
  --release-tag "${release_tag}" \
  --mode prepare \
  --surface-report-json "${surface_report}" \
  --surface-report-md "${surface_report_md}" \
  --release-manifest-json "${release_manifest}" \
  --release-manifest-md "${release_manifest_md}" \
  --baseline-report-json "${report}" \
  --trex-report-json "${trex_report}" \
  --root-report-json "${root_report}" \
  --out-dir "${release_candidate_bundle_dir}"
echo "OK. Release candidate bundle: ${release_candidate_bundle_dir}"
