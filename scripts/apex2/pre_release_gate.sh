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
#   - APEX2_PERF_POLICY: auto|enforce|advisory (default: auto)
#   - APEX2_CANONICAL_PERF_RUNNER: set to 1 on the canonical perf runner; auto policy
#       enforces perf only there and treats local/dev hardware as advisory.
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
perf_policy="${APEX2_PERF_POLICY:-auto}"
canonical_perf_runner="${APEX2_CANONICAL_PERF_RUNNER:-0}"
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
gate_summary_json="tmp/apex2_pre_release_gate_summary.json"
gate_summary_md="tmp/apex2_pre_release_gate_summary.md"
gate_governance_steps="tmp/apex2_pre_release_gate_governance_steps.tsv"
gate_performance_steps="tmp/apex2_pre_release_gate_performance_steps.tsv"

exit_governance=20
exit_performance=21
exit_infrastructure=22

rm -f "${gate_governance_steps}" "${gate_performance_steps}"
mkdir -p "$(dirname "${gate_governance_steps}")"

record_step() {
  local layer="$1"
  local step_id="$2"
  local status="$3"
  local label="${4:-$2}"
  local file
  if [[ "${layer}" == "governance" ]]; then
    file="${gate_governance_steps}"
  else
    file="${gate_performance_steps}"
  fi
  printf '%s\t%s\t%s\n' "${step_id}" "${status}" "${label}" >> "${file}"
}

resolve_perf_policy() {
  case "${perf_policy}" in
    enforce|advisory)
      printf '%s' "${perf_policy}"
      ;;
    auto)
      if [[ "${canonical_perf_runner}" == "1" ]]; then
        printf 'enforce'
      else
        printf 'advisory'
      fi
      ;;
    *)
      echo "Invalid APEX2_PERF_POLICY=${perf_policy}; expected auto|enforce|advisory." >&2
      exit "${exit_infrastructure}"
      ;;
  esac
}

render_gate_summary() {
  local exit_code="$1"
  local failure_step="${2:-}"
  local message="${3:-}"
  PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.apex2.pre_release_gate_summary \
    --governance-steps "${gate_governance_steps}" \
    --performance-steps "${gate_performance_steps}" \
    --out-json "${gate_summary_json}" \
    --out-md "${gate_summary_md}" \
    --exit-code "${exit_code}" \
    --failure-step "${failure_step}" \
    --message "${message}" \
    --version "${version}" \
    --release-tag "${release_tag}"
}

fail_gate() {
  local layer="$1"
  local step_id="$2"
  local exit_code="$3"
  local message="$4"
  record_step "${layer}" "${step_id}" failed "${step_id}"
  render_gate_summary "${exit_code}" "${step_id}" "${message}"
  echo "${message}" >&2
  echo "Gate summary: ${gate_summary_json}" >&2
  echo "Gate summary (md): ${gate_summary_md}" >&2
  exit "${exit_code}"
}

perf_advisory() {
  local step_id="$1"
  local message="$2"
  record_step "performance" "${step_id}" advisory "${step_id}"
  echo "PERF ADVISORY: ${message}" >&2
}

# Normalize the selected Python to an absolute executable path while preserving
# the venv shim path. `realpath()` would unwrap `.venv/bin/python` to the
# Homebrew-managed base interpreter and break local wheel installs under PEP 668.
if ! py="$("${py}" -c 'import os, sys; print(os.path.abspath(sys.executable))')"; then
  echo "Failed to resolve APEX2_PY to an absolute interpreter path." >&2
  exit "${exit_infrastructure}"
fi

version="$(grep '^version' Cargo.toml | head -1 | sed 's/.*\"\(.*\)\"/\1/')"
release_tag="v${version}"
effective_perf_policy="$(resolve_perf_policy)"

if [[ "${allow_dirty}" != "1" ]]; then
  if command -v git >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
      fail_gate "governance" "clean_worktree" "${exit_infrastructure}" \
        "Git working tree is dirty. Commit/stash changes before release gating, or set APEX2_ALLOW_DIRTY=1 to override."
    fi
  fi
fi

echo "Validating release surface matrix..."
if ! PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_surface_matrix \
  --check \
  --out-json "${surface_report}" \
  --out-md "${surface_report_md}"; then
  fail_gate "governance" "release_surface_matrix" "${exit_governance}" \
    "Release surface matrix validation failed."
fi
record_step "governance" "release_surface_matrix" ok "release_surface_matrix"
echo "OK. Release surface report: ${surface_report}"
echo "OK. Release surface summary: ${surface_report_md}"
echo

echo "Validating HEP surface matrix..."
if ! "${py}" scripts/hep_surface_matrix.py --check; then
  fail_gate "governance" "hep_surface_matrix" "${exit_governance}" \
    "HEP surface matrix validation failed."
fi
record_step "governance" "hep_surface_matrix" ok "hep_surface_matrix"
echo "OK. HEP surface matrix is up to date."
echo

echo "Validating HEP validation bundle (141/141 stable)..."
if ! PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.hep_validation_bundle --check; then
  fail_gate "governance" "hep_validation_bundle" "${exit_governance}" \
    "HEP validation bundle check failed."
fi
record_step "governance" "hep_validation_bundle" ok "hep_validation_bundle"
echo "OK. HEP validation bundle check passed."
echo
echo "Rendering release manifest..."
if ! PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_manifest \
  --release-tag "${release_tag}" \
  --mode prepare \
  --out-json "${release_manifest}" \
  --out-md "${release_manifest_md}"; then
  fail_gate "governance" "release_manifest" "${exit_governance}" \
    "Release manifest render failed."
fi
record_step "governance" "release_manifest" ok "release_manifest"
echo "OK. Release manifest: ${release_manifest}"
echo "OK. Release manifest summary: ${release_manifest_md}"
echo

echo "Running local full-fidelity release simulation..."
if ! PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_full_fidelity_simulation \
  --release-tag "${release_tag}" \
  --mode prepare \
  --out-dir "${release_dry_run_dir}" \
  --out-json "${release_dry_run_report}" \
  --out-md "${release_dry_run_report_md}"; then
  fail_gate "governance" "release_full_fidelity_simulation" "${exit_governance}" \
    "Local full-fidelity release simulation failed."
fi
record_step "governance" "release_full_fidelity_simulation" ok "release_full_fidelity_simulation"
echo "OK. Release dry-run report: ${release_dry_run_report}"
echo "OK. Release dry-run summary: ${release_dry_run_report_md}"
echo

if [[ "${skip_cargo}" != "1" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    fail_gate "governance" "cargo_toolchain" "${exit_infrastructure}" \
      "Missing cargo in PATH; set APEX2_SKIP_CARGO=1 to skip."
  fi

  # ── CI-parity: cargo fmt (catches formatting drift before CI) ──
  echo "Running cargo fmt --all --check..."
  if ! cargo fmt --all --check; then
    fail_gate "governance" "cargo_fmt" "${exit_governance}" \
      "cargo fmt --all --check failed."
  fi
  record_step "governance" "cargo_fmt" ok "cargo_fmt"
  echo

  # ── CI-parity: clippy -D warnings (exact same flags as release-candidate.yml) ──
  echo "Running cargo clippy --workspace --all-targets -- -D warnings..."
  if ! cargo clippy --workspace --all-targets -- -D warnings; then
    fail_gate "governance" "cargo_clippy" "${exit_governance}" \
      "cargo clippy --workspace --all-targets -- -D warnings failed."
  fi
  record_step "governance" "cargo_clippy" ok "cargo_clippy"
  echo

  echo "Running cargo build (${cargo_build_args})..."
  if ! cargo build ${cargo_build_args}; then
    fail_gate "governance" "cargo_build" "${exit_governance}" \
      "cargo build failed."
  fi
  record_step "governance" "cargo_build" ok "cargo_build"
  echo
  echo "Running cargo test (${cargo_test_args})..."
  if ! cargo test ${cargo_test_args}; then
    fail_gate "governance" "cargo_test" "${exit_governance}" \
      "cargo test failed."
  fi
  record_step "governance" "cargo_test" ok "cargo_test"
  echo
fi

if [[ "${skip_maturin}" != "1" ]]; then
  if [[ ! -x "./.venv/bin/maturin" ]]; then
    fail_gate "governance" "maturin_toolchain" "${exit_infrastructure}" \
      "Missing ./.venv/bin/maturin; set APEX2_SKIP_MATURIN=1 to skip."
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
  if ! (cd bindings/ns-cli-py && \
    CARGO_TARGET_DIR="${repo_root}/tmp/cargo_target_maturin" \
    ../../.venv/bin/maturin build --release -o "${wheelhouse}"); then
    fail_gate "governance" "maturin_build_cli" "${exit_governance}" \
      "Building nextstat-cli wheel failed."
  fi
  record_step "governance" "maturin_build_cli" ok "maturin_build_cli"
  echo

  echo "Building nextstat wheel (bindings/ns-py)..."
  if ! (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${repo_root}/tmp/cargo_target_maturin" \
    ../../.venv/bin/maturin build --release --interpreter "${py}" -o "${wheelhouse}"); then
    fail_gate "governance" "maturin_build_nextstat" "${exit_governance}" \
      "Building nextstat wheel failed."
  fi
  record_step "governance" "maturin_build_nextstat" ok "maturin_build_nextstat"
  echo

  echo "Installing from local wheelhouse (--no-index, no PyPI)..."
  if ! "${py}" -m pip install --force-reinstall --no-deps --no-index \
    --find-links "${wheelhouse}" \
    "nextstat-cli==${version}" "nextstat==${version}"; then
    fail_gate "governance" "wheelhouse_install" "${exit_governance}" \
      "Installing nextstat packages from local wheelhouse failed."
  fi
  record_step "governance" "wheelhouse_install" ok "wheelhouse_install"
  echo

  # After local install, nextstat + _core.so live in venv site-packages.
  # Clear py_path so PYTHONPATH does NOT shadow the installed package with
  # the source-only wrapper at bindings/ns-py/python/.
  py_path=""
fi

if [[ "${skip_pytest}" != "1" ]]; then
  if [[ ! -x "./.venv/bin/pytest" ]]; then
    fail_gate "governance" "pytest_toolchain" "${exit_infrastructure}" \
      "Missing ./.venv/bin/pytest; set APEX2_SKIP_PYTEST=1 to skip."
  fi
  echo "Running pytest (-m \"${pytest_marker}\")..."
  pytest_argv=(-q -m "${pytest_marker}")
  if [[ -n "${pytest_extra_args}" ]]; then
    read -r -a extra_argv <<<"${pytest_extra_args}"
    pytest_argv+=("${extra_argv[@]}")
  fi
  read -r -a paths_argv <<<"${pytest_paths}"
  pytest_argv+=("${paths_argv[@]}")
  if ! PYTHONPATH="${py_path}" "${py}" -m pytest "${pytest_argv[@]}"; then
    fail_gate "governance" "pytest" "${exit_governance}" \
      "pytest prerelease suite failed."
  fi
  record_step "governance" "pytest" ok "pytest"
  echo
fi

if [[ ! -f "${manifest}" ]]; then
  if [[ "${effective_perf_policy}" == "enforce" ]]; then
    fail_gate "performance" "baseline_manifest" "${exit_infrastructure}" \
      "Missing baseline manifest: ${manifest}. Record baselines first with make apex2-baseline-record."
  fi
  perf_advisory "baseline_manifest" \
    "Missing baseline manifest: ${manifest}. Perf lane skipped because policy=${effective_perf_policy}."
else
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
    if [[ "${effective_perf_policy}" == "enforce" ]]; then
      fail_gate "performance" "baseline_compare" "${exit_performance}" \
        "Apex2 baseline compare reported performance or baseline regressions."
    fi
    perf_advisory "baseline_compare" \
      "Apex2 baseline compare reported performance or baseline regressions (policy=${effective_perf_policy})."
  else
    record_step "performance" "baseline_compare" ok "baseline_compare"
    echo
    echo "OK. Report: ${report}"
  fi
fi

if [[ "${skip_trex}" != "1" ]]; then
  if [[ ! -f "${trex_manifest}" ]]; then
    fail_gate "governance" "trex_manifest" "${exit_infrastructure}" \
      "TREx analysis-spec baseline missing: ${trex_manifest}. Record baselines first with make apex2-baseline-record."
  fi
  echo
  echo "Running TREx analysis-spec baseline compare..."
  echo "  manifest: ${trex_manifest}"
  echo "  report:   ${trex_report}"
  echo
  if ! PYTHONPATH="${py_path}" "${py}" tests/compare_trex_analysis_spec_with_latest_baseline.py \
    --manifest "${trex_manifest}" \
    --out "${trex_report}" \
    ${trex_compare_args}; then
    fail_gate "governance" "trex_analysis_spec_compare" "${exit_governance}" \
      "TREx analysis-spec baseline compare failed."
  fi
  record_step "governance" "trex_analysis_spec_compare" ok "trex_analysis_spec_compare"
  echo
  echo "OK. TREx report: ${trex_report}"
fi

# ── ROOT suite baseline ──────────────────────────────────────────────────
if [[ "${skip_root_suite}" != "1" ]]; then
  if [[ ! -f "${root_manifest}" ]]; then
    if [[ "${effective_perf_policy}" == "enforce" ]]; then
      fail_gate "performance" "root_manifest" "${exit_infrastructure}" \
        "ROOT suite baseline missing: ${root_manifest}. Record the ROOT baseline first."
    fi
    perf_advisory "root_manifest" \
      "ROOT suite baseline missing: ${root_manifest}. Perf lane skipped because policy=${effective_perf_policy}."
  else
    echo
    echo "Running ROOT suite baseline compare (allows expected ROOT divergences)..."
    echo "  manifest: ${root_manifest}"
    echo "  cases:    ${root_cases}"
    echo "  report:   ${root_report}"
    echo

    baseline_root_cases="$("${py}" -c "import json; print((json.load(open('${root_manifest}')).get('baselines',{}).get('root_cases',{}) or {}).get('path',''))")"
    baseline_root_suite="$("${py}" -c "import json; print((json.load(open('${root_manifest}')).get('baselines',{}).get('root_suite',{}) or {}).get('path',''))")"
    if [[ -z "${baseline_root_cases}" || -z "${baseline_root_suite}" ]]; then
      if [[ "${effective_perf_policy}" == "enforce" ]]; then
        fail_gate "performance" "root_manifest" "${exit_infrastructure}" \
          "Invalid ROOT manifest (missing root_cases/root_suite): ${root_manifest}"
      fi
      perf_advisory "root_manifest" \
        "Invalid ROOT manifest (missing root_cases/root_suite): ${root_manifest} (policy=${effective_perf_policy})."
    else
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
      if ! PYTHONPATH="${py_path}" "${py}" tests/compare_apex2_root_suite_to_baseline.py \
        --baseline "${baseline_root_suite}" \
        --current "${root_report}" \
        --out "${root_perf_report}"; then
        if [[ "${effective_perf_policy}" == "enforce" ]]; then
          fail_gate "performance" "root_suite_compare" "${exit_performance}" \
            "ROOT suite compare reported performance or regression failures."
        fi
        perf_advisory "root_suite_compare" \
          "ROOT suite compare reported performance or regression failures (policy=${effective_perf_policy})."
      else
        record_step "performance" "root_suite_compare" ok "root_suite_compare"
        echo
        echo "OK. ROOT suite report: ${root_report}"
        echo "OK. ROOT suite perf:   ${root_perf_report}"
      fi
    fi
  fi
fi

echo
echo "Building release candidate bundle..."
if ! PYTHONPATH="${repo_root}${py_path:+:${py_path}}" "${py}" -m scripts.release_candidate_bundle \
  --release-tag "${release_tag}" \
  --mode prepare \
  --surface-report-json "${surface_report}" \
  --surface-report-md "${surface_report_md}" \
  --release-manifest-json "${release_manifest}" \
  --release-manifest-md "${release_manifest_md}" \
  --baseline-report-json "${report}" \
  --trex-report-json "${trex_report}" \
  --root-report-json "${root_report}" \
  --out-dir "${release_candidate_bundle_dir}"; then
  fail_gate "governance" "release_candidate_bundle" "${exit_governance}" \
    "Release candidate bundle build failed."
fi
record_step "governance" "release_candidate_bundle" ok "release_candidate_bundle"
echo "OK. Release candidate bundle: ${release_candidate_bundle_dir}"
if [[ "${effective_perf_policy}" == "advisory" ]] && grep -q $'\tadvisory\t' "${gate_performance_steps}" 2>/dev/null; then
  render_gate_summary 0 "" \
    "Apex2 pre-release gate passed with performance advisories on non-canonical hardware."
else
  render_gate_summary 0 "" "Apex2 pre-release gate passed."
fi
echo "OK. Gate summary: ${gate_summary_json}"
echo "OK. Gate summary (md): ${gate_summary_md}"
