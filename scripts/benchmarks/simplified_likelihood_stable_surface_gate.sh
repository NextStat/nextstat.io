#!/usr/bin/env bash
set -euo pipefail

# Stable-surface verification gate for the promoted simplified-likelihood subset.
#
# Scope:
#   - nextstat_simplified_likelihood_v0 input contract
#   - nextstat_simplified_likelihood_audit_v0 audit contract
#   - fit / upper-limit / audit consume path for simplified-likelihood JSON
#   - Apex2 simplified-likelihood report contract and CI performance gate
#
# Optional env vars:
#   - SL_STABLE_PY: Python executable (default: ./.venv/bin/python, else python3, else python)
#   - SL_STABLE_MATURIN: maturin executable (default: ./.venv/bin/maturin, else maturin)
#   - SL_STABLE_PYTHONPATH: pythonpath for local bindings (default: bindings/ns-py/python)
#   - SL_STABLE_CARGO_TARGET_DIR: isolated cargo target dir
#   - SL_STABLE_OUT_DIR: directory for emitted Apex2 report
#   - SL_STABLE_SKIP_MATURIN: set to 1 to skip `maturin develop --release`
#   - SL_STABLE_SKIP_DOC_CHECKS: set to 1 to skip release-governance doc checks
#   - SL_STABLE_SUITE: Apex2 suite preset (default: ci)
#   - SL_STABLE_FIT_REPEAT: fit timing repeats for the Apex2 runner (default: 1)
#   - SL_STABLE_UPPER_LIMIT_REPEAT: upper-limit timing repeats for the Apex2 runner (default: 1)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

skip_maturin="${SL_STABLE_SKIP_MATURIN:-0}"
skip_doc_checks="${SL_STABLE_SKIP_DOC_CHECKS:-0}"
py_path="${SL_STABLE_PYTHONPATH:-bindings/ns-py/python}"
suite="${SL_STABLE_SUITE:-ci}"
fit_repeat="${SL_STABLE_FIT_REPEAT:-1}"
upper_limit_repeat="${SL_STABLE_UPPER_LIMIT_REPEAT:-1}"
cargo_target_dir="${SL_STABLE_CARGO_TARGET_DIR:-${repo_root}/tmp/cargo_target_sl_stable_surface}"
out_dir="${SL_STABLE_OUT_DIR:-${repo_root}/tmp/simplified-likelihood-stable-surface}"
report_json="${out_dir}/apex2_simplified_likelihood_report.json"
bundle_dir="${out_dir}/promotion_bundle"

if [[ -n "${SL_STABLE_PY:-}" ]]; then
  py="${SL_STABLE_PY}"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  py="${repo_root}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="python3"
else
  py="python"
fi

if [[ -n "${SL_STABLE_MATURIN:-}" ]]; then
  maturin_cmd=("${SL_STABLE_MATURIN}")
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

mkdir -p "${cargo_target_dir}" "${out_dir}"

if [[ "${skip_doc_checks}" != "1" ]]; then
  required_files=(
    "docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08.md"
    "docs/references/simplified-likelihood-artifacts.md"
    "docs/schemas/hep/simplified_likelihood_v0.schema.json"
    "docs/schemas/hep/simplified_likelihood_audit_v0.schema.json"
    "docs/schemas/apex2/simplified_likelihood_report_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_check_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.schema.json"
    "docs/schemas/benchmarks/snapshot_index_v1.schema.json"
    "docs/specs/hep/simplified_likelihood_v0.example.json"
    "docs/specs/hep/simplified_likelihood_audit_v0.example.json"
    "docs/specs/apex2_simplified_likelihood_report_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_promotion_evidence_check_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.example.json"
    ".github/workflows/simplified-likelihood-stable-surface.yml"
    ".gitignore"
    "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json"
    "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json"
    "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json"
    "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json"
    "scripts/benchmarks/apex2_simplified_likelihood_remote.sh"
    "scripts/benchmarks/_simplified_likelihood_promotion_bundle.py"
    "scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py"
    "scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py"
    "scripts/benchmarks/promote_simplified_likelihood_promotion_bundle.py"
    "scripts/benchmarks/write_snapshot_index.py"
  )
  for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || {
      echo "Missing simplified-likelihood stable-surface evidence file: ${file}" >&2
      exit 8
    }
  done

  grep -qF "make simplified-likelihood-stable-surface-gate" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified-likelihood-stable-surface.yml" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified-likelihood-support-matrix-2026-03-08" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified-likelihood-release-notes-2026-03-08" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified-likelihood-promotion-runbook-2026-03-08.md" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_promotion_evidence_bundle_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_promotion_evidence_check_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_promotion_bundle_promotion_report_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified-likelihood-support-matrix-2026-03-08.md" docs/README.md
  grep -qF "simplified-likelihood-release-notes-2026-03-08.md" docs/README.md
  grep -qF "simplified-likelihood-release-pr-checklist-2026-03-08.md" docs/README.md
  grep -qF "simplified_likelihood_promotion_evidence_bundle_v0.example.json" docs/README.md
  grep -qF "simplified_likelihood_promotion_evidence_check_v0.example.json" docs/README.md
  grep -qF "simplified_likelihood_promotion_bundle_promotion_report_v0.example.json" docs/README.md
  grep -qF "simplified-likelihood-support-matrix-2026-03-08" docs/benchmarks.md
  grep -qF "simplified-likelihood-release-notes-2026-03-08" docs/benchmarks.md
  grep -qF "simplified-likelihood-promotion-runbook-2026-03-08" docs/benchmarks.md
  grep -qF "simplified_likelihood_promotion_evidence_bundle_v0.example.json" docs/benchmarks.md
  grep -qF "simplified_likelihood_promotion_evidence_check_v0.example.json" docs/benchmarks.md
  grep -qF "simplified_likelihood_promotion_bundle_promotion_report_v0.example.json" docs/benchmarks.md
fi

if [[ "${skip_maturin}" != "1" ]]; then
  echo "Building local wheelhouse for simplified-likelihood stable-surface gate..."
  sl_wheels="${repo_root}/tmp/sl_stable_wheels"
  rm -rf "${sl_wheels}"
  mkdir -p "${sl_wheels}"
  (cd bindings/ns-cli-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release -o "${sl_wheels}")
  (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release -o "${sl_wheels}")
  "${py}" -m pip install --force-reinstall --no-deps \
    "${sl_wheels}"/nextstat_cli-*.whl \
    "${sl_wheels}"/nextstat-*.whl
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

echo "Running simplified-likelihood translation gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-translate simplified --lib -- --test-threads=1
echo

echo "Running simplified-likelihood audit dispatcher gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-translate audit_workspace_json --lib -- --test-threads=1
echo

echo "Running simplified-likelihood CLI gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_simplified_likelihood
echo

echo "Running simplified-likelihood CLI schema gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_config_schema
echo

echo "Running simplified-likelihood HS3 reject gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-cli --test cli_hs3_scope audit_rejects_hs3_input -- --nocapture
echo

echo "Running simplified-likelihood server gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-server tool_execute_workspace_audit_accepts_simplified_likelihood_workspace -- --nocapture
echo

echo "Running simplified-likelihood Python gate..."
NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="${py_path}" "${py}" -m pytest -q \
  tests/python/test_simplified_likelihood_schema_smoke.py \
  tests/python/test_apex2_simplified_likelihood_report_smoke.py \
  tests/python/test_simplified_likelihood_promotion_evidence_bundle_smoke.py \
  tests/python/test_simplified_likelihood_promotion_evidence_check_smoke.py \
  tests/python/test_simplified_likelihood_promotion_bundle_promotion_smoke.py \
  tests/python/test_bindings_api.py \
  -k "simplified_likelihood or workspace_audit_accepts_simplified_likelihood_workspace"
echo

echo "Running simplified-likelihood Apex2 CI gate..."
# Keep the operational gate artifact non-deterministic so the performance payload
# (`summary.bench` / per-case `bench`) survives canonicalization and can enforce
# the CI speedup threshold directly from the archived JSON.
NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="${py_path}" "${py}" tests/apex2_simplified_likelihood_report.py \
  --suite "${suite}" \
  --fit-repeat "${fit_repeat}" \
  --upper-limit-repeat "${upper_limit_repeat}" \
  --include-public-fixtures \
  --out "${report_json}"
echo

echo "Validating simplified-likelihood Apex2 artifact..."
"${py}" - "${report_json}" "${repo_root}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])

try:
    import jsonschema  # type: ignore
except Exception as exc:  # pragma: no cover - operational dependency
    raise SystemExit(f"jsonschema is required for the stable-surface gate: {exc}") from exc

schema_path = repo_root / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
schema = json.loads(schema_path.read_text(encoding="utf-8"))
jsonschema.validate(instance=report, schema=schema)

assert report["schema_version"] == "nextstat_apex2_simplified_likelihood_report_v0"
summary = report["summary"]
assert summary["status"] == "ok"
assert summary["all_schema_valid"] is True
assert summary["all_fidelity_gates_pass"] is True
assert summary["all_performance_gates_pass"] is True
assert float(summary["bench"]["min_speedup_end_to_end_upper_limit"]) >= 3.0

print(
    "validated",
    f"cases={summary['case_count']}",
    f"min_e2e_ul_speedup={summary['bench']['min_speedup_end_to_end_upper_limit']:.2f}x",
)
PY
echo

echo "Building simplified-likelihood promotion bundle..."
"${py}" scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py \
  --benchmark-artifact "${report_json}" \
  --bundle-dir "${bundle_dir}"
echo

echo "Verifying simplified-likelihood promotion bundle..."
"${py}" scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py \
  --bundle-dir "${bundle_dir}"
echo

echo "Checking formatting..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo fmt --all --check
echo

echo "OK. Simplified-likelihood stable-surface gate passed."
echo "Artifact: ${report_json}"
