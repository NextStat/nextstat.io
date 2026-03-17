#!/usr/bin/env bash
set -euo pipefail

# Gate for the simplified-likelihood exporter acceptance contour.
#
# This validates the committed nextstat-bench exporter snapshot plus targeted
# tests for the derive/export path. The narrow exporter subset is now stable,
# while wider fallback modes remain research-grade.
#
# Optional env vars:
#   - SL_EXPORTER_PY: Python executable (default: ./.venv/bin/python, else python3, else python)
#   - SL_EXPORTER_MATURIN: maturin executable (default: ./.venv/bin/maturin, else maturin)
#   - SL_EXPORTER_PYTHONPATH: pythonpath for local bindings (default: bindings/ns-py/python)
#   - SL_EXPORTER_CARGO_TARGET_DIR: isolated cargo target dir
#   - SL_EXPORTER_FRESHNESS_REFERENCE_DATE: override YYYY-MM-DD used by the live freshness check
#   - SL_EXPORTER_SKIP_MATURIN: set to 1 to skip local wheel build/install
#   - SL_EXPORTER_SKIP_DOC_CHECKS: set to 1 to skip doc/link checks

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

skip_doc_checks="${SL_EXPORTER_SKIP_DOC_CHECKS:-0}"
skip_maturin="${SL_EXPORTER_SKIP_MATURIN:-0}"
py_path="${SL_EXPORTER_PYTHONPATH:-bindings/ns-py/python}"
cargo_target_dir="${SL_EXPORTER_CARGO_TARGET_DIR:-${repo_root}/tmp/cargo_target_sl_exporter_surface}"
freshness_reference_date="${SL_EXPORTER_FRESHNESS_REFERENCE_DATE:-$(date -u +%F)}"
current_dir="${repo_root}/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current"
artifact_json="${current_dir}/apex2_simplified_likelihood_report.json"
snapshot_report_json="${current_dir}/export_benchmark_snapshot_report.json"
public_validation_report_json="${current_dir}/export_public_validation_report.json"
snapshot_index_json="${current_dir}/snapshot_index.json"
accepted_bundle_dir="${repo_root}/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted"
exporter_bundle_tmp_dir="${repo_root}/tmp/sl_exporter_bundle_gate_${$}"

if [[ -n "${SL_EXPORTER_PY:-}" ]]; then
  py="${SL_EXPORTER_PY}"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
  py="${repo_root}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  py="python3"
else
  py="python"
fi

if [[ -n "${SL_EXPORTER_MATURIN:-}" ]]; then
  maturin_cmd=("${SL_EXPORTER_MATURIN}")
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

if [[ "${skip_maturin}" != "1" ]]; then
  echo "Building local wheelhouse for simplified-likelihood exporter gate..."
  sl_exporter_wheels="${repo_root}/tmp/sl_exporter_wheels"
  rm -rf "${sl_exporter_wheels}"
  mkdir -p "${sl_exporter_wheels}"
  (cd bindings/ns-cli-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release --interpreter "${py}" -o "${sl_exporter_wheels}")
  (cd bindings/ns-py && \
    CARGO_TARGET_DIR="${cargo_target_dir}" run_maturin build --release --interpreter "${py}" -o "${sl_exporter_wheels}")
  "${py}" -m pip install --force-reinstall --no-deps \
    "${sl_exporter_wheels}"/nextstat_cli-*.whl \
    "${sl_exporter_wheels}"/nextstat-*.whl
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

if [[ "${skip_doc_checks}" != "1" ]]; then
  required_files=(
    "docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md"
    "docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md"
    "docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md"
    "docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md"
    "docs/references/simplified-likelihood-artifacts.md"
    "docs/schemas/apex2/simplified_likelihood_report_v0.schema.json"
    "docs/schemas/hep/simplified_likelihood_derive_v0.schema.json"
    "docs/schemas/hep/simplified_likelihood_export_report_v0.schema.json"
    "docs/schemas/apex2/simplified_likelihood_export_public_case_catalog_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_export_public_validation_report_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json"
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
    "docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_export_public_validation_report_v0.example.json"
    "docs/specs/apex2_simplified_likelihood_report_v0.example.json"
    "docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json"
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json"
    "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json"
    "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json"
    "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json"
    "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json"
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json"
    ".github/workflows/simplified-likelihood-exporter-surface.yml"
    "scripts/benchmarks/_simplified_likelihood_exporter_promotion_bundle.py"
    "scripts/benchmarks/_simplified_likelihood_exporter_stable_review.py"
    "scripts/benchmarks/_simplified_likelihood_exporter_stable_candidate.py"
    "scripts/benchmarks/_simplified_likelihood_exporter_stable_evidence_policy.py"
    "scripts/benchmarks/_simplified_likelihood_exporter_stable_evidence_freshness.py"
    "scripts/benchmarks/_simplified_likelihood_exporter_stable_source_semantics.py"
    "scripts/benchmarks/_simplified_likelihood_exporter_stable_promotion.py"
    "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py"
    "scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py"
    "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py"
    "scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py"
    "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py"
    "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py"
    "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py"
    "scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py"
    "scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py"
    "scripts/benchmarks/promote_simplified_likelihood_exporter_promotion_bundle.py"
    "scripts/benchmarks/persist_simplified_likelihood_export_benchmark.py"
    "scripts/benchmarks/_simplified_likelihood_export_public_validation.py"
    "scripts/benchmarks/build_simplified_likelihood_export_public_validation_report.py"
    "scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh"
    "tests/python/_simplified_likelihood_export_public_case_catalog.py"
    "tests/python/test_simplified_likelihood_export_benchmark_snapshot_smoke.py"
    "tests/python/test_simplified_likelihood_export_public_validation_report_smoke.py"
    "tests/python/test_simplified_likelihood_export_public_case_catalog_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_gate_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_promotion_bundle_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_source_semantics_boundary_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_review_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_candidate_review_packet_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_evidence_policy_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_evidence_freshness_smoke.py"
    "tests/python/test_simplified_likelihood_exporter_stable_promotion_decision_smoke.py"
    "tests/python/test_release_workflow_simplified_likelihood_exporter_smoke.py"
  )
  for file in "${required_files[@]}"; do
    [[ -f "${file}" ]] || {
      echo "Missing simplified-likelihood exporter gate file: ${file}" >&2
      exit 8
    }
  done

  grep -qF "simplified-likelihood-exporter-acceptance-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-runtime-gate" docs/README.md
  grep -qF "simplified-likelihood-exporter-promotion-runbook-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-public-validation-surface-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-review-checklist-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09" docs/README.md
  grep -qF "simplified-likelihood-exporter-release-pr-checklist-2026-03-09" docs/README.md
  grep -qF "simplified_likelihood_exporter_stable_evidence_policy_v0.example.json" docs/README.md
  grep -qF "simplified-likelihood-exporter-acceptance-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-runtime-gate" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-promotion-runbook-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-public-validation-surface-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-review-checklist-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-release-pr-checklist-2026-03-09" docs/benchmarks.md
  grep -qF "simplified_likelihood_exporter_stable_evidence_policy_v0.example.json" docs/benchmarks.md
  grep -qF "apex2_simplified_likelihood_export_public_case_catalog_v0.example.json" docs/README.md
  grep -qF "simplified_likelihood_export_public_validation_report_v0.example.json" docs/README.md
  grep -qF "apex2_simplified_likelihood_export_public_case_catalog_v0.example.json" docs/benchmarks.md
  grep -qF "simplified_likelihood_export_public_validation_report_v0.example.json" docs/benchmarks.md
  grep -qF "simplified-likelihood-exporter-acceptance-2026-03-09" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified-likelihood-exporter-runtime-gate" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_promotion_evidence_bundle_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_review_assessment_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_evidence_policy_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_evidence_freshness_report_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_source_semantics_boundary_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_candidate_review_packet_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_exporter_stable_promotion_decision_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_export_public_case_catalog_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF "simplified_likelihood_export_public_validation_report_v0" docs/references/simplified-likelihood-artifacts.md
  grep -qF 'export matrix public reinterpretation-style case count: `9`' docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md
  grep -qF "export_public_validation_report.json" docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md
  grep -qF "stable_evidence_policy.json" docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md
  grep -qF "stable_evidence_freshness_report.json" docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md
  grep -qF "min_net_end_to_end_upper_limit_speedup >= 1.25x" docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md
  grep -qF "stable_evidence_policy.json" docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md
  grep -qF "stable_evidence_freshness_report.json" docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md
  grep -qF "min_net_end_to_end_upper_limit_speedup >= 1.25x" docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md
fi

echo "Running exporter translation gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -p ns-translate simplified --lib -- --test-threads=1
echo

echo "Running exporter CLI gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_simplified_likelihood
echo

echo "Running exporter CLI schema gate..."
CARGO_TARGET_DIR="${cargo_target_dir}" cargo test -q -p ns-cli --test cli_config_schema
echo

echo "Running exporter Python gate..."
NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="${py_path}" "${py}" -m pytest -q \
  tests/python/test_simplified_likelihood_schema_smoke.py \
  tests/python/test_apex2_simplified_likelihood_report_smoke.py \
  tests/python/test_simplified_likelihood_export_benchmark_snapshot_smoke.py \
  tests/python/test_simplified_likelihood_export_public_validation_report_smoke.py \
  tests/python/test_simplified_likelihood_export_public_case_catalog_smoke.py \
  tests/python/test_simplified_likelihood_exporter_gate_smoke.py \
  tests/python/test_simplified_likelihood_exporter_promotion_bundle_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_source_semantics_boundary_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_review_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_candidate_review_packet_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_evidence_policy_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_evidence_freshness_smoke.py \
  tests/python/test_simplified_likelihood_exporter_stable_promotion_decision_smoke.py \
  tests/python/test_release_workflow_simplified_likelihood_exporter_smoke.py
echo

echo "Validating committed exporter benchmark artifact..."
"${py}" - "${artifact_json}" "${snapshot_report_json}" "${public_validation_report_json}" "${snapshot_index_json}" "${repo_root}" <<'PY'
import json
import sys
from pathlib import Path

artifact_path = Path(sys.argv[1])
snapshot_report_path = Path(sys.argv[2])
public_validation_report_path = Path(sys.argv[3])
snapshot_index_path = Path(sys.argv[4])
repo_root = Path(sys.argv[5])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for the exporter gate: {exc}") from exc

artifact_schema_path = repo_root / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
snapshot_report_schema_path = repo_root / "docs" / "schemas" / "benchmarks" / "simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json"
public_validation_report_schema_path = repo_root / "docs" / "schemas" / "benchmarks" / "simplified_likelihood_export_public_validation_report_v0.schema.json"
snapshot_index_schema_path = repo_root / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json"

artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
snapshot_report = json.loads(snapshot_report_path.read_text(encoding="utf-8"))
public_validation_report = json.loads(public_validation_report_path.read_text(encoding="utf-8"))
snapshot_index = json.loads(snapshot_index_path.read_text(encoding="utf-8"))

artifact_schema = json.loads(artifact_schema_path.read_text(encoding="utf-8"))
snapshot_report_schema = json.loads(snapshot_report_schema_path.read_text(encoding="utf-8"))
public_validation_report_schema = json.loads(public_validation_report_schema_path.read_text(encoding="utf-8"))
snapshot_index_schema = json.loads(snapshot_index_schema_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=artifact, schema=artifact_schema)
jsonschema.validate(instance=snapshot_report, schema=snapshot_report_schema)
jsonschema.validate(instance=public_validation_report, schema=public_validation_report_schema)
jsonschema.validate(instance=snapshot_index, schema=snapshot_index_schema)

assert artifact["schema_version"] == "nextstat_apex2_simplified_likelihood_report_v0"
assert artifact["environment"]["hostname"] == "nextstat-bench"
summary = artifact["summary"]
assert summary["status"] == "ok"
assert summary["all_schema_valid"] is True
assert summary["all_fidelity_gates_pass"] is True
assert summary["export_matrix_included"] is True
assert summary["export_matrix_status"] == "ok"
assert int(summary["export_matrix_case_count"]) >= 11
assert int(summary["export_matrix_public_reinterpretation_style_case_count"]) >= 9
assert set(summary["export_matrix_case_kinds"]) == {"public_reinterpretation_style", "synthetic"}

export_summary = artifact["export_matrix"]["summary"]
assert export_summary["status"] == "ok"
assert export_summary["all_schema_valid"] is True
assert export_summary["all_fidelity_gates_pass"] is True
assert int(export_summary["public_reinterpretation_style_case_count"]) >= 9
assert int(export_summary["synthetic_case_count"]) >= 1
assert set(export_summary["case_kinds"]) == {"public_reinterpretation_style", "synthetic"}
assert float(export_summary["max_abs_q_mu_diff"]) <= 0.1
assert float(export_summary["max_upper_limit_ratio_deviation"]) <= 0.05
synthetic_min_speedup = min(
    float(case["bench"]["speedup"]["net_end_to_end_upper_limit"])
    for case in artifact["export_matrix"]["cases"]
    if case["case_kind"] == "synthetic"
)
assert synthetic_min_speedup >= 1.25

assert snapshot_report["schema_version"] == "nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0"
assert snapshot_report["status"] == "persisted"
assert snapshot_report["persisted"] is True
assert snapshot_report["source_summary"]["benchmark_host"] == "nextstat-bench"
assert snapshot_report["source_summary"]["export_matrix_status"] == "ok"
assert int(snapshot_report["source_summary"]["export_matrix_case_count"]) >= 11
assert int(snapshot_report["source_summary"]["export_matrix_public_reinterpretation_style_case_count"]) >= 9
assert set(snapshot_report["source_summary"]["export_matrix_case_kinds"]) == {"public_reinterpretation_style", "synthetic"}
assert float(snapshot_report["source_summary"]["export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup"]) >= 1.25

assert public_validation_report["schema_version"] == "nextstat_simplified_likelihood_export_public_validation_report_v0"
assert public_validation_report["status"] == "ok"
assert public_validation_report["summary"]["benchmark_host"] == "nextstat-bench"
assert int(public_validation_report["summary"]["public_case_count"]) >= 9
assert public_validation_report["summary"]["all_schema_valid"] is True
assert public_validation_report["summary"]["all_fidelity_gates_pass"] is True
assert public_validation_report["summary"]["all_performance_gates_pass"] is True
assert float(public_validation_report["summary"]["max_abs_q_mu_diff"]) <= 0.1
assert float(public_validation_report["summary"]["max_upper_limit_ratio_deviation"]) <= 0.05
assert float(public_validation_report["summary"]["min_net_end_to_end_upper_limit_speedup"]) >= 0.75
assert int(public_validation_report["summary"]["cases_outside_promoted_stable_runtime_boundary"]) == 0

assert snapshot_index["suite"] == "simplified_likelihood_export_benchmark_snapshot"
artifact_paths = {entry["path"] for entry in snapshot_index["artifacts"]}
assert "apex2_simplified_likelihood_report.json" in artifact_paths
assert "export_benchmark_snapshot_report.json" in artifact_paths
assert "export_public_validation_report.json" in artifact_paths

print(
    "validated",
    f"host={snapshot_report['source_summary']['benchmark_host']}",
    f"export_cases={snapshot_report['source_summary']['export_matrix_case_count']}",
    f"synthetic_min_net_e2e_ul={snapshot_report['source_summary']['export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup']:.3f}x",
    f"public_min_net_e2e_ul={public_validation_report['summary']['min_net_end_to_end_upper_limit_speedup']:.3f}x",
)
PY
echo

echo "Regenerating and checking committed exporter public validation report..."
"${py}" scripts/benchmarks/build_simplified_likelihood_export_public_validation_report.py \
  --benchmark-artifact "${artifact_json}" \
  --out "${exporter_bundle_tmp_dir}/export_public_validation_report.json" \
  --deterministic
"${py}" - "${public_validation_report_json}" "${exporter_bundle_tmp_dir}/export_public_validation_report.json" <<'PY'
import json
import sys
from pathlib import Path

committed_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])

committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["status"] == "ok"
print(
    "validated",
    f"status={committed['status']}",
    f"public_cases={committed['summary']['public_case_count']}",
    "outside_promoted_runtime_boundary="
    f"{committed['summary']['cases_outside_promoted_stable_runtime_boundary']}",
)
PY
echo

echo "Building and verifying exporter promotion-readiness bundle..."
"${py}" scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py \
  --benchmark-artifact "${artifact_json}" \
  --snapshot-report "${snapshot_report_json}" \
  --snapshot-index "${snapshot_index_json}" \
  --bundle-dir "${exporter_bundle_tmp_dir}" \
  --deterministic
"${py}" scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py \
  --bundle-dir "${exporter_bundle_tmp_dir}" \
  --out "${exporter_bundle_tmp_dir}/promotion_evidence_check.json" \
  --require-promotion-ready \
  --deterministic
echo

echo "Verifying committed accepted exporter promotion bundle..."
"${py}" scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py \
  --bundle-dir "${accepted_bundle_dir}" \
  --out "${exporter_bundle_tmp_dir}/accepted_promotion_evidence_check.json" \
  --require-promotion-ready \
  --deterministic
echo

echo "Regenerating and checking committed exporter stable-review assessment..."
"${py}" scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py \
  --bundle-dir "${accepted_bundle_dir}" \
  --out "${exporter_bundle_tmp_dir}/stable_review_assessment.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_review_assessment.json" "${exporter_bundle_tmp_dir}/stable_review_assessment.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-review validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_review_assessment_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["summary"]["status"] == "review_ready"
assert committed["automatic_stable_promotion"] is False
assert committed["stable_review"]["ready"] is True
assert committed["stable_review"]["status"] == "review_ready"
print(
    "validated",
    f"status={committed['summary']['status']}",
    f"host={committed['summary']['benchmark_host']}",
)
PY
echo

echo "Regenerating and checking committed exporter stable source-semantics boundary..."
"${py}" scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py \
  --out "${exporter_bundle_tmp_dir}/stable_source_semantics_boundary.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_source_semantics_boundary.json" "${exporter_bundle_tmp_dir}/stable_source_semantics_boundary.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-source-semantics validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
generated_cmp = dict(generated)
generated_cmp["bundle_dir"] = committed.get("bundle_dir")

assert committed == generated_cmp
assert committed["status"] == "published"
assert committed["support_class"] == "research-grade"
assert committed["target_support_class"] == "stable"
assert committed["automatic_stable_promotion"] is False
assert committed["future_stable_boundary"]["source_workspace_formats"] == ["pyhf"]
assert committed["future_stable_boundary"]["poi_scope"] == "single_poi"
assert committed["future_stable_boundary"]["supported_constraint_covariance_source"] == "source_model_constraints"
assert committed["future_stable_boundary"]["supported_source_constraint_families"] == ["gaussian"]
assert committed["future_stable_boundary"]["source_level_nuisance_identity_preserved"] is False
print(
    "validated",
    f"status={committed['summary']['status']}",
    f"bundle_dir={committed['bundle_dir']}",
)
PY
echo

echo "Regenerating and checking committed exporter stable-candidate blocker matrix..."
"${py}" scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py \
  --bundle-dir "${accepted_bundle_dir}" \
  --out "${exporter_bundle_tmp_dir}/stable_candidate_blocker_matrix.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_candidate_blocker_matrix.json" "${exporter_bundle_tmp_dir}/stable_candidate_blocker_matrix.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-candidate validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["summary"]["status"] == "ready"
assert committed["support_class"] == "stable"
assert committed["automatic_stable_promotion"] is False
assert committed["stable_candidate"]["ready"] is True
assert committed["stable_candidate"]["open_blocker_count"] == 0
public_blocker = next(
    blocker
    for blocker in committed["blockers"]
    if blocker["blocker_id"] == "public_exporter_matrix_not_yet_part_of_stable_candidate_evidence"
)
assert public_blocker["status"] == "resolved"
assert public_blocker["blocking"] is False
source_semantics_blocker = next(
    blocker
    for blocker in committed["blockers"]
    if blocker["blocker_id"] == "stable_source_semantics_boundary_not_yet_promoted"
)
assert source_semantics_blocker["status"] == "resolved"
assert source_semantics_blocker["blocking"] is False
review_packet_blocker = next(
    blocker
    for blocker in committed["blockers"]
    if blocker["blocker_id"] == "stable_candidate_review_packet_not_yet_published"
)
assert review_packet_blocker["status"] == "resolved"
assert review_packet_blocker["blocking"] is False
decision_blocker = next(
    blocker
    for blocker in committed["blockers"]
    if blocker["blocker_id"] == "stable_release_promotion_decision_not_yet_taken"
)
assert decision_blocker["status"] == "resolved"
assert decision_blocker["blocking"] is False
print(
    "validated",
    f"status={committed['summary']['status']}",
    f"host={committed['summary']['benchmark_host']}",
    f"open_blockers={committed['stable_candidate']['open_blocker_count']}",
)
PY
echo

echo "Regenerating and checking committed exporter stable-candidate review packet..."
"${py}" scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py \
  --bundle-dir "${accepted_bundle_dir}" \
  --out "${exporter_bundle_tmp_dir}/stable_candidate_review_packet.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_candidate_review_packet.json" "${exporter_bundle_tmp_dir}/stable_candidate_review_packet.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-candidate review-packet validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["summary"]["status"] == "ready"
assert committed["support_class"] == "stable"
assert committed["automatic_stable_promotion"] is False
assert committed["review_packet"]["ready"] is True
assert committed["review_packet"]["recommendation_status"] == "stable_promoted"
assert committed["review_packet"]["open_blocker_count"] == 0
print(
    "validated",
    f"status={committed['summary']['status']}",
    f"host={committed['summary']['benchmark_host']}",
    f"open_blockers={committed['review_packet']['open_blocker_count']}",
)
PY
echo

echo "Regenerating and checking committed exporter stable-promotion decision..."
"${py}" scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py \
  --bundle-dir "${accepted_bundle_dir}" \
  --out "${exporter_bundle_tmp_dir}/stable_promotion_decision.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_promotion_decision.json" "${exporter_bundle_tmp_dir}/stable_promotion_decision.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-promotion decision validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["status"] == "accepted"
assert committed["support_class"] == "stable"
assert committed["automatic_stable_promotion"] is False
assert committed["summary"]["release_assets_wired"] is True
assert committed["summary"]["stable_scope_promoted"] is True
assert committed["summary"]["open_blocker_count_observed"] == 0
print(
    "validated",
    f"status={committed['status']}",
    f"host={committed['summary']['benchmark_host']}",
)
PY
echo

echo "Regenerating and checking committed exporter stable-evidence policy..."
"${py}" scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py \
  --benchmark-artifact "${artifact_json}" \
  --public-validation-report "${public_validation_report_json}" \
  --stable-promotion-decision "${accepted_bundle_dir}/stable_promotion_decision.json" \
  --out "${exporter_bundle_tmp_dir}/stable_evidence_policy.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_evidence_policy.json" "${exporter_bundle_tmp_dir}/stable_evidence_policy.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-evidence policy validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["status"] == "accepted"
assert committed["support_class"] == "stable"
assert committed["stable_evidence_floor"]["min_total_export_matrix_case_count"] == 11
assert committed["stable_evidence_floor"]["min_public_reinterpretation_style_case_count"] == 9
assert committed["maintenance_cadence"]["refresh_cadence"] == "on_every_exporter_release_pr_or_public_case_admission"
assert committed["current_evidence_summary"]["export_matrix_case_count"] >= 11
assert committed["current_evidence_summary"]["public_case_count"] >= 9
print(
    "validated",
    f"status={committed['status']}",
    f"public_cases={committed['current_evidence_summary']['public_case_count']}",
    f"export_cases={committed['current_evidence_summary']['export_matrix_case_count']}",
)
PY
echo

echo "Regenerating and checking committed exporter stable-evidence freshness report..."
"${py}" scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py \
  --snapshot-report "${snapshot_report_json}" \
  --public-validation-report "${public_validation_report_json}" \
  --stable-evidence-policy "${accepted_bundle_dir}/stable_evidence_policy.json" \
  --stable-promotion-decision "${accepted_bundle_dir}/stable_promotion_decision.json" \
  --out "${exporter_bundle_tmp_dir}/stable_evidence_freshness_report.json" \
  --deterministic
"${py}" - "${repo_root}" "${accepted_bundle_dir}/stable_evidence_freshness_report.json" "${exporter_bundle_tmp_dir}/stable_evidence_freshness_report.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
committed_path = Path(sys.argv[2])
generated_path = Path(sys.argv[3])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for exporter stable-evidence freshness validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
committed = json.loads(committed_path.read_text(encoding="utf-8"))
generated = json.loads(generated_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=committed, schema=schema)
jsonschema.validate(instance=generated, schema=schema)
committed.pop("generated_at_utc", None)
generated.pop("generated_at_utc", None)
assert committed == generated
assert committed["status"] == "fresh"
assert committed["support_class"] == "stable"
assert committed["freshness_policy"]["max_snapshot_age_days"] == 45
assert committed["freshness_observation"]["snapshot_age_days"] == 0
print(
    "validated",
    f"status={committed['status']}",
    f"snapshot_age_days={committed['freshness_observation']['snapshot_age_days']}",
)
PY
echo

echo "Running live exporter stable-evidence freshness check..."
"${py}" scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py \
  --snapshot-report "${snapshot_report_json}" \
  --public-validation-report "${public_validation_report_json}" \
  --stable-evidence-policy "${accepted_bundle_dir}/stable_evidence_policy.json" \
  --stable-promotion-decision "${accepted_bundle_dir}/stable_promotion_decision.json" \
  --reference-date "${freshness_reference_date}" \
  --out "${exporter_bundle_tmp_dir}/stable_evidence_freshness_report_live.json"
"${py}" - "${repo_root}" "${exporter_bundle_tmp_dir}/stable_evidence_freshness_report_live.json" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
report_path = Path(sys.argv[2])

try:
    import jsonschema  # type: ignore
except Exception as exc:
    raise SystemExit(f"jsonschema is required for live exporter stable-evidence freshness validation: {exc}") from exc

schema_path = (
    repo_root
    / "docs"
    / "schemas"
    / "benchmarks"
    / "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json"
)
schema = json.loads(schema_path.read_text(encoding="utf-8"))
report = json.loads(report_path.read_text(encoding="utf-8"))

jsonschema.validate(instance=report, schema=schema)
assert report["status"] == "fresh"
assert report["validity"]["passed"] is True
assert report["freshness_observation"]["snapshot_age_days"] <= report["freshness_policy"]["max_snapshot_age_days"]
print(
    "validated",
    f"status={report['status']}",
    f"reference_date={report['reference_date']}",
    f"snapshot_age_days={report['freshness_observation']['snapshot_age_days']}",
)
PY
echo

echo "OK. Simplified-likelihood exporter surface gate passed."
echo "Artifact: ${artifact_json}"
