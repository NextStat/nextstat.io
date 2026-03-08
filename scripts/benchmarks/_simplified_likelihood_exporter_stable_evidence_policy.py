from __future__ import annotations

from pathlib import Path
from typing import Any

from _simplified_likelihood_exporter_promotion_bundle import (
    DEFAULT_ACCEPTED_BUNDLE_DIR,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
    REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT,
    REQUIRED_MIN_TOTAL_EXPORT_MATRIX_CASE_COUNT,
    relative_or_absolute,
    now_utc,
)
from _simplified_likelihood_exporter_stable_promotion import (
    DECISION_DOC,
    RELEASE_PR_CHECKLIST_DOC,
    RELEASE_WORKFLOW_PATH,
    STANDALONE_WORKFLOW_PATH,
)
from _simplified_likelihood_exporter_stable_source_semantics import (
    ACCEPTANCE_DOC,
    ARTIFACT_REFERENCE_DOC,
    BOUNDARY_DOC,
    PROMOTION_RUNBOOK_DOC,
    RELEASE_NOTES_DOC,
    RUNTIME_GATE_DOC,
    SUPPORT_MATRIX_DOC,
)


SCHEMA_VERSION = "nextstat_simplified_likelihood_exporter_stable_evidence_policy_v0"
POLICY_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md"
)
POLICY_SCHEMA = (
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json"
)
POLICY_EXAMPLE = (
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json"
)
DEFAULT_OUT_PATH = DEFAULT_ACCEPTED_BUNDLE_DIR / "stable_evidence_policy.json"

REFRESH_CADENCE = "on_every_exporter_release_pr_or_public_case_admission"
REQUIRED_PUBLIC_VALIDATION_MIN_SPEEDUP = 0.75
REQUIRED_CASE_KIND = "public_reinterpretation_style"
REQUIRED_SOURCE_FORMATS = ["pyhf"]
REQUIRED_CONSTRAINT_COVARIANCE_SOURCE = "source_model_constraints"
REQUIRED_SOURCE_CONSTRAINT_FAMILIES = ["gaussian"]
REQUIRED_SOURCE_WORKSPACE_MODIFIER_TYPES = ["histosys", "lumi", "normfactor", "normsys"]

REQUIRED_REFRESH_TRIGGERS = [
    "exporter_release_pr_touches_runtime_or_governance_surface",
    "accepted_public_case_catalog_changes",
    "committed_nextstat_bench_exporter_snapshot_refresh",
    "promoted_runtime_boundary_or_support_wording_changes",
]

REQUIRED_REFRESH_SCOPE = [
    "committed_current_exporter_snapshot",
    "committed_public_validation_report",
    "accepted_exporter_bundle",
    "release_workflow_uploads",
    "standalone_exporter_workflow_uploads",
    "release_facing_docs_and_checklists",
]

ALIGNMENT_DOCUMENTS = [
    POLICY_DOC,
    ACCEPTANCE_DOC,
    RUNTIME_GATE_DOC,
    PROMOTION_RUNBOOK_DOC,
    BOUNDARY_DOC,
    DECISION_DOC,
    RELEASE_PR_CHECKLIST_DOC,
    SUPPORT_MATRIX_DOC,
    RELEASE_NOTES_DOC,
    ARTIFACT_REFERENCE_DOC,
]

RELEASE_CONSUMERS = {
    "workflow_paths": [RELEASE_WORKFLOW_PATH, STANDALONE_WORKFLOW_PATH],
    "doc_paths": ALIGNMENT_DOCUMENTS,
    "required_release_artifacts": [
        "export_public_validation_report.json",
        "stable_evidence_policy.json",
        "stable_source_semantics_boundary.json",
        "stable_promotion_decision.json",
    ],
}


def build_policy(
    *,
    benchmark_artifact_path: Path,
    benchmark: dict[str, Any],
    public_validation_report_path: Path,
    public_validation_report: dict[str, Any],
    stable_promotion_decision_path: Path,
    stable_promotion_decision: dict[str, Any],
    deterministic: bool,
) -> dict[str, Any]:
    decision_status = stable_promotion_decision.get("status")
    decision_support_class = stable_promotion_decision.get("support_class")
    if decision_status != "accepted" or decision_support_class != "stable":
        raise ValueError(
            "stable evidence policy requires an accepted stable_promotion_decision.json artifact"
        )

    benchmark_host = (
        benchmark.get("environment", {}).get("hostname")
        if isinstance(benchmark.get("environment"), dict)
        else None
    )
    if benchmark_host != REQUIRED_BENCHMARK_HOST:
        raise ValueError(
            f"benchmark artifact must come from {REQUIRED_BENCHMARK_HOST}, got {benchmark_host!r}"
        )

    benchmark_summary = benchmark.get("summary")
    if not isinstance(benchmark_summary, dict):
        raise ValueError("benchmark artifact missing summary object")
    export_matrix = benchmark.get("export_matrix")
    if not isinstance(export_matrix, dict):
        raise ValueError("benchmark artifact missing export_matrix object")
    export_cases = export_matrix.get("cases")
    if not isinstance(export_cases, list):
        raise ValueError("benchmark artifact missing export_matrix cases")
    public_summary = public_validation_report.get("summary")
    if not isinstance(public_summary, dict):
        raise ValueError("public validation report missing summary object")
    if public_validation_report.get("status") != "ok":
        raise ValueError("public validation report must be green before publishing stable policy")

    public_case_names = public_summary.get("public_case_names")
    if not isinstance(public_case_names, list):
        raise ValueError("public validation report missing public_case_names")

    synthetic_speeds = [
        float(case["bench"]["speedup"]["net_end_to_end_upper_limit"])
        for case in export_cases
        if isinstance(case, dict)
        and case.get("case_kind") == "synthetic"
        and isinstance(case.get("bench"), dict)
        and isinstance(case["bench"].get("speedup"), dict)
        and "net_end_to_end_upper_limit" in case["bench"]["speedup"]
    ]
    if not synthetic_speeds:
        raise ValueError("benchmark artifact missing synthetic exporter speedup cases")

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "simplified_likelihood_exporter_stable_evidence_policy_v0",
        "status": "accepted",
        "support_class": "stable",
        "generated_at_utc": now_utc(deterministic),
        "benchmark_host": REQUIRED_BENCHMARK_HOST,
        "stable_evidence_floor": {
            "min_total_export_matrix_case_count": REQUIRED_MIN_TOTAL_EXPORT_MATRIX_CASE_COUNT,
            "min_public_reinterpretation_style_case_count": REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT,
            "min_synthetic_control_net_end_to_end_upper_limit_speedup": REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
            "min_public_validation_net_end_to_end_upper_limit_speedup": REQUIRED_PUBLIC_VALIDATION_MIN_SPEEDUP,
        },
        "admission_policy": {
            "required_case_kind": REQUIRED_CASE_KIND,
            "required_benchmark_host": REQUIRED_BENCHMARK_HOST,
            "required_source_workspace_formats": list(REQUIRED_SOURCE_FORMATS),
            "required_poi_scope": "single_poi",
            "required_constraint_covariance_source": REQUIRED_CONSTRAINT_COVARIANCE_SOURCE,
            "required_source_constraint_families": list(REQUIRED_SOURCE_CONSTRAINT_FAMILIES),
            "required_source_workspace_modifier_types": list(REQUIRED_SOURCE_WORKSPACE_MODIFIER_TYPES),
            "require_cases_inside_promoted_stable_runtime_boundary": True,
            "allow_source_level_nuisance_identity_claims": False,
            "allow_silent_stable_boundary_expansion": False,
        },
        "maintenance_cadence": {
            "refresh_cadence": REFRESH_CADENCE,
            "required_refresh_triggers": list(REQUIRED_REFRESH_TRIGGERS),
            "required_refresh_scope": list(REQUIRED_REFRESH_SCOPE),
            "no_silent_boundary_expansion": True,
        },
        "release_consumers": RELEASE_CONSUMERS,
        "source_artifacts": {
            "benchmark_artifact_path": relative_or_absolute(benchmark_artifact_path),
            "public_validation_report_path": relative_or_absolute(public_validation_report_path),
            "stable_promotion_decision_path": relative_or_absolute(stable_promotion_decision_path),
        },
        "current_evidence_summary": {
            "export_matrix_case_count": int(benchmark_summary["export_matrix_case_count"]),
            "public_case_count": int(public_summary["public_case_count"]),
            "public_case_names": list(public_case_names),
            "synthetic_min_net_end_to_end_upper_limit_speedup": min(synthetic_speeds),
            "public_min_net_end_to_end_upper_limit_speedup": float(
                public_summary["min_net_end_to_end_upper_limit_speedup"]
            ),
            "max_abs_q_mu_diff": float(public_summary["max_abs_q_mu_diff"]),
            "max_upper_limit_ratio_deviation": float(
                public_summary["max_upper_limit_ratio_deviation"]
            ),
            "cases_outside_promoted_stable_runtime_boundary": int(
                public_summary["cases_outside_promoted_stable_runtime_boundary"]
            ),
            "observed_constraint_covariance_sources": list(
                public_summary["observed_constraint_covariance_sources"]
            ),
        },
    }
