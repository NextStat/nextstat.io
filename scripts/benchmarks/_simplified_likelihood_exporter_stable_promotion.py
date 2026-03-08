from __future__ import annotations

from _simplified_likelihood_exporter_stable_source_semantics import (
    ACCEPTANCE_DOC,
    ARTIFACT_REFERENCE_DOC,
    BOUNDARY_DOC,
    CANONICAL_BOUNDARY_CLAIMS,
    CLI_DOC,
    PYTHON_API_DOC,
    RELEASE_NOTES_DOC,
    RESEARCH_GRADE_ONLY_OR_REJECTED,
    REQUIRED_OUTPUT_SOURCE_FORMAT,
    RUNTIME_GATE_DOC,
    RUST_API_DOC,
    SERVER_API_DOC,
    SUPPORTED_CONSTRAINT_COVARIANCE_SOURCE,
    SUPPORTED_POI_SCOPE,
    SUPPORTED_SOURCE_CONSTRAINT_FAMILIES,
    SUPPORTED_SOURCE_FORMATS,
    SUPPORT_MATRIX_DOC,
)


DECISION_SCHEMA_VERSION = "nextstat_simplified_likelihood_exporter_stable_promotion_decision_v0"
DECISION_SCHEMA = (
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
)
DECISION_EXAMPLE = (
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json"
)
DECISION_DOC = "docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md"
RELEASE_PR_CHECKLIST_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md"
)
POLICY_DOC = "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md"
RELEASE_WORKFLOW_PATH = ".github/workflows/release.yml"
STANDALONE_WORKFLOW_PATH = ".github/workflows/simplified-likelihood-exporter-surface.yml"
DEFAULT_DECISION_OUT_PATH = (
    "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/"
    "stable_promotion_decision.json"
)
RELEASE_JOB_NAME = "Simplified Likelihood Exporter Surface Gate"
RELEASE_ARTIFACT_NAME = "simplified-likelihood-exporter-surface-report"

PROMOTED_COMMAND_NAMES = ["simplify workspace"]
PROMOTED_SCHEMA_NAMES = [
    "simplified_likelihood_derive_v0",
    "simplified_likelihood_export_report_v0",
    "simplified_likelihood_exporter_stable_evidence_policy_v0",
    "simplified_likelihood_exporter_stable_evidence_freshness_report_v0",
    "simplified_likelihood_exporter_stable_promotion_decision_v0",
]
RELEASE_ARTIFACT_FILENAMES = [
    "apex2_simplified_likelihood_report.json",
    "export_benchmark_snapshot_report.json",
    "snapshot_index.json",
    "promotion_evidence.json",
    "promotion_evidence_check.json",
    "promotion_bundle_promotion_report.json",
    "stable_review_assessment.json",
    "stable_evidence_policy.json",
    "stable_evidence_freshness_report.json",
    "stable_source_semantics_boundary.json",
    "stable_candidate_blocker_matrix.json",
    "stable_candidate_review_packet.json",
    "stable_promotion_decision.json",
]

STABLE_SCOPE = {
    "source_workspace_formats": list(SUPPORTED_SOURCE_FORMATS),
    "poi_scope": SUPPORTED_POI_SCOPE,
    "required_output_source_format": REQUIRED_OUTPUT_SOURCE_FORMAT,
    "supported_constraint_covariance_source": SUPPORTED_CONSTRAINT_COVARIANCE_SOURCE,
    "supported_source_constraint_families": list(SUPPORTED_SOURCE_CONSTRAINT_FAMILIES),
    "source_level_nuisance_identity_preserved": False,
    "ranking_source_level_breakdown_supported": False,
    "canonical_claims": list(CANONICAL_BOUNDARY_CLAIMS),
}

ALIGNMENT_DOCUMENTS = {
    "decision_doc": DECISION_DOC,
    "policy_doc": POLICY_DOC,
    "release_pr_checklist_doc": RELEASE_PR_CHECKLIST_DOC,
    "acceptance_doc": ACCEPTANCE_DOC,
    "runtime_gate_doc": RUNTIME_GATE_DOC,
    "support_matrix_doc": SUPPORT_MATRIX_DOC,
    "release_notes_doc": RELEASE_NOTES_DOC,
    "artifact_reference_doc": ARTIFACT_REFERENCE_DOC,
    "boundary_doc": BOUNDARY_DOC,
    "cli_doc": CLI_DOC,
    "python_api_doc": PYTHON_API_DOC,
    "server_api_doc": SERVER_API_DOC,
    "rust_api_doc": RUST_API_DOC,
    "release_workflow_path": RELEASE_WORKFLOW_PATH,
    "standalone_workflow_path": STANDALONE_WORKFLOW_PATH,
    "schema_path": DECISION_SCHEMA,
    "example_path": DECISION_EXAMPLE,
}

RESEARCH_GRADE_EXCEPTIONS = list(RESEARCH_GRADE_ONLY_OR_REJECTED)
