from __future__ import annotations

from _simplified_likelihood_promotion_bundle import (
    bundle_path_for,
    derive_stamp_from_path,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
    REPO_ROOT,
)


BUNDLE_SCHEMA_VERSION = "nextstat_simplified_likelihood_exporter_promotion_evidence_bundle_v0"
CHECK_SCHEMA_VERSION = "nextstat_simplified_likelihood_exporter_promotion_evidence_check_v0"
PROMOTION_REPORT_SCHEMA_VERSION = (
    "nextstat_simplified_likelihood_exporter_promotion_bundle_promotion_report_v0"
)

REQUIRED_BENCHMARK_HOST = "nextstat-bench"
REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP = 1.25
REQUIRED_MAX_ABS_Q_MU_DIFF = 0.1
REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION = 0.05
REQUIRED_MIN_TOTAL_EXPORT_MATRIX_CASE_COUNT = 8
REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT = 6

PROMOTION_ARTIFACT_SUITE = "simplified_likelihood_exporter_promotion_bundle"
DEFAULT_ACCEPTED_BUNDLE_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "simplified_likelihood_exporter_promotion_bundles"
    / REQUIRED_BENCHMARK_HOST
    / "accepted"
)
DEFAULT_ACCEPTED_HISTORY_DIR = DEFAULT_ACCEPTED_BUNDLE_DIR.parent / "history"

ACCEPTED_COMMAND_NAMES = ["simplify workspace"]
ACCEPTED_SCHEMA_NAMES = [
    "simplified_likelihood_derive_v0",
    "simplified_likelihood_export_report_v0",
    "simplified_likelihood_export_benchmark_snapshot_report_v0",
    "simplified_likelihood_exporter_stable_evidence_policy_v0",
    "simplified_likelihood_exporter_stable_evidence_freshness_report_v0",
]
EXPLICIT_BOUNDARIES = [
    "research-grade support class",
    "future stable exporter claim remains pyhf-only on the source side",
    "future stable exporter claim remains single-POI only",
    "future stable exporter claim only promises constraint_covariance_source=source_model_constraints",
    "future stable exporter claim only covers Gaussian-constrained source nuisances on that path",
    "partial channel bin selection rejected",
    "derived artifacts remain reduced-coordinate models rather than source-level nuisance replicas",
    "ranking and impact views on derived artifacts are not a source-level breakdown",
    "no automatic release gating",
    "future stable review only",
]

STATIC_ARTIFACTS: list[dict[str, str]] = [
    {
        "role": "exporter_acceptance_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md",
    },
    {
        "role": "exporter_runtime_gate_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md",
    },
    {
        "role": "exporter_promotion_runbook_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
    },
    {
        "role": "exporter_stable_evidence_policy_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md",
    },
    {
        "role": "exporter_stable_evidence_freshness_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md",
    },
    {
        "role": "exporter_stable_source_semantics_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
    },
    {
        "role": "benchmark_snapshot_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md",
    },
    {
        "role": "support_matrix_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md",
    },
    {
        "role": "release_notes_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md",
    },
    {
        "role": "artifact_reference_doc",
        "kind": "doc",
        "source_path": "docs/references/simplified-likelihood-artifacts.md",
    },
    {
        "role": "derive_schema",
        "kind": "schema",
        "source_path": "docs/schemas/hep/simplified_likelihood_derive_v0.schema.json",
    },
    {
        "role": "export_report_schema",
        "kind": "schema",
        "source_path": "docs/schemas/hep/simplified_likelihood_export_report_v0.schema.json",
    },
    {
        "role": "export_snapshot_report_schema",
        "kind": "schema",
        "source_path": "docs/schemas/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json",
    },
    {
        "role": "export_public_case_catalog_schema",
        "kind": "schema",
        "source_path": "docs/schemas/apex2/simplified_likelihood_export_public_case_catalog_v0.schema.json",
    },
    {
        "role": "exporter_stable_source_semantics_schema",
        "kind": "schema",
        "source_path": "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json",
    },
    {
        "role": "exporter_stable_evidence_policy_schema",
        "kind": "schema",
        "source_path": "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json",
    },
    {
        "role": "exporter_stable_evidence_freshness_schema",
        "kind": "schema",
        "source_path": "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json",
    },
    {
        "role": "derive_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_derive_v0.example.json",
    },
    {
        "role": "derived_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_derived_from_workspace_v0.example.json",
    },
    {
        "role": "export_report_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_export_report_v0.example.json",
    },
    {
        "role": "export_snapshot_report_example",
        "kind": "example",
        "source_path": "docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json",
    },
    {
        "role": "export_public_case_catalog_example",
        "kind": "example",
        "source_path": "docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json",
    },
    {
        "role": "exporter_stable_source_semantics_example",
        "kind": "example",
        "source_path": "docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json",
    },
    {
        "role": "exporter_stable_evidence_policy_example",
        "kind": "example",
        "source_path": "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json",
    },
    {
        "role": "exporter_stable_evidence_freshness_example",
        "kind": "example",
        "source_path": "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json",
    },
]

REQUIRED_ROLES = [
    "benchmark_artifact",
    "current_snapshot_report",
    "current_snapshot_index",
    *[artifact["role"] for artifact in STATIC_ARTIFACTS],
]
