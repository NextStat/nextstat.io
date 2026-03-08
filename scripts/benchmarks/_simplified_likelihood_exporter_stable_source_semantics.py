from __future__ import annotations

from _simplified_likelihood_exporter_promotion_bundle import DEFAULT_ACCEPTED_BUNDLE_DIR


SCHEMA_VERSION = (
    "nextstat_simplified_likelihood_exporter_stable_source_semantics_boundary_v0"
)
BOUNDARY_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md"
)
BOUNDARY_SCHEMA = (
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
)
BOUNDARY_EXAMPLE = (
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json"
)
CLI_DOC = "docs/references/cli.md"
PYTHON_API_DOC = "docs/references/python-api.md"
SERVER_API_DOC = "docs/references/server-api.md"
RUST_API_DOC = "docs/references/rust-api.md"
SUPPORT_MATRIX_DOC = "docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md"
RELEASE_NOTES_DOC = "docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md"
ARTIFACT_REFERENCE_DOC = "docs/references/simplified-likelihood-artifacts.md"
ACCEPTANCE_DOC = "docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md"
RUNTIME_GATE_DOC = "docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md"
PROMOTION_RUNBOOK_DOC = "docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md"
STABLE_REVIEW_CHECKLIST_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md"
)

SUPPORTED_SOURCE_FORMATS = ["pyhf"]
SUPPORTED_POI_SCOPE = "single_poi"
SUPPORTED_CONSTRAINT_COVARIANCE_SOURCE = "source_model_constraints"
SUPPORTED_SOURCE_CONSTRAINT_FAMILIES = ["gaussian"]
REQUIRED_OUTPUT_SOURCE_FORMAT = "derived_from_workspace"

CANONICAL_BOUNDARY_CLAIMS = [
    "future stable exporter claim remains pyhf-only on the source side",
    "future stable exporter claim remains single-POI only",
    "future stable exporter claim only promises constraint_covariance_source=source_model_constraints",
    "future stable exporter claim only covers Gaussian-constrained source nuisances on that path",
    "derived artifacts remain reduced-coordinate models rather than source-level nuisance replicas",
    "ranking and impact views on derived artifacts are not a source-level breakdown",
]

RESEARCH_GRADE_ONLY_OR_REJECTED = [
    {
        "id": "aligned_fit_covariance",
        "status": "research-grade-only",
        "detail": (
            "constraint_covariance_source=aligned_fit_covariance remains available as a "
            "compatibility fallback, but it is outside the future stable exporter claim"
        ),
    },
    {
        "id": "non_gaussian_or_unconstrained_source_nuisances",
        "status": "explicit_reject_or_research-grade-only",
        "detail": (
            "the future stable exporter claim does not promise non-Gaussian or "
            "unconstrained source nuisances on the source_model_constraints path"
        ),
    },
    {
        "id": "partial_per_channel_bin_selection",
        "status": "explicit_reject",
        "detail": (
            "partial per-channel bin selections stay rejected explicitly instead of "
            "silently dropping source-model semantics"
        ),
    },
    {
        "id": "multi_poi_export",
        "status": "explicit_reject_or_out_of_scope",
        "detail": "multi-POI reduced export remains outside the future stable exporter claim",
    },
    {
        "id": "source_level_nuisance_identity_preservation",
        "status": "not_promised",
        "detail": (
            "derived reduced artifacts do not preserve original nuisance identities "
            "through reduction"
        ),
    },
    {
        "id": "source_level_ranking_breakdown",
        "status": "not_promised",
        "detail": (
            "ranking and impact outputs on derived reduced artifacts remain reduced-"
            "coordinate diagnostics rather than source-level systematic breakdowns"
        ),
    },
]

ALIGNMENT_DOCUMENTS = [
    BOUNDARY_DOC,
    ACCEPTANCE_DOC,
    RUNTIME_GATE_DOC,
    PROMOTION_RUNBOOK_DOC,
    STABLE_REVIEW_CHECKLIST_DOC,
    SUPPORT_MATRIX_DOC,
    RELEASE_NOTES_DOC,
    ARTIFACT_REFERENCE_DOC,
    CLI_DOC,
    PYTHON_API_DOC,
    SERVER_API_DOC,
    RUST_API_DOC,
]

DEFAULT_OUT_PATH = DEFAULT_ACCEPTED_BUNDLE_DIR / "stable_source_semantics_boundary.json"
