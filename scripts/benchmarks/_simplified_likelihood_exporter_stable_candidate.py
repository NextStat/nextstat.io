from __future__ import annotations

from _simplified_likelihood_exporter_promotion_bundle import (
    DEFAULT_ACCEPTED_BUNDLE_DIR,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MAX_ABS_Q_MU_DIFF,
    REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
    REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
)
from _simplified_likelihood_exporter_stable_source_semantics import (
    BOUNDARY_DOC as SOURCE_SEMANTICS_BOUNDARY_DOC,
    SCHEMA_VERSION as SOURCE_SEMANTICS_BOUNDARY_SCHEMA_VERSION,
)
from _simplified_likelihood_exporter_stable_promotion import (
    DECISION_DOC as STABLE_PROMOTION_DECISION_DOC,
    DECISION_SCHEMA_VERSION as STABLE_PROMOTION_DECISION_SCHEMA_VERSION,
    RELEASE_PR_CHECKLIST_DOC,
)
from _simplified_likelihood_exporter_stable_review import (
    ACCEPTANCE_DOC,
    ARTIFACT_REFERENCE_DOC,
    CHECKLIST_DOC as STABLE_REVIEW_CHECKLIST_DOC,
    PROMOTION_RUNBOOK_DOC,
    RUNTIME_GATE_DOC,
)


BLOCKER_MATRIX_SCHEMA_VERSION = (
    "nextstat_simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0"
)
REVIEW_PACKET_SCHEMA_VERSION = (
    "nextstat_simplified_likelihood_exporter_stable_candidate_review_packet_v0"
)
BLOCKER_MATRIX_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md"
)
REVIEW_PACKET_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md"
)

OPEN_BLOCKERS: list[dict[str, object]] = [
    {
        "blocker_id": "public_exporter_matrix_not_yet_part_of_stable_candidate_evidence",
        "title": "Public reinterpretation-style exporter matrix is not yet part of the stable-candidate evidence set",
        "category": "external_evidence",
        "why_blocking_stable": (
            "The committed exporter matrix on nextstat-bench is still synthetic-only. "
            "A future stable claim needs public reinterpretation-style export cases, "
            "not only synthetic covariance tiers."
        ),
        "exit_criteria": [
            "publish machine-readable exporter case classification for synthetic vs public reinterpretation-style cases",
            "run nextstat-bench exporter matrix with at least three public reinterpretation-style export cases",
            "commit the resulting evidence under the exporter benchmark and accepted bundle paths",
        ],
    },
    {
        "blocker_id": "stable_source_semantics_boundary_not_yet_promoted",
        "title": "Stable exporter source-semantics boundary is not yet promoted",
        "category": "scope_boundary",
        "why_blocking_stable": (
            "The exporter still documents source semantics as research-grade. "
            "A stable claim requires a narrow but explicit source boundary for "
            "what source-model semantics are promised."
        ),
        "exit_criteria": [
            "publish the future stable exporter claim as pyhf-only, single-POI, Gaussian-constrained source scope",
            "keep unsupported source semantics as explicit rejects or research-grade-only paths",
            "align CLI, Python, server, and release wording around the same narrowed boundary",
        ],
    },
    {
        "blocker_id": "stable_candidate_review_packet_not_yet_published",
        "title": "Validator-facing stable-candidate review packet is not yet published",
        "category": "review_package",
        "why_blocking_stable": (
            "The accepted bundle and stable-review assessment exist, but there is "
            "not yet a single review packet that merges accepted evidence, blocker "
            "state, and maintainer-facing recommendation."
        ),
        "exit_criteria": [
            "publish a versioned stable-candidate review packet contract",
            "build the review packet from the accepted exporter bundle plus the blocker matrix",
            "add smoke tests and gate checks for the committed accepted review packet",
        ],
    },
    {
        "blocker_id": "stable_release_promotion_decision_not_yet_taken",
        "title": "Explicit stable promotion decision and release wiring are not yet in place",
        "category": "release_governance",
        "why_blocking_stable": (
            "The exporter surface intentionally stays outside automatic release "
            "promotion. A stable claim requires an explicit promotion decision "
            "and release-facing consumption of the accepted evidence."
        ),
        "exit_criteria": [
            "publish a formal exporter stable-promotion decision note or checklist",
            "wire accepted exporter stable-candidate artifacts into release-facing governance",
            "change support-matrix and release-note wording only after the explicit promotion decision lands",
        ],
    },
]
