# Simplified Likelihood Stable-Surface Support Matrix

**Date**: 2026-03-08  
**Status**: Executed support matrix  
**Scope**: simplified-likelihood reinterpretation surface for HEP

## Purpose

This document is the short operational matrix for the promoted
simplified-likelihood subset.

It answers one narrow question:

- what is `stable` now
- what remains `research-grade`
- which CLI, Python, and server surfaces are covered by the March 8, 2026
  promotion

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | public compatibility promise for the named simplified-likelihood subset |
| `research-grade` | versioned and tested, but still evolving without stable-surface promise |

## Stable CLI matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat config schema --name simplified_likelihood_v0` | `stable` | published input contract |
| `nextstat config schema --name simplified_likelihood_audit_v0` | `stable` | published audit contract |
| `nextstat config schema --name simplified_likelihood_derive_v0` | `stable` | promoted narrow exporter request contract: `pyhf` source only, single-POI only, Gaussian-constrained `source_model_constraints`, explicit provenance, and reduced-coordinate output semantics |
| `nextstat config schema --name simplified_likelihood_export_report_v0` | `stable` | promoted narrow exporter report contract for `nextstat simplify workspace --report ...` within the same narrow stable boundary |
| `nextstat config schema --name simplified_likelihood_export_benchmark_snapshot_report_v0` | `research-grade` | published machine-readable persistence contract for committed `nextstat-bench` exporter evidence; this governs benchmark curation, not exporter promotion |
| `nextstat config schema --name simplified_likelihood_exporter_promotion_evidence_bundle_v0` | `research-grade` | published machine-readable exporter bundle contract for future stable-review evidence built from the committed exporter snapshot |
| `nextstat config schema --name simplified_likelihood_exporter_promotion_evidence_check_v0` | `research-grade` | published verification report contract for exporter promotion-readiness evidence bundles |
| `nextstat config schema --name simplified_likelihood_exporter_promotion_bundle_promotion_report_v0` | `research-grade` | published persistence contract for moving an accepted exporter evidence bundle under a stable committed path |
| `nextstat config schema --name simplified_likelihood_exporter_stable_review_assessment_v0` | `research-grade` | published formal stable-review assessment contract derived from the committed accepted exporter bundle; it is review governance, not a stable promotion |
| `nextstat config schema --name simplified_likelihood_exporter_stable_evidence_policy_v0` | `stable` | published machine-readable release-facing admission policy and maintenance cadence for the accepted `9 public / 11 total` exporter stable-evidence floor |
| `nextstat config schema --name simplified_likelihood_exporter_stable_evidence_freshness_report_v0` | `stable` | published machine-readable freshness-breach report for the accepted exporter stable-evidence floor; it enforces the `45-day` freshness window for `stable_evidence_freshness_report.json` |
| `nextstat config schema --name simplified_likelihood_exporter_stable_source_semantics_boundary_v0` | `research-grade` | published machine-readable future-stable source-semantics boundary for the exporter; it narrows any later stable claim to `pyhf` source, single-POI, Gaussian-constrained `source_model_constraints`, and reduced-coordinate output semantics |
| `nextstat config schema --name simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0` | `research-grade` | published blocker-matrix contract for the exporter stable-candidate lifecycle; current accepted artifact now resolves all blockers for the promoted narrow subset |
| `nextstat config schema --name simplified_likelihood_exporter_stable_promotion_decision_v0` | `stable` | explicit stable-promotion decision contract for the narrow exporter subset and release-facing governance consumption |
| `nextstat audit` | `stable` | pyhf + simplified-likelihood audit path; HS3 remains an explicit reject |
| `nextstat fit` | `stable` | direct consume path for simplified-likelihood JSON |
| `nextstat hypotest` | `stable` | asymptotic CLs consume path for simplified-likelihood JSON |
| `nextstat upper-limit` | `stable` | promoted reinterpretation benchmark path |
| `nextstat scan` | `stable` | promoted reduced-model profile-scan path |
| `nextstat simplify workspace` | `stable` | promoted narrow exporter runtime: `pyhf` source only, single-POI only, `constraint_covariance_source="source_model_constraints"` for Gaussian-constrained sources, explicit provenance, no partial-bin selection, reduced-coordinate rather than source-level nuisance semantics, and explicit research-grade fallback outside that boundary |
| `nextstat significance` | `stable` | deterministic asymptotic discovery significance (q0, z0, p0) |
| `nextstat hypotest-toys` | `research-grade` | simplified-likelihood input is compatibility-tested, but toy CLs is outside the promoted stable subset |
| ranking / impact surfaces | `research-grade` | simplified-likelihood input is compatibility-tested, but ranking acts on reduced nuisance coordinates rather than source-level systematics; covariance-form and `derived_from_workspace` source semantics remain outside the promoted stable subset |
| export / publication commands | `research-grade` | advanced export/publication flows beyond the narrow stable boundary remain outside the promoted subset as research-grade fallback, including `aligned_fit_covariance`, broader source nuisance semantics, and source-level identity preservation |

## Stable Python matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat.workspace_audit(...)` | `stable` | published `nextstat_simplified_likelihood_audit_v0` artifact for simplified-likelihood inputs |
| `nextstat.tools.execute_tool("nextstat_workspace_audit", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_fit", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_hypotest", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_upper_limit", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_scan", ...)` | `stable` | local and server tool surface |
| `nextstat.tools.execute_tool("nextstat_discovery_asymptotic", ...)` | `stable` | deterministic asymptotic discovery significance tool |
| `nextstat.tools.execute_tool("nextstat_hypotest_toys", ...)` | `research-grade` | simplified-likelihood input is compatibility-tested, but toy CLs is outside the promoted stable subset |
| `nextstat.tools.execute_tool("nextstat_ranking", ...)` | `research-grade` | simplified-likelihood input is compatibility-tested, but ranking acts on reduced nuisance coordinates rather than source-level systematics |

## Stable server matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat_workspace_audit` | `stable` | remote server-safe audit path for simplified-likelihood JSON |
| `nextstat_fit` | `stable` | remote server-safe fit path for simplified-likelihood JSON |
| `nextstat_hypotest` | `stable` | remote server-safe asymptotic CLs path |
| `nextstat_upper_limit` | `stable` | remote server-safe reinterpretation benchmark path |
| `nextstat_scan` | `stable` | remote server-safe reduced-model profile scan |
| `nextstat_discovery_asymptotic` | `stable` | deterministic asymptotic discovery significance |
| `nextstat_hypotest_toys` | `research-grade` | simplified-likelihood input is compatibility-tested, but toy CLs is outside the promoted stable subset |
| `nextstat_ranking` | `research-grade` | simplified-likelihood input is compatibility-tested, but ranking acts on reduced nuisance coordinates rather than source-level systematics |

## Boundaries that remain research-grade

- derive/export workflows from full workspaces beyond the promoted narrow `nextstat simplify workspace` path
- source-model-constraint export outside Gaussian-constrained nuisance sources
- covariance-only ranking and source-semantics guarantees
- source-level nuisance mapping for `derived_from_workspace` reduced artifacts
- arbitrary non-Gaussian reduced nuisance models
- multi-POI reduced likelihood support
- bench-host external validation beyond the current synthetic Apex2 promotion
  matrix; curated public-style consume fixtures now participate in an Apex2
  runtime matrix, but they are still not promotion-grade benchmark evidence on
  `nextstat-bench`

## Evidence and gate

The promoted stable subset is backed by:

- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)

Companion Apex2 evidence now also exists for curated public-style consume
fixtures through the optional `--include-public-fixtures` matrix in
`tests/apex2_simplified_likelihood_report.py`. That matrix is runtime evidence
for public-style basis/covariance/derived examples, not a replacement for the
paired synthetic speedup artifact or the `nextstat-bench` promotion gate.

Release-facing promotion evidence is now expected to travel as a
three-artifact set: the frozen Apex2 report, `promotion_evidence.json`, and
`promotion_evidence_check.json`.

That release-facing trio is now also persisted under a stable committed path:

- `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
- `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
- `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`

Research-grade exporter benchmark evidence is now also persisted under a stable
committed path:

- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json`
- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json`
- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json`

Exporter-specific future-promotion governance now also exists:

- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Review Checklist](/docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Source-Semantics Boundary](/docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Candidate Blocker Matrix](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Promotion Decision](/docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md)
- [Simplified Likelihood Exporter Release PR Checklist](/docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md)

Accepted exporter promotion-readiness evidence is now also persisted under:

- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`

Committed stable public exporter validation evidence now also exists at:

- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`

That committed exporter evidence is no longer synthetic-only. As of the March 9,
2026 `nextstat-bench` refresh, the current exporter matrix contains one
`public_reinterpretation_style` case plus the synthetic control cases. The
published stable-source-semantics boundary and explicit stable-promotion
decision close the old wording/release blockers, so the accepted blocker matrix
now reports `open_blocker_count = 0`.
The same release-facing evidence now also publishes
`export_public_validation_report.json` as the stable evidence surface for the
curated public exporter matrix.
The same accepted path now also publishes `stable_evidence_freshness_report.json`
as the machine-readable freshness-breach surface for that stable evidence set.

Operational gate:

- script:
  [simplified_likelihood_stable_surface_gate.sh](/scripts/benchmarks/simplified_likelihood_stable_surface_gate.sh)
- workflow:
  [simplified-likelihood-stable-surface.yml](/.github/workflows/simplified-likelihood-stable-surface.yml)
- make target:
  `make simplified-likelihood-stable-surface-gate`

## Bottom line

The stable product promise is intentionally narrow:

- audit, fit, asymptotic CLs, upper-limit, scan, and discovery significance are `stable`
- ranking and toy CLs remain available but stay
  `research-grade` for simplified-likelihood inputs
- export/publication and broader reduced-model semantics outside the published
  narrow source boundary remain outside the current promoted subset
- the committed exporter benchmark snapshot is audit-friendly research-grade
  evidence, not a stable-promotion artifact
