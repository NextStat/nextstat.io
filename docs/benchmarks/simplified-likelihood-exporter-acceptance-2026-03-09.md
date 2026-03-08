# Simplified Likelihood Exporter Acceptance

**Date**: 2026-03-09  
**Status**: Narrow stable-surface acceptance and explicit boundary policy  
**Scope**: `nextstat simplify workspace` and the `full -> derived -> reinterpret` exporter evidence path

## Purpose

This document defines the acceptance criteria for the narrow simplified-likelihood
exporter surface that is now promoted to `stable`.

It also records the explicit limits that keep the wider exporter surface out of
that stable claim.

## Support class

The narrow exporter subset is now `stable`.

Broader exporter behavior outside that subset remains `research-grade`.

## In-scope surface

This acceptance policy applies to:

- `nextstat simplify workspace`
- `nextstat_simplified_likelihood_derive_v0`
- `nextstat_simplified_likelihood_export_report_v0`
- `nextstat_simplified_likelihood_exporter_stable_evidence_policy_v0`
- `nextstat_simplified_likelihood_exporter_stable_evidence_freshness_report_v0`
- `nextstat_simplified_likelihood_exporter_stable_promotion_decision_v0`
- `nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0`
- `nextstat_simplified_likelihood_export_public_validation_report_v0`
- the committed `nextstat-bench` exporter benchmark snapshot under
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/`

## Out of scope

This policy does not imply:

- support for non-Gaussian source constraint families on the
  `source_model_constraints` path
- support for `aligned_fit_covariance` inside any future stable claim
- stable source-level nuisance semantics for reduced covariance directions
- multi-POI reduced-likelihood export support
- automatic widening beyond the explicit release-gated stable boundary

## Acceptance criteria

All items below must hold for exporter acceptance.

### 1. Contract

- `nextstat_simplified_likelihood_derive_v0` is versioned and published
- `nextstat_simplified_likelihood_export_report_v0` is versioned and published
- `nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0` is
  versioned and published
- CLI behavior remains explicit for supported and rejected exporter inputs
- `reduction.constraint_covariance_source` remains explicit in the contract and
  docs

### 2. Verification matrix

- Rust simplified-likelihood translation tests are green
- CLI simplified-likelihood tests are green
- CLI schema publication tests are green
- Python schema/report/exporter snapshot smoke tests are green
- no contract drift exists between docs, examples, and committed exporter
  artifacts

### 3. Fidelity gates

For the committed `nextstat-bench` `full -> derived -> reinterpret` matrix:

- `max_abs_q_mu_diff <= 0.1`
- `upper_limit_ratio` remains within `[0.95, 1.05]`
- exported matrix `all_fidelity_gates_pass = true`
- exported matrix `all_schema_valid = true`

### 4. Performance gate

For the committed `nextstat-bench` exporter matrix:

- synthetic control floor
  `synthetic_min_net_end_to_end_upper_limit_speedup >= 1.25x`
- curated public validation evidence remains green under its separate
  stable-evidence threshold:
  `public_validation.min_net_end_to_end_upper_limit_speedup >= 0.75x`

This threshold is intentionally lower than the consume-path promotion gate.
The exporter path pays derivation overhead, so the acceptance question is net
benefit after export on the synthetic control lane, not raw reinterpretation
speed alone. The public evidence lane is tracked separately so real-world cases
do not silently rewrite the stable-promotion floor.

### 5. Provenance and persistence

- the current exporter artifact is committed under
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/`
- the matching machine-readable persistence report exists
- the matching machine-readable public validation report exists:
  `export_public_validation_report.json`
- the matching machine-readable stable evidence policy exists:
  `stable_evidence_policy.json`
- the matching machine-readable stable evidence freshness report exists:
  `stable_evidence_freshness_report.json`
- the matching `snapshot_index.json` exists
- the committed benchmark host is `nextstat-bench`
- the committed exporter matrix covers at least five benchmark cases
- the committed exporter matrix includes at least three
  `public_reinterpretation_style` export cases in addition to the synthetic
  control cases

## Operational gate

The exporter acceptance contour is enforced through:

- local/CI gate:
  `make simplified-likelihood-exporter-surface-gate`
- gate script:
  `scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh`
- dedicated workflow:
  `.github/workflows/simplified-likelihood-exporter-surface.yml`

## Decision rule

The narrow exporter path is accepted as `stable` only if:

1. all contract artifacts exist and are documented
2. all targeted verification layers are green
3. all exporter fidelity gates pass
4. the exporter net end-to-end speedup gate passes on `nextstat-bench`
5. the committed exporter snapshot/report/index trio is present and valid
6. the committed accepted exporter bundle can still produce a `review_ready`
   formal stable-review assessment as historical evidence
7. the published stable source-semantics boundary stays explicit:
   `pyhf`-only source, single-POI only, Gaussian-constrained
   `source_model_constraints`, and reduced-coordinate rather than source-level
   nuisance semantics
8. the committed accepted exporter bundle includes a versioned
   `stable_promotion_decision.json` artifact, a versioned
   `stable_evidence_policy.json` artifact, a versioned
   `stable_evidence_freshness_report.json` artifact, and release-facing
   workflow consumption for the accepted `8 public / 10 total` floor
9. the accepted stable evidence stays inside the published freshness window:
   `max_snapshot_age_days = 45`, and any freshness breach is treated as a
   governance failure instead of a silent stale-evidence fallback

If any one of these fails, the exporter remains `research-grade` without
promotion-readiness.

## Current decision for March 9, 2026

Based on the current repository evidence, the narrow exporter subset satisfies
the acceptance contour above and is now promoted to `stable`.

Supporting evidence:

- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Freshness](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md)
- [Simplified Likelihood Exporter Public Validation Surface](/docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Review Checklist](/docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Source-Semantics Boundary](/docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Candidate Blocker Matrix](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Candidate Review Packet](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Promotion Decision](/docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md)
- [Simplified Likelihood Exporter Release PR Checklist](/docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)

The current `nextstat-bench` exporter snapshot shows:

- `export_matrix_case_count = 10`
- `export_matrix_public_reinterpretation_style_case_count = 8`
- `export_matrix_min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`
- `export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup = 2.1792428549098894x`
- `public_validation.min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`
- `max_abs_q_mu_diff = 0.09618848026584459`
- `max_upper_limit_ratio_deviation = 0.011190668120821257`
- the stable source boundary is now published separately as:
  `pyhf` source only, single-POI only, Gaussian-constrained
  `source_model_constraints`, and reduced-coordinate rather than source-level
  nuisance semantics
- the explicit stable-promotion decision is now committed under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`
- the release-facing maintenance/admission policy is now committed under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
- the release-facing freshness report is now committed under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`
- the committed public validation report is now published under
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`

## Bottom line

This policy makes the narrow exporter path stably governable:

- contract expectations are explicit
- `nextstat-bench` thresholds are explicit
- the committed benchmark evidence is explicit
- the accepted exporter promotion bundle is explicit under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`
- the formal stable-review assessment is explicit under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json`
- the stable-candidate blocker matrix is explicit under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`
- the validator-facing stable-candidate review packet is explicit under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json`
- the explicit stable-promotion decision is explicit under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`
- the stable-evidence freshness state is explicit under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`
- the committed exporter matrix now includes machine-classified public
  reinterpretation-style cases, so future stable review is no longer blocked on
  synthetic-only exporter evidence
- release-facing stable governance can now be argued from versioned artifacts
  instead of terminal output
