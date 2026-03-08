# Simplified Likelihood Exporter Stable-Candidate Blocker Matrix

**Date**: 2026-03-09  
**Status**: release-hardening blocker matrix for the promoted narrow stable subset  
**Scope**: `nextstat simplify workspace` and the accepted exporter evidence bundle on `nextstat-bench`

## Purpose

This note answers a narrower question than the exporter acceptance note and the
stable-review checklist:

- not "is the exporter evidence package real?"
- not "is the wider research-grade fallback governed?"
- but "what still blocks the narrow stable exporter claim?"

The current answer is:

- the exporter has strong accepted evidence
- the exporter is formal-review ready
- the narrow exporter subset is now promoted to `stable`
- wider fallback modes still remain outside that stable claim

## Current evidence base

Read this blocker matrix together with:

- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Review Checklist](/docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md)
- [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)

The committed accepted exporter evidence now includes:

- `promotion_evidence.json`
- `promotion_evidence_check.json`
- `promotion_bundle_promotion_report.json`
- `stable_review_assessment.json`
- `stable_source_semantics_boundary.json`
- `stable_candidate_blocker_matrix.json`
- `stable_candidate_review_packet.json`
- `stable_promotion_decision.json`
- `snapshot_index.json`

all under
`benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`.

## What is already strong

The following exporter claims are now evidence-backed:

1. The exporter is not an undocumented prototype.
   It has versioned derive/export contracts, a machine-readable export report,
   a committed `nextstat-bench` snapshot, an accepted evidence bundle, and a
   formal stable-review assessment.

2. The accepted bundle is repeatable.
   The exporter surface gate rebuilds and rechecks the accepted bundle and the
   stable-review assessment deterministically.

3. The current research-grade path is numerically strong.
   The committed exporter matrix is green on fidelity and net end-to-end speedup
   for the current mixed public-plus-synthetic `full -> derived -> reinterpret`
   cases.

4. The support-class boundary is explicit.
   Public docs and accepted artifacts now keep only the narrow subset `stable`,
   while wider fallback modes remain `research-grade`, and the accepted
   assessment keeps automatic stable promotion disabled.

## What still blocks a stable candidate

No blockers remain for the narrow promoted subset.

The previously open `public reinterpretation-style exporter matrix` blocker is
resolved.

The previously open `stable source semantics boundary` blocker is resolved.

The previously open `stable-candidate review packet` blocker is also resolved.

The previously open `stable release promotion decision` blocker is now resolved
through the committed stable-promotion decision and release workflow
consumption.

Current accepted evidence on `nextstat-bench` includes:

- `export_matrix_case_count = 8`
- `export_matrix_public_reinterpretation_style_case_count = 6`
- public cases:
  `atlas_public_dual_sr_dual_cr_gaussian_export_stable_example`,
  `atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example`,
  `atlas_public_sr_cr_gaussian_export_stable_example`,
  `cms_public_sr_cr_export_stable_example`,
  `cms_public_sr_cr_asymmetric_gaussian_export_stable_example`,
  `cms_public_dual_sr_cr_gaussian_export_stable_example`
- current `stable_candidate.open_blocker_count = 0`

## Machine-readable blocker matrix

The canonical JSON contract for this note is:

- schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json`
- example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json`
- generator:
  `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py`

The committed accepted artifact is:

- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`

The machine-readable matrix must remain explicit about governance state:

- `support_class = "stable"` for the narrow promoted subset
- `automatic_stable_promotion = false`
- `stable_candidate.status = "ready"` means the blocker contour is fully closed
- wider fallback modes remain outside the promoted stable subset

## Recommendation

The correct next step is not to widen the exporter claim in place.

The correct next step is:

1. preserve the promoted narrow stable subset through release gating and
   accepted evidence refreshes
2. treat any broader exporter behavior as a separate hardening program

## Current March 9, 2026 state

The current stable-candidate blocker matrix is now `ready`.

That is the correct and intended state for the narrow stable subset.

It means:

- the exporter evidence and release governance are aligned
- the public exporter-evidence blocker is closed
- the stable source-semantics boundary blocker is closed
- the release-promotion decision blocker is closed
- the narrow exporter subset is `stable`
