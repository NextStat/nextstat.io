# Simplified Likelihood Exporter Stable Promotion Decision

**Date**: 2026-03-09  
**Status**: Accepted narrow stable-promotion decision  
**Scope**: `nextstat simplify workspace` on the accepted `nextstat-bench` exporter evidence path

## Decision

The narrow exporter subset is now promoted to `stable`.

This promotion is explicit, versioned, and machine-verifiable through
`stable_promotion_decision.json`. It is not automatic and it does not widen the
stable claim beyond the already published boundary.

## Stable subset

The promoted `stable` subset for `nextstat simplify workspace` is:

- `pyhf` source workspaces only
- `single-POI` exporter scope only
- `constraint_covariance_source = "source_model_constraints"` only
- `Gaussian-constrained` source nuisances only
- reduced-coordinate output semantics only
- explicit `derived_from_workspace` provenance in the exported artifact

The accepted decision artifact lives at:

- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`

## Research-grade fallback

The stable claim remains intentionally narrow. The following paths stay
`research-grade` fallback or explicit reject behavior:

- `aligned_fit_covariance`
- non-Gaussian or unconstrained source nuisance families
- multi-POI export
- partial per-channel bin selection
- source-level nuisance identity guarantees after reduction
- source-level ranking breakdown on reduced coordinates

## Release-facing governance

This decision is consumed by `.github/workflows/release.yml` and by the
dedicated exporter workflow:

- `.github/workflows/simplified-likelihood-exporter-surface.yml`
- `.github/workflows/release.yml`

Release-facing governance now expects the accepted exporter bundle to include:

- `promotion_evidence.json`
- `promotion_evidence_check.json`
- `promotion_bundle_promotion_report.json`
- `stable_review_assessment.json`
- `stable_evidence_policy.json`
- `stable_source_semantics_boundary.json`
- `stable_candidate_blocker_matrix.json`
- `stable_candidate_review_packet.json`
- `stable_promotion_decision.json`
- `snapshot_index.json`

## Evidence basis

The accepted decision is based on:

- the committed `nextstat-bench` exporter snapshot under
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/`
- the accepted exporter promotion bundle under
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/`
- the published source boundary note
- the release-facing checklist that wires the accepted exporter artifacts into
  `.github/workflows/release.yml`

The decision keeps `automatic_stable_promotion = false`. Maintainers still take
the stable decision deliberately; they do not get it as a side effect of the
benchmark or the gate.

## Bottom line

`nextstat simplify workspace` is now `stable` only for the narrow source model
boundary above. Wider fallback modes remain available, but stay
`research-grade` and must continue to be documented as such.
