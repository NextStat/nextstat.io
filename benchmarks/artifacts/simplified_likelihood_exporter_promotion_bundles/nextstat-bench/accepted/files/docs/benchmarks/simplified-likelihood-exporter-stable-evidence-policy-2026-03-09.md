# Simplified Likelihood Exporter Stable Evidence Policy

**Date**: 2026-03-09  
**Status**: Accepted release-facing admission and maintenance policy  
**Scope**: committed `nextstat-bench` exporter stable evidence for the promoted narrow exporter subset

## Purpose

This note defines the release-facing policy for maintaining the accepted
exporter stable-evidence floor.

The runtime support claim is already governed by the narrow stable source
boundary and the explicit `stable_promotion_decision.json` artifact.

This policy covers the separate question that release governance must answer:

- when a public exporter case is admissible into the accepted stable-evidence matrix
- when the committed current snapshot and accepted bundle must be refreshed
- which docs/workflows must move in lockstep when the `8 public / 10 total`
  floor changes

## Machine-readable policy artifact

The canonical machine-readable policy is:

- schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json`
- example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- committed accepted artifact:
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
- builder:
  `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py`

## Stable-evidence floor

The current accepted floor is:

- `export_matrix_case_count >= 10`
- `public_reinterpretation_style` case count `>= 8`
- synthetic control floor:
  `synthetic_min_net_end_to_end_upper_limit_speedup >= 1.25x`
- public stable-evidence floor:
  `public_validation.min_net_end_to_end_upper_limit_speedup >= 0.75x`
- `cases_outside_promoted_stable_runtime_boundary = 0`

This floor is release-facing governance for the evidence surface.

It does not silently widen the stable runtime claim for
`nextstat simplify workspace`.

## Admission policy

A case is admissible into the accepted public exporter stable-evidence matrix
only if all of the following hold:

- case kind is `public_reinterpretation_style`
- evidence comes from committed `nextstat-bench` exporter snapshots
- source workspace is `pyhf`
- POI scope is single-POI
- `constraint_covariance_source = "source_model_constraints"`
- source nuisance family is Gaussian-constrained on that path
- observed source modifiers stay within `histosys`, `normsys`, `lumi`, `normfactor`
- the case remains inside the promoted stable runtime boundary
- the case does not require source-level nuisance identity preservation
- the case does not force silent widening of the promoted runtime/source-semantics boundary

If any one of these fails, the case can still exist as research-grade evidence,
but it must not be counted toward the accepted `8 public / 10 total` stable floor.

## Maintenance cadence

Refresh cadence is:

- on every exporter release PR that touches runtime or governance surface
- on every accepted public-case catalog change
- on every committed `nextstat-bench` exporter snapshot refresh
- on every change to promoted boundary wording or support-class wording

Each refresh must update, together:

- committed current exporter snapshot
- committed public validation report
- accepted exporter bundle
- release workflow uploads
- standalone exporter workflow uploads
- release-facing docs and checklists

This is deliberate. The floor is only useful if the committed artifacts,
workflows, and wording stay synchronized.

## Release consumers

The current release-facing consumers are:

- `.github/workflows/release.yml`
- `.github/workflows/simplified-likelihood-exporter-surface.yml`
- `docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md`
- `docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`
- `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`
- `docs/references/simplified-likelihood-artifacts.md`

Required uploaded artifacts now explicitly include:

- `export_public_validation_report.json`
- `stable_evidence_policy.json`
- `stable_source_semantics_boundary.json`
- `stable_promotion_decision.json`

## Current March 9, 2026 accepted state

The committed accepted policy reflects the current `nextstat-bench` evidence:

- `export_matrix_case_count = 10`
- `public_case_count = 8`
- `public_case_names = ["atlas_public_dual_sr_dual_cr_gaussian_export_stable_example", "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example", "atlas_public_sr_cr_gaussian_export_stable_example", "cms_public_sr_cr_export_stable_example", "cms_public_sr_cr_asymmetric_gaussian_export_stable_example", "cms_public_dual_sr_cr_gaussian_export_stable_example", "cms_public_sr_dual_cr_gaussian_export_stable_example", "cms_public_sr_vr_dual_cr_gaussian_export_stable_example"]`
- `synthetic_min_net_end_to_end_upper_limit_speedup = 2.1792428549098894x`
- `public_min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`
- `max_abs_q_mu_diff = 0.09618848026584459`
- `max_upper_limit_ratio_deviation = 0.011190668120821257`
- `cases_outside_promoted_stable_runtime_boundary = 0`

## Bottom line

The stable exporter runtime claim is already explicit.

This policy makes the stable evidence floor explicit too:

- which public cases can count
- when the accepted evidence must be refreshed
- which release surfaces must stay aligned

That keeps the accepted `8 public / 10 total` floor governable as a stable
product surface instead of a one-off benchmark snapshot.
