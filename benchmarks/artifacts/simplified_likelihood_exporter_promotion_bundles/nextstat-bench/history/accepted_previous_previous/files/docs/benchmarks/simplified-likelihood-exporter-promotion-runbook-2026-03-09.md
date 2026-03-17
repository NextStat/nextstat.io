# Simplified Likelihood Exporter Promotion Runbook

**Date**: 2026-03-09  
**Status**: Accepted exporter evidence-bundle lifecycle for the promoted narrow stable subset  
**Scope**: machine-readable exporter evidence bundles built from the committed
`nextstat-bench` exporter snapshot

## Purpose

This runbook governs the exporter-specific evidence-bundle lifecycle.

It defines how to turn the committed exporter benchmark snapshot into a
verifiable, accepted evidence bundle, historical stable-review artifacts, and
the explicit stable-promotion decision for the narrow exporter subset.

## Contracts

- bundle schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json`
- bundle example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json`
- verifier schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json`
- verifier example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.example.json`
- promotion report schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.schema.json`
- promotion report example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json`
- stable evidence policy schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json`
- stable evidence policy example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json`
- stable evidence freshness schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json`
- stable evidence freshness example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json`

## Scripts

- builder:
  `scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py`
- verifier:
  `scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py`
- promoter:
  `scripts/benchmarks/promote_simplified_likelihood_exporter_promotion_bundle.py`
- canonical remote bench runner:
  `scripts/benchmarks/apex2_simplified_likelihood_remote.sh`
- stable evidence policy builder:
  `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py`
- stable evidence freshness builder:
  `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py`

The canonical remote runner builds the Python extension from the synced source
tree with `maturin develop --skip-install` and executes the Apex2 harness with
`PYTHONPATH=$REPO/bindings/ns-py/python`, so the `nextstat-bench` lane does not
depend on a prepublished `nextstat-cli` wheel matching the repo version.

## Inputs

The canonical input is the committed current exporter snapshot under:

- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json`
- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json`
- `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json`

## Promotion-readiness thresholds

The exporter bundle is promotion-ready for future stable review only if:

- benchmark host is `nextstat-bench`
- `max_abs_q_mu_diff <= 0.1`
- `upper_limit_ratio` remains within `[0.95, 1.05]`
- synthetic control floor
  `min_net_end_to_end_upper_limit_speedup >= 1.25x`
- committed exporter snapshot status is `persisted`
- committed exporter matrix case count is at least `10`
- committed exporter matrix includes at least `9 public reinterpretation-style cases`

The bundle also carries the overall public-case minimum separately:

- `overall_min_net_end_to_end_upper_limit_speedup`
- `public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup`

Those numbers are evidence-only and do not replace the synthetic stable-review
control floor.

## Accepted artifact path

The committed accepted exporter bundle lives under:

- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`

## Commands

Build:

```bash
python3 scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py \
  --benchmark-artifact benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json \
  --snapshot-report benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json \
  --snapshot-index benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json \
  --bundle-dir tmp/simplified_likelihood_exporter_promotion_bundle_<timestamp>/nextstat-bench \
  --deterministic
```

Verify:

```bash
python3 scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py \
  --bundle-dir tmp/simplified_likelihood_exporter_promotion_bundle_<timestamp>/nextstat-bench \
  --out tmp/simplified_likelihood_exporter_promotion_bundle_<timestamp>/nextstat-bench/promotion_evidence_check.json \
  --require-promotion-ready \
  --deterministic
```

Promote:

```bash
python3 scripts/benchmarks/promote_simplified_likelihood_exporter_promotion_bundle.py \
  --source-bundle-dir tmp/simplified_likelihood_exporter_promotion_bundle_<timestamp>/nextstat-bench \
  --deterministic
```

Assess for formal stable review:

```bash
python3 scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py \
  --bundle-dir benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json \
  --deterministic
```

Publish the future stable source-semantics boundary:

```bash
python3 scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json \
  --deterministic
```

Assess the remaining delta-to-stable blockers:

```bash
python3 scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py \
  --bundle-dir benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json \
  --deterministic
```

Build the validator-facing stable-candidate review packet:

```bash
python3 scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py \
  --bundle-dir benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json \
  --deterministic
```

Record the explicit stable-promotion decision:

```bash
python3 scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py \
  --bundle-dir benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json \
  --deterministic
```

Publish the release-facing stable-evidence maintenance/admission policy:

```bash
python3 scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py \
  --benchmark-artifact benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json \
  --public-validation-report benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json \
  --stable-promotion-decision benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json \
  --deterministic
```

Publish the release-facing stable-evidence freshness report:

```bash
python3 scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py \
  --snapshot-report benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json \
  --public-validation-report benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json \
  --stable-evidence-policy benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json \
  --stable-promotion-decision benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json \
  --deterministic
```

## Maintenance cadence

The accepted `9 public / 11 total` floor is refreshed:

- on every exporter release PR touching runtime or governance surface
- on every accepted public-case catalog change
- on every committed `nextstat-bench` exporter snapshot refresh
- on every promoted boundary/support wording change

That cadence is now recorded machine-readably in
`stable_evidence_policy.json`; it is no longer only prose in the runbook.
The operational freshness window is recorded separately in
`stable_evidence_freshness_report.json`; a freshness breach is an explicit
governance failure rather than a silent stale-evidence fallback.

## Boundary

This accepted bundle path records exporter evidence and governance state.

The narrow exporter subset is now stable only because the accepted bundle also
includes `stable_promotion_decision.json` and release-facing workflow
consumption.

Everything outside the published narrow source boundary remains
`research-grade`.

Current March 11, 2026 committed exporter evidence meets those thresholds with:

- `export_matrix_case_count = 11`
- `export_matrix_public_reinterpretation_style_case_count = 9`
- `min_net_end_to_end_upper_limit_speedup = 2.1792428549098894x`
- `overall_min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`
- `public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup = 0.8777103422927768x`
- `max_snapshot_age_days = 45`
