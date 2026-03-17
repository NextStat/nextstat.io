# Simplified Likelihood Exporter Runtime Gate

This document defines the operational gate for the simplified-likelihood
exporter acceptance contour.

The gate validates the promoted narrow stable exporter subset and also checks
that wider fallback paths remain outside that stable claim. It stays aligned
with the explicit acceptance policy from
[Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md).

## Entry points

- script:
  `scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh`
- make target:
  `make simplified-likelihood-exporter-surface-gate`
- workflow:
  `.github/workflows/simplified-likelihood-exporter-surface.yml`

## What the gate checks

1. required docs/contracts exist
2. targeted Rust and CLI tests for simplified-likelihood/export are green
3. targeted Python smoke tests are green
4. the committed `nextstat-bench` exporter artifact trio exists:
   - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json`
   - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json`
   - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`
   - `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json`
5. JSON Schema validation passes for the committed trio
6. exporter-specific thresholds pass:
   - `max_abs_q_mu_diff <= 0.1`
   - `upper_limit_ratio` remains within `[0.95, 1.05]`
   - synthetic control floor
     `export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup >= 1.25x`
   - the committed exporter matrix includes at least three
     `public_reinterpretation_style` cases and at least five total exporter
     cases
   - the stable evidence artifact `export_public_validation_report.json`
     remains schema-valid and green
   - the public stable-evidence lane keeps
     `min_net_end_to_end_upper_limit_speedup >= 0.75x` and
     `cases_outside_promoted_stable_runtime_boundary = 0`
7. the committed accepted exporter promotion bundle is present and verifiable:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json`
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`
8. the committed accepted exporter stable-review assessment exists and matches
   a deterministic regeneration from the accepted bundle:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json`
   - `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md`
9. the committed accepted exporter stable-source-semantics boundary exists and
   matches a deterministic regeneration:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json`
   - `scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
10. the committed accepted exporter stable-candidate blocker matrix exists and
   matches a deterministic regeneration from the accepted bundle:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json`
   - `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md`
11. the committed accepted exporter stable-candidate review packet exists and
    matches a deterministic regeneration from the accepted bundle:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json`
   - `scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md`
12. the committed accepted exporter stable-promotion decision exists and
    matches a deterministic regeneration from the accepted bundle:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json`
   - `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md`
   - `docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md`
13. the committed accepted exporter stable-evidence policy exists and
    matches a deterministic regeneration from the committed current snapshot
    plus the accepted stable-promotion decision:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
   - `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md`
14. the committed accepted exporter stable-evidence freshness report exists,
    matches a deterministic regeneration from the committed current snapshot,
    and does not show a freshness breach:
   - `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`
   - `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py`
   - `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md`
   - `max_snapshot_age_days = 45`
   - a live freshness breach is treated as gate failure, not as a warning-only
     condition

## Run locally

```bash
make simplified-likelihood-exporter-surface-gate
```

Optional environment variables:

- `SL_EXPORTER_PY`
- `SL_EXPORTER_PYTHONPATH`
- `SL_EXPORTER_CARGO_TARGET_DIR`
- `SL_EXPORTER_FRESHNESS_REFERENCE_DATE`
- `SL_EXPORTER_SKIP_DOC_CHECKS`

## Why this gate exists

The consume path already has a promoted stable-surface gate.

The exporter path is different:

- it includes derivation overhead
- it still has explicit research-grade boundaries
- it depends on committed `nextstat-bench` `full -> derived -> reinterpret`
  evidence rather than the consume-path promotion bundle

That is why exporter acceptance uses a synthetic control floor of
`export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup >= 1.25x`
instead of the stable consume-path `>= 10x` reinterpretation claim.
The public validation surface has its own fidelity-first threshold and is
tracked separately as stable evidence.

## Notes

- this gate validates a committed `nextstat-bench` snapshot; it does not rerun
  the remote benchmark
- exporter bundle lifecycle details live in:
  [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md)
- public validation evidence semantics live in:
  [Simplified Likelihood Exporter Public Validation Surface](/docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md)
- release-facing stable-evidence cadence and admission policy live in:
  [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md)
- freshness-breach semantics and the `45-day` operational window live in:
  [Simplified Likelihood Exporter Stable Evidence Freshness](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md)
- that public-validation note defines the separate stable evidence surface for
  curated public exporter cases
- this gate preserves the promoted narrow stable exporter subset; it does not
  widen the exporter beyond the published source boundary
- use the benchmark snapshot note for the concrete March 8, 2026 evidence:
  [Simplified Likelihood Benchmark Snapshot: 2026-03-08](/docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md)
