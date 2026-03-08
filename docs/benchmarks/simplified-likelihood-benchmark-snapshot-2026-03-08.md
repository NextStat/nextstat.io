# Simplified Likelihood Benchmark Snapshot: 2026-03-08

**Date**: 2026-03-08  
**Host**: `nextstat-bench`  
**Stable artifact**: [apex2_simplified_likelihood_report.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/files/benchmark/apex2_simplified_likelihood_report.json)
**Stable bundle**: [promotion_evidence.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json)
**Stable check**: [promotion_evidence_check.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json)
**Stable snapshot index**: [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json)
**Exporter current artifact**: [apex2_simplified_likelihood_report.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json)
**Exporter current snapshot report**: [export_benchmark_snapshot_report.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json)
**Exporter public validation report**: [export_public_validation_report.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json)
**Exporter current snapshot index**: [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json)
**Exporter public case catalog**: [simplified_likelihood_export_public_case_catalog_v0.example.json](/docs/specs/apex2_simplified_likelihood_export_public_case_catalog_v0.example.json)
**Exporter accepted bundle**: [promotion_evidence.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json)
**Exporter accepted check**: [promotion_evidence_check.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json)
**Exporter accepted promotion report**: [promotion_bundle_promotion_report.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json)
**Exporter accepted review packet**: [stable_candidate_review_packet.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json)
**Exporter accepted stable-review assessment**: [stable_review_assessment.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json)
**Exporter accepted stable-candidate blocker matrix**: [stable_candidate_blocker_matrix.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json)
**Exporter accepted stable-evidence policy**: [stable_evidence_policy.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json)
**Exporter accepted stable-evidence freshness report**: [stable_evidence_freshness_report.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json)
**Exporter accepted stable-promotion decision**: [stable_promotion_decision.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json)
**Exporter accepted snapshot index**: [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json)

## Purpose

This note records the current benchmark evidence for the simplified-likelihood
stable-surface acceptance decision.

It is the benchmark companion to:

- [Simplified Likelihood Stable-Surface Acceptance](/docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Support Matrix](/docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md)
- [Simplified Likelihood Stable-Surface Release Notes](/docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md)
- [Simplified Likelihood Artifacts](/docs/references/simplified-likelihood-artifacts.md)
- [Simplified Likelihood Exporter Acceptance](/docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md)
- [Simplified Likelihood Exporter Runtime Gate](/docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md)
- [Simplified Likelihood Exporter Promotion Runbook](/docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Policy](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Evidence Freshness](/docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md)
- [Simplified Likelihood Exporter Public Validation Surface](/docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Review Checklist](/docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Source-Semantics Boundary](/docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md)
- [Simplified Likelihood Exporter Stable-Candidate Blocker Matrix](/docs/benchmarks/simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md)
- [Simplified Likelihood Exporter Stable Promotion Decision](/docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md)

The validator-facing promotion bundle was built from the frozen benchmark
artifact above and records the same promotion claim in a single inventory-backed
package. The paired promotion check report verifies the bundle schema, copied
inventory, and promotion-ready claim from that frozen handoff directory. The
local promotion handoff under `tmp/` is intentionally not committed; the public
repo keeps the persisted stable accepted bundle path
at `benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/`
with a committed [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json)
for auditability and release linking.

The exporter current artifact is a separate `nextstat-bench` run used to
verify the research-grade `full -> derived -> reinterpret` matrix after the
explicit `constraint_covariance_source` fix. It does not replace the accepted
stable-surface promotion bundle above.

That exporter follow-up artifact is now also persisted under a committed
research-grade path:

- current exporter artifact:
  [apex2_simplified_likelihood_report.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json)
- current exporter snapshot report:
  [export_benchmark_snapshot_report.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json)
- current exporter public validation report:
  [export_public_validation_report.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json)
- current exporter snapshot index:
  [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json)

That same committed exporter snapshot now also has an accepted promotion-readiness
bundle under:

- accepted exporter bundle:
  [promotion_evidence.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json)
- accepted exporter verification report:
  [promotion_evidence_check.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json)
- accepted exporter promotion report:
  [promotion_bundle_promotion_report.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json)
- accepted exporter stable-review assessment:
  [stable_review_assessment.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json)
- accepted exporter stable-candidate blocker matrix:
  [stable_candidate_blocker_matrix.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json)
- accepted exporter stable-evidence policy:
  [stable_evidence_policy.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json)
- accepted exporter stable-evidence freshness report:
  [stable_evidence_freshness_report.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json)
- accepted exporter stable-promotion decision:
  [stable_promotion_decision.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json)
- accepted exporter snapshot index:
  [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json)

Accepted bundle inventory for the first persisted promotion:

- accepted bundle entrypoint:
  [promotion_evidence.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json)
- accepted verification report:
  [promotion_evidence_check.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json)
- accepted promotion report:
  [promotion_bundle_promotion_report.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json)
- accepted benchmark artifact copy:
  [apex2_simplified_likelihood_report.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/files/benchmark/apex2_simplified_likelihood_report.json)
- accepted snapshot index:
  [snapshot_index.json](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json)
- promoted history snapshot:
  [accepted_20260308T173340Z_promoted](/benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/history/accepted_20260308T173340Z_promoted)

## Run command

```bash
BENCH_HOST=nextstat-bench \
BENCH_SUITE=bench \
bash scripts/benchmarks/apex2_simplified_likelihood_remote.sh
```

## Acceptance summary

All current acceptance gates passed.

Suite summary:

- status: `ok`
- case count: `2`
- all schema valid: `true`
- all fidelity gates pass: `true`
- all performance gates pass: `true`
- public fixture matrix included: `true`
- public fixture matrix status: `ok`
- public fixture matrix JSON Schema validation available: `true`
- max `delta_mu_hat / sigma_mu_full`: `0.0`
- max `q_mu` absolute difference: `2.393811229239873e-07`
- max upper-limit ratio deviation from `1.0`: `0.0`
- max reduced nuisance fraction: `0.08333333333333333`
- max JSON size fraction: `0.12117248240539445`
- minimum fit speedup: `94.7609528384387x`
- minimum upper-limit speedup: `10.652118204493238x`
- minimum end-to-end upper-limit speedup: `10.325036821081905x`

Public-style fixture matrix summary:

- fixture count: `3`
- all runtime gates pass: `true`
- all derived fidelity gates pass: `true`
- source formats covered: `basis`, `covariance`, `derived_from_workspace`

Exporter follow-up matrix summary (`nextstat-bench`, March 11, 2026 exporter refresh):

- export matrix included: `true`
- export matrix status: `ok`
- export matrix case count: `10`
- export matrix case kinds: `public_reinterpretation_style`, `synthetic`
- export matrix public reinterpretation-style case count: `8`
- exported public validation artifact path:
  `benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json`
- export matrix all schema valid: `true`
- export matrix all fidelity gates pass: `true`
- export matrix max `q_mu` absolute difference: `0.09618848026584459`
- export matrix max upper-limit ratio deviation from `1.0`: `0.011190668120821257`
- export matrix overall minimum net end-to-end upper-limit speedup: `0.8777103422927768x`
- export matrix synthetic control minimum net end-to-end upper-limit speedup: `2.1792428549098894x`
- export matrix public reinterpretation-style minimum net end-to-end upper-limit speedup: `0.8777103422927768x`
- accepted exporter blocker matrix open blocker count: `0`

## Case results

### `synthetic_covariance_medium`

- full nuisance count: `72`
- reduced nuisance count: `6`
- reduced nuisance fraction: `0.08333333333333333`
- full JSON bytes: `36775`
- simplified JSON bytes: `4037`
- JSON size fraction: `0.10977566281441196`
- fit speedup: `94.7609528384387x`
- upper-limit speedup: `10.652118204493238x`
- end-to-end upper-limit speedup: `10.325036821081905x`
- max `q_mu` absolute difference: `2.393811229239873e-07`
- upper-limit ratio: `1.0`

### `synthetic_covariance_large`

- full nuisance count: `128`
- reduced nuisance count: `8`
- reduced nuisance fraction: `0.0625`
- full JSON bytes: `108556`
- simplified JSON bytes: `13154`
- JSON size fraction: `0.12117248240539445`
- fit speedup: `249.69333999261687x`
- upper-limit speedup: `18.294003650276608x`
- end-to-end upper-limit speedup: `17.502768873140017x`
- max `q_mu` absolute difference: `6.294365562098392e-08`
- upper-limit ratio: `1.0`

## Interpretation

The current evidence supports the stable-surface claim for the simplified
likelihood consume/audit path:

- fidelity gates are not marginal; they are comfortably green
- reduction gates are well inside budget
- the bench-host promotion target of `>= 10x` end-to-end upper-limit speedup is met
- the refreshed artifact also carries a green public-style runtime matrix for
  curated basis/covariance/derived consume fixtures

In practical terms, the current evidence is strong enough to support the
product claim:

> the simplified-likelihood path delivers real reinterpretation acceleration,
> and the current benchmark evidence already reaches the `~10x` target class on
> `nextstat-bench`

For the research-grade exporter path, the March 9, 2026 follow-up bench-host
evidence now shows two concrete things:

- the explicit `constraint_covariance_source` split remains green on the
  synthetic medium/large export controls
- the committed exporter matrix now also includes seven machine-classified public
  reinterpretation-style export cases on `nextstat-bench`
- those public-style exporter cases are now also published as a dedicated
  stable evidence artifact instead of living only inside the aggregate Apex2
  report
- the accepted exporter bundle now also publishes a machine-readable
  `stable_evidence_freshness_report.json`, so stale evidence breaches are
  explicit and release-consumable
- the public stable-evidence lane now stays inside the promoted runtime
  boundary on the `source_model_constraints` path, while the synthetic control
  lane remains the stable-review performance floor

That closes the old synthetic-only exporter evidence gap. The public exporter
blocker is now resolved in the accepted
`stable_candidate_blocker_matrix.json`, the stable source-semantics
boundary is now published explicitly, the validator-facing
`stable_candidate_review_packet.json` is committed alongside it, and the
explicit `stable_promotion_decision.json` closes the release-facing governance
loop for the narrow stable exporter subset.

The public exporter validation surface is stable as evidence and now stays
inside the promoted runtime boundary for its committed case. It still remains a
separate evidence lane rather than the synthetic promotion-control floor used
for stable review.

## Caveat

This note is evidence for the currently scoped stable subset only.
It does not by itself promote export/publication or broader reduced-model
surfaces that are still outside the accepted scope. The embedded public fixture
matrix is useful external-validation evidence, but it is not a replacement for
the paired synthetic speedup gates that still drive the stable promotion claim.
The accepted bundle path above is a persistence/curation surface for already
verified evidence; it is not a substitute for rerunning the canonical
`nextstat-bench` workflow when runtime behavior or performance claims change.
The committed exporter snapshot path above is persistence for the bench-host
evidence, not a substitute for the accepted exporter promotion bundle. The
narrow `nextstat simplify workspace` stable subset is now governed by the
accepted bundle plus `stable_promotion_decision.json`; wider exporter behavior
still remains outside that stable subset.
