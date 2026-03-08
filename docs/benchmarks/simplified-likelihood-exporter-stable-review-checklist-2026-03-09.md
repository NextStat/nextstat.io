# Simplified Likelihood Exporter Stable-Review Checklist

**Date**: 2026-03-09  
**Status**: formal stable-review process whose accepted output now feeds the promoted narrow stable subset  
**Scope**: `nextstat simplify workspace` and the committed exporter
promotion-readiness bundle under `nextstat-bench`

## Purpose

This checklist is the formal maintainer-side review surface for the
simplified-likelihood exporter.

It exists to answer one narrow question:

- is the current exporter evidence package strong enough to back an explicit
  stable-promotion decision?

The answer from the accepted assessment is **yes**, but only for the narrow
stable subset defined by the published source boundary.

## Required input evidence

The review must start from the committed accepted bundle under:

- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json`
- `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/snapshot_index.json`

## Machine-readable assessment

The canonical assessment contract is:

- schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.schema.json`
- example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json`
- generator:
  `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py`

The assessment must remain explicit about governance state:

- `support_class = "research-grade"`
- `automatic_stable_promotion = false`
- `summary.status = "review_ready"` means only that the exporter may enter an
  explicit stable-promotion decision
- `review_ready` does not by itself promote `nextstat simplify workspace` to
  `stable`

This does not by itself promote `nextstat simplify workspace` to `stable`.

## Checklist

### Contract and policy

- [ ] exporter acceptance note is current:
  - `docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md`
- [ ] exporter runtime gate note is current:
  - `docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md`
- [ ] exporter promotion runbook is current:
  - `docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md`
- [ ] stable evidence policy note is current:
  - `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md`
- [ ] stable source-semantics boundary note is current:
  - `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
- [ ] stable-review checklist note is current:
  - `docs/benchmarks/simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md`
- [ ] artifacts reference is current:
  - `docs/references/simplified-likelihood-artifacts.md`
- [ ] public docs state the narrow exporter subset as `stable` and wider
  fallback modes as `research-grade`
- [ ] no public text claims automatic stable promotion
- [ ] the future stable source boundary still states `pyhf`-only source,
  single-POI only, Gaussian-constrained `source_model_constraints`, and
  reduced-coordinate rather than source-level nuisance semantics

### Evidence validity

- [ ] benchmark host remains `nextstat-bench`
- [ ] accepted bundle check status is `passed`
- [ ] accepted promotion report status is `promoted`
- [ ] accepted stable-review assessment exists:
  - `stable_review_assessment.json`
- [ ] fidelity gates remain within policy:
  - `max_abs_q_mu_diff <= 0.1`
  - `max_upper_limit_ratio_deviation <= 0.05`
- [ ] performance gate remains within policy:
  - synthetic control floor
    `min_net_end_to_end_upper_limit_speedup >= 1.25x`
  - public stable-evidence floor
    `public_validation.min_net_end_to_end_upper_limit_speedup >= 0.75x`
- [ ] committed exporter matrix still includes at least three
  `public_reinterpretation_style` cases alongside the synthetic controls

### Automation and repeatability

- [ ] one-command gate exists:
  - `make simplified-likelihood-exporter-surface-gate`
- [ ] gate script is current:
  - `scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh`
- [ ] assessment script is current:
  - `scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py`
- [ ] dedicated workflow is current:
  - `.github/workflows/simplified-likelihood-exporter-surface.yml`
- [ ] workflow archives the accepted `stable_review_assessment.json`

### Messaging boundary

- [ ] release-facing docs keep wider fallback modes outside the promoted
  stable subset
- [ ] `review_ready` is described as historical evidence, not as automatic
  promotion
- [ ] any future promotion decision cites the accepted JSON artifacts, not
  terminal output

## Command

```bash
python3 scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py \
  --bundle-dir benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted \
  --out benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json \
  --deterministic
```

## Current March 9, 2026 review state

The current exporter assessment is `review_ready`.

That historical assessment is now consumed by the explicit stable-promotion
decision for the narrow exporter subset.

It still does **not** mean every exporter-compatible path is `stable`.
