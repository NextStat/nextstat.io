# Simplified Likelihood Exporter Release PR Checklist

**Date**: 2026-03-09  
**Status**: Release-facing checklist for the promoted narrow exporter subset  
**Scope**: `nextstat simplify workspace`

Use this checklist when a release PR includes exporter-facing changes.

## Required artifacts

- confirm the accepted bundle contains `promotion_evidence.json`
- confirm the accepted bundle contains `promotion_evidence_check.json`
- confirm the accepted bundle contains `stable_candidate_blocker_matrix.json`
- confirm the accepted bundle contains `stable_candidate_review_packet.json`
- confirm the accepted bundle contains `stable_evidence_policy.json`
- confirm the accepted bundle contains `stable_evidence_freshness_report.json`
- confirm the accepted bundle contains `stable_promotion_decision.json`
- confirm `.github/workflows/release.yml` consumes the exporter gate and uploads
  the accepted exporter artifacts
- confirm `.github/workflows/simplified-likelihood-exporter-surface.yml` stays
  green and uploads the same committed accepted bundle
- confirm the committed `snapshot_index.json` in the accepted bundle includes
  `stable_promotion_decision.json`

## Required wording

- keep `nextstat simplify workspace` marked as `stable` only for the narrow
  published boundary
- keep broader exporter modes marked as `research-grade fallback`
- keep `aligned_fit_covariance` and source-level nuisance identity claims out of
  the `stable` subset
- keep the accepted stable-evidence floor explicit as `8 public / 10 total`
- keep exporter public-case admission policy explicit instead of silently
  counting broader cases toward the accepted floor
- keep maintenance cadence explicit for exporter release PRs and public-case
  admissions
- keep the `45-day` stable-evidence freshness window explicit and treat any
  freshness breach as a release blocker

## Required release references

- `.github/workflows/release.yml`
- `.github/workflows/simplified-likelihood-exporter-surface.yml`
- `docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`
- `docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md`

## Bottom line

Do not widen the exporter stable claim in a release PR. The only accepted
stable surface is the narrow subset recorded in `stable_promotion_decision.json`;
everything else remains an explicit research-grade fallback. A freshness breach
in `stable_evidence_freshness_report.json` is a hard stop for the release PR.
