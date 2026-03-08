# Simplified Likelihood Exporter Stable-Evidence Freshness

**Date**: 2026-03-09  
**Status**: Accepted operational freshness guard for the promoted exporter stable subset  
**Scope**: committed `nextstat-bench` exporter stable-evidence floor

## Purpose

The exporter stable subset already has a published admission policy in
`stable_evidence_policy.json`.

This note adds the missing operational guard: a machine-readable freshness
report so the accepted `5 public / 7 total` floor cannot silently age past the
review window.

## Contract

- schema:
  `docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json`
- example:
  `docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json`
- builder:
  `scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py`
- accepted artifact:
  `benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json`

## Freshness Rule

The accepted exporter stable-evidence floor is fresh only if all of the
following remain true:

- benchmark host is `nextstat-bench`
- committed current exporter snapshot status is `persisted`
- committed public validation status is `ok`
- accepted stable-evidence policy remains `accepted`
- accepted stable-promotion decision remains `accepted`
- accepted stable-promotion support class remains `stable`
- committed current snapshot still satisfies the published `7 total / 5 public`
  floor
- committed current snapshot still has zero cases outside the promoted runtime
  boundary
- snapshot age does not exceed `45` calendar days at the report reference date

## Boundary

This report is an operational guard, not a new stable-claim expansion.

It does not widen source semantics, runtime support, or admission thresholds.
It only makes staleness explicit and machine-checkable.

## Current Accepted State

For the March 9, 2026 accepted evidence set:

- `snapshot_id = export-20260309T160200Z`
- `snapshot_age_days = 0`
- `max_snapshot_age_days = 45`
- `status = fresh`

If the committed `nextstat-bench` exporter evidence is not refreshed before the
45-day window is exceeded, the freshness report moves to `breached` and the
exporter surface gate must fail explicitly.
