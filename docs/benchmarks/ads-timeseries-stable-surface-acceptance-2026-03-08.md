# Ads + Time Series Stable-Surface Acceptance

**Date**: 2026-03-08  
**Status**: Release-hardening acceptance policy  
**Scope**: ads-native observation, variance-reduction, and weekly state-space convenience surface

## Purpose

This document defines the acceptance criteria for treating the current
ads-native observation / response helpers plus the fixed weekly Kalman surface
as a stable product surface rather than an implementation detail.

It answers one narrow question:

- what must be true before we say this ads + weekly time-series surface is
  acceptable as `stable`?

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | versioned public contract with explicit verification and promotion workflow |
| `research-grade` | versioned and tested, but not covered by the promoted stable-surface promise |

## Stable subset in scope

The stable subset is intentionally narrow:

- Rust:
  - `ns_inference::BetaBinomialModel`
  - `ns_inference::DelayCorrectionModel`
  - `ns_inference::cuped_adjust`
  - `ns_inference::cure_adjust`
  - `ns_inference::hill`
  - `ns_inference::adstock_geometric`
  - `ns_inference::timeseries::kalman::KalmanModel::local_level_weekly`
  - `ns_inference::timeseries::kalman::KalmanModel::local_linear_trend_weekly`
- CLI:
  - weekly Kalman JSON aliases `local_level_weekly`
  - weekly Kalman JSON aliases `local_linear_trend_weekly`
- Python:
  - `nextstat.BetaBinomialModel`
  - `nextstat.DelayCorrectionModel`
  - `nextstat.ads.*`
  - `nextstat.timeseries.local_level_weekly_model(...)`
  - `nextstat.timeseries.local_linear_trend_weekly_model(...)`

Out of scope for the current stable acceptance promise:

- `hierarchical_segment_lift_summary(...)`
- generic seasonal builders beyond the fixed weekly aliases
- richer delay families beyond the current single-rate exponential model
- any future MMM-style parameter fitting beyond the current primitive helpers

## Acceptance criteria

All items below must be true for acceptance.

### 1. Contract

- Rust API is documented in `docs/references/rust-api.md`
- Python API is documented in `docs/references/python-api.md`
- CLI contract is documented in `docs/references/cli.md`
- Python type stubs are synchronized with the exported native surface
- stable vs `research-grade` boundaries are explicit in the support matrix

### 2. Verification matrix

- Rust ads tests are green
- Rust variance-reduction tests are green
- Rust weekly-builder tests are green
- CLI weekly contract tests are green
- Python ads / variance-reduction / weekly builder smoke tests are green
- benchmark schema/example smoke tests are green
- no unresolved drift exists between docs/examples and the emitted benchmark artifact

### 3. Determinism and validation

- the promoted weekly CLI fixtures pass under a deterministic local gate
- the promoted Python ads helpers validate their outputs on fixed committed inputs
- the benchmark harness records CUPED/CURE Python contract cases alongside the existing ads/time-series surface
- committed reference fixtures under `tests/fixtures/variance_reduction/` pass in both Rust integration tests and Python public-surface parity tests
- the benchmark harness records both CLI and Python surfaces in one machine-readable artifact
- the benchmark artifact schema pins the exact promoted 9-case surface rather
  than accepting the older 7-case subset

### 4. Promotion evidence

- a current benchmark artifact exists from `nextstat-bench`
- the artifact uses `scripts/benchmarks/bench_ads_timeseries_surface.py`
- the accepted baseline exists at
  `benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json`
- compare/promotion/gate reports have committed schemas and examples
- compare status against the accepted baseline is `passed` before runtime
  wording changes are merged, except for the explicit reviewed
  `--allow-case-set-change` widening path
- the artifact is linked from the release PR when runtime wording is changed

## Operational gate surface

The stable subset is enforced through these operational entry points:

- local/CI gate:
  `make ads-timeseries-stable-surface-gate`
- dedicated workflow:
  `.github/workflows/ads-timeseries-stable-surface.yml`
- runtime gate:
  `docs/benchmarks/ads-timeseries-runtime-gate.md`
- release PR checklist:
  `docs/benchmarks/ads-timeseries-release-pr-checklist-2026-03-08.md`
- bench-host promotion runbook:
  `docs/benchmarks/ads-timeseries-promotion-runbook-2026-03-08.md`

## Acceptance decision rule

The surface is acceptable as `stable` only if:

1. contract docs and type stubs exist
2. the local/CI verification matrix is green
3. the benchmark harness artifact contract exists and validates
4. a current `nextstat-bench` artifact is available whenever runtime or performance wording changes

If any one of these fails, the affected surface remains `research-grade`.

## Current decision for March 8, 2026

Based on the current repository evidence, the API contract and local/CI
verification requirements are satisfied for the narrow subset listed above.

What is still intentionally scoped:

- the API stability claim is accepted for the listed surface
- any runtime wording or benchmark narrative still requires the canonical
  `nextstat-bench` artifact from the promotion runbook

## What acceptance does not imply

Acceptance does not mean:

- blanket MMM performance claims
- stable promotion of all generic seasonal state-space helpers
- stable promotion of every ads-native helper in `ns_inference::ads`
- approval to skip the bench-host artifact when public runtime wording changes
