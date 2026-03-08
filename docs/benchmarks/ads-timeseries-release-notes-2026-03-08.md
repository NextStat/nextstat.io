# Ads + Time Series Stable-Surface Release Notes

**Date**: 2026-03-08  
**Status**: release-hardening note  
**Scope**: ads-native observation, variance-reduction, and weekly state-space convenience surface

## Promoted stable subset

This release-hardening wave promotes a narrow product surface:

- `BetaBinomialModel`
- `DelayCorrectionModel`
- `cuped_adjust(...)`
- `cure_adjust(...)`
- `hill(...)`
- `adstock_geometric(...)`
- weekly Kalman constructors in Rust
- weekly Kalman JSON aliases in the CLI
- `nextstat.ads.*` plus weekly Python builders

## What changed

- the Python surface is now fully wired through PyO3, lazy module exports, and type stubs
- CUPED is now treated architecturally as the one-covariate case of the shared CURE layer
- the shared CURE surface now reports method/solver/covariate diagnostics, typed covariate provenance, and guards against collinearity with SVD or ridge fallback
- `ns_inference::ads` now re-exports the shared CUPED/CURE primitives in addition to the crate-root export
- committed fixture-grade reference datasets now cover binary, revenue, ratio-style, low-conversion, multi-channel, and ridge-fallback variance-reduction cases
- the weekly CLI aliases are covered by committed contract fixtures
- the stable subset is now documented as an explicit support matrix rather than an implicit code path
- a dedicated stable-surface gate exists for Rust, CLI, Python, and benchmark-smoke coverage
- a canonical `nextstat-bench` promotion runbook exists for runtime evidence
- an accepted `nextstat-bench` baseline is now committed for the promoted subset
- compare / promotion / gate reports now have versioned schemas and examples
- the benchmark schemas now pin the promoted 9-case runtime surface and require
  the committed CUPED/CURE benchmark detail contract
- benchmark identity widenings now go through an explicit reviewed-promotion
  path before the accepted baseline is re-pinned
- a separate realistic ads variance-reduction matrix now compares `naive`,
  `CUPED`, and `CURE` across sample size, covariate count, sparsity, and
  collinearity, including an explicit ridge-fallback stress case

## Boundaries that remain intentional

This note does **not** widen the stable promise to:

- `hierarchical_segment_lift_summary(...)`
- generic seasonal builders beyond the fixed weekly aliases
- richer delay or MMM calibration families beyond the current primitives
- post-treatment covariates for CUPED/CURE

## Operational surface

- local/CI gate:
  `make ads-timeseries-stable-surface-gate`
- benchmark harness:
  `python3 scripts/benchmarks/bench_ads_timeseries_surface.py`
- remote benchmark runner:
  `bash scripts/benchmarks/bench_ads_timeseries_surface_remote.sh`
- runtime compare:
  `python3 scripts/benchmarks/compare_ads_timeseries_benchmark.py`
- one-shot runtime gate:
  `python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py`

## Runtime wording

This promotion note is an API-stability note, not a blanket performance claim.

Any public runtime wording must be scoped to the current `nextstat-bench`
artifact produced from the canonical runtime gate / promotion runbook.
