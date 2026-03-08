# Ads + Time Series Benchmark Snapshot: 2026-03-08

**Date**: 2026-03-08  
**Host**: `nextstat-bench`  
**Artifact**: [accepted.json](/benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json)

## Purpose

This note records the current benchmark evidence for the ads + weekly
time-series stable surface on `nextstat-bench`.

It is the benchmark companion to:

- [Ads + Time Series Stable-Surface Acceptance](/docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08.md)
- [Ads + Time Series Stable-Surface Support Matrix](/docs/benchmarks/ads-timeseries-support-matrix-2026-03-08.md)
- [Ads + Time Series Stable-Surface Release Notes](/docs/benchmarks/ads-timeseries-release-notes-2026-03-08.md)
- [Ads + Time Series Stable-Surface Release PR Checklist](/docs/benchmarks/ads-timeseries-release-pr-checklist-2026-03-08.md)
- [Ads + Time Series Runtime Gate](/docs/benchmarks/ads-timeseries-runtime-gate.md)
- [Ads + Time Series Promotion Runbook](/docs/benchmarks/ads-timeseries-promotion-runbook-2026-03-08.md)
- [Ads Variance-Reduction Matrix Runbook](/docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md)
- [Ads Variance-Reduction Benchmark](/docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md)

## Run command

```bash
python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py \
  --promotion-mode dry_run
```

## Source of truth

The public machine-readable evidence for this snapshot is the committed promoted
baseline:

- accepted baseline: [accepted.json](/benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json)

The compare/promotion/gate bundles used during promotion are intentionally left
under local `tmp/` workdirs and are not committed to the public repo.

## Acceptance summary

The current benchmark evidence is fully green for the promoted stable subset:

- schema version: `nextstat.ads_timeseries_benchmark_result.v1`
- suite: `ads_timeseries_surface`
- host policy: `nextstat-bench`
- hostname: `nextstat-bench`
- build profile: `release`
- runs: `5`
- warmups: `1`
- case count: `9`
- python case count: `7`
- cli case count: `2`
- all cases ok: `true`
- compare status vs accepted baseline: `passed`
- dry-run gate status: `passed`
- slowest case: `cli_kalman_local_linear_trend_weekly_filter`
- slowest median runtime: `0.010056s`

## Case summary

| Case | Surface | Median (s) | Notes |
| --- | --- | ---: | --- |
| `python_beta_binomial_fit_from_counts` | `python` | `0.000018` | EB prior fit + posterior update |
| `python_delay_correction_fit_from_lag_buckets` | `python` | `0.000252` | lag-bucket fit + censoring correction |
| `python_cuped_adjust` | `python` | `0.000037` | one-covariate CUPED adjustment with selected covariates and typed provenance |
| `python_cure_adjust` | `python` | `0.000054` | multivariate CURE adjustment with typed provenance and ridge diagnostics |
| `python_response_curve_helpers` | `python` | `0.000016` | Hill + geometric adstock |
| `python_kalman_local_level_weekly_filter` | `python` | `0.000223` | weekly local-level seasonal filter |
| `python_kalman_local_linear_trend_weekly_filter` | `python` | `0.000192` | weekly local-linear-trend seasonal filter |
| `cli_kalman_local_level_weekly_filter` | `cli` | `0.008947` | release CLI JSON contract path |
| `cli_kalman_local_linear_trend_weekly_filter` | `cli` | `0.010056` | release CLI JSON contract path |

## Interpretation

This snapshot supports the current stable-surface claim for the ads + weekly
time-series surface:

- the published Rust/Python/CLI subset has reproducible benchmark provenance on
  `nextstat-bench`
- the shared CUPED/CURE Python contract cases are now part of the promoted
  machine-readable surface alongside the existing ads/time-series subset, with
  selected covariates, typed provenance, provenance validation, and solver diagnostics
- compare against the accepted baseline is green with `0` failed cases and `0`
  review cases
- the post-promotion dry-run gate is green with `requires_review = false`
- the two CLI weekly-contract paths stay in the low-10ms class on the bench
  host with a fresh release binary and current accepted baseline
- the Python helpers stay effectively free at operator scale relative to any
  real campaign-analysis workflow

In practical terms, the stable-surface claim is now backed by both local/CI
contract gates and current bench-host baseline-management evidence.

For direct `naive` vs `CUPED` vs `CURE` performance and variance-reduction
trade-offs on realistic ads matrices, use the dedicated variance-reduction
benchmark note rather than this stable-surface runtime snapshot.
