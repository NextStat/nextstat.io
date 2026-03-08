# Ads + Time Series Stable-Surface Support Matrix

**Date**: 2026-03-08  
**Status**: Executed support matrix  
**Scope**: ads-native observation, variance-reduction, and weekly state-space convenience surface

## Purpose

This document is the short operational matrix for the promoted ads + weekly
time-series subset.

It answers one narrow question:

- what is `stable` now
- what remains `research-grade`

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | public compatibility promise for the named ads + weekly subset |
| `research-grade` | versioned and tested, but still evolving without stable-surface promise |

## Stable Rust matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `ns_inference::BetaBinomialModel` | `stable` | empirical-Bayes beta-binomial prior fit + posterior update |
| `ns_inference::DelayCorrectionModel` | `stable` | single-rate exponential delay correction |
| `ns_inference::cuped_adjust(...)` | `stable` | shared one-covariate variance-reduction primitive; pre-treatment only |
| `ns_inference::cure_adjust(...)` | `stable` | shared multivariate regression-adjustment primitive with SVD/ridge guardrails |
| `ns_inference::hill` | `stable` | Hill saturation primitive |
| `ns_inference::adstock_geometric` | `stable` | geometric adstock primitive |
| `KalmanModel::local_level_weekly(...)` | `stable` | fixed weekly alias for `period=7` |
| `KalmanModel::local_linear_trend_weekly(...)` | `stable` | fixed weekly alias for `period=7` |
| `hierarchical_segment_lift_summary(...)` | `research-grade` | useful shrinkage helper, but outside the promoted narrow stable subset |
| generic `local_*_seasonal(period, ...)` builders | `research-grade` | documented and supported, but not widened by this promotion note |

## Stable CLI matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat timeseries kalman-filter` with `local_level_weekly` JSON | `stable` | fixed weekly JSON alias |
| `nextstat timeseries kalman-filter` with `local_linear_trend_weekly` JSON | `stable` | fixed weekly JSON alias |
| generic seasonal JSON forms | `research-grade` | available and documented, but outside this narrow promotion |

## Stable Python matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat.BetaBinomialModel` | `stable` | native top-level class export |
| `nextstat.DelayCorrectionModel` | `stable` | native top-level class export |
| `nextstat.ads.BetaBinomialModel` | `stable` | lazy convenience module |
| `nextstat.ads.DelayCorrectionModel` | `stable` | lazy convenience module |
| `nextstat.ads.cuped_adjust(...)` | `stable` | shared one-covariate CUPED wrapper with diagnostics, typed provenance, and fail-fast pre-treatment validation |
| `nextstat.ads.cure_adjust(...)` | `stable` | shared multivariate CURE wrapper with diagnostics, typed provenance, and ridge fallback reporting |
| `nextstat.ads.hill(...)` | `stable` | Hill saturation helper |
| `nextstat.ads.adstock_geometric(...)` | `stable` | geometric adstock helper |
| `nextstat.timeseries.local_level_weekly_model(...)` | `stable` | fixed weekly builder |
| `nextstat.timeseries.local_linear_trend_weekly_model(...)` | `stable` | fixed weekly builder |
| generic seasonal builders under `nextstat.timeseries` | `research-grade` | available and documented, but outside this narrow promotion note |

## Evidence and gate

The promoted stable subset is backed by:

- [Ads + Time Series Stable-Surface Acceptance](/docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08.md)
- [Ads + Time Series Stable-Surface Release Notes](/docs/benchmarks/ads-timeseries-release-notes-2026-03-08.md)
- [Ads + Time Series Runtime Gate](/docs/benchmarks/ads-timeseries-runtime-gate.md)
- accepted baseline:
  [accepted.json](/benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json)
- benchmark schema:
  [ads_timeseries_benchmark_result_v1.schema.json](/docs/schemas/benchmarks/ads_timeseries_benchmark_result_v1.schema.json)
- variance-reduction matrix runbook:
  [ads-variance-reduction-runbook-2026-03-08.md](/docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md)
- variance-reduction matrix benchmark note:
  [ads-variance-reduction-benchmark-2026-03-08.md](/docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md)
- compare schema:
  [ads_timeseries_benchmark_compare_report_v1.schema.json](/docs/schemas/benchmarks/ads_timeseries_benchmark_compare_report_v1.schema.json)
- promotion schema:
  [ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json](/docs/schemas/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json)
- gate schema:
  [ads_timeseries_benchmark_gate_report_v1.schema.json](/docs/schemas/benchmarks/ads_timeseries_benchmark_gate_report_v1.schema.json)

Operational gate:

- script:
  [ads_timeseries_stable_surface_gate.sh](/scripts/benchmarks/ads_timeseries_stable_surface_gate.sh)
- compare script:
  [compare_ads_timeseries_benchmark.py](/scripts/benchmarks/compare_ads_timeseries_benchmark.py)
- promotion script:
  [promote_ads_timeseries_benchmark_baseline.py](/scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py)
- one-shot gate:
  [run_ads_timeseries_benchmark_gate.py](/scripts/benchmarks/run_ads_timeseries_benchmark_gate.py)
- workflow:
  [ads-timeseries-stable-surface.yml](/.github/workflows/ads-timeseries-stable-surface.yml)
- make target:
  `make ads-timeseries-stable-surface-gate`

## Bottom line

The stable product promise is intentionally narrow:

- ads observation / response primitives are `stable`
- shared CUPED/CURE variance-reduction primitives are `stable`
- shared CUPED/CURE fixtures under `tests/fixtures/variance_reduction/` are part of the stable validation bar
- the benchmark contract for `python_cuped_adjust` / `python_cure_adjust` is `stable`, including selected covariates, typed provenance, provenance validation, and solver diagnostics
- the supplemental `naive` / `CUPED` / `CURE` matrix benchmark is part of the performance-evidence story for ads variance reduction, even though it is not itself the accepted runtime baseline gate
- fixed weekly state-space aliases are `stable`
- broader seasonal and segment-shrinkage helpers remain available but stay
  `research-grade` for this promotion wave
