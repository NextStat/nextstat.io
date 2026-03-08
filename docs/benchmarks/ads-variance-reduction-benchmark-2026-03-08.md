# Ads Variance-Reduction Benchmark: 2026-03-08

**Date**: 2026-03-08  
**Host**: `nextstat-bench`  
**Primary artifact**: [accepted.json](/benchmarks/artifacts/ads_variance_reduction_baselines/nextstat-bench/accepted.json)
**Accepted baseline**: [accepted.json](/benchmarks/artifacts/ads_variance_reduction_baselines/nextstat-bench/accepted.json)

Local compare/gate/promotion outputs are intentionally ephemeral and stay under
`tmp/` during promotion rather than being committed to the public repo. The
ephemeral filenames are still `ads_variance_reduction_benchmark.json`,
`compare_report.json`, `gate_report.json`, and `promotion_report.json`.

## Purpose

This benchmark is the practical performance companion for the shared ads
variance-reduction layer. It complements the promoted ads/time-series gate by
answering the question that the stable-surface benchmark intentionally does not
answer:

- how `naive`, `CUPED`, and `CURE` compare on realistic ads scenarios
- where multivariate adjustment buys real variance reduction
- whether the ridge fallback path is actually exercised under collinearity

For the product-surface acceptance gate and CI-safe local verification path,
see:

- [ads-variance-reduction-stable-surface-acceptance-2026-03-09.md](/docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md)
- [ads-variance-reduction-runtime-gate.md](/docs/benchmarks/ads-variance-reduction-runtime-gate.md)

## Headline result

The `nextstat-bench` run is clean and useful:

- `schema_version`: `nextstat.ads_variance_reduction_benchmark_result.v1`
- `suite`: `ads_variance_reduction_matrix`
- `all_cases_ok=true`
- `case_count=12`
- `scenario_count=4`
- `method_count=3`
- `ridge_case_count=1`
- slowest case: `python_sparse_new_user_conversion_cure` at `0.248373s`
- compare against accepted baseline: `passed`
- gate status: `passed`
- promotion status: `promoted`

## Scenario summary

### `revenue_dense_signal`

- `n_per_arm=20000`
- `naive=0.009297s`, `CUPED=0.030328s`, `CURE=0.057242s`
- `CUPED/naive=3.262128x`, `CURE/naive=6.15704x`
- `CUPED SE=0.094313202763`, `CURE SE=0.092221430558`
- `CUPED variance ratio=0.16123239`, `CURE variance ratio=0.15415976`
- `CUPED effective sample multiplier=6.20222763x`
- `CURE effective sample multiplier=6.48677715x`
- `CURE solver=svd`

### `ratio_style_efficiency`

- `n_per_arm=24000`
- `naive=0.011319s`, `CUPED=0.039632s`, `CURE=0.163329s`
- `CUPED/naive=3.501369x`, `CURE/naive=14.429632x`
- `CUPED SE=0.000375170931`, `CURE SE=0.000288545045`
- `CUPED variance ratio=0.40359978`, `CURE variance ratio=0.23873672`
- `CUPED effective sample multiplier=2.4776946x`
- `CURE effective sample multiplier=4.18869651x`
- `CURE solver=svd`

### `sparse_new_user_conversion`

- `n_per_arm=50000`
- `naive=0.019323s`, `CUPED=0.085185s`, `CURE=0.248373s`
- `CUPED/naive=4.408477x`, `CURE/naive=12.853749x`
- `CUPED SE=0.001643556884`, `CURE SE=0.001535756408`
- `CUPED variance ratio=0.99527468`, `CURE variance ratio=0.86899691`
- `CUPED effective sample multiplier=1.00474763x`
- `CURE effective sample multiplier=1.15075216x`
- `CURE solver=svd`

### `collinear_account_history`

- `n_per_arm=18000`
- `naive=0.012861s`, `CUPED=0.037762s`, `CURE=0.067894s`
- `CUPED/naive=2.936164x`, `CURE/naive=5.279061x`
- `CUPED SE=0.088665029151`, `CURE SE=0.086232154469`
- `CUPED variance ratio=0.04952642`, `CURE variance ratio=0.0468458`
- `CUPED effective sample multiplier=20.19124438x`
- `CURE effective sample multiplier=21.34663056x`
- `CURE solver=ridge`, `ridge_lambda=107.365577843049`

## Interpretation

The numbers are in the right shape for a production ads surface:

- `CUPED` stays cheap enough to be operationally negligible.
- `CURE` is more expensive, but still comfortably sub-second on the slowest
  realistic Python case.
- Dense and ratio-style scenarios show real multivariate upside over the
  one-covariate baseline.
- Sparse new-user conversion remains hard; the benchmark keeps that visible
  instead of overfitting the story to only easy dense revenue cases.
- The collinearity stress case proves that ridge fallback is not theoretical.

In short: runtime cost goes up, but the cost is still small relative to real
experiment-analysis workflows, and the statistical upside is real where ads
workloads actually have multiple informative pre-treatment covariates.
