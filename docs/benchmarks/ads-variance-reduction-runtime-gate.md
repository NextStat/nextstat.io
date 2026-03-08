# Ads Variance Reduction Runtime Gate

This surface now has two distinct gates and they are intentionally not the same.

## 1. Local stable-surface gate

Purpose:

- prove the promoted Rust, Python, local-tool, and server-tool contracts still
  build and agree on a clean checkout
- stay safe for GitHub Actions and developer laptops without requiring
  `nextstat-bench`

Entry points:

- `scripts/benchmarks/ads_variance_reduction_stable_surface_gate.sh`
- `make ads-variance-reduction-stable-surface-gate`
- `.github/workflows/ads-variance-reduction-stable-surface.yml`

What it runs:

- `maturin develop --release`
- CUPED/CURE Rust tests and fixture tests
- Python ads variance-reduction tests
- focused local-tool tests for `nextstat_ads_cuped_adjust` and
  `nextstat_ads_cure_adjust`
- focused `ns-server` tool-runtime tests and local/server golden checks
- manifest / schema / example / doc / golden sync checks
- deterministic benchmark smoke for
  `scripts/benchmarks/bench_ads_variance_reduction_matrix.py`

Output artifact:

- `tmp/ads-variance-reduction-stable-surface/ads_variance_reduction_benchmark.json`

## 2. Bench-host runtime gate

Purpose:

- measure realistic `naive` vs `CUPED` vs `CURE` runtime and statistical
  behaviour on `nextstat-bench`
- compare against the accepted bench-host baseline

Entry points:

- `scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh`
- `make ads-variance-reduction-bench`
- `scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py`
- `make ads-variance-reduction-bench-gate`

Canonical accepted baseline:

- `benchmarks/artifacts/ads_variance_reduction_baselines/nextstat-bench/accepted.json`

Promotion / compare / gate companions:

- `scripts/benchmarks/compare_ads_variance_reduction_benchmark.py`
- `scripts/benchmarks/promote_ads_variance_reduction_benchmark_baseline.py`
- `scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py`

## Policy

- CI must rely on the local stable-surface gate.
- performance promotion decisions must rely on the `nextstat-bench` gate.
- neither gate substitutes for the other.
