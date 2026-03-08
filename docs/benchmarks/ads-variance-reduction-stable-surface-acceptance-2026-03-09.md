# Ads Variance Reduction Stable-Surface Acceptance: 2026-03-09

## Accepted surface

The promoted ads variance-reduction surface now includes all four layers below:

- Shared Rust primitive:
  - `ns_inference::cuped_adjust(...)`
  - `ns_inference::cure_adjust(...)`
- Stable Python helpers:
  - `nextstat.ads.cuped_adjust(...)`
  - `nextstat.ads.cure_adjust(...)`
- Stable local tool surface:
  - `nextstat.tools.execute_tool("nextstat_ads_cuped_adjust", ...)`
  - `nextstat.tools.execute_tool("nextstat_ads_cure_adjust", ...)`
- Stable server-safe tool surface:
  - `GET /v1/tools/schema`
  - `POST /v1/tools/execute` for `nextstat_ads_cuped_adjust`
  - `POST /v1/tools/execute` for `nextstat_ads_cure_adjust`

Architectural rule:

- `CUPED = CURE with one covariate`
- only pre-treatment covariates are allowed
- provenance validation is fail-fast
- ill-conditioned multivariate designs fall back to ridge and report `ridge_lambda`

## Acceptance evidence

Stable-surface evidence is split deliberately:

- local CI-safe gate:
  - `scripts/benchmarks/ads_variance_reduction_stable_surface_gate.sh`
  - `make ads-variance-reduction-stable-surface-gate`
  - `.github/workflows/ads-variance-reduction-stable-surface.yml`
- remote runtime benchmark gate on `nextstat-bench`:
  - `scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh`
  - `scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py`
  - `benchmarks/artifacts/ads_variance_reduction_baselines/nextstat-bench/accepted.json`

## Required checks

The accepted gate requires:

- Rust inference tests for CUPED/CURE and fixture-grade validation
- Python public-surface tests for ads helpers
- local tool contract tests for `nextstat_ads_cuped_adjust` / `nextstat_ads_cure_adjust`
- server-tool Rust tests for execution, capability metadata, and local/server golden parity
- benchmark smoke validation against `ads_variance_reduction_benchmark_result_v1`
- tool manifest / schema / example / golden / reference-doc sync checks

## Canonical references

- `docs/benchmarks/ads-variance-reduction-runtime-gate.md`
- `docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md`
- `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`
- `docs/references/python-api.md`
- `docs/references/rust-api.md`
- `docs/references/tool-api.md`
- `docs/references/server-api.md`
