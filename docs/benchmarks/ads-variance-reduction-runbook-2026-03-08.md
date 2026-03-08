# Ads Variance-Reduction Matrix Runbook

**Date**: 2026-03-08  
**Status**: reproducible benchmark runbook  
**Scope**: realistic `naive` vs `CUPED` vs `CURE` comparisons on the stable ads Rust/Python/tool surface

## Purpose

This runbook exists to answer a different question than the promoted
ads/time-series stable-surface gate.

For the local CI-safe stable-surface acceptance gate and product boundary, see:

- `docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md`
- `docs/benchmarks/ads-variance-reduction-runtime-gate.md`

The stable-surface gate proves that the shipped subset is reproducible and
regression-controlled. This benchmark runbook measures the practical trade-off
between:

- `naive` difference in means
- `CUPED` using the designated primary pre-treatment covariate
- `CURE` using the full pre-treatment covariate set

The benchmark matrix is designed around realistic ads workloads and sweeps the
dimensions that matter operationally:

- sample size (`n`)
- number of covariates (`p`)
- sparsity / new-user history
- collinearity and ridge fallback

## Source of truth

Benchmark inputs:

- scenario manifest:
  `tests/fixtures/variance_reduction_benchmark/scenario_matrix.json`

Machine-readable output:

- result schema:
  `docs/schemas/benchmarks/ads_variance_reduction_benchmark_result_v1.schema.json`
- result example:
  `docs/specs/benchmarks/ads_variance_reduction_benchmark_result_v1.example.json`
- compare schema:
  `docs/schemas/benchmarks/ads_variance_reduction_benchmark_compare_report_v1.schema.json`
- compare example:
  `docs/specs/benchmarks/ads_variance_reduction_benchmark_compare_report_v1.example.json`
- gate schema:
  `docs/schemas/benchmarks/ads_variance_reduction_benchmark_gate_report_v1.schema.json`
- gate example:
  `docs/specs/benchmarks/ads_variance_reduction_benchmark_gate_report_v1.example.json`
- promotion schema:
  `docs/schemas/benchmarks/ads_variance_reduction_benchmark_baseline_promotion_report_v1.schema.json`
- promotion example:
  `docs/specs/benchmarks/ads_variance_reduction_benchmark_baseline_promotion_report_v1.example.json`
- accepted baseline:
  `benchmarks/artifacts/ads_variance_reduction_baselines/nextstat-bench/accepted.json`

Operator entry points:

- local stable-surface gate:
  `bash scripts/benchmarks/ads_variance_reduction_stable_surface_gate.sh`
- local harness:
  `python3 scripts/benchmarks/bench_ads_variance_reduction_matrix.py`
- remote harness:
  `bash scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh`
- one-shot gate:
  `python3 scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py`
- make stable-surface gate target:
  `make ads-variance-reduction-stable-surface-gate`
- make target:
  `make ads-variance-reduction-bench`
- make gate target:
  `make ads-variance-reduction-bench-gate`

## Local smoke

For quick contract verification:

```bash
PYTHONPATH=bindings/ns-py/python \
  ./.venv/bin/python scripts/benchmarks/bench_ads_variance_reduction_matrix.py \
  --nextstat-bin target/release/nextstat \
  --scenario-manifest tests/fixtures/variance_reduction_benchmark/scenario_matrix.json \
  --out tmp/ads_variance_reduction_benchmark_smoke.json \
  --markdown-out tmp/ads_variance_reduction_benchmark_smoke.md \
  --smoke \
  --deterministic
```

## Canonical bench-host run

The canonical evidence path is `nextstat-bench`:

```bash
bash scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh
```

By default this path:

1. rsyncs the current working tree to `nextstat-bench`
2. builds the local Python bindings via `maturin develop --release`
3. builds a release `nextstat` CLI binary for the canonical `--nextstat-bin`
   path
4. runs the deterministic matrix harness against
   `tests/fixtures/variance_reduction_benchmark/scenario_matrix.json`
5. syncs the JSON + Markdown artifacts back into `tmp/`

Useful environment knobs:

- `BENCH_RUNS`
- `BENCH_WARMUPS`
- `BENCH_SMOKE=1`
- `BENCH_SKIP_BUILD=1`
  Reuse an existing remote venv + target directory instead of rebuilding.
- `BENCH_SCENARIO_MANIFEST`
- `BENCH_REMOTE_REPO`, `BENCH_REMOTE_VENV`, `BENCH_REMOTE_TARGET`

## Compare / gate / promotion

If the goal is a complete acceptance bundle instead of a raw benchmark run:

```bash
python3 scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py \
  --current tmp/ads_variance_reduction_benchmark_<stamp>/nextstat-bench/ads_variance_reduction_benchmark.json \
  --promotion-mode dry_run
```

This emits:

- `compare_report.json`
- `gate_report.json`
- `promotion_report.json`

Use `--allow-review` only for intentional, investigated baseline re-pins, and
only if a post-promotion rerun returns compare status `passed`.

## What the artifact records

Each case records:

- realistic scenario metadata (`n`, `p`, sparsity, collinearity)
- realized primary-covariate correlation with the outcome
- realized max pairwise covariate correlation
- method-level runtime for `naive`, `CUPED`, and `CURE`
- `r_squared`, `variance_reduction_factor`, and `effective_sample_multiplier`
- solver choice, regression rank, condition number, and `ridge_lambda`
- whether `CURE` beat `CUPED` on variance reduction

Top-level derived metrics include:

- slowest per-method case across the matrix
- ridge-fallback case count / fraction
- ridge-fallback scenario ids
- per-scenario runtime / standard-error summaries

## Interpretation rules

Read this benchmark as a product-performance companion for the stable variance-
reduction layer, not as a public baseline gate.

- `naive` is the unadjusted planning and analysis baseline
- `CUPED` is the one-covariate baseline using the nominated primary pre-period
  covariate
- `CURE` is the full multivariate adjustment path

The benchmark is successful when it shows all of the following:

- `CURE` stays operationally cheap relative to realistic ads workflows
- `CURE` usually improves variance reduction over `CUPED` once `p > 1`
- sparse new-user scenarios are explicitly represented, even when variance
  reduction is modest
- the collinearity stress case actually exercises ridge fallback
