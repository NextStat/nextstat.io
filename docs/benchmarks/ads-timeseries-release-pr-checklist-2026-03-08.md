# Ads + Time Series Stable-Surface Release PR Checklist

Use this checklist for PRs that affect the promoted ads + weekly time-series
stable subset.

## Contract and docs

- [ ] support matrix is present:
  - `docs/benchmarks/ads-timeseries-support-matrix-2026-03-08.md`
- [ ] acceptance policy is present:
  - `docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08.md`
- [ ] release note is present:
  - `docs/benchmarks/ads-timeseries-release-notes-2026-03-08.md`
- [ ] release PR checklist is present:
  - `docs/benchmarks/ads-timeseries-release-pr-checklist-2026-03-08.md`
- [ ] promotion runbook is present:
  - `docs/benchmarks/ads-timeseries-promotion-runbook-2026-03-08.md`
- [ ] runtime gate note is present:
  - `docs/benchmarks/ads-timeseries-runtime-gate.md`
- [ ] variance-reduction matrix runbook is present:
  - `docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md`
- [ ] variance-reduction benchmark note is present:
  - `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`
- [ ] variance-reduction stable-surface acceptance is present:
  - `docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md`
- [ ] variance-reduction runtime gate note is present:
  - `docs/benchmarks/ads-variance-reduction-runtime-gate.md`
- [ ] benchmark schema is present:
  - `docs/schemas/benchmarks/ads_timeseries_benchmark_result_v1.schema.json`
- [ ] benchmark example is present:
  - `docs/specs/benchmarks/ads_timeseries_benchmark_result_v1.example.json`
- [ ] variance-reduction benchmark schemas and examples are present:
  - `docs/schemas/benchmarks/ads_variance_reduction_benchmark_result_v1.schema.json`
  - `docs/specs/benchmarks/ads_variance_reduction_benchmark_result_v1.example.json`
  - `docs/schemas/benchmarks/ads_variance_reduction_benchmark_compare_report_v1.schema.json`
  - `docs/specs/benchmarks/ads_variance_reduction_benchmark_compare_report_v1.example.json`
  - `docs/schemas/benchmarks/ads_variance_reduction_benchmark_gate_report_v1.schema.json`
  - `docs/specs/benchmarks/ads_variance_reduction_benchmark_gate_report_v1.example.json`
  - `docs/schemas/benchmarks/ads_variance_reduction_benchmark_baseline_promotion_report_v1.schema.json`
  - `docs/specs/benchmarks/ads_variance_reduction_benchmark_baseline_promotion_report_v1.example.json`
- [ ] compare schema and example are present:
  - `docs/schemas/benchmarks/ads_timeseries_benchmark_compare_report_v1.schema.json`
  - `docs/specs/benchmarks/ads_timeseries_benchmark_compare_report_v1.example.json`
- [ ] promotion schema and example are present:
  - `docs/schemas/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json`
  - `docs/specs/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.example.json`
- [ ] gate schema and example are present:
  - `docs/schemas/benchmarks/ads_timeseries_benchmark_gate_report_v1.schema.json`
  - `docs/specs/benchmarks/ads_timeseries_benchmark_gate_report_v1.example.json`
- [ ] accepted baseline is present:
  - `benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json`
- [ ] CLI / Python / Rust references reflect the promoted subset

## Verification

- [ ] local gate passes:
  - `make ads-timeseries-stable-surface-gate`
- [ ] Rust ads tests are green
- [ ] Rust variance-reduction tests are green
- [ ] committed variance-reduction fixture matrix is present and green:
  - `tests/fixtures/variance_reduction/`
  - `cargo test -p ns-inference --test variance_reduction_fixtures`
- [ ] Rust weekly-builder tests are green
- [ ] CLI weekly contract tests are green
- [ ] Python ads / variance-reduction / weekly builder smoke tests are green
- [ ] Python provenance-aware fixture parity is green:
  - `tests/python/test_ads_variance_reduction_fixtures.py`
- [ ] provenance fail-fast coverage is green for post-treatment / unknown covariates
- [ ] benchmark smoke tests are green
- [ ] benchmark CUPED/CURE cases record selected covariates, typed provenance, and validation diagnostics
- [ ] local tool / server-tool CUPED+CURE contract tests are green
- [ ] variance-reduction matrix benchmark smoke is green
- [ ] collinearity stress case actually exercises ridge fallback in the matrix benchmark

## Automation

- [ ] gate script is present:
  - `scripts/benchmarks/ads_timeseries_stable_surface_gate.sh`
- [ ] benchmark harness is present:
  - `scripts/benchmarks/bench_ads_timeseries_surface.py`
- [ ] remote benchmark runner is present:
  - `scripts/benchmarks/bench_ads_timeseries_surface_remote.sh`
- [ ] variance-reduction benchmark harnesses are present:
  - `scripts/benchmarks/bench_ads_variance_reduction_matrix.py`
  - `scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh`
- [ ] variance-reduction stable-surface gate is present:
  - `scripts/benchmarks/ads_variance_reduction_stable_surface_gate.sh`
- [ ] variance-reduction compare / promote / gate scripts are present:
  - `scripts/benchmarks/compare_ads_variance_reduction_benchmark.py`
  - `scripts/benchmarks/promote_ads_variance_reduction_benchmark_baseline.py`
  - `scripts/benchmarks/run_ads_variance_reduction_benchmark_gate.py`
- [ ] compare / promote / gate scripts are present:
  - `scripts/benchmarks/compare_ads_timeseries_benchmark.py`
  - `scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py`
  - `scripts/benchmarks/run_ads_timeseries_benchmark_gate.py`
- [ ] dedicated workflow is present:
  - `.github/workflows/ads-timeseries-stable-surface.yml`
- [ ] dedicated variance-reduction workflow is present:
  - `.github/workflows/ads-variance-reduction-stable-surface.yml`
- [ ] Makefile target is present:
  - `make ads-timeseries-stable-surface-gate`
- [ ] variance-reduction Makefile target is present:
  - `make ads-variance-reduction-stable-surface-gate`

## Promotion evidence

- [ ] if the PR changes runtime wording or benchmark claims, a fresh `nextstat-bench` artifact exists
- [ ] if the PR changes CUPED/CURE performance or solver/collinearity wording, a fresh `nextstat-bench` variance-reduction matrix artifact exists
- [ ] compare status against the accepted baseline is `passed`
- [ ] if benchmark case identity changed intentionally, reviewed promotion used `--allow-review` and a post-promotion rerun returns compare status `passed`
- [ ] the `nextstat-bench` artifact is linked from the PR
- [ ] no blanket runtime claim is made without the linked artifact
- [ ] docs-only and test-only changes are not blocked on a new `nextstat-bench` run
