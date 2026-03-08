# Ads + Time Series Runtime Gate

**Date**: 2026-03-08  
**Status**: runtime baseline gate  
**Scope**: release-grade `nextstat-bench` evidence for the ads + weekly time-series stable surface

## Purpose

This gate is the release-grade runtime path for the promoted ads + weekly
time-series subset.

It defines:

- how to compare a fresh `nextstat-bench` artifact against the accepted baseline
- how to dry-run or apply baseline promotion
- which machine-readable reports are the source of truth for that decision
- how provenance-aware CUPED/CURE benchmark identity changes are reviewed and promoted

Questions about the practical runtime and variance-reduction trade-off between
`naive`, `CUPED`, and `CURE` are answered by the separate realistic matrix
benchmark:

- `docs/benchmarks/ads-variance-reduction-runbook-2026-03-08.md`
- `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`

## Stable evidence surface

Benchmark result contract:

- schema: `docs/schemas/benchmarks/ads_timeseries_benchmark_result_v1.schema.json`
- example: `docs/specs/benchmarks/ads_timeseries_benchmark_result_v1.example.json`
- accepted baseline:
  `benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json`

The promoted benchmark result contract pins the full 9-case runtime surface,
including CUPED/CURE detail payloads with selected covariates, typed
pre-treatment provenance, provenance validation, and solver diagnostics.

Compare / promotion contracts:

- compare schema:
  `docs/schemas/benchmarks/ads_timeseries_benchmark_compare_report_v1.schema.json`
- compare example:
  `docs/specs/benchmarks/ads_timeseries_benchmark_compare_report_v1.example.json`
- promotion schema:
  `docs/schemas/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json`
- promotion example:
  `docs/specs/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.example.json`
- gate schema:
  `docs/schemas/benchmarks/ads_timeseries_benchmark_gate_report_v1.schema.json`
- gate example:
  `docs/specs/benchmarks/ads_timeseries_benchmark_gate_report_v1.example.json`

Operator entry points:

- remote runner:
  `bash scripts/benchmarks/bench_ads_timeseries_surface_remote.sh`
- compare:
  `python3 scripts/benchmarks/compare_ads_timeseries_benchmark.py`
- promote:
  `python3 scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py`
- one-shot gate:
  `python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py`

## Preconditions

Before touching `nextstat-bench`, the local/CI contract gate must already be
green:

```bash
make ads-timeseries-stable-surface-gate
```

This runtime gate is promotion evidence, not a replacement for contract
verification.

## Canonical dry-run gate

Recommended operator path:

```bash
make ads-timeseries-runtime-gate
```

Equivalent direct command:

```bash
python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py \
  --promotion-mode dry_run
```

This path:

1. runs the canonical remote runner on `nextstat-bench`
2. materializes a fresh benchmark artifact
3. compares it against `benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json`
4. emits a dry-run promotion report without mutating the baseline

## Compare-only path

If a fresh artifact already exists, compare it directly:

```bash
python3 scripts/benchmarks/compare_ads_timeseries_benchmark.py \
  --baseline benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json \
  --current tmp/ads_timeseries_benchmark_<STAMP>/nextstat-bench/ads_timeseries_benchmark.json \
  --out tmp/ads_timeseries_benchmark_<STAMP>/nextstat-bench/compare_report.json
```

To make review states fail CI/operator automation, add:

```bash
--fail-on-review
```

## Promotion path

After a reviewed dry-run, apply promotion explicitly:

```bash
python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py \
  --current tmp/ads_timeseries_benchmark_<STAMP>/nextstat-bench/ads_timeseries_benchmark.json \
  --promotion-mode apply
```

Direct promotion is also available:

```bash
python3 scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py \
  --accepted benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json \
  --current tmp/ads_timeseries_benchmark_<STAMP>/nextstat-bench/ads_timeseries_benchmark.json \
  --compare-report tmp/ads_timeseries_benchmark_<STAMP>/nextstat-bench/compare_report.json \
  --report tmp/ads_timeseries_benchmark_<STAMP>/nextstat-bench/promotion_report.json
```

Review states are blocked unless the operator opts in with:

```bash
--allow-review
```

This reviewed path is required when the benchmark case identity changes
intentionally, for example when CUPED/CURE detail payloads widen to include new
stable diagnostics without changing the case set.

If the support matrix intentionally adds new required benchmark cases, the first
baseline promotion must opt in explicitly:

```bash
--allow-case-set-change
```

## Pass conditions

The runtime gate is green only if all of the following are true:

- the fresh artifact validates against the result schema
- `host_policy == nextstat-bench`
- `hostname == nextstat-bench`
- `build_profile == release`
- `smoke == false`
- `deterministic == true`
- `protocol.runs == 5`
- `protocol.warmups == 1`
- the exact promoted 9-case surface is present, including
  `python_cuped_adjust` and `python_cure_adjust`
- the case set matches the accepted baseline exactly, except for the explicit
  one-time widening path via direct promotion with `--allow-case-set-change`
- CLI case medians do not exceed the compare policy thresholds
- compare status is `passed`, or `review` only when explicitly accepted

After any reviewed promotion, rerun the dry-run gate once more. The post-
promotion compare must return `passed` against the newly accepted baseline.

## Baseline history

Accepted baseline history lives under:

```text
benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/history/
```

Each successful promotion archives:

- the previous accepted baseline
- the newly promoted accepted snapshot

## Output reports

The one-shot gate emits:

- benchmark artifact: `ads_timeseries_benchmark.json`
- compare report: `compare_report.json`
- promotion report: `promotion_report.json`
- gate report: `gate_report.json`

These reports are the machine-readable audit trail for release promotion.
