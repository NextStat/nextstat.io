# Ads + Time Series Promotion Runbook

**Date**: 2026-03-08  
**Status**: promotion runbook  
**Scope**: `nextstat-bench` evidence for the ads + weekly time-series stable surface, including CUPED/CURE Python contract cases

## Purpose

This runbook defines the canonical promotion workflow for the ads + weekly
time-series stable subset after the local/CI gate is green.

It answers one operational question:

- how do maintainers produce and review the benchmark artifact for this surface
  on `nextstat-bench`?

## Ownership

Promotion requires three explicit roles:

- release owner: decides whether any runtime wording changes in the release PR
- bench operator: runs the canonical `nextstat-bench` benchmark artifact
- reviewer: confirms the artifact matches the current code snapshot and support matrix

## When this runbook is required

Run this promotion workflow when a change affects any of:

- `BetaBinomialModel`
- `DelayCorrectionModel`
- `cuped_adjust(...)`
- `cure_adjust(...)`
- `hill(...)`
- `adstock_geometric(...)`
- weekly Kalman constructors or JSON aliases
- the benchmark harness or its schema
- CUPED/CURE benchmark identity payloads such as selected covariates, typed provenance, or solver diagnostics
- public runtime wording tied to this surface

Docs-only edits that do not change behavior or wording do not require a new
bench-host run.

## Artifact contract

The benchmark artifact is the machine-readable JSON report:

- Schema:
  `docs/schemas/benchmarks/ads_timeseries_benchmark_result_v1.schema.json`
- Example:
  `docs/specs/benchmarks/ads_timeseries_benchmark_result_v1.example.json`
- Accepted baseline:
  `benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json`
- Runner:
  `python3 scripts/benchmarks/bench_ads_timeseries_surface.py`
- Remote runner:
  `bash scripts/benchmarks/bench_ads_timeseries_surface_remote.sh`
- Compare:
  `python3 scripts/benchmarks/compare_ads_timeseries_benchmark.py`
- Promote:
  `python3 scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py`
- One-shot gate:
  `python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py`

## Precondition

Before touching `nextstat-bench`, the local/CI gate must already be green:

```bash
make ads-timeseries-stable-surface-gate
```

This ensures the bench-host run is promotion evidence, not a substitute for the
correctness gate.

## Canonical nextstat-bench run

Recommended command:

```bash
BENCH_HOST=nextstat-bench \
bash scripts/benchmarks/bench_ads_timeseries_surface_remote.sh
```

If the local SSH alias is not configured correctly, pass explicit overrides:

```bash
BENCH_HOST=<host> \
BENCH_SSH_USER=<user> \
BENCH_SSH_PORT=<port> \
BENCH_SSH_KEY=/path/to/key \
bash scripts/benchmarks/bench_ads_timeseries_surface_remote.sh
```

Recommended preserved output layout:

- local synced artifact dir:
  `tmp/ads_timeseries_benchmark_<timestamp>/<host>/`
- report JSON:
  `ads_timeseries_benchmark.json`
- report Markdown:
  `ads_timeseries_benchmark.md`

## Recommended operator flow

Dry-run the canonical gate first:

```bash
python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py \
  --promotion-mode dry_run
```

If the benchmark case identity changed intentionally but the case set stayed
fixed, review the compare report and use the reviewed promotion path:

```bash
python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py \
  --promotion-mode apply \
  --allow-review
```

If the artifact is already present locally, compare directly:

```bash
python3 scripts/benchmarks/compare_ads_timeseries_benchmark.py \
  --baseline benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json \
  --current tmp/ads_timeseries_benchmark_<timestamp>/nextstat-bench/ads_timeseries_benchmark.json \
  --out tmp/ads_timeseries_benchmark_<timestamp>/nextstat-bench/compare_report.json
```

Apply promotion only after review:

```bash
python3 scripts/benchmarks/run_ads_timeseries_benchmark_gate.py \
  --current tmp/ads_timeseries_benchmark_<timestamp>/nextstat-bench/ads_timeseries_benchmark.json \
  --promotion-mode apply
```

When the support matrix is intentionally widened with new required benchmark
cases, allow the first baseline promotion explicitly:

```bash
python3 scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py \
  --accepted benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json \
  --current tmp/ads_timeseries_benchmark_<timestamp>/nextstat-bench/ads_timeseries_benchmark.json \
  --compare-report tmp/ads_timeseries_benchmark_<timestamp>/nextstat-bench/compare_report.json \
  --report tmp/ads_timeseries_benchmark_<timestamp>/nextstat-bench/promotion_report.json \
  --allow-case-set-change
```

## Pass conditions

Promotion evidence is acceptable only if all of the following are true:

- the artifact validates against the benchmark schema
- every benchmark case has `status = "ok"`
- `derived.all_cases_ok == true`
- the exact promoted 9-case surface is present, including
  `python_cuped_adjust` and `python_cure_adjust`
- the artifact came from `nextstat-bench`
- compare status against the accepted baseline is `passed`, or the reviewed
  exception is explicitly approved with `--allow-review` for intentional
  identity widening or `--allow-case-set-change` for intentional case-set widening
- any reviewed promotion is followed by a post-promotion rerun whose compare
  status returns to `passed`
- any runtime wording in the release note is scoped to the linked artifact

## Review checklist

The reviewer must confirm:

- the artifact came from `nextstat-bench`
- the artifact path is linked from the PR if runtime wording changed
- the support matrix does not widen the claim beyond the promoted narrow subset
- no release note implies performance guarantees beyond the linked artifact

## Failure handling

If the bench-host run fails:

- do not add new runtime wording for this surface
- keep the stable API subset as-is
- record the failed artifact path in the PR for traceability
- open or update follow-up tasks before promotion proceeds
