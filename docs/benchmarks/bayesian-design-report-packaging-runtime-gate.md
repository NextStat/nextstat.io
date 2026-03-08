---
title: "Bayesian Design Report Packaging Runtime Gate"
status: stable
---

# Bayesian Design Report Packaging Runtime Gate

This runbook defines the reproducible benchmark gate for runtime-affecting
changes to frozen Bayesian design report packaging.

It is the canonical companion to the acceptance policy in:

- `docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md`

## When this gate is required

Run this gate when a change affects any of:

- `write_beta_binomial_design_report_bundle(...)`
- `write_normal_normal_design_report_bundle(...)`
- `nextstat.audit.write_bundle(..., deterministic=True)` behavior as consumed by the Bayesian report surface
- bundle layout population, Markdown rendering, or manifest rewrite behavior
- frozen-report packaging latency or emitted bundle size

Docs-only, internal-only, and acceptance-text-only changes do not require this gate.

## Artifact contract

The benchmark artifact is a machine-readable JSON report:

- Schema: `docs/schemas/benchmarks/bayesian_design_report_bundle_benchmark_result_v1.schema.json`
- Canonical example: `docs/specs/pharma/bayesian_design_report_bundle_benchmark_result_v1.example.json`
- Budget manifest: `scripts/bayesian_design_report_bundle_performance_budget_v1.json`
- Budget schema: `docs/schemas/benchmarks/nextstat_bayesian_design_report_bundle_performance_budget_v1.schema.json`
- Budget helper: `python3 scripts/bayesian_design_report_bundle_performance_budget.py --format json`
- Runner: `python3 scripts/benchmarks/bench_bayesian_design_report_bundle.py`

The report records, per case:

- bundle write timing
- manifest regeneration timing
- total bundle bytes
- manifest bytes
- validation of deterministic bundle invariants
- pass/fail against committed performance budgets

## Fixture policy

The gate is intentionally deterministic and offline-first:

- frozen reports are built from local `nextstat.bayes_design` builders using fixed specs, observed data, and prior-sensitivity campaigns
- the timed path measures packaging from a frozen on-disk report artifact
- no external network access is required

## Local smoke run

Use this for fast validation while developing the gate:

```bash
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  scripts/benchmarks/bench_bayesian_design_report_bundle.py \
  --smoke \
  --deterministic \
  --out bench_results/bayesian_design_report_bundle/summary.json
```

This local smoke run is for development only. It is not promotion evidence.

## Canonical nextstat-bench run

On `nextstat-bench`, run the full benchmark gate with the committed release repeat policy:

```bash
ssh nextstat-bench
cd /path/to/nextstat.io
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
  scripts/benchmarks/bench_bayesian_design_report_bundle.py \
  --deterministic \
  --out bench_results/bayesian_design_report_bundle/summary.json
```

Recommended preserved output layout:

- summary: `bench_results/bayesian_design_report_bundle/summary.json`
- workdir: `bench_results/bayesian_design_report_bundle/work/`

## Pass conditions

The gate passes only if:

- `bench_bayesian_design_report_bundle.py` exits zero
- every benchmark case has `status = "ok"`
- every case reports `created_unix_ms_zero = true`
- every case reports `summary_deterministic = true`
- every case reports `required_artifacts_present = true`
- every case stays within the committed duration and size budgets

For v1, the gate is a fixed-budget packaging runtime gate with canonical
promotion evidence from `nextstat-bench`. If a future release wants to make a
stronger host-specific product claim, that must ship as a separate benchmark
snapshot and acceptance addendum.
