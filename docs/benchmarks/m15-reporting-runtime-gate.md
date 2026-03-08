---
title: "M15 Reporting Runtime Gate"
status: stable
---

# M15 Reporting Runtime Gate

This runbook defines the reproducible benchmark gate for runtime-affecting
changes to the M15 reporting surface.

It is the canonical performance companion to:

- `docs/references/m15-reporting.md`

## When this gate is required

Run this gate when a change affects any of:

- `nextstat m15 assessment-table`
- `nextstat m15 map`
- `nextstat m15 mar`
- `nextstat m15 bundle`
- `validation-pack/render_validation_pack.sh --m15-config ...`
- deterministic artifact chaining or integrity verification for M15 outputs

Docs-only and acceptance-text-only changes do not require this gate.

## Artifact contract

The benchmark artifact is a machine-readable JSON report:

- Schema: `docs/schemas/benchmarks/m15_reporting_benchmark_result_v1.schema.json`
- Canonical example: `docs/specs/pharma/m15_reporting_benchmark_result_v1.example.json`
- Runner: `python3 scripts/benchmarks/bench_m15_reporting.py`
- Remote runner: `bash scripts/benchmarks/bench_m15_reporting_remote.sh`
- Stable-surface gate: `bash scripts/benchmarks/m15_reporting_stable_surface_gate.sh`
- Accepted baseline: `benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json`
- Compare schema: `docs/schemas/benchmarks/m15_reporting_benchmark_compare_report_v1.schema.json`
- Compare example: `docs/specs/pharma/m15_reporting_benchmark_compare_report_v1.example.json`
- Compare runner: `python3 scripts/benchmarks/compare_m15_reporting_benchmark.py`
- Dedicated workflow: `.github/workflows/m15-reporting-stable-surface.yml`

The report records, per case:

- assessment-table timing
- MAP timing
- MAR timing
- bundle timing
- JSON-only validation-pack timing with and without M15 enabled
- schema-version validation for emitted artifacts

## Fixture policy

The gate is deterministic and offline-first:

- frozen inputs come from committed example/fixture artifacts
- validation-pack runs use `--skip-pharma-validation`
- the M15 validation-pack case pre-seeds `pharma_validation.json` and must not trigger hidden Apex2 or pharma re-execution

## Local smoke run

Use this only for development of the harness itself:

```bash
cargo build -p ns-cli
python3 scripts/benchmarks/bench_m15_reporting.py \
  --nextstat-bin target/debug/nextstat \
  --smoke \
  --deterministic \
  --out bench_results/m15_reporting/summary.json
```

This smoke run is not promotion evidence.

## Canonical nextstat-bench run

Promotion evidence must come from `nextstat-bench` with a current M15-enabled
`release` binary built from the same snapshot:

```bash
make m15-reporting-stable-surface-gate

# or, for the remote/manual protocol:
bash scripts/benchmarks/bench_m15_reporting_remote.sh
```

Recommended preserved output layout:

- JSON: `tmp/m15_reporting_benchmark_<STAMP>/nextstat-bench/m15_reporting_benchmark.json`
- Markdown: `tmp/m15_reporting_benchmark_<STAMP>/nextstat-bench/m15_reporting_benchmark.md`

Then compare the fresh artifact against the accepted release baseline:

```bash
python3 scripts/benchmarks/compare_m15_reporting_benchmark.py \
  --baseline benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json \
  --current tmp/m15_reporting_benchmark_<STAMP>/nextstat-bench/m15_reporting_benchmark.json \
  --out tmp/m15_reporting_benchmark_<STAMP>/nextstat-bench/m15_reporting_compare.json
```

The automated stable-surface gate uses the stricter promotion path:

- actual host provenance must resolve to `hostname = nextstat-bench`
- the compare step runs with `--fail-on-review`
- CI/release promotion therefore requires `status = "passed"` rather than a merely reviewable outcome

The same gate is wired into:

- `.github/workflows/m15-reporting-stable-surface.yml`
- `.github/workflows/python-tests.yml`
- `.github/workflows/release.yml`

## Pass conditions

The gate passes only if:

- `bench_m15_reporting.py` exits zero
- `compare_m15_reporting_benchmark.py` exits zero
- every benchmark case has `status = "ok"`
- every validated artifact reports the expected `schema_version`
- the benchmark is run on `nextstat-bench`
- the comparison report records `hostname = nextstat-bench`
- promotion evidence uses a current M15-enabled `release` binary
- the automated promotion gate status is `passed`

For v1 this gate establishes reproducible release-grade baseline evidence for
the M15 reporting surface with an accepted nextstat-bench release baseline and
explicit compare policy.
