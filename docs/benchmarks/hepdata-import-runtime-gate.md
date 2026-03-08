---
title: "HEPData Import Runtime Gate"
status: stable
---

# HEPData Import Runtime Gate

This runbook defines the reproducible benchmark gate for runtime-affecting
changes to `nextstat import hepdata`.

It is the canonical companion to the acceptance policy in:

- `docs/specs/hep/hepdata_import_acceptance_v1.md`

The current published frozen benchmark evidence is:

- `docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md`
- `docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md`
- `docs/benchmarks/hepdata-import-release-notes-2026-03-08.md`
- `docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md`
- `docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md`

## When this gate is required

Run this gate on `nextstat-bench` when a change affects any of:

- HEPData download flow
- cache behavior
- archive extraction / nested extraction
- patch materialization
- output lockfile/provenance generation coupled to runtime behavior
- downstream fit/runtime behavior of imported workspaces

Docs-only and test-only changes do not require this gate.

## Artifact contract

The benchmark artifact is a machine-readable JSON report:

- Schema: `docs/schemas/benchmarks/hepdata_import_benchmark_result_v1.schema.json`
- Canonical example: `docs/specs/hepdata_import_benchmark_result_v1.example.json`
- Accepted baseline: `benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json`
- Runner: `python scripts/benchmarks/bench_hepdata_import.py`
- Remote runner: `bash scripts/benchmarks/bench_hepdata_import_remote.sh`
- Drift comparator: `python scripts/benchmarks/compare_hepdata_import_benchmark.py`
- Baseline promotion helper: `python scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py`
- One-shot operator gate: `python scripts/benchmarks/run_hepdata_import_benchmark_gate.py`
- Comparator report schema: `docs/schemas/benchmarks/hepdata_import_benchmark_compare_report_v1.schema.json`
- Comparator example: `docs/specs/hepdata_import_benchmark_compare_report_v1.example.json`
- Promotion report schema: `docs/schemas/benchmarks/hepdata_import_benchmark_baseline_promotion_report_v1.schema.json`
- Promotion example: `docs/specs/hepdata_import_benchmark_baseline_promotion_report_v1.example.json`
- Gate report schema: `docs/schemas/benchmarks/hepdata_import_benchmark_gate_report_v1.schema.json`
- Gate report example: `docs/specs/hepdata_import_benchmark_gate_report_v1.example.json`

The report records:

- the exact `nextstat` command used
- environment metadata
- stage-level timing breakdown (`discovery_s`, `download_s`, `extract_s`, `materialize_s`, `fit_s`, `import_total_s`)
- curated catalog timing
- direct DOI patch-catalog timing (cached)
- curated materialization timing (offline cached fixture)
- direct DOI materialization timing (local network fixture)
- optional downstream `nextstat fit` smoke timing on materialized workspaces

For stable identity, the benchmark artifact records the logical HEPData DOI for
the direct network case. The ephemeral local fixture transport URL
(`127.0.0.1:<port>/download`) is not part of the dataset identity contract.

Stage breakdowns come from the versioned HEPData import summary itself. For
the direct DOI materialization benchmark, the `discovery_s` / `extract_s`
values include the explicit preflight patch-discovery call used to choose the
benchmark patch in a reproducible way.

## Fixture policy

The gate is intentionally offline-first and deterministic:

- fixture archives come from checked-in HEPData cache artifacts under `tests/hepdata/_cache/`
- direct DOI network mode is exercised through a local HTTP server serving the checked-in archive
- no external HEPData network availability is required for the benchmark itself

## Local smoke run

Use this for quick validation while developing the gate:

```bash
python3 scripts/benchmarks/bench_hepdata_import.py \
  --smoke \
  --deterministic \
  --out bench_results/hepdata_import_benchmark/summary.json
```

## Canonical nextstat-bench run

On `nextstat-bench`, run:

```bash
cd /path/to/nextstat.io
python3 scripts/benchmarks/bench_hepdata_import.py \
  --deterministic \
  --out bench_results/hepdata_import_benchmark/summary.json
```

Canonical automated path:

```bash
BENCH_SSH_USER=<user> bash scripts/benchmarks/bench_hepdata_import_remote.sh
```

Use `BENCH_SSH_USER`, `BENCH_SSH_PORT`, and `BENCH_SSH_KEY` when the local
`nextstat-bench` ssh alias is not already configured with the correct
credentials.

Recommended preserved output layout:

- summary: `bench_results/hepdata_import_benchmark/summary.json`
- workdir: `bench_results/hepdata_import_benchmark/work/`
- compare report: `bench_results/hepdata_import_benchmark/compare_report.json`
- promotion report: `bench_results/hepdata_import_benchmark/promotion_report.json`
- gate report: `bench_results/hepdata_import_benchmark/gate_report.json`

## Canonical one-shot operator flow

For `nextstat-bench`, the preferred operator path is the one-shot gate:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py \
  --promotion-mode dry_run
```

Single-line form:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run
```

This command:

- runs the canonical remote runner (`bench_hepdata_import_remote.sh`) by default
- compares the fetched artifact against the accepted baseline
- optionally runs baseline promotion semantics in `dry_run` or `apply` mode
- writes machine-readable compare / promotion / gate reports next to the artifact

When a benchmark artifact already exists locally, reuse it explicitly:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py \
  --current bench_results/hepdata_import_benchmark/summary.json \
  --promotion-mode dry_run
```

Promotion mode semantics:

- `--promotion-mode none`: benchmark + compare only
- `--promotion-mode dry_run`: benchmark + compare + promotion plan without mutating `accepted.json`
- `--promotion-mode apply`: benchmark + compare + actual promotion via the promotion helper

## Baseline compare gate

After a `nextstat-bench` run, compare the fresh artifact against the accepted
baseline:

```bash
python3 scripts/benchmarks/compare_hepdata_import_benchmark.py \
  --current bench_results/hepdata_import_benchmark/summary.json \
  --out bench_results/hepdata_import_benchmark/compare_report.json
```

This remains the low-level comparator contract used by the one-shot gate.

Comparator policy (v1):

- hard fail on missing/invalid benchmark artifacts, case-set drift, host-policy mismatch, or correctness drift in benchmark case metadata
- hard fail when `import_total_s` exceeds the accepted baseline by more than `1.35x`
- hard fail when `fit_s` exceeds the accepted baseline by more than `1.25x`
- mark `review` when `import_total_s` exceeds `1.15x`, `fit_s` exceeds `1.10x`, or any stage metric exceeds `1.50x`
- ignore ratio gates for very small baselines (`import_total_s < 0.01s`, `fit_s < 1.0s`, stage metric `< 0.05s`) to avoid timer-noise regressions

`review` is not an automatic failure by default, but it is a release-review
blocker until the regression is explained or the accepted baseline is
intentionally refreshed.

## Baseline promotion workflow

When the fresh `nextstat-bench` artifact is intentionally accepted as the new
baseline, promote it through the helper instead of copying files by hand:

```bash
python3 scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py \
  --current bench_results/hepdata_import_benchmark/summary.json \
  --compare-report bench_results/hepdata_import_benchmark/compare_report.json \
  --report bench_results/hepdata_import_benchmark/promotion_report.json
```

This remains the low-level promotion contract used by the one-shot gate when
`--promotion-mode apply` is selected.

Promotion semantics:

- the helper reruns the comparator against `accepted.json`
- promotion is blocked on compare `status = "failed"`
- promotion is blocked on compare `status = "review"` unless `--allow-review` is supplied
- the previous accepted baseline is archived under `benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/history/`
- the promoted candidate is also archived under that history directory
- `accepted.json` is updated only after the compare gate has passed
- `--dry-run` validates the promotion plan and writes a machine-readable report without changing the baseline

## Downstream HEP benchmark follow-up

If the change plausibly affects the runtime behavior of the produced workspaces
beyond import itself, pair the import gate with the standard HEP harness:

```bash
cd benchmarks/nextstat-public-benchmarks
python3 suites/hep/run.py --deterministic --out out/hep_simple_nll.json
python3 suites/hep/run.py --deterministic --fit --fit-repeat 3 --out out/hep_simple_nll_fit.json
```

This keeps import/runtime evidence aligned with the public HEP benchmark suite.

## Pass conditions

The gate passes only if:

- `bench_hepdata_import.py` exits zero
- `run_hepdata_import_benchmark_gate.py` reports `status = "passed"` or an explicitly reviewed `status = "review"`
- `compare_hepdata_import_benchmark.py` reports `status = "passed"` or an explicitly reviewed `status = "review"`
- any accepted baseline refresh goes through `promote_hepdata_import_benchmark_baseline.py`
- every benchmark case has `status = "ok"`
- every materialization case writes the expected lockfile/workspaces
- every enabled fit smoke reports `converged = true`
- no correctness/sanity failure occurs in the paired HEP suite when that follow-up is required

For v1, this runbook is an evidence gate, not a fixed slowdown budget gate.
Release review should compare the new artifact against the previous accepted
`nextstat-bench` artifact and investigate material regressions before promotion.
