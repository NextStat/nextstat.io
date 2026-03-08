# HEPData Import Stable-Surface Release Notes

**Date**: 2026-03-08  
**Status**: release-hardening note  
**Scope**: `nextstat import hepdata` open-likelihood ingestion surface

## Summary

The HEPData import surface is now promoted as a stable operator-facing product
subset.

This promotion is intentionally narrow. It covers curated catalog import,
explicit direct DOI import, versioned summary/lockfile contracts, and the
runtime evidence/promotion workflow on `nextstat-bench`. It does not widen the
surface to hidden DOI inference, implicit patch selection, or undocumented
archive heuristics.

## Promoted to stable

### CLI and bundle surface

- `nextstat import hepdata --list`
- `nextstat import hepdata --dataset <catalog-id> ...`
- `nextstat import hepdata --doi <url> --dataset-id <id> ...`
- `nextstat import hepdata --list-patches --doi <url> --dataset-id <id>`
- explicit `--patch <id>` and `--patch <id>=<patch_name>`
- bundle integration for `import hepdata`

### Contracts

- `nextstat.hepdata_import.v1`
- `nextstat.hepdata_lock.v1`
- `nextstat.hepdata_import_benchmark_result.v1`
- `nextstat.hepdata_import_benchmark_compare_report.v1`
- `nextstat.hepdata_import_benchmark_baseline_promotion_report.v1`
- `nextstat.hepdata_import_benchmark_gate_report.v1`

### Operator workflow

- `python3 scripts/check_io_contracts.py --family hepdata`
- `python3 scripts/benchmarks/bench_hepdata_import.py`
- `bash scripts/benchmarks/bench_hepdata_import_remote.sh`
- `python3 scripts/benchmarks/compare_hepdata_import_benchmark.py`
- `python3 scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py`
- `python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py`

## What remains intentionally outside the stable promise

- hidden DOI-to-dataset inference
- automatic patch selection without explicit operator input
- undocumented archive heuristics
- arbitrary transport URLs as dataset identity
- a new Python-native ingest API outside the CLI/operator path

## What changed in the hardening wave

- HEPData import now has a single product implementation path; the Python fetch
  helper is only a compatibility wrapper over the CLI
- the runtime gate is fully versioned end-to-end: benchmark, compare,
  promotion, and one-shot gate reports are all published contracts
- the direct network benchmark case now records the logical HEPData DOI rather
  than the ephemeral local fixture URL, which makes baseline comparison stable
  across runs
- the frozen `nextstat-bench` baseline now has published snapshot evidence plus
  archived promotion provenance

## Evidence behind the promotion

This promotion is backed by:

- [HEPData Import Acceptance Criteria (Stable Surface v1)](/docs/specs/hep/hepdata_import_acceptance_v1.md)
- [HEPData Import Runtime Gate](/docs/benchmarks/hepdata-import-runtime-gate.md)
- [HEPData Import Stable-Surface Support Matrix](/docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md)
- [HEPData Import Benchmark Snapshot: 2026-03-08](/docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md)
- [HEPData Import Promotion Runbook](/docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md)
- [HEPData Import Release PR Checklist](/docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md)
- [HEPData CLI Reference](/docs/references/cli.md)

## Runtime wording

This release note is a stable-surface note, not a blanket performance claim.

Any public runtime wording should be scoped to the promoted `nextstat-bench`
artifact and the published benchmark snapshot. The accepted March 8, 2026
baseline is:

- [accepted.json](/benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json)

## Bottom line

This release does not claim that every possible HEPData ingestion workflow is
stable. It claims something narrower and stronger:

- explicit curated and direct DOI import are stable
- published provenance/runtime promotion contracts are stable
- hidden inference and heuristic widening remain outside the promoted subset
