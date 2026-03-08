# HEPData Import Stable-Surface Support Matrix

**Date**: 2026-03-08  
**Status**: Executed support matrix  
**Scope**: `nextstat import hepdata` open-likelihood ingestion surface

## Purpose

This document is the short operational matrix for the promoted HEPData import
subset.

It answers one narrow question:

- what is `stable` now
- what remains intentionally outside the stable promise

## Support classes

| Class | Meaning |
| --- | --- |
| `stable` | public compatibility promise for the named HEPData import subset |
| `research-grade` | available or documented boundary, but not promoted into the stable promise |

## Stable CLI matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat import hepdata --list` | `stable` | deterministic curated catalog surface |
| `nextstat import hepdata --dataset <catalog-id> ...` | `stable` | curated materialization path with versioned summary + lockfile |
| `nextstat import hepdata --doi <url> --dataset-id <id> ...` | `stable` | direct DOI materialization path with explicit dataset identity |
| `nextstat import hepdata --list-patches --doi <url> --dataset-id <id>` | `stable` | read-only direct DOI patch discovery |
| `nextstat import hepdata --patch <id>` | `stable` | explicit patch selection by stable output id |
| `nextstat import hepdata --patch <id>=<patch_name>` | `stable` | explicit patch-name mapping for direct DOI materialization |
| `nextstat import hepdata --offline` | `stable` | explicit offline/cache behavior in the public contract |
| `nextstat bundle ... import hepdata ...` | `stable` | bundle path preserves the same HEPData contract semantics |

## Stable contract matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `nextstat.hepdata_import.v1` | `stable` | published import summary contract for catalog, patch-discovery, and materialize modes |
| `nextstat.hepdata_lock.v1` | `stable` | published lockfile/provenance contract |
| canonical examples in `docs/specs/hepdata_import_v1.*` | `stable` | schema-validated and reproducibly generated |
| `nextstat.hepdata_import_benchmark_result.v1` | `stable` | runtime evidence contract for `nextstat-bench` |
| `nextstat.hepdata_import_benchmark_compare_report.v1` | `stable` | accepted-baseline drift contract |
| `nextstat.hepdata_import_benchmark_baseline_promotion_report.v1` | `stable` | promotion provenance contract |
| `nextstat.hepdata_import_benchmark_gate_report.v1` | `stable` | one-shot operator gate contract |

## Stable operator matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `python3 scripts/check_io_contracts.py --family hepdata` | `stable` | reproducibility/doc/schema gate |
| `python3 scripts/benchmarks/bench_hepdata_import.py` | `stable` | deterministic runtime gate runner |
| `bash scripts/benchmarks/bench_hepdata_import_remote.sh` | `stable` | canonical `nextstat-bench` remote runner |
| `python3 scripts/benchmarks/compare_hepdata_import_benchmark.py` | `stable` | baseline drift gate |
| `python3 scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py` | `stable` | baseline promotion contract |
| `python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py` | `stable` | canonical one-shot operator flow |
| `tests/hepdata/fetch_workspaces.py` | `stable` | compatibility wrapper over the product CLI; not a separate implementation surface |

## Boundaries that remain intentional

The following are not widened by the current stable promise:

- hidden DOI-to-dataset inference
- automatic patch selection without explicit operator input
- unsupported archive heuristics beyond the documented filenames and flags
- arbitrary external HEPData transport URLs as dataset identity
- a separate fixed performance budget beyond the accepted `nextstat-bench` baseline gate
- any new Python-native HEPData ingest API outside the product CLI + compatibility wrapper path

## Evidence and gate

The promoted stable subset is backed by:

- [HEPData Import Acceptance Criteria (Stable Surface v1)](/docs/specs/hep/hepdata_import_acceptance_v1.md)
- [HEPData Import Runtime Gate](/docs/benchmarks/hepdata-import-runtime-gate.md)
- [HEPData Import Benchmark Snapshot: 2026-03-08](/docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md)
- [HEPData Import Promotion Runbook](/docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md)
- [HEPData Import Release PR Checklist](/docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md)
- [HEPData CLI Reference](/docs/references/cli.md)

## Bottom line

The stable product promise is intentionally narrow and operator-facing:

- curated and direct DOI HEPData import are `stable`
- versioned summary/lockfile and benchmark/provenance contracts are `stable`
- one-shot `nextstat-bench` gating and baseline promotion are `stable`
- hidden inference and heuristic widening remain outside the promoted subset
