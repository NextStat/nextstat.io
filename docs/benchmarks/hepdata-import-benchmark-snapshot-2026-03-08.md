# HEPData Import Benchmark Snapshot: 2026-03-08

**Date**: 2026-03-08  
**Host**: `nextstat-bench`  
**Status**: Published frozen baseline evidence  
**Artifact**: [accepted.json](/benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json)

## Purpose

This note records the benchmark evidence behind the frozen `nextstat import hepdata`
stable-surface baseline after promotion on `nextstat-bench`.

It is the published evidence companion to:

- [HEPData Import Runtime Gate](/docs/benchmarks/hepdata-import-runtime-gate.md)
- [HEPData Import Acceptance Criteria (Stable Surface v1)](/docs/specs/hep/hepdata_import_acceptance_v1.md)
- [HEPData Import Stable-Surface Support Matrix](/docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md)
- [HEPData Import Stable-Surface Release Notes](/docs/benchmarks/hepdata-import-release-notes-2026-03-08.md)
- [HEPData Import Promotion Runbook](/docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md)
- [HEPData Import Release PR Checklist](/docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md)

## Source of truth

The promoted machine-readable evidence for this snapshot is:

- accepted baseline: [accepted.json](/benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json)

Promotion history for this freeze:

- previous baseline snapshot: [accepted_20260308T171936Z_previous.json](/benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/history/accepted_20260308T171936Z_previous.json)
- promoted baseline snapshot: [accepted_20260308T171936Z_promoted.json](/benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/history/accepted_20260308T171936Z_promoted.json)

The local compare/promotion/apply gate reports used during the freeze remain
ephemeral `tmp/` artifacts and are intentionally not committed.

## Canonical operator flow

The freeze was produced through the published one-shot gate:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py \
  --current tmp/hepdata_import_benchmark_20260308T164825Z/nextstat-bench/summary.json \
  --promotion-mode apply
```

This uses the canonical remote runner on `nextstat-bench`:

```bash
BENCH_SSH_USER=<user> bash scripts/benchmarks/bench_hepdata_import_remote.sh
```

## Gate summary

The promoted evidence is fully green:

- gate status: `passed`
- benchmark step: `passed`
- compare status: `passed`
- promotion status: `promoted`
- post-promotion compare against `accepted.json`: `passed`
- failed cases after promotion: `0`
- review cases after promotion: `0`

## Case summary

| Case | Source mode | Import best | Fit best | Notes |
| --- | --- | ---: | ---: | --- |
| `curated_catalog` | `curated` | `0.022485s` | n/a | deterministic catalog discovery |
| `direct_patch_catalog_cached` | `direct_doi` | `4.342169s` | n/a | cached direct DOI patch discovery |
| `curated_materialize_offline` | `curated` | `8.401761s` | `40.318278s` | offline cached materialization + fit smoke |
| `direct_materialize_network` | `direct_doi` | `12.828479s` | `39.896699s` | local fixture transport, logical DOI identity |

Stage breakdown of the promoted network materialization case:

- discovery: `6.555457s`
- download: `0.008478s`
- extract: `1.688345s`
- materialize: `3.843482s`
- fit: `39.896699s`

## Interpretation

This snapshot supports the current stable-surface claim for `nextstat import hepdata`:

- curated catalog remains effectively free at operator scale
- direct DOI patch discovery stays in the low-single-digit seconds range on `nextstat-bench`
- curated and direct materialization produce comparable downstream fit cost
- the benchmark identity contract is stable across runs because the direct network case records the logical HEPData DOI rather than the ephemeral local fixture URL

In practical terms, this means the HEPData v1 runtime gate is no longer just
defined in docs; it now has a published promoted baseline with archived
promotion provenance.
