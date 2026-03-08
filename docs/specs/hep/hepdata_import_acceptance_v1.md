---
title: "HEPData Import Acceptance Criteria (Stable Surface v1)"
status: stable
---

# HEPData Import Acceptance Criteria (Stable Surface v1)

This document defines the release acceptance criteria for the stable product
surface behind `nextstat import hepdata`.

The surface is considered accepted only when **all** criteria below are true.

## 1. Public CLI surface is explicit

The accepted stable commands are:

- `nextstat import hepdata --list`
- `nextstat import hepdata --list-patches --doi <url> --dataset-id <id>`
- `nextstat import hepdata --dataset <catalog-id> --out-dir <dir> [--cache-dir ...] [--lock ...] [--offline]`
- `nextstat import hepdata --doi <url> --dataset-id <id> --out-dir <dir> [--display-name ...] [--bkgonly-filename ...] [--patchset-filename ...] [--patch ...] [--cache-dir ...] [--lock ...] [--offline]`

Explicitness requirements:

- curated catalog mode and direct DOI mode are separate public paths
- `--doi` and `--dataset-id` are required together
- direct DOI mode does not rely on hidden DOI inference or implicit dataset discovery
- `--list-patches` is read-only discovery and does not materialize outputs
- patch materialization uses explicit `--patch <id>` or `--patch <id>=<patch_name>`

## 2. JSON contracts are versioned and published

The accepted public JSON contracts are:

- `schema_version = "nextstat.hepdata_import.v1"`
- `schema_version = "nextstat.hepdata_lock.v1"`

The following must exist and stay in sync:

- published schemas:
  - `nextstat config schema --name hepdata_import_v1`
  - `nextstat config schema --name hepdata_lock_v1`
- canonical examples:
  - `docs/specs/hepdata_import_v1.catalog.example.json`
  - `docs/specs/hepdata_import_v1.list_patches.example.json`
  - `docs/specs/hepdata_import_v1.materialize.example.json`
  - `docs/specs/hepdata_lock_v1.example.json`
- aggregate runner report schema:
  - `docs/schemas/io/nextstat_io_contract_runner_report_v1.schema.json`

## 3. Provenance and reproducibility are mandatory

Accepted outputs must provide reproducible provenance:

- import summary reports `mode` and `source_mode`
- materialization outputs record per-dataset download provenance
- lockfiles record download provenance and output hashes
- offline/cache behavior is explicit in the output contract
- bundle integration preserves the same HEPData contract semantics

## 4. Canonical tooling path is singular

The accepted implementation path is the product CLI surface.

This means:

- Python compatibility wrappers delegate to `nextstat import hepdata`
- there is no separate feature-complete legacy fetch implementation
- examples, docs, bundle flows, and smoke tests all point to the same CLI path

## 5. Mandatory verification gates

The stable surface is accepted only if all of the following pass:

- Rust CLI contract tests:
  - `cargo test -p ns-cli --test cli_import_hepdata`
  - `cargo test -p ns-cli --test cli_bundle_more_commands`
- Python schema/runtime/docs gate:
  - `pytest -q tests/python/test_hepdata_schema_smoke.py`
- reproducibility runner:
  - `python scripts/check_io_contracts.py --family hepdata [--report-json ...]`
- whitespace / patch hygiene:
  - `git diff --check -- <touched files>`

For release-quality changes, the acceptance evidence should include the exact
commands that were run and their pass/fail result.

## 6. Documentation must expose the supported contract

The following published docs are part of the acceptance surface:

- `docs/references/cli.md`
- `docs/specs/hep/hepdata_import_acceptance_v1.md`
- `docs/benchmarks/hepdata-import-runtime-gate.md`
- `docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md`
- `docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md`
- `docs/benchmarks/hepdata-import-release-notes-2026-03-08.md`
- `docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md`
- `docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md`

Those docs must explicitly expose:

- supported CLI modes
- schema and example locations
- regen/check workflow
- reproducibility gate
- lockfile/public provenance expectations

## 7. nextstat-bench rule

`nextstat-bench` is a **conditional acceptance gate**:

- docs-only and test-only changes do **not** require a bench run
- refactors that do not change HEPData runtime behavior do **not** require a bench run
- changes that affect download, cache, extraction, patch materialization, or downstream fit/runtime behavior **do** require a bench run on `nextstat-bench`

When the bench gate is required, the minimum evidence is:

- run the canonical one-shot gate on `nextstat-bench`:
  - `python scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run`
- run the dedicated import runtime gate:
  - `python scripts/benchmarks/bench_hepdata_import.py --deterministic --out bench_results/hepdata_import_benchmark/summary.json`
- compare the fresh artifact against the accepted `nextstat-bench` baseline:
  - `python scripts/benchmarks/compare_hepdata_import_benchmark.py --current bench_results/hepdata_import_benchmark/summary.json --out bench_results/hepdata_import_benchmark/compare_report.json`
- if the fresh artifact is intentionally accepted, promote it through the helper:
  - `python scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py --current bench_results/hepdata_import_benchmark/summary.json --compare-report bench_results/hepdata_import_benchmark/compare_report.json --report bench_results/hepdata_import_benchmark/promotion_report.json`
- store the machine-readable artifact described by:
  - `docs/schemas/benchmarks/hepdata_import_benchmark_result_v1.schema.json`
  - `docs/specs/hepdata_import_benchmark_result_v1.example.json`
- store the accepted baseline + comparison contract described by:
  - `benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json`
  - `docs/schemas/benchmarks/hepdata_import_benchmark_compare_report_v1.schema.json`
  - `docs/specs/hepdata_import_benchmark_compare_report_v1.example.json`
- store the promotion contract described by:
  - `docs/schemas/benchmarks/hepdata_import_benchmark_baseline_promotion_report_v1.schema.json`
  - `docs/specs/hepdata_import_benchmark_baseline_promotion_report_v1.example.json`
- store the orchestration gate contract described by:
  - `docs/schemas/benchmarks/hepdata_import_benchmark_gate_report_v1.schema.json`
  - `docs/specs/hepdata_import_benchmark_gate_report_v1.example.json`
- record the benchmark command and output artifact location
- confirm every import benchmark case reports `status = "ok"`
- document any observed `review` or `failed` comparison status before promotion

Canonical harness entry points:

- `python scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run`
- `python scripts/benchmarks/run_hepdata_import_benchmark_gate.py --current bench_results/hepdata_import_benchmark/summary.json --promotion-mode apply`
- `python scripts/benchmarks/bench_hepdata_import.py --smoke --deterministic --out bench_results/hepdata_import_benchmark/summary.json`
- `python scripts/benchmarks/bench_hepdata_import.py --deterministic --out bench_results/hepdata_import_benchmark/summary.json`
- `python scripts/benchmarks/compare_hepdata_import_benchmark.py --current bench_results/hepdata_import_benchmark/summary.json --out bench_results/hepdata_import_benchmark/compare_report.json`
- `python scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py --current bench_results/hepdata_import_benchmark/summary.json --compare-report bench_results/hepdata_import_benchmark/compare_report.json --report bench_results/hepdata_import_benchmark/promotion_report.json`
- `bash scripts/benchmarks/bench_hepdata_import_remote.sh`
- `python suites/hep/run.py --deterministic --out out/hep_simple_nll.json`
- `python suites/hep/run.py --deterministic --fit --fit-repeat 3 --out out/hep_simple_nll_fit.json`
- `make hep`
- `make hep-fit`

## 8. Out of scope for v1 acceptance

The following are not part of the stable acceptance contract for v1:

- hidden DOI-to-dataset inference
- automatic patch selection without explicit operator input
- arbitrary unsupported archive heuristics beyond the documented filenames and flags
- a dedicated HEPData import performance budget separate from the benchmark correctness gate
