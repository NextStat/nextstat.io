# HEPData Import Promotion Runbook

**Date**: 2026-03-08  
**Status**: promotion runbook  
**Scope**: `nextstat-bench` promotion evidence for the stable HEPData import surface

## Purpose

This runbook defines the canonical promotion workflow for the HEPData import
stable subset after local and contract gates are green.

It answers one operational question:

- how do maintainers produce, review, and promote the `nextstat-bench`
  evidence for `nextstat import hepdata`?

## Ownership

Promotion requires three explicit roles:

- release owner: decides whether the stable HEPData import claim is restated in the release PR
- bench operator: runs the canonical `nextstat-bench` benchmark artifact
- reviewer: confirms the artifact matches the current accepted support matrix and promotion contract

For a small release, the same maintainer may hold all three roles, but the
artifact review still needs to be explicit in the PR.

## When this runbook is required

Run this workflow when a change affects any of:

- `nextstat import hepdata` CLI behavior
- curated catalog or direct DOI import semantics
- patch discovery / patch mapping semantics
- HEPData provenance or lockfile contracts
- bundle integration for HEPData import
- benchmark, compare, promotion, or one-shot gate logic
- public runtime wording tied to the HEPData import surface

Docs-only edits that do not change the stable claim or runtime wording do not
require a new `nextstat-bench` run.

## Artifact contract

The promotion evidence is the machine-readable benchmark and gate contract set:

- benchmark schema:
  `docs/schemas/benchmarks/hepdata_import_benchmark_result_v1.schema.json`
- compare schema:
  `docs/schemas/benchmarks/hepdata_import_benchmark_compare_report_v1.schema.json`
- promotion schema:
  `docs/schemas/benchmarks/hepdata_import_benchmark_baseline_promotion_report_v1.schema.json`
- gate schema:
  `docs/schemas/benchmarks/hepdata_import_benchmark_gate_report_v1.schema.json`
- runner:
  `python3 scripts/benchmarks/bench_hepdata_import.py`
- remote runner:
  `bash scripts/benchmarks/bench_hepdata_import_remote.sh`
- compare helper:
  `python3 scripts/benchmarks/compare_hepdata_import_benchmark.py`
- promotion helper:
  `python3 scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py`
- one-shot gate:
  `python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py`

The current published evidence notes are:

- `docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md`
- `docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md`
- `docs/benchmarks/hepdata-import-release-notes-2026-03-08.md`

## Precondition

Before touching `nextstat-bench`, the local correctness/contract gates must
already be green:

```bash
cargo test -p ns-cli --test cli_import_hepdata
cargo test -p ns-cli --test cli_bundle_more_commands
pytest -q tests/python/test_hepdata_schema_smoke.py
pytest -q tests/python/test_hepdata_import_benchmark_smoke.py
python3 scripts/check_io_contracts.py --family hepdata
```

This ensures the bench-host run is promotion evidence, not a substitute for the
standard correctness gate.

## Canonical nextstat-bench run

Recommended command:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run
```

If the local SSH alias is not configured correctly, the canonical remote runner
can still be driven with explicit overrides:

```bash
BENCH_HOST=<host> \
BENCH_SSH_USER=<user> \
BENCH_SSH_PORT=<port> \
BENCH_SSH_KEY=/path/to/key \
bash scripts/benchmarks/bench_hepdata_import_remote.sh
```

If a fresh artifact already exists locally, the canonical promotion path is:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py \
  --current tmp/hepdata_import_benchmark_<timestamp>/nextstat-bench/summary.json \
  --promotion-mode apply
```

Recommended preserved output layout:

- local synced artifact dir:
  `tmp/hepdata_import_benchmark_<timestamp>/nextstat-bench/`
- benchmark artifact:
  `summary.json`
- compare report:
  `compare_report.json`
- promotion report:
  `promotion_report.json`
- gate reports:
  `gate_report.json` / `gate_report_apply.json`

## Pass conditions

Promotion evidence is acceptable only if all of the following are true:

- one-shot gate reports `status = "passed"` or an explicitly reviewed `status = "review"`
- benchmark step reports `status = "passed"`
- compare status is `passed` before promotion
- promotion status is `promoted` for the apply step
- post-promotion compare against `accepted.json` reports `status = "passed"`
- every benchmark case has `status = "ok"`
- the artifact came from `nextstat-bench`
- direct network identity is stable at the logical HEPData DOI rather than an ephemeral fixture URL

## Review checklist

The reviewer must confirm:

- the artifact came from `nextstat-bench`
- the artifact path is linked from the benchmark snapshot note
- the support matrix does not widen the claim beyond the promoted subset
- the release note does not widen the claim beyond the promoted subset
- the accepted baseline and history snapshots are updated only through the promotion contract

## Failure handling

If the bench-host run fails:

- do not widen the stable HEPData import claim
- do not update `accepted.json` by hand
- record the failed artifact path in the PR for traceability
- open or update follow-up tasks before promotion proceeds
