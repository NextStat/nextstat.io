# HEPData Import Release PR Checklist

**Date**: 2026-03-08  
**Status**: release hardening checklist  
**Scope**: stable HEPData import surface

## Purpose

This checklist is the maintainer-side `PR-ready` gate for the currently
accepted HEPData import stable subset.

Use it after implementation and docs are complete, but before cutting or
shipping a release that claims HEPData import as a stable product surface.

## Release PR scope

This checklist applies only to the promoted stable subset:

- curated catalog listing and materialization
- explicit direct DOI listing and materialization
- explicit patch discovery / patch mapping
- versioned HEPData import summary and lockfile contracts
- bundle integration preserving the same HEPData semantics
- benchmark / compare / promotion / one-shot gate contracts

It does not promote:

- hidden DOI-to-dataset inference
- automatic patch selection without explicit operator input
- undocumented archive heuristics
- arbitrary transport URLs as dataset identity
- a new Python-native ingest API outside the CLI/operator path

Those remain intentionally outside the stable subset.

## Pre-PR checklist

### Contract

- [ ] stable CLI modes are named explicitly in:
  - `docs/references/cli.md`
- [ ] stable vs out-of-scope boundary is stated explicitly in public docs
- [ ] `nextstat.hepdata_import.v1` and `nextstat.hepdata_lock.v1` remain versioned and published
- [ ] benchmark, compare, promotion, and gate report contracts remain versioned and published

### Evidence

- [ ] acceptance policy is present:
  - `docs/specs/hep/hepdata_import_acceptance_v1.md`
- [ ] runtime gate doc is present:
  - `docs/benchmarks/hepdata-import-runtime-gate.md`
- [ ] benchmark snapshot is present:
  - `docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md`
- [ ] support matrix is present:
  - `docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md`
- [ ] release notes are present:
  - `docs/benchmarks/hepdata-import-release-notes-2026-03-08.md`
- [ ] release PR checklist is present:
  - `docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md`
- [ ] promotion runbook is present:
  - `docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md`

### Verification

- [ ] Rust CLI tests pass:

```bash
cargo test -p ns-cli --test cli_import_hepdata
cargo test -p ns-cli --test cli_bundle_more_commands
```

- [ ] Python schema/runtime/docs gates pass:

```bash
pytest -q tests/python/test_hepdata_schema_smoke.py
pytest -q tests/python/test_hepdata_import_benchmark_smoke.py
```

- [ ] IO contract runner passes:

```bash
python3 scripts/check_io_contracts.py --family hepdata
```

- [ ] benchmark gate reports green on `nextstat-bench` when required:

```bash
python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run
```

- [ ] whitespace / patch hygiene is clean:

```bash
git diff --check -- <touched files>
```

### Promotion evidence

- [ ] current `nextstat-bench` artifact exists and is linked from the benchmark snapshot note
- [ ] current `nextstat-bench` artifact passes compare against the accepted baseline
- [ ] any accepted baseline refresh goes through:
  - `python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --current <artifact> --promotion-mode apply`
- [ ] no stable claim relies on terminal-only output without archived JSON artifacts

### Messaging

- [ ] PR summary names the promoted stable subset explicitly
- [ ] PR summary names the intentional out-of-scope boundaries explicitly
- [ ] no blanket claim is made about every possible HEPData ingestion workflow being stable
- [ ] runtime wording is scoped to the linked `nextstat-bench` artifact and snapshot note

## Recommended PR summary structure

Use a short structure:

1. what is stable now
2. what remains intentionally outside the stable subset
3. what evidence backs the stable claim
4. how to rerun the canonical operator gate

## Exit condition

The release PR is ready only when every checkbox above is green and the
published HEPData claim stays within the accepted subset defined by the March
8, 2026 acceptance policy.
