---
title: "ADR-0001: RNTuple Support Policy"
status: accepted
date: 2026-02-16
owners:
  - ns-root maintainers
---

# ADR-0001: RNTuple Support Policy

## Status

Accepted (updated on 2026-02-16: full native-support track remains active with CI correctness/reliability/perf gates in place, including verified large mixed-layout decode coverage).

## Context

NextStat currently supports ROOT `TH1*` and `TTree` ingestion through `ns-root`, with production
pipelines already centered on Arrow/Parquet for event-level workloads:

- `ns-root` provides native `TH1D/TH1F` and `TTree/TBranch` read paths.
- `nextstat convert` and `EventStore::from_parquet()` provide a ROOT-to-Parquet bridge and a
  fast Parquet-first analysis path.
- The roadmap already treats Arrow/Parquet as a primary interchange format.

RNTuple is a strategic format for HEP workflows. Prior policy deferred native support, but current
product direction now prioritizes direct RNTuple ingestion in `ns-root` to reduce conversion friction
and remove dependency on bridge-only workflows.

## Decision

We adopt **Option 4** now:

1. Build a **full native RNTuple reader** in `ns-root` (production support target).
2. Deliver in phases, but keep end-goal as broad compatibility rather than a permanently minimal subset.
3. Keep Arrow/Parquet as interoperable output/input path, not as a mandatory bridge.
4. Track delivery under a dedicated implementation epic and release gates.

## Options considered

### Option 1: No native RNTuple reader now (rejected)

- Use current bridge strategy (`ROOT/TTree -> Parquet/Arrow`) for production.
- Lowest immediate engineering cost and lowest parser maintenance risk.

### Option 2: Minimal native RNTuple reader (rejected as end-state)

- Scope: flat numeric columns plus simple nested cases only.
- Adds new binary parser surface (footer/schema/page/cluster semantics).
- Detailed estimate: `docs/references/rntuple-minimal-reader-estimate.md`.

### Option 3: Separate RNTuple-to-Arrow/Parquet converter tool (secondary)

- Useful as migration/interop support and fallback.
- Does not replace need for full native ingestion.
- Detailed estimate: `docs/references/rntuple-minimal-reader-estimate.md`.

### Option 4: Full native RNTuple reader (chosen)

- Goal: broad production-grade RNTuple support in `ns-root`.
- Includes robust metadata/layout handling, wider type coverage, and compatibility testing.
- Detailed estimate: `docs/references/rntuple-minimal-reader-estimate.md`.

## Execution gates

Before declaring support complete:

1. Compatibility gate: agreed matrix of ROOT/RNTuple producer versions covered by CI fixtures.
2. Correctness gate: golden tests for schema/page/cluster decoding and complex/nested type mapping.
3. Reliability gate: corruption/edge-case behavior is deterministic and documented.
4. Performance gate: no unacceptable regressions versus existing TTree read-path expectations.
5. API gate: stable public API semantics for RNTuple reads in `ns-root`.

Gate evidence references:

- Compatibility + correctness rows: `docs/references/rntuple-compatibility-matrix.md`
- Rollout scope/limits + migration notes: `docs/references/rntuple-rollout-v1.md`
- Stable perf gate command/path: `make rntuple-perf-gate` and `.github/workflows/rust-tests.yml`
- Host-level throughput evidence (`ns-root` vs ROOT on `nextstat-bench`): `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md`
- Mixed-layout regression evidence: `crates/ns-root/tests/rntuple_discovery.rs` (`large_mixed_layout_fixture_decodes_all_cluster_groups`) and compatibility row for `tests/fixtures/rntuple_bench_large.root` marked `verified`.

## Consequences

### Positive

- Removes mandatory conversion hop for RNTuple-first users.
- Improves product completeness for modern ROOT workflows.
- Reduces long-term strategic gap in native ROOT format coverage.

### Negative

- Higher near-term engineering cost and delivery risk.
- Larger long-term maintenance surface in binary format parsing.
- Requires stricter compatibility testing infrastructure.

## Scope boundaries (initial release)

Initial full-support release excludes:

- Write support.
- Non-essential tooling outside ingestion/read path.

These can be tracked as follow-up work once core read compatibility is stable.

## Related artifacts

- Effort estimate: `docs/references/rntuple-minimal-reader-estimate.md`
- Compatibility matrix: `docs/references/rntuple-compatibility-matrix.md`
- Rollout v1 notes: `docs/references/rntuple-rollout-v1.md`
- Current Arrow/Parquet strategy: `docs/references/arrow-parquet-io.md`
- Roadmap context: `docs/ROADMAP.md`
