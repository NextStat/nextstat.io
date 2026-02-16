---
title: "RNTuple Support Effort Estimate (Minimal, Converter, Full)"
status: reference
date: 2026-02-16
---

# RNTuple Support Effort Estimate (Minimal, Converter, Full)

This estimate supports ADR policy work in `docs/rfcs/rntuple-support-policy.md`.
For current rollout scope/limits, see `docs/references/rntuple-rollout-v1.md`.

Implementation snapshot (as of 2026-02-16):

- Option 4 (full native reader track) is the active policy path.
- Compatibility/correctness/reliability/performance CI gates are now implemented and documented.
- This file remains a planning/effort reference, not a live progress tracker.

## Baseline and assumptions

Scope assumptions:

- One experienced Rust engineer on `ns-root` internals.
- Full-support option targets broad production support (not strict 100% ROOT parity).
- Existing `ns-root` complexity for TTree/branch read path is already substantial
  (core read-path files are ~4.5k LOC excluding tests and tooling).
- Estimate includes implementation, tests, docs, and hardening for production use.

Not included:

- Full schema evolution support across all ROOT/RNTuple edge cases.
- Write path.
- Advanced nested object model parity.

## Option 2: Minimal native RNTuple reader (flat numeric + simple nested)

Target: direct read path in `ns-root` for a constrained subset.

Work breakdown (person-weeks):

| Workstream | Effort |
|---|---:|
| Binary format parsing (header/footer/schema/page descriptors) | 2.0 |
| Cluster/page navigation and read API integration | 1.5 |
| Type mapping (flat numerics + simple nested), errors | 1.0 |
| Test corpus + golden fixtures + fuzz/smoke gates | 1.5 |
| Perf pass, docs, stabilization buffer | 1.0 |
| **Total** | **7.0 pw** |

Risk-adjusted range:

- Optimistic: 5.5 pw
- Most likely: 7.0 pw
- Conservative: 9.0 pw

Primary risks:

- Hidden complexity in page/cluster edge cases.
- Version-specific metadata differences.
- Maintenance overhead once parser is shipped.

## Option 3: Separate RNTuple -> Arrow/Parquet converter tool

Target: keep inference path Parquet-first; add conversion entrypoint.

Work breakdown (person-weeks):

| Workstream | Effort |
|---|---:|
| Converter architecture + ingestion abstraction | 1.0 |
| RNTuple read adapter (subset) for conversion needs | 2.0 |
| Arrow/Parquet mapping, schema contract, validation | 1.0 |
| CLI/tool UX, failure modes, fixtures | 1.0 |
| Benchmarking, docs, release hardening | 1.0 |
| **Total** | **6.0 pw** |

Risk-adjusted range:

- Optimistic: 4.5 pw
- Most likely: 6.0 pw
- Conservative: 8.0 pw

Primary risks:

- Data type/shape mismatches during conversion.
- Extra operational burden from converter lifecycle and support.

## Option 4: Full native RNTuple reader (production support)

Target: broad native RNTuple ingestion in `ns-root` with compatibility matrix and release gates.

Work breakdown (person-weeks):

| Workstream | Effort |
|---|---:|
| Format architecture + compatibility matrix definition | 2.0 |
| Metadata/schema decoding + schema-evolution handling | 3.0 |
| Page/cluster navigation across supported layouts | 2.5 |
| Full type mapping (primitives, arrays, nested structures) | 3.0 |
| API integration + deterministic error model | 1.5 |
| Large fixture corpus, regression tests, fuzz/negative tests | 3.0 |
| Performance tuning + benchmark gates + docs/release hardening | 3.0 |
| **Total** | **18.0 pw** |

Risk-adjusted range:

- Optimistic: 14.0 pw
- Most likely: 18.0 pw
- Conservative: 24.0 pw

Primary risks:

- Significant edge-case spread across writer/version combinations.
- Hidden complexity in nested type semantics and schema evolution.
- Ongoing maintenance burden after initial release.

## Comparative summary

| Option | Most likely effort | Notes |
|---|---:|---|
| 2. Minimal native reader | 7.0 pw | Better direct UX, larger parser maintenance commitment |
| 3. Converter tool | 6.0 pw | Slightly cheaper, aligned with current Parquet-first strategy |
| 4. Full native reader | 18.0 pw | Strategic end-state, highest delivery and maintenance cost |

Recommendation from effort perspective:

- If strategic priority is full native support, run Option 4 in phased delivery with strict gates.
- Option 3 remains useful as migration/fallback tooling, not as a substitute for Option 4.
