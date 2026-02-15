---
title: "RNTuple Compatibility Matrix (Apex2 Baseline)"
status: draft
date: 2026-02-15
---

# RNTuple Compatibility Matrix (Apex2 Baseline)

This document is the Planning/Exploration baseline for full native RNTuple support in `ns-root`.
It tracks the compatibility targets, fixture sources, and verification status.

Related:

- ADR policy: `docs/rfcs/rntuple-support-policy.md`
- Effort estimates: `docs/references/rntuple-minimal-reader-estimate.md`

## Matrix dimensions

- Producer version line (ROOT release family).
- Feature class (flat primitive, arrays, nested structures, schema evolution).
- Compression/path variations where relevant.
- Verification status (`planned`, `in_progress`, `verified`, `blocked`).

## Compatibility matrix (initial)

| Producer line | Feature class | Fixture source | Verification status | Notes |
|---|---|---|---|---|
| ROOT `v6.38` | Key discovery + raw payload extraction | `tests/fixtures/rntuple_simple.root` | verified | Covered by `crates/ns-root/tests/rntuple_discovery.rs` |
| ROOT `v6.38` | Anchor metadata parse (`ROOT::RNTuple`) | `tests/fixtures/rntuple_simple.root` | verified | `read_rntuple_anchor()` values cross-checked against ROOT getters |
| ROOT `v6.38` | Header/footer envelope blob decode | `tests/fixtures/rntuple_simple.root` | verified | `read_rntuple_envelopes()` validates decompressed lengths |
| ROOT `v6.38` | Envelope framing parse + header summary strings | `tests/fixtures/rntuple_simple.root` | verified | `read_rntuple_metadata_summary()` validates type/length and extracts `Events`, `ROOT v6.38.00`, plus best-effort field tokens (`pt:float`, `n:int*`) |
| ROOT `v6.38` | Metadata schema projection (field name + scalar type) | `tests/fixtures/rntuple_simple.root` | verified | `read_rntuple_schema_summary()` maps `pt -> F32`, `n -> I32` |
| ROOT `v6.x` (line A) | Flat primitives (typed decode) | `tests/fixtures/rntuple_simple.root` + `tests/fixtures/rntuple_complex.root` | in_progress | Metadata-level type mapping verified (`Primitive`), data-page decode still pending |
| ROOT `v6.38` | Arrays / variable-length (metadata mapping) | `tests/fixtures/rntuple_complex.root` + synthetic tokens | verified | `std::array<T,N>`, `T[N]`, `std::vector<T>`/`RVec<T>` mapped to `FixedArray`/`VariableArray` with element scalar type |
| ROOT `v6.38` | Nested structures (metadata mapping) | `tests/fixtures/rntuple_complex.root` + synthetic tokens | verified | `std::pair<...>` and record-like tokens mapped to `Nested` via `read_rntuple_schema_summary()` |
| ROOT `v6.x` (line B) | Schema evolution cases | TBD | planned | Backward/forward compat checks |
| Mixed producer samples | Corrupted/negative cases | TBD | planned | Deterministic error behavior |

## Acceptance gates

1. Every matrix row has at least one reproducible fixture in CI.
2. Golden correctness tests pass for all `verified` rows.
3. Regression suite covers at least one negative case per feature class.
4. Performance gate is green for baseline read scenarios before release.

## Apex2 tracking

- `Planning`: matrix structure and acceptance gates defined.
- `Exploration`: fixture discovery and parser assumptions documented.
- `Execution`: parser/type/API implementation mapped to matrix rows.
- `Verification`: rows move to `verified` only with CI evidence.
