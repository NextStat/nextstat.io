---
title: "RNTuple Compatibility Matrix (Apex2 Status)"
status: draft
date: 2026-02-16
---

# RNTuple Compatibility Matrix (Apex2 Status)

This document tracks current native RNTuple compatibility status in `ns-root`.
It records fixture-backed coverage and verification state for release gating.

Related:

- ADR policy: `docs/rfcs/rntuple-support-policy.md`
- Effort estimates: `docs/references/rntuple-minimal-reader-estimate.md`
- Rollout notes: `docs/references/rntuple-rollout-v1.md`

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
| ROOT `v6.38` | Footer cluster-group summary parse | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root`, `tests/fixtures/rntuple_multicluster.root`, `tests/fixtures/rntuple_schema_evolution.root` | verified | `read_rntuple_footer_summary()` extracts cluster-group entry span and page-list locator (`nbytes`, `position`) across single- and multi-group fixtures |
| ROOT `v6.38` | Page-list envelope load (`type=0x03`) | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_pagelist_envelope()` loads/decompresses the referenced page-list envelope and validates framing |
| ROOT `v6.38` | Page-list descriptor parse (`nbytes`, `position`) | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_pagelist_summary()` parses ROOT 6.38 page records and exposes stable page locators plus raw record markers |
| ROOT `v6.38` | Page-list framing variant (non-uniform preamble/record sections across cluster groups) | `tests/fixtures/rntuple_bench_large_primitive.root` | verified | `read_rntuple_pagelist_summary()` now accepts observed larger-fixture framing variants (not only fixed 72+40 layout) |
| ROOT `v6.38` | Raw page-blob load by descriptor index | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_page_blob()` reads on-storage page bytes for each parsed descriptor (no typed decode yet) |
| ROOT `v6.38` | Flat primitives (typed decode) | `tests/fixtures/rntuple_simple.root` + `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_primitive_columns_f64()` decodes primitive fields (`pt`, `n`) via page-list binding + byte-shuffle + zigzag, validated against fixture values |
| ROOT `v6.38` | Compressed primitive page decode in large multi-cluster fixture | `tests/fixtures/rntuple_bench_large_primitive.root` | verified | Primitive decode path no longer assumes `nbytes_on_storage == uncompressed_size`; pages are decompressed against expected element-width payload length before decode |
| ROOT `v6.38` | Fixed-array pages (typed decode) | `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_fixed_array_columns_f64()` decodes `arr_fixed` as `[1,2,3]` for the fixture entry |
| ROOT `v6.38` | Variable-array pages (typed decode) | `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_variable_array_columns_f64()` decodes offset+data pages for `arr_var` as `[10,20,30]` |
| ROOT `v6.38` | Nested pair pages (typed decode) | `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_pair_columns_f64()` decodes `pair_ff` as `(4.0,5.0)` |
| ROOT `v6.38` | Nested pair + variable right pages (typed decode) | `tests/fixtures/rntuple_pair_scalar_variable.root` | verified | `read_rntuple_pair_scalar_variable_columns_f64()` decodes `std::pair<primitive,std::vector<primitive>>` (fixture `pair_f_vec` -> `(3.0,[10,20])`) |
| ROOT `v6.38` | Nested pair + variable left pages (typed decode) | `tests/fixtures/rntuple_pair_variable_scalar.root` | verified | `read_rntuple_pair_variable_scalar_columns_f64()` decodes `std::pair<std::vector<primitive>,primitive>` (fixture `pair_vec_f` -> `([10,20],6.5)`) |
| ROOT `v6.38` | Nested pair + variable both sides (typed decode) | `tests/fixtures/rntuple_pair_variable_variable.root` | verified | `read_rntuple_pair_variable_variable_columns_f64()` decodes `std::pair<std::vector<primitive>,std::vector<primitive>>` (fixture `pair_vec_vec` -> `([1,2],[30,40.5])`) |
| ROOT `v6.38` | Large mixed layout stress (`primitive + variable-array + nested-pair`, 20 clusters, high-entropy payload) | `tests/fixtures/rntuple_bench_large.root` | verified | `read_rntuple_decoded_columns_all_clusters_f64()` decodes all 20 cluster groups (`2,000,000` entries total); regression coverage in `crates/ns-root/tests/rntuple_discovery.rs` (`large_mixed_layout_fixture_decodes_all_cluster_groups`) |
| ROOT `v6.38` | Single-pass supported-field decode bundle | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root` | verified | `read_rntuple_decoded_columns_f64()` returns grouped decoded columns in one pass (schema-order page binding) |
| ROOT `v6.38` | All-cluster decode orchestration | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root`, `tests/fixtures/rntuple_multicluster.root`, `tests/fixtures/rntuple_schema_evolution.root` | verified | `read_rntuple_decoded_columns_all_clusters_f64()` iterates every parsed cluster group and returns per-group decoded bundle + entry-range metadata |
| ROOT `v6.38` | Arrays / variable-length (metadata mapping) | `tests/fixtures/rntuple_complex.root` + synthetic tokens | verified | `std::array<T,N>`, `T[N]`, `std::vector<T>`/`RVec<T>` mapped to `FixedArray`/`VariableArray` with element scalar type |
| ROOT `v6.38` | Nested structures (metadata mapping) | `tests/fixtures/rntuple_complex.root` + synthetic tokens | verified | `std::pair<...>` and record-like tokens mapped to `Nested` via `read_rntuple_schema_summary()` |
| ROOT `v6.38` | Schema evolution cases | `tests/fixtures/rntuple_schema_evolution.root` | verified | Schema summary now merges header and footer field tokens; all-cluster decode handles cluster-local field availability (new field `n` appears only in later cluster group) |
| Synthetic producer variants | Footer cluster-group frame variants | `crates/ns-root/src/rntuple.rs` unit payloads | verified | `parse_cluster_groups()` is covered for record-with-preamble and direct-record frame variants, plus scan fallback when list-frame preamble is absent |
| Synthetic malformed payloads | Envelope parser robustness (fuzz-like) | `crates/ns-root/src/rntuple.rs` unit payloads | verified | deterministic pseudo-random malformed footer/page-list envelope payloads are exercised to ensure parser returns `Result` without panic |
| ROOT `v6.38` | Unsupported nested decode path | `tests/fixtures/rntuple_unsupported_nested.root` | verified | `read_rntuple_decoded_columns_f64()` returns deterministic `UnsupportedClass` for nested types outside current decode surface (e.g. `std::pair<float,std::pair<float,float>>`) |
| Mixed producer samples | Corrupted/negative cases | `tests/fixtures/rntuple_complex.root`, `tests/fixtures/rntuple_pair_scalar_variable.root`, `tests/fixtures/rntuple_pair_variable_scalar.root`, `tests/fixtures/rntuple_pair_variable_variable.root`, `tests/fixtures/rntuple_unsupported_nested.root` (mutated in tests) | verified | `rntuple_discovery` covers corrupted offsets in variable arrays and nested pair jagged sides (`Deserialization`), plus unsupported nested layouts (`UnsupportedClass`) with deterministic messages |
| CI stable gate | Decode performance baseline | `tests/fixtures/rntuple_simple.root`, `tests/fixtures/rntuple_complex.root`, `tests/fixtures/rntuple_multicluster.root`, `tests/fixtures/rntuple_schema_evolution.root`, `tests/fixtures/rntuple_pair_scalar_variable.root`, `tests/fixtures/rntuple_pair_variable_scalar.root`, `tests/fixtures/rntuple_pair_variable_variable.root` | verified | `crates/ns-root/tests/rntuple_perf_gate.rs` (`rntuple_decode_perf_gate_baseline`) runs via `make rntuple-perf-gate` in `rust-tests.yml`; thresholds tunable with `NS_ROOT_RNTUPLE_PERF_*` env vars |
| Host benchmark evidence | `ns-root` vs ROOT decode throughput (`nextstat-bench`) | `tests/fixtures/rntuple_bench_large_primitive.root` | verified | Repro note and commands: `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md` |

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
