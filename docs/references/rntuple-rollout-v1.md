---
title: "RNTuple Rollout v1 (Support Scope, Limits, Migration)"
status: draft
date: 2026-02-16
---

# RNTuple Rollout v1 (Support Scope, Limits, Migration)

This document defines the rollout contract for current native RNTuple support in `ns-root`.
It is the operator-facing companion to:

- ADR policy: `docs/rfcs/rntuple-support-policy.md`
- compatibility matrix: `docs/references/rntuple-compatibility-matrix.md`
- Rust API reference: `docs/references/rust-api.md`

## Scope (v1)

Native read support is available for matrix-verified ROOT `v6.38` fixtures with:

- key discovery and anchor/envelope/metadata parse;
- footer cluster-group and page-list navigation;
- page-list framing variants with non-uniform preamble/record sections observed in larger multi-cluster fixtures;
- typed decode (`f64` representation) for:
- primitive numeric fields;
- fixed arrays (`std::array<T,N>`, `T[N]`);
- variable arrays (`std::vector<T>`, `RVec<T>`);
- nested pair layouts:
- `std::pair<primitive, primitive>`;
- `std::pair<primitive, std::vector<primitive>>`;
- `std::pair<std::vector<primitive>, primitive>`;
- `std::pair<std::vector<primitive>, std::vector<primitive>>`;
- large mixed-layout stress fixtures that combine primitives + variable arrays + nested pairs across many cluster groups (ROOT `v6.38` fixture line);
- schema-evolution handling where footer-only fields are surfaced and decoded only in cluster groups where pages exist.

## Known Limits (v1)

- Write path is out of scope.
- Nested types outside the supported pair layouts return deterministic `UnsupportedClass`.
- Decode behavior is guaranteed for compatibility-matrix fixtures and parser variants covered by CI; out-of-matrix producer/layout combinations may fail with deterministic `Deserialization` errors.
- API returns decoded numeric values as `f64`; original scalar widths are exposed in schema/column metadata.

## Migration Notes

For workflows previously using conversion as a mandatory bridge:

1. Discover RNTuples using `RootFile::list_rntuples()` / `RootFile::has_rntuples()`.
2. Use `read_rntuple_decoded_columns_all_clusters_f64()` as the default high-level decode path.
3. Use specialized APIs (`read_rntuple_pair_*`) only when consumers require explicit nested-column grouping.
4. Keep converter/Parquet path as fallback for out-of-matrix layouts or unsupported nested types.

## Release Gates

Current CI coverage includes:

- correctness + matrix fixtures: `crates/ns-root/tests/rntuple_discovery.rs`;
- large mixed-layout regression fixture: `large_mixed_layout_fixture_decodes_all_cluster_groups` (`tests/fixtures/rntuple_bench_large.root`, `20` cluster groups, `2,000,000` entries);
- parser robustness + malformed payload coverage: `crates/ns-root/src/rntuple.rs` unit tests;
- schema/page mismatch deterministic failure coverage: `crates/ns-root/src/file.rs` unit tests;
- stable performance gate:
- test: `crates/ns-root/tests/rntuple_perf_gate.rs`;
- command: `make rntuple-perf-gate`;
- optional mixed-layout release gate: `make rntuple-perf-gate-large-mixed`;
- CI wiring: `.github/workflows/rust-tests.yml`.
- benchmark note (host-level comparison): `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md`.

Thresholds for the performance gate are tunable with:

- `NS_ROOT_RNTUPLE_PERF_ITERS`
- `NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS`
- `NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC`
- `NS_ROOT_RNTUPLE_PERF_CASES` (optional comma-separated case override; supports fixture names or absolute paths)
