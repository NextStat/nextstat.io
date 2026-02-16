# RNTuple Reader Tutorial

Native RNTuple reading without ROOT installation. NextStat's `ns-root` crate decodes ROOT v6.38 RNTuple files directly — primitives, arrays, nested pairs, schema evolution — with full CI-backed correctness gates and a stable performance baseline.

> **Note:** RNTuple support is **read-only**. Write path is out of scope. For file creation, use ROOT or `uproot`.

## Contents

1. [What Is RNTuple?](#1-what-is-rntuple)
2. [Supported Types](#2-supported-types)
3. [CLI Tutorial](#3-cli-tutorial)
4. [Rust API](#4-rust-api)
5. [Compatibility Matrix](#5-compatibility-matrix)
6. [Known Limits](#6-known-limits)
7. [Migration from Converter Workflows](#7-migration-from-converter-workflows)
8. [Performance](#8-performance)

## 1. What Is RNTuple?

RNTuple is the next-generation columnar I/O format in ROOT, replacing TTree. It offers better compression, faster reads, and a cleaner on-disk layout. As ATLAS, CMS, and other experiments migrate to RNTuple, NextStat provides native read support — no ROOT installation required.

## 2. Supported Types

All decode paths return `f64` values. Original scalar widths are preserved in schema/column metadata.

| Type Class | C++ Types | Status |
|---|---|---|
| Flat primitives | `float`, `double`, `int32_t`, `uint64_t`, … | ✓ verified |
| Fixed arrays | `std::array<T,N>`, `T[N]` | ✓ verified |
| Variable arrays | `std::vector<T>`, `RVec<T>` | ✓ verified |
| Pair (scalar, scalar) | `std::pair<float, float>` | ✓ verified |
| Pair (scalar, vector) | `std::pair<float, std::vector<float>>` | ✓ verified |
| Pair (vector, scalar) | `std::pair<std::vector<float>, float>` | ✓ verified |
| Pair (vector, vector) | `std::pair<std::vector<float>, std::vector<float>>` | ✓ verified |
| Schema evolution | Fields added in later cluster groups | ✓ verified |
| Deeply nested | `std::pair<float, std::pair<…>>` | UnsupportedClass |

## 3. CLI Tutorial

### Step 1: Discover RNTuples in a ROOT file

```bash
$ nextstat root inspect data.root

RNTuples found: 2
  Events    — 150,000 entries, 3 cluster groups
  Metadata  — 1 entry, 1 cluster group
```

### Step 2: View schema (fields and types)

```bash
$ nextstat root schema data.root --ntuple Events

Fields:
  pt          F32   (primitive)
  eta         F32   (primitive)
  jet_pt      F32   (variable array → std::vector<float>)
  pair_ff     F32   (nested pair → std::pair<float,float>)
```

### Step 3: Decode columns to CSV / Arrow

```bash
# Decode all primitive columns across all clusters
$ nextstat root decode data.root --ntuple Events --format csv > events.csv

# Or pipe to Arrow IPC for zero-copy handoff to Polars/DuckDB
$ nextstat root decode data.root --ntuple Events --format arrow > events.arrow
```

## 4. Rust API

The Rust API in `ns-root` provides layered access from low-level envelope parsing to high-level decoded columns.

```rust
use ns_root::RootFile;

let file = RootFile::open("data.root")?;

// Discovery
let has = file.has_rntuples();
let names = file.list_rntuples();

// Schema inspection
let schema = file.read_rntuple_schema_summary("Events")?;
for field in &schema.fields {
    println!("{}: {:?}", field.name, field.field_type);
}

// High-level decode (all clusters, f64 representation)
let decoded = file.read_rntuple_decoded_columns_all_clusters_f64("Events")?;
for (cluster_idx, bundle) in decoded.iter().enumerate() {
    println!("Cluster {}: {} entries", cluster_idx, bundle.entry_range.len());
    for col in &bundle.columns {
        println!("  {} → {} values", col.name, col.values.len());
    }
}
```

For nested pair columns, use specialized APIs: `read_rntuple_pair_columns_f64()`, `read_rntuple_pair_scalar_variable_columns_f64()`, etc.

## 5. Compatibility Matrix

Every row in the matrix is backed by a CI fixture and golden correctness test. Current verification target: **ROOT v6.38**.

| Feature | Fixture | Status |
|---|---|---|
| Anchor + envelope parse | `rntuple_simple.root` | ✓ |
| Primitive decode (byte-shuffle + zigzag) | `rntuple_simple.root`, `rntuple_complex.root` | ✓ |
| Fixed-array decode | `rntuple_complex.root` | ✓ |
| Variable-array decode (offset + data) | `rntuple_complex.root` | ✓ |
| All nested pair variants | `rntuple_pair_*.root` (4 fixtures) | ✓ |
| Multi-cluster (20 groups, 2M entries) | `rntuple_bench_large.root` | ✓ |
| Compressed pages (zlib/lz4) | `rntuple_bench_large_primitive.root` | ✓ |
| Schema evolution | `rntuple_schema_evolution.root` | ✓ |
| Malformed payload robustness | Unit test fuzz payloads | ✓ |

Full matrix with details: `docs/references/rntuple-compatibility-matrix.md`.

## 6. Known Limits

- **Read-only** — no write path.
- **Nested depth ≤ 1** — deeply nested types (pair-of-pair) return `UnsupportedClass`.
- **ROOT v6.38** — verified against v6.38 fixtures. Older/newer producer versions may work but are not CI-guaranteed.
- **f64 output** — all numeric values returned as f64. Original widths in schema metadata.

## 7. Migration from Converter Workflows

If your workflow previously required a TTree → Parquet or TTree → CSV conversion step, you can now read RNTuple files directly:

| Step | Before | Now |
|---|---|---|
| 1. Discovery | `rootls data.root` | `nextstat root inspect data.root` |
| 2. Schema | `rootls -t data.root:Events` | `nextstat root schema data.root` |
| 3. Decode | `root2csv` → pandas | `nextstat root decode` → Arrow/CSV |
| 4. Fallback | N/A | Keep converter for out-of-matrix layouts |

## 8. Performance

CI includes a stable performance gate (`make rntuple-perf-gate`) that runs on every PR. Thresholds are tunable via environment variables:

```
NS_ROOT_RNTUPLE_PERF_ITERS=100
NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS=5
NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC=200
```

The large mixed-layout stress fixture (20 cluster groups, 2M entries) is available as an optional release gate: `make rntuple-perf-gate-large-mixed`.

Benchmark comparison against ROOT's own RNTuple reader: `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md`.

## Related Documents

- ADR policy: `docs/rfcs/rntuple-support-policy.md`
- Compatibility matrix: `docs/references/rntuple-compatibility-matrix.md`
- Rollout notes: `docs/references/rntuple-rollout-v1.md`
- Rust API reference: `docs/references/rust-api.md`
- Benchmark: `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md`
