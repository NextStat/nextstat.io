# Binned Histogram Parquet Schema (v2)

Extended specification for binned HistFactory data stored in Apache Parquet format, including systematic modifiers.

**Status:** v2 draft (2026-02-10)  
**Supersedes:** implicit v1 (yields-only schema in `arrow/mod.rs`)

---

## Overview

A HistFactory workspace in Parquet is represented by **two tables** stored as separate Parquet files (or row groups):

1. **Yields table** — one row per (channel, sample), containing bin counts and statistical errors.
2. **Modifiers table** — one row per (channel, sample, modifier), containing systematic uncertainty data.

This two-table design avoids deeply nested Arrow structures and enables efficient column-pruning and predicate pushdown.

---

## Table 1: Yields

Same as the existing v1 schema (`arrow/ingest.rs`).

| Column | Arrow Type | Required | Description |
|--------|-----------|----------|-------------|
| `channel` | `Utf8` | **yes** | Channel (region) name |
| `sample` | `Utf8` | **yes** | Sample (process) name |
| `yields` | `List<Float64>` | **yes** | Expected event counts per bin |
| `stat_error` | `List<Float64>` | no | Per-bin statistical uncertainties (σ) |

## Table 2: Modifiers

| Column | Arrow Type | Required | Description |
|--------|-----------|----------|-------------|
| `channel` | `Utf8` | **yes** | Channel name (join key) |
| `sample` | `Utf8` | **yes** | Sample name (join key) |
| `modifier_name` | `Utf8` | **yes** | Modifier parameter name (e.g. `"alpha_JES"`) |
| `modifier_type` | `Utf8` | **yes** | One of: `normfactor`, `normsys`, `histosys`, `shapesys`, `shapefactor`, `staterror`, `lumi` |
| `data_hi` | `List<Float64>` | no | Up-variation data (histosys `hi_data`, or normsys `[hi]`) |
| `data_lo` | `List<Float64>` | no | Down-variation data (histosys `lo_data`, or normsys `[lo]`) |
| `data` | `List<Float64>` | no | Generic data (shapesys per-bin σ, staterror per-bin σ) |

### Column semantics by modifier type

| `modifier_type` | `data_hi` | `data_lo` | `data` |
|-----------------|-----------|-----------|--------|
| `normfactor` | — | — | — |
| `normsys` | `[hi_factor]` (1 element) | `[lo_factor]` (1 element) | — |
| `histosys` | per-bin up template | per-bin down template | — |
| `shapesys` | — | — | per-bin σ |
| `shapefactor` | — | — | — |
| `staterror` | — | — | per-bin σ |
| `lumi` | — | — | — |

---

## Table 3: Observations (optional)

| Column | Arrow Type | Required | Description |
|--------|-----------|----------|-------------|
| `channel` | `Utf8` | **yes** | Channel name |
| `data` | `List<Float64>` | **yes** | Observed bin counts |

If absent, Asimov data (sum of yields) is used.

---

## Table 4: Measurements (optional)

| Column | Arrow Type | Required | Description |
|--------|-----------|----------|-------------|
| `name` | `Utf8` | **yes** | Measurement name |
| `poi` | `Utf8` | **yes** | Parameter of interest name |

---

## File Layout

### Single-file (recommended for small workspaces)

All tables concatenated with a `_table` discriminator column, or stored as separate row groups.

### Multi-file (recommended for production)

```
workspace/
├── yields.parquet
├── modifiers.parquet
├── observations.parquet   # optional
├── measurements.parquet   # optional
└── manifest.json
```

### Manifest

```json
{
  "schema_version": "nextstat_binned_parquet_v2",
  "yields": "yields.parquet",
  "modifiers": "modifiers.parquet",
  "observations": "observations.parquet",
  "measurements": "measurements.parquet",
  "poi": "mu",
  "created": "2026-02-10T14:30:00Z",
  "nextstat_version": "0.9.0"
}
```

---

## Roundtrip Guarantees

- **pyhf JSON → Parquet → pyhf JSON** must produce an equivalent workspace (modulo JSON key ordering).
- All modifier types from the pyhf spec are supported.
- The `normfactor` and `shapefactor` types carry no numeric data — their rows in the modifiers table serve as declarations.

---

## Usage

### Export (Rust)

```rust
use ns_translate::arrow::export::{yields_to_record_batch, modifiers_to_record_batch};
use ns_translate::arrow::parquet::write_parquet;

let yields_batch = yields_to_record_batch(&model, params)?;
let modifiers_batch = modifiers_to_record_batch(&workspace)?;

write_parquet(Path::new("yields.parquet"), &[yields_batch])?;
write_parquet(Path::new("modifiers.parquet"), &[modifiers_batch])?;
```

### Ingest (Rust)

```rust
use ns_translate::arrow::parquet::read_parquet_batches;
use ns_translate::arrow::ingest::from_record_batches_with_modifiers;

let yields = read_parquet_batches(Path::new("yields.parquet"))?;
let modifiers = read_parquet_batches(Path::new("modifiers.parquet"))?;
let workspace = from_record_batches_with_modifiers(&yields, &modifiers, &config)?;
```

### Python

```python
import nextstat

# Export workspace to Parquet directory
nextstat.to_parquet(model, "workspace/", what="full")

# Read back
model = nextstat.from_parquet("workspace/", poi="mu")
```

---

## Comparison with pyhf JSON

| Aspect | pyhf JSON | Parquet v2 |
|--------|-----------|-----------|
| Size (100-channel workspace) | ~50 MB | ~5 MB (Zstd) |
| Query specific channel | Parse entire file | Row group pruning |
| Column selection | N/A | Read only `yields` column |
| Schema validation | JSON Schema (optional) | Arrow schema (enforced) |
| Streaming read | No | Yes (row group iteration) |
| Tool compatibility | pyhf, cabiern | DuckDB, Polars, Spark |
