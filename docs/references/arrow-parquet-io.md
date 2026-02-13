---
title: "Arrow / Parquet I/O (Histogram Tables)"
status: stable
---

# Arrow / Parquet I/O (Histogram Tables)

NextStat supports direct interchange with the Arrow ecosystem (PyArrow, Polars, DuckDB, Spark)
for **HistFactory-style histogram tables**.

There are two use-cases:

- **Ingest**: Parquet/Arrow -> `HistFactoryModel` (`nextstat.from_parquet`, `nextstat.from_arrow`).
- **Export**: `HistFactoryModel` -> Arrow (`nextstat.to_arrow` / IPC bytes).

## Histogram Table Contract (v1)

The histogram table schema is:

| Column | Arrow type | Required | Meaning |
|---|---|---|---|
| `channel` | `Utf8` (or `LargeUtf8`) | yes | Channel / region name |
| `sample` | `Utf8` (or `LargeUtf8`) | yes | Sample / process name |
| `yields` | `List<Float64>` (or `LargeList<Float64>`) | yes | Expected event counts per bin |
| `stat_error` | `List<Float64>` | no | Per-bin statistical uncertainties (optional) |

**Bin count rule (stability)**: within each `channel`, all rows must have the same `len(yields)`.
NextStat will fail-fast on inconsistent bin counts.

## Reading Parquet (no PyArrow required)

`nextstat.from_parquet(...)` reads Parquet via Rust `parquet`/`arrow` crates, so the **runtime
machine does not need PyArrow**:

```python
import nextstat

model = nextstat.from_parquet("histograms.parquet", poi="mu")
fit = nextstat.fit(model)
```

You only need PyArrow/Polars/etc. on the machine that *writes* the Parquet file.

## Writing Parquet + Manifest (recommended)

For reproducible pipelines, we recommend writing a small **manifest JSON** next to the Parquet file.

- Manifest schema: `docs/schemas/io/histograms_parquet_manifest_v1.schema.json`
- Python helper module: `nextstat.arrow_io` (requires `pyarrow`)

Install:

```bash
pip install "nextstat[io]"
```

Example:

```python
import pyarrow as pa
import nextstat
from nextstat.arrow_io import write_histograms_parquet

table = pa.table({
    "channel": ["SR", "SR", "CR"],
    "sample":  ["signal", "background", "background"],
    "yields":  [[5., 10., 15.], [100., 200., 150.], [500., 600.]],
})

manifest = write_histograms_parquet(table, "histograms.parquet", compression="zstd")

# Later (or on a different machine):
model = nextstat.from_parquet("histograms.parquet", poi="mu")
fit = nextstat.fit(model)
```

The manifest stores:

- Parquet file SHA-256
- Arrow schema fingerprint (string + SHA-256)
- Per-channel `n_bins` summary
- Optional `observations_path` if you provide observed counts (otherwise Asimov is used).

## Polars / DuckDB / Spark Interop

Any Arrow-compatible source can be turned into a PyArrow table:

```python
import polars as pl
import nextstat

df = pl.read_parquet("histograms.parquet")
model = nextstat.from_arrow(df.to_arrow(), poi="mu")
```

## High-Performance Unbinned Reads (Rust)

For unbinned event-level Parquet files (written by `nextstat convert` or `EventStore::to_parquet`),
`ns-translate` provides optimized readers:

| Function | Feature |
|----------|---------|
| `read_parquet_mmap()` | memmap2 zero-copy, no heap alloc for I/O |
| `read_parquet_mmap_projected()` | + column projection (35–50% faster) |
| `read_parquet_mmap_filtered()` | + row group predicate pushdown via min/max stats (up to 5×) |
| `read_parquet_mmap_parallel()` | + rayon parallel row group decode |
| `read_parquet_events_soa()` | All of the above → GPU-ready SoA f64 layout |

Benchmarks (MacBook M5, 100k events, 4 observables):

- Full decode: **557 µs**
- With 10% predicate filter: **159 µs** (5× speedup from row group pruning)

### Direct-to-GPU Upload

`read_parquet_events_soa()` produces a `ParquetEventData` struct that can be uploaded
directly to GPU memory without intermediate copies:

```rust
use ns_translate::arrow::parquet::read_parquet_events_soa;
use ns_compute::metal_parquet::upload_events_to_metal;

let data = read_parquet_events_soa(path, &columns, bounds, None)?;
let gpu = upload_events_to_metal(&device, &data.soa, data.weights.as_deref(), data.n_events, data.n_obs)?;
```

| Target | Module | Precision |
|--------|--------|-----------|
| CUDA | `ns_compute::cuda_parquet` | f64 (native) |
| Metal | `ns_compute::metal_parquet` | f32 (converted at upload) |

See `docs/gpu-contract.md` § Direct-to-GPU Parquet Pipeline for tolerance and performance details.

## Notes on Type Mapping

- Strings: `Utf8` and `LargeUtf8` are both accepted.
- Lists: `List<Float64>` and `LargeList<Float64>` are both accepted.
- Float64 is required for yields/stat_error to keep numeric behavior stable across languages.

