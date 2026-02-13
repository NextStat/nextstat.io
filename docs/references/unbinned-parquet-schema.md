# Unbinned Event-Level Parquet Schema (v1)

Specification for event-level data stored in Apache Parquet format, used as input to NextStat's unbinned likelihood fits.

**Status:** v1 (2026-02-10)

---

## Motivation

ROOT TTree is the primary event-level data source for HEP analyses, but many ML pipelines (Polars, DuckDB, Spark, PyArrow) work natively with Parquet. This schema enables:

- Unbinned fits without ROOT files
- Data exchange with the Arrow/Parquet ecosystem
- Reproducible archival of event-level datasets

---

## Schema

### Event Table (required)

One Parquet file containing per-event rows. Each observable is a top-level `Float64` column.

| Column | Arrow Type | Required | Description |
|--------|-----------|----------|-------------|
| *`<observable_name>`* | `Float64` | **yes** (one per observable) | Observable value for this event (e.g. `mass`, `pt`, `eta`) |
| `_weight` | `Float64` | no | Per-event weight. Default: 1.0 if absent. Must be non-negative and finite. |
| `_channel` | `Utf8` | no | Channel label for multi-channel files. When present, NextStat can filter a single channel via `channels[].data.channel` in the unbinned spec. |

**Constraints:**
- Column names must match the `name` field of the corresponding `ObservableSpec`.
- All observable values must be finite (`!NaN`, `!Inf`).
- Values should lie within the declared observable bounds `(lo, hi)`. Out-of-bounds values are rejected.
- Extra columns are silently ignored.

### Example

A 2-observable dataset (invariant mass + transverse momentum) with per-event weights:

```
mass (Float64) | pt (Float64) | _weight (Float64)
─────────────────────────────────────────────────
125.3          | 45.2         | 1.0
126.1          | 67.8         | 0.95
124.7          | 33.1         | 1.02
```

---

## Manifest (optional)

A sidecar JSON file linking the Parquet data to an unbinned spec YAML.

**Filename convention:** `<name>_manifest.json`

```json
{
  "schema_version": "nextstat_unbinned_parquet_v1",
  "data_files": [
    {
      "path": "events.parquet",
      "sha256": "abc123...",
      "n_events": 100000,
      "observables": ["mass", "pt"],
      "has_weights": true
    }
  ],
  "created": "2026-02-10T14:00:00Z",
  "nextstat_version": "0.9.0"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | **yes** | Must be `nextstat_unbinned_parquet_v1` |
| `data_files` | array | **yes** | List of Parquet data files |
| `data_files[].path` | string | **yes** | Relative path to Parquet file |
| `data_files[].sha256` | string | no | SHA-256 of the Parquet file for integrity |
| `data_files[].n_events` | integer | no | Row count (informational) |
| `data_files[].observables` | array of string | no | Observable column names present |
| `data_files[].has_weights` | boolean | no | Whether `weight` column exists |
| `created` | string (ISO 8601) | no | Creation timestamp |
| `nextstat_version` | string | no | NextStat version that wrote the file |

---

## Usage

### Rust (ns-translate)

```rust
use ns_translate::arrow::unbinned::{event_store_from_parquet, event_store_to_parquet};
use ns_unbinned::event_store::ObservableSpec;

let observables = vec![
    ObservableSpec::branch("mass", (100.0, 200.0)),
    ObservableSpec::branch("pt", (0.0, 500.0)),
];

// Read
let store = event_store_from_parquet(Path::new("events.parquet"), &observables)?;

// Write
event_store_to_parquet(&store, Path::new("output.parquet"))?;
```

### Python

```python
import nextstat

# Read Parquet → unbinned fit (planned API)
model = nextstat.unbinned_from_parquet(
    "events.parquet",
    config="model.yaml",
)
result = nextstat.unbinned_fit(model)
```

### Unbinned Spec YAML Integration

Reference a Parquet file in the channel `data` block:

```yaml
schema_version: nextstat_unbinned_spec_v0
model:
  poi: mu
  parameters:
    - name: mu
      init: 1.0
      bounds: [0.0, 10.0]
channels:
  - name: signal_region
    data:
      file: events.parquet       # Parquet or ROOT
      format: parquet             # explicit format hint
      # Optional: if the Parquet file contains a `_channel` column, select one channel:
      # channel: "SR"
    observables:
      - name: mass
        bounds: [100.0, 200.0]
    processes:
      - name: signal
        pdf: { type: gaussian, observable: mass }
        yield: { expr: "mu * 100" }
```

---

## Comparison with Binned Histogram Schema

| Aspect | Binned (existing) | Unbinned Event-Level (this spec) |
|--------|-------------------|----------------------------------|
| Row semantics | One row per (channel, sample) | One row per event |
| Columns | `channel`, `sample`, `yields`, `stat_error` | Observable floats + optional `weight` |
| Use case | HistFactory binned fits | Unbinned likelihood fits |
| Modifier support | Via pyhf JSON (normsys, histosys, etc.) | Via unbinned spec YAML (rate modifiers) |

---

## Future Extensions (v2)

- **Per-event systematic weights** — additional `Float64` columns for weight variations (e.g. `weight_sys_up`, `weight_sys_down`)
- **Multi-channel** — `channel` column (`Utf8`) to partition events by analysis region within a single file
- **Parquet footer metadata** — embed `nextstat_version`, `schema_version`, observable bounds in Parquet key-value metadata
