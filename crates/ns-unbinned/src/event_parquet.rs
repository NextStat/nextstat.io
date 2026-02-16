//! Parquet / Arrow event I/O for unbinned analysis.
//!
//! Defines the **unbinned event table schema v1** and provides read/write
//! functions that bridge Arrow [`RecordBatch`] ↔ [`EventStore`].
//!
//! # Schema: `nextstat_unbinned_events_v1`
//!
//! ## Columns
//!
//! | Column             | Arrow Type  | Required | Description                       |
//! |--------------------|-------------|----------|-----------------------------------|
//! | `<observable>`     | `Float64`   | ≥ 1      | One column per observable         |
//! | `_weight`          | `Float64`   | no       | Per-event weight                  |
//! | `_channel`         | `Utf8`      | no       | Channel name (multi-channel file) |
//!
//! ## Parquet key-value metadata
//!
//! | Key                          | Value                                        |
//! |------------------------------|----------------------------------------------|
//! | `nextstat.schema_version`    | `"nextstat_unbinned_events_v1"`              |
//! | `nextstat.observables`       | JSON array: `[{"name":"x","bounds":[a,b]}]`  |
//!
//! The `nextstat.observables` metadata is **required** when writing and used
//! during reading to reconstruct [`ObservableSpec`] bounds.  Without it the
//! reader falls back to `(-inf, inf)` bounds (caller must supply bounds).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, AsArray, Float64Array, LargeStringArray, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::event_store::{EventStore, ObservableSpec};
use ns_core::{Error, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Schema version string embedded in Parquet key-value metadata.
pub const UNBINNED_EVENTS_SCHEMA_V1: &str = "nextstat_unbinned_events_v1";

/// Parquet metadata key for the schema version.
pub const META_KEY_SCHEMA_VERSION: &str = "nextstat.schema_version";

/// Parquet metadata key for observable definitions (JSON).
pub const META_KEY_OBSERVABLES: &str = "nextstat.observables";

/// Reserved column name for per-event weights.
pub const WEIGHT_COLUMN: &str = "_weight";

/// Reserved column name for channel labels in multi-channel files.
pub const CHANNEL_COLUMN: &str = "_channel";

// ---------------------------------------------------------------------------
// Observable metadata (de)serialization
// ---------------------------------------------------------------------------

/// JSON-serializable observable descriptor stored in Parquet metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObservableMeta {
    /// Observable name (matches column name in Parquet/Arrow).
    pub name: String,
    /// Support bounds `[low, high]` for PDF normalization.
    pub bounds: [f64; 2],
}

impl From<&ObservableSpec> for ObservableMeta {
    fn from(o: &ObservableSpec) -> Self {
        Self { name: o.name.clone(), bounds: [o.bounds.0, o.bounds.1] }
    }
}

impl ObservableMeta {
    /// Convert to [`ObservableSpec`] (expr == name for Parquet sources).
    pub fn to_observable_spec(&self) -> ObservableSpec {
        ObservableSpec::branch(&self.name, (self.bounds[0], self.bounds[1]))
    }
}

// ---------------------------------------------------------------------------
// Write: EventStore → Arrow RecordBatch (→ Parquet)
// ---------------------------------------------------------------------------

/// Build an Arrow [`RecordBatch`] from an [`EventStore`].
///
/// The schema embeds observable bounds in its metadata under
/// [`META_KEY_OBSERVABLES`].
pub fn event_store_to_record_batch(store: &EventStore) -> Result<RecordBatch> {
    let obs_meta: Vec<ObservableMeta> = store
        .column_names()
        .iter()
        .map(|name| {
            let bounds = store.bounds(name).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            ObservableMeta { name: name.clone(), bounds: [bounds.0, bounds.1] }
        })
        .collect();

    let obs_json = serde_json::to_string(&obs_meta)
        .map_err(|e| Error::Validation(format!("failed to serialize observable metadata: {e}")))?;

    let mut fields: Vec<Field> =
        store.column_names().iter().map(|n| Field::new(n, DataType::Float64, false)).collect();

    if store.weights().is_some() {
        fields.push(Field::new(WEIGHT_COLUMN, DataType::Float64, false));
    }

    let metadata = HashMap::from([
        (META_KEY_SCHEMA_VERSION.to_string(), UNBINNED_EVENTS_SCHEMA_V1.to_string()),
        (META_KEY_OBSERVABLES.to_string(), obs_json),
    ]);

    let schema = Arc::new(Schema::new(fields).with_metadata(metadata));

    let mut arrays: Vec<Arc<dyn Array>> = store
        .column_names()
        .iter()
        .map(|name| {
            let col = store.column(name).unwrap();
            Arc::new(Float64Array::from(col.to_vec())) as Arc<dyn Array>
        })
        .collect();

    if let Some(w) = store.weights() {
        arrays.push(Arc::new(Float64Array::from(w.to_vec())) as Arc<dyn Array>);
    }

    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::Validation(format!("failed to build RecordBatch: {e}")))
}

/// Write an [`EventStore`] to a Parquet file.
///
/// Uses Zstd compression if the `arrow-io-zstd` feature is enabled, otherwise Snappy.
pub fn write_event_parquet(store: &EventStore, path: &Path) -> Result<()> {
    let batch = event_store_to_record_batch(store)?;
    let file = std::fs::File::create(path)
        .map_err(|e| Error::Validation(format!("failed to create {}: {e}", path.display())))?;

    let compression = default_compression();
    let props =
        parquet::file::properties::WriterProperties::builder().set_compression(compression).build();

    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| Error::Validation(format!("failed to create Parquet writer: {e}")))?;
    writer.write(&batch).map_err(|e| Error::Validation(format!("failed to write Parquet: {e}")))?;
    writer
        .close()
        .map_err(|e| Error::Validation(format!("failed to close Parquet writer: {e}")))?;
    Ok(())
}

/// Write an [`EventStore`] to Parquet bytes in memory.
pub fn write_event_parquet_bytes(store: &EventStore) -> Result<Vec<u8>> {
    let batch = event_store_to_record_batch(store)?;
    let compression = default_compression();
    let props =
        parquet::file::properties::WriterProperties::builder().set_compression(compression).build();

    let mut buf = Vec::new();
    let mut writer = parquet::arrow::ArrowWriter::try_new(&mut buf, batch.schema(), Some(props))
        .map_err(|e| Error::Validation(format!("failed to create Parquet writer: {e}")))?;
    writer.write(&batch).map_err(|e| Error::Validation(format!("failed to write Parquet: {e}")))?;
    writer
        .close()
        .map_err(|e| Error::Validation(format!("failed to close Parquet writer: {e}")))?;
    Ok(buf)
}

/// Write multiple channel `EventStore`s into a single Parquet file with a `_channel` column.
///
/// All stores must have identical observable columns (same names in the same order) and must
/// either all have per-event weights or none have weights.
pub fn write_event_parquet_multi_channel(
    stores: &[(String, EventStore)],
    path: &Path,
) -> Result<()> {
    let bytes = write_event_parquet_multi_channel_bytes(stores)?;
    std::fs::write(path, bytes)
        .map_err(|e| Error::Validation(format!("failed to write {}: {e}", path.display())))?;
    Ok(())
}

/// Write multiple channel `EventStore`s into Parquet bytes with a `_channel` column.
pub fn write_event_parquet_multi_channel_bytes(stores: &[(String, EventStore)]) -> Result<Vec<u8>> {
    if stores.is_empty() {
        return Err(Error::Validation(
            "write_event_parquet_multi_channel requires at least one store".into(),
        ));
    }

    let (first_ch, first_store) = &stores[0];
    if first_ch.trim().is_empty() {
        return Err(Error::Validation("channel name must be non-empty".into()));
    }

    let col_names = first_store.column_names().to_vec();
    if col_names.is_empty() {
        return Err(Error::Validation(
            "EventStore must have at least one observable column".into(),
        ));
    }

    let has_weights = first_store.weights().is_some();

    for (ch, s) in stores {
        if ch.trim().is_empty() {
            return Err(Error::Validation("channel name must be non-empty".into()));
        }
        if s.column_names() != col_names.as_slice() {
            return Err(Error::Validation(format!(
                "multi-channel Parquet write requires identical column_names across stores; got '{}' columns {:?}, expected {:?}",
                ch,
                s.column_names(),
                col_names
            )));
        }
        if s.weights().is_some() != has_weights {
            return Err(Error::Validation(
                "multi-channel Parquet write requires all stores to either have weights or not have weights"
                    .into(),
            ));
        }
        for name in &col_names {
            let a = first_store.bounds(name).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            let b = s.bounds(name).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            if a != b {
                return Err(Error::Validation(format!(
                    "multi-channel Parquet write requires identical bounds for column '{}'; channel '{}' has {:?}, expected {:?}",
                    name, ch, b, a
                )));
            }
        }
    }

    let total_rows: usize = stores.iter().map(|(_, s)| s.n_events()).sum();
    if total_rows == 0 {
        return Err(Error::Validation(
            "multi-channel Parquet write requires at least one row".into(),
        ));
    }

    // Build schema metadata from the first store (consistent with single-channel writer).
    let obs_meta: Vec<ObservableMeta> = col_names
        .iter()
        .map(|name| {
            let bounds = first_store.bounds(name).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            ObservableMeta { name: name.clone(), bounds: [bounds.0, bounds.1] }
        })
        .collect();
    let obs_json = serde_json::to_string(&obs_meta)
        .map_err(|e| Error::Validation(format!("failed to serialize observable metadata: {e}")))?;

    let mut fields: Vec<Field> =
        col_names.iter().map(|n| Field::new(n, DataType::Float64, false)).collect();
    if has_weights {
        fields.push(Field::new(WEIGHT_COLUMN, DataType::Float64, false));
    }
    fields.push(Field::new(CHANNEL_COLUMN, DataType::Utf8, false));

    let metadata = HashMap::from([
        (META_KEY_SCHEMA_VERSION.to_string(), UNBINNED_EVENTS_SCHEMA_V1.to_string()),
        (META_KEY_OBSERVABLES.to_string(), obs_json),
    ]);
    let schema = Arc::new(Schema::new(fields).with_metadata(metadata));

    // Concatenate columns across stores.
    let mut arrays: Vec<Arc<dyn Array>> =
        Vec::with_capacity(col_names.len() + usize::from(has_weights) + 1);
    for name in &col_names {
        let mut out = Vec::<f64>::with_capacity(total_rows);
        for (_, s) in stores {
            out.extend_from_slice(s.column(name).ok_or_else(|| {
                Error::Validation(format!("missing column '{}' in EventStore", name))
            })?);
        }
        arrays.push(Arc::new(Float64Array::from(out)) as Arc<dyn Array>);
    }

    if has_weights {
        let mut out = Vec::<f64>::with_capacity(total_rows);
        for (_, s) in stores {
            out.extend_from_slice(s.weights().unwrap());
        }
        arrays.push(Arc::new(Float64Array::from(out)) as Arc<dyn Array>);
    }

    // Channel column.
    let mut b = StringBuilder::new();
    for (ch, s) in stores {
        for _ in 0..s.n_events() {
            b.append_value(ch);
        }
    }
    arrays.push(Arc::new(b.finish()) as Arc<dyn Array>);

    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::Validation(format!("failed to build RecordBatch: {e}")))?;

    let compression = default_compression();
    let props =
        parquet::file::properties::WriterProperties::builder().set_compression(compression).build();

    let mut buf = Vec::new();
    let mut writer = parquet::arrow::ArrowWriter::try_new(&mut buf, batch.schema(), Some(props))
        .map_err(|e| Error::Validation(format!("failed to create Parquet writer: {e}")))?;
    writer.write(&batch).map_err(|e| Error::Validation(format!("failed to write Parquet: {e}")))?;
    writer
        .close()
        .map_err(|e| Error::Validation(format!("failed to close Parquet writer: {e}")))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Read: Parquet / Arrow RecordBatch → EventStore
// ---------------------------------------------------------------------------

/// Extract observable metadata from an Arrow schema's key-value metadata.
///
/// Returns `None` if the metadata key is absent.
pub fn observables_from_schema_metadata(schema: &Schema) -> Result<Option<Vec<ObservableMeta>>> {
    let meta = schema.metadata();
    let Some(json_str) = meta.get(META_KEY_OBSERVABLES) else {
        return Ok(None);
    };
    let obs: Vec<ObservableMeta> = serde_json::from_str(json_str)
        .map_err(|e| Error::Validation(format!("invalid {META_KEY_OBSERVABLES} metadata: {e}")))?;
    Ok(Some(obs))
}

/// Build an [`EventStore`] from an Arrow [`RecordBatch`].
///
/// If `observables` is `None`, the function reads observable definitions from
/// the schema's key-value metadata ([`META_KEY_OBSERVABLES`]).  If metadata is
/// also absent, all `Float64` columns (except `_weight` / `_channel`) are
/// treated as observables with `(-inf, inf)` bounds.
pub fn event_store_from_record_batch(
    batch: &RecordBatch,
    observables: Option<&[ObservableSpec]>,
) -> Result<EventStore> {
    let schema = batch.schema();

    // Resolve observable specs.
    let obs_specs: Vec<ObservableSpec> = if let Some(provided) = observables {
        provided.to_vec()
    } else if let Some(meta_obs) = observables_from_schema_metadata(&schema)? {
        meta_obs.iter().map(|m| m.to_observable_spec()).collect()
    } else {
        // Fallback: all Float64 columns except reserved names.
        schema
            .fields()
            .iter()
            .filter(|f| {
                f.data_type() == &DataType::Float64
                    && f.name() != WEIGHT_COLUMN
                    && f.name() != CHANNEL_COLUMN
            })
            .map(|f| ObservableSpec::branch(f.name(), (f64::NEG_INFINITY, f64::INFINITY)))
            .collect()
    };

    if obs_specs.is_empty() {
        return Err(Error::Validation("no observable columns found in RecordBatch".into()));
    }

    // Extract columns.
    let mut columns: Vec<(String, Vec<f64>)> = Vec::with_capacity(obs_specs.len());
    for obs in &obs_specs {
        let col_idx = schema.index_of(&obs.name).map_err(|_| {
            Error::Validation(format!("missing observable column '{}' in RecordBatch", obs.name))
        })?;
        let arr = batch.column(col_idx);
        if arr.data_type() != &DataType::Float64 {
            return Err(Error::Validation(format!(
                "column '{}' has type {:?}, expected Float64",
                obs.name,
                arr.data_type()
            )));
        }
        let f64_arr = arr.as_primitive::<arrow::datatypes::Float64Type>();
        columns.push((obs.name.clone(), f64_arr.values().to_vec()));
    }

    // Extract optional weights.
    let weights = if let Ok(idx) = schema.index_of(WEIGHT_COLUMN) {
        let arr = batch.column(idx);
        if arr.data_type() != &DataType::Float64 {
            return Err(Error::Validation(format!(
                "column '{WEIGHT_COLUMN}' has type {:?}, expected Float64",
                arr.data_type()
            )));
        }
        let f64_arr = arr.as_primitive::<arrow::datatypes::Float64Type>();
        Some(f64_arr.values().to_vec())
    } else {
        None
    };

    EventStore::from_columns(obs_specs, columns, weights)
}

/// Build [`EventStore`]s from Arrow [`RecordBatch`]es, optionally splitting by `_channel`.
///
/// If `_channel` column is present, returns one `EventStore` per unique channel value
/// (keyed by channel name). Otherwise returns a single entry keyed by `channel_name`.
pub fn event_stores_from_record_batches(
    batches: &[RecordBatch],
    observables: Option<&[ObservableSpec]>,
    default_channel: &str,
) -> Result<Vec<(String, EventStore)>> {
    if batches.is_empty() {
        return Err(Error::Validation("no RecordBatches provided".into()));
    }

    // Check if _channel column exists in the first batch.
    let has_channel = batches[0].schema().index_of(CHANNEL_COLUMN).is_ok();

    if !has_channel {
        // Single-channel: merge all batches into one EventStore.
        // For simplicity, concat row-wise.
        let merged = arrow::compute::concat_batches(&batches[0].schema(), batches)
            .map_err(|e| Error::Validation(format!("failed to concat batches: {e}")))?;
        let store = event_store_from_record_batch(&merged, observables)?;
        return Ok(vec![(default_channel.to_string(), store)]);
    }

    // Multi-channel: partition by _channel value.
    let mut channel_batches: HashMap<String, Vec<RecordBatch>> = HashMap::new();

    for batch in batches {
        let ch_idx = batch.schema().index_of(CHANNEL_COLUMN).map_err(|_| {
            Error::Validation("inconsistent schema: _channel missing in some batches".into())
        })?;
        let ch_col = batch.column(ch_idx);

        // Get unique channels in this batch and filter rows.
        let mut channels_in_batch: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..batch.num_rows() {
            let ch = channel_row_value(ch_col, i)?;
            channels_in_batch.entry(ch).or_default().push(i);
        }

        for (ch_name, indices) in channels_in_batch {
            let n_rows = batch.num_rows();
            let mut mask = vec![false; n_rows];
            for &i in &indices {
                mask[i] = true;
            }
            let bool_arr = arrow::array::BooleanArray::from(mask);
            let filtered_batch =
                arrow::compute::filter_record_batch(batch, &bool_arr).map_err(|e| {
                    Error::Validation(format!(
                        "failed to filter batch for channel '{ch_name}': {e}"
                    ))
                })?;
            channel_batches.entry(ch_name).or_default().push(filtered_batch);
        }
    }

    let mut result = Vec::with_capacity(channel_batches.len());
    for (ch_name, ch_batches) in channel_batches {
        let schema = ch_batches[0].schema();
        let merged = arrow::compute::concat_batches(&schema, &ch_batches).map_err(|e| {
            Error::Validation(format!("failed to concat batches for '{ch_name}': {e}"))
        })?;

        // Drop the _channel column before constructing EventStore.
        let ch_idx = merged.schema().index_of(CHANNEL_COLUMN).unwrap();
        let fields: Vec<Field> = merged
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ch_idx)
            .map(|(_, f)| f.as_ref().clone())
            .collect();
        let arrays: Vec<Arc<dyn Array>> = (0..merged.num_columns())
            .filter(|i| *i != ch_idx)
            .map(|i| merged.column(i).clone())
            .collect();
        let new_schema =
            Arc::new(Schema::new(fields).with_metadata(merged.schema().metadata().clone()));
        let merged = RecordBatch::try_new(new_schema, arrays)
            .map_err(|e| Error::Validation(format!("failed to drop _channel column: {e}")))?;

        let store = event_store_from_record_batch(&merged, observables)?;
        result.push((ch_name, store));
    }

    // Sort by channel name for deterministic order.
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

/// Read a Parquet file into an [`EventStore`].
///
/// Observable bounds are read from Parquet key-value metadata if available.
/// Override with explicit `observables` to use custom bounds.
pub fn read_event_parquet(
    path: &Path,
    observables: Option<&[ObservableSpec]>,
) -> Result<EventStore> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Validation(format!("failed to open {}: {e}", path.display())))?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::Validation(format!("failed to read Parquet: {e}")))?;

    // Capture the Arrow schema (with key-value metadata) before building the reader.
    let full_schema = builder.schema().clone();

    let reader = builder
        .build()
        .map_err(|e| Error::Validation(format!("failed to build Parquet reader: {e}")))?;

    let batches: std::result::Result<Vec<_>, _> = reader.collect();
    let batches =
        batches.map_err(|e| Error::Validation(format!("failed to read Parquet batches: {e}")))?;

    if batches.is_empty() {
        return Err(Error::Validation("Parquet file contains no data".into()));
    }

    let merged = concat_batches_with_schema(&full_schema, &batches)?;
    event_store_from_record_batch(&merged, observables)
}

/// Read a multi-channel Parquet file and select events for a given `_channel` value.
pub fn read_event_parquet_channel(
    path: &Path,
    observables: Option<&[ObservableSpec]>,
    channel: &str,
) -> Result<EventStore> {
    if channel.trim().is_empty() {
        return Err(Error::Validation("channel must be non-empty".into()));
    }

    let file = std::fs::File::open(path)
        .map_err(|e| Error::Validation(format!("failed to open {}: {e}", path.display())))?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::Validation(format!("failed to read Parquet: {e}")))?;

    // Capture the Arrow schema (with key-value metadata) before building the reader.
    let full_schema = builder.schema().clone();
    if full_schema.index_of(CHANNEL_COLUMN).is_err() {
        return Err(Error::Validation(format!(
            "Parquet file has no '{CHANNEL_COLUMN}' column; cannot select channel='{channel}'"
        )));
    }

    let reader = builder
        .build()
        .map_err(|e| Error::Validation(format!("failed to build Parquet reader: {e}")))?;

    let batches: std::result::Result<Vec<_>, _> = reader.collect();
    let batches =
        batches.map_err(|e| Error::Validation(format!("failed to read Parquet batches: {e}")))?;

    if batches.is_empty() {
        return Err(Error::Validation("Parquet file contains no data".into()));
    }

    let stores = event_stores_from_record_batches(&batches, observables, "__default")?;
    let mut available = Vec::with_capacity(stores.len());
    for (name, store) in stores {
        if name == channel {
            return Ok(store);
        }
        available.push(name);
    }
    available.sort();
    Err(Error::Validation(format!(
        "requested channel='{channel}' not found in Parquet file; available channels: {:?}",
        available
    )))
}

/// Read a Parquet file into an [`EventStore`] from in-memory bytes.
pub fn read_event_parquet_bytes(
    data: &[u8],
    observables: Option<&[ObservableSpec]>,
) -> Result<EventStore> {
    let buf = bytes::Bytes::copy_from_slice(data);
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(buf)
        .map_err(|e| Error::Validation(format!("failed to read Parquet bytes: {e}")))?;

    let full_schema = builder.schema().clone();

    let reader = builder
        .build()
        .map_err(|e| Error::Validation(format!("failed to build Parquet reader: {e}")))?;

    let batches: std::result::Result<Vec<_>, _> = reader.collect();
    let batches =
        batches.map_err(|e| Error::Validation(format!("failed to read Parquet batches: {e}")))?;

    if batches.is_empty() {
        return Err(Error::Validation("Parquet bytes contain no data".into()));
    }

    let merged = concat_batches_with_schema(&full_schema, &batches)?;
    event_store_from_record_batch(&merged, observables)
}

/// Read multi-channel Parquet bytes and select events for a given `_channel` value.
pub fn read_event_parquet_bytes_channel(
    data: &[u8],
    observables: Option<&[ObservableSpec]>,
    channel: &str,
) -> Result<EventStore> {
    if channel.trim().is_empty() {
        return Err(Error::Validation("channel must be non-empty".into()));
    }

    let buf = bytes::Bytes::copy_from_slice(data);
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(buf)
        .map_err(|e| Error::Validation(format!("failed to read Parquet bytes: {e}")))?;

    let full_schema = builder.schema().clone();
    if full_schema.index_of(CHANNEL_COLUMN).is_err() {
        return Err(Error::Validation(format!(
            "Parquet bytes have no '{CHANNEL_COLUMN}' column; cannot select channel='{channel}'"
        )));
    }

    let reader = builder
        .build()
        .map_err(|e| Error::Validation(format!("failed to build Parquet reader: {e}")))?;

    let batches: std::result::Result<Vec<_>, _> = reader.collect();
    let batches =
        batches.map_err(|e| Error::Validation(format!("failed to read Parquet batches: {e}")))?;

    if batches.is_empty() {
        return Err(Error::Validation("Parquet bytes contain no data".into()));
    }

    let stores = event_stores_from_record_batches(&batches, observables, "__default")?;
    let mut available = Vec::with_capacity(stores.len());
    for (name, store) in stores {
        if name == channel {
            return Ok(store);
        }
        available.push(name);
    }
    available.sort();
    Err(Error::Validation(format!(
        "requested channel='{channel}' not found in Parquet bytes; available channels: {:?}",
        available
    )))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Concat batches using the full schema (with key-value metadata from the Parquet footer).
///
/// `arrow::compute::concat_batches` uses the schema from the first batch, which may lack
/// key-value metadata that was in the original Parquet file.  This helper merges using the
/// provided `full_schema` so metadata (e.g. `nextstat.observables`) survives roundtrip.
fn concat_batches_with_schema(
    full_schema: &Arc<Schema>,
    batches: &[RecordBatch],
) -> Result<RecordBatch> {
    let merged = arrow::compute::concat_batches(full_schema, batches)
        .map_err(|e| Error::Validation(format!("failed to concat Parquet batches: {e}")))?;
    Ok(merged)
}

fn channel_row_value(col: &Arc<dyn Array>, row: usize) -> Result<String> {
    if col.is_null(row) {
        return Err(Error::Validation(format!("row {row}: '{CHANNEL_COLUMN}' must not be null")));
    }
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                Error::Validation(format!(
                    "column '{CHANNEL_COLUMN}' has type {:?}, expected Utf8",
                    col.data_type()
                ))
            })?;
            Ok(arr.value(row).to_string())
        }
        DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
                Error::Validation(format!(
                    "column '{CHANNEL_COLUMN}' has type {:?}, expected LargeUtf8",
                    col.data_type()
                ))
            })?;
            Ok(arr.value(row).to_string())
        }
        _ => Err(Error::Validation(format!(
            "column '{CHANNEL_COLUMN}' has type {:?}, expected Utf8/LargeUtf8",
            col.data_type()
        ))),
    }
}

fn default_compression() -> parquet::basic::Compression {
    #[cfg(feature = "arrow-io-zstd")]
    {
        parquet::basic::Compression::ZSTD(Default::default())
    }
    #[cfg(not(feature = "arrow-io-zstd"))]
    {
        parquet::basic::Compression::SNAPPY
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use arrow::array::{Float64Array, LargeStringBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn make_test_store() -> EventStore {
        let obs = vec![
            ObservableSpec::branch("mass", (100.0, 180.0)),
            ObservableSpec::branch("pt", (0.0, 500.0)),
        ];
        let columns = vec![
            ("mass".to_string(), vec![120.0, 130.0, 140.0, 150.0]),
            ("pt".to_string(), vec![50.0, 100.0, 200.0, 300.0]),
        ];
        let weights = Some(vec![1.0, 0.8, 1.2, 0.5]);
        EventStore::from_columns(obs, columns, weights).unwrap()
    }

    #[test]
    fn test_record_batch_roundtrip() {
        let store = make_test_store();
        let batch = event_store_to_record_batch(&store).unwrap();

        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 3); // mass, pt, _weight

        // Check metadata.
        let schema = batch.schema();
        let meta = schema.metadata();
        assert_eq!(meta.get(META_KEY_SCHEMA_VERSION).unwrap(), UNBINNED_EVENTS_SCHEMA_V1);
        assert!(meta.contains_key(META_KEY_OBSERVABLES));

        // Roundtrip back to EventStore (using metadata for bounds).
        let store2 = event_store_from_record_batch(&batch, None).unwrap();
        assert_eq!(store2.n_events(), 4);
        assert_eq!(store2.column("mass").unwrap(), &[120.0, 130.0, 140.0, 150.0]);
        assert_eq!(store2.column("pt").unwrap(), &[50.0, 100.0, 200.0, 300.0]);
        assert_eq!(store2.weights().unwrap(), &[1.0, 0.8, 1.2, 0.5]);
        assert_eq!(store2.bounds("mass").unwrap(), (100.0, 180.0));
        assert_eq!(store2.bounds("pt").unwrap(), (0.0, 500.0));
    }

    #[test]
    fn test_parquet_bytes_roundtrip() {
        let store = make_test_store();
        let bytes = write_event_parquet_bytes(&store).unwrap();
        assert!(!bytes.is_empty());

        let store2 = read_event_parquet_bytes(&bytes, None).unwrap();
        assert_eq!(store2.n_events(), 4);
        assert_eq!(store2.column("mass").unwrap(), store.column("mass").unwrap());
        assert_eq!(store2.column("pt").unwrap(), store.column("pt").unwrap());
        assert_eq!(store2.weights().unwrap(), store.weights().unwrap());
        assert_eq!(store2.bounds("mass").unwrap(), (100.0, 180.0));
    }

    #[test]
    fn test_parquet_multi_channel_selects_channel() {
        let sr = make_test_store();
        let cr = {
            let obs = vec![
                ObservableSpec::branch("mass", (100.0, 180.0)),
                ObservableSpec::branch("pt", (0.0, 500.0)),
            ];
            let columns = vec![
                ("mass".to_string(), vec![101.0, 102.0]),
                ("pt".to_string(), vec![10.0, 20.0]),
            ];
            let weights = Some(vec![2.0, 3.0]);
            EventStore::from_columns(obs, columns, weights).unwrap()
        };

        let bytes = write_event_parquet_multi_channel_bytes(&[
            ("SR".to_string(), sr.clone()),
            ("CR".to_string(), cr.clone()),
        ])
        .unwrap();

        let sr2 = read_event_parquet_bytes_channel(&bytes, None, "SR").unwrap();
        assert_eq!(sr2.n_events(), 4);
        assert_eq!(sr2.column("mass").unwrap(), sr.column("mass").unwrap());
        assert_eq!(sr2.weights().unwrap(), sr.weights().unwrap());

        let cr2 = read_event_parquet_bytes_channel(&bytes, None, "CR").unwrap();
        assert_eq!(cr2.n_events(), 2);
        assert_eq!(cr2.column("mass").unwrap(), cr.column("mass").unwrap());
        assert_eq!(cr2.weights().unwrap(), cr.weights().unwrap());
    }

    #[test]
    fn test_record_batch_no_weights() {
        let obs = vec![ObservableSpec::branch("x", (0.0, 10.0))];
        let columns = vec![("x".to_string(), vec![1.0, 2.0, 3.0])];
        let store = EventStore::from_columns(obs, columns, None).unwrap();

        let batch = event_store_to_record_batch(&store).unwrap();
        assert_eq!(batch.num_columns(), 1); // x only

        let store2 = event_store_from_record_batch(&batch, None).unwrap();
        assert_eq!(store2.n_events(), 3);
        assert!(store2.weights().is_none());
    }

    #[test]
    fn test_explicit_observables_override_metadata() {
        let store = make_test_store();
        let batch = event_store_to_record_batch(&store).unwrap();

        // Override bounds via explicit observables.
        let custom = vec![
            ObservableSpec::branch("mass", (110.0, 170.0)),
            ObservableSpec::branch("pt", (10.0, 400.0)),
        ];
        let store2 = event_store_from_record_batch(&batch, Some(&custom)).unwrap();
        assert_eq!(store2.bounds("mass").unwrap(), (110.0, 170.0));
        assert_eq!(store2.bounds("pt").unwrap(), (10.0, 400.0));
    }

    #[test]
    fn test_missing_column_error() {
        let obs = vec![ObservableSpec::branch("x", (0.0, 10.0))];
        let columns = vec![("x".to_string(), vec![1.0, 2.0])];
        let store = EventStore::from_columns(obs, columns, None).unwrap();
        let batch = event_store_to_record_batch(&store).unwrap();

        let bad_obs = vec![ObservableSpec::branch("nonexistent", (0.0, 1.0))];
        let result = event_store_from_record_batch(&batch, Some(&bad_obs));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("nonexistent"));
    }

    #[test]
    fn test_event_stores_from_record_batches_accepts_largeutf8_channel() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("mass", DataType::Float64, false),
            Field::new(CHANNEL_COLUMN, DataType::LargeUtf8, false),
        ]));

        let mass = Arc::new(Float64Array::from(vec![120.0, 130.0, 140.0, 150.0])) as Arc<dyn Array>;
        let mut ch_builder = LargeStringBuilder::new();
        ch_builder.append_value("SR");
        ch_builder.append_value("SR");
        ch_builder.append_value("CR");
        ch_builder.append_value("CR");
        let ch = Arc::new(ch_builder.finish()) as Arc<dyn Array>;

        let batch = RecordBatch::try_new(schema, vec![mass, ch]).unwrap();
        let obs = vec![ObservableSpec::branch("mass", (100.0, 180.0))];
        let stores = event_stores_from_record_batches(&[batch], Some(&obs), "__default").unwrap();

        assert_eq!(stores.len(), 2);
        assert_eq!(stores[0].0, "CR");
        assert_eq!(stores[0].1.n_events(), 2);
        assert_eq!(stores[1].0, "SR");
        assert_eq!(stores[1].1.n_events(), 2);
    }
}
