//! Arrow RecordBatch → unbinned [`EventStore`] bridge.
//!
//! Converts columnar event-level data from Arrow (or Parquet) into an
//! [`EventStore`] suitable for unbinned likelihood fits. This enables
//! ML pipelines (Polars, DuckDB, Spark) to feed data directly into
//! NextStat without requiring ROOT files.
//!
//! # Event-Level Parquet Schema (v1)
//!
//! | Column        | Type      | Required | Description                          |
//! |--------------|-----------|----------|--------------------------------------|
//! | *observable*  | `Float64` | yes      | One column per observable (e.g. `mass`, `pt`) |
//! | `weight`      | `Float64` | no       | Per-event weight (default: 1.0)      |
//!
//! Observable columns are matched by name against the provided
//! [`ObservableSpec`] list. Extra columns are ignored.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float64Array};
use arrow::record_batch::RecordBatch;

use ns_unbinned::event_store::{EventStore, ObservableSpec};

/// Error type for unbinned Arrow/Parquet ingestion.
#[derive(Debug, thiserror::Error)]
pub enum UnbinnedArrowError {
    #[error("missing observable column '{0}' in RecordBatch")]
    MissingColumn(String),

    #[error("column '{col}' has wrong type: expected Float64, got {actual}")]
    WrongType { col: String, actual: String },

    #[error("EventStore construction failed: {0}")]
    EventStore(#[from] ns_core::Error),

    #[error("Parquet error: {0}")]
    Parquet(#[from] super::parquet::ParquetError),
}

/// Convert Arrow [`RecordBatch`]es into an unbinned [`EventStore`].
///
/// Each observable in `observables` must correspond to a `Float64` column
/// in the batches. An optional `weight` column provides per-event weights.
///
/// # Example
///
/// ```ignore
/// use ns_unbinned::event_store::ObservableSpec;
/// let obs = vec![ObservableSpec::branch("mass", (100.0, 200.0))];
/// let store = event_store_from_record_batches(&batches, &obs)?;
/// ```
pub fn event_store_from_record_batches(
    batches: &[RecordBatch],
    observables: &[ObservableSpec],
) -> Result<EventStore, UnbinnedArrowError> {
    if batches.is_empty() {
        return Err(UnbinnedArrowError::EventStore(ns_core::Error::Validation(
            "no RecordBatches provided".into(),
        )));
    }

    // Pre-calculate total row count for capacity hints.
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    // Initialize column accumulators.
    let mut columns: Vec<(String, Vec<f64>)> =
        observables.iter().map(|o| (o.name.clone(), Vec::with_capacity(total_rows))).collect();

    let mut weights: Vec<f64> = Vec::new();
    let mut has_weight_col = false;

    for (batch_idx, batch) in batches.iter().enumerate() {
        let schema = batch.schema();

        // Extract observable columns.
        for (col_idx, obs) in observables.iter().enumerate() {
            let arr_idx = schema
                .index_of(&obs.name)
                .map_err(|_| UnbinnedArrowError::MissingColumn(obs.name.clone()))?;

            let col = batch.column(arr_idx);
            let f64_arr = col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                UnbinnedArrowError::WrongType {
                    col: obs.name.clone(),
                    actual: format!("{:?}", col.data_type()),
                }
            })?;

            columns[col_idx].1.extend(f64_arr.values().iter().copied());
        }

        // Extract optional weight column.
        if let Ok(w_idx) = schema.index_of("weight") {
            has_weight_col = true;
            let col = batch.column(w_idx);
            let f64_arr = col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                UnbinnedArrowError::WrongType {
                    col: "weight".into(),
                    actual: format!("{:?}", col.data_type()),
                }
            })?;

            if batch_idx == 0 {
                weights.reserve(total_rows);
            }
            weights.extend(f64_arr.values().iter().copied());
        }
    }

    let w = if has_weight_col { Some(weights) } else { None };
    let store = EventStore::from_columns(observables.to_vec(), columns, w)?;
    Ok(store)
}

/// Read a Parquet file into an unbinned [`EventStore`].
///
/// The Parquet file must contain one `Float64` column per observable,
/// optionally a `weight` column.
pub fn event_store_from_parquet(
    path: &Path,
    observables: &[ObservableSpec],
) -> Result<EventStore, UnbinnedArrowError> {
    let batches = super::parquet::read_parquet_batches(path)?;
    event_store_from_record_batches(&batches, observables)
}

/// Read Parquet bytes into an unbinned [`EventStore`].
pub fn event_store_from_parquet_bytes(
    data: &[u8],
    observables: &[ObservableSpec],
) -> Result<EventStore, UnbinnedArrowError> {
    let batches = super::parquet::read_parquet_bytes(data)?;
    event_store_from_record_batches(&batches, observables)
}

/// Write an [`EventStore`] to Parquet format.
///
/// Exports observable columns and optional weights as a single RecordBatch.
pub fn event_store_to_parquet(store: &EventStore, path: &Path) -> Result<(), UnbinnedArrowError> {
    let batch = event_store_to_record_batch(store)?;
    super::parquet::write_parquet(path, &[batch]).map_err(UnbinnedArrowError::Parquet)
}

/// Convert an [`EventStore`] to an Arrow [`RecordBatch`].
pub fn event_store_to_record_batch(store: &EventStore) -> Result<RecordBatch, UnbinnedArrowError> {
    use arrow::datatypes::{DataType, Field, Schema};

    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

    for name in store.column_names() {
        fields.push(Field::new(name, DataType::Float64, false));
        let col = store.column(name).unwrap_or(&[]);
        arrays.push(Arc::new(Float64Array::from(col.to_vec())));
    }

    if let Some(w) = store.weights() {
        fields.push(Field::new("weight", DataType::Float64, false));
        arrays.push(Arc::new(Float64Array::from(w.to_vec())));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| UnbinnedArrowError::EventStore(ns_core::Error::Validation(e.to_string())))?;
    Ok(batch)
}
