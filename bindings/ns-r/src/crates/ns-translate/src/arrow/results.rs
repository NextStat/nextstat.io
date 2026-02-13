//! Export fit results and toy study outputs to Arrow RecordBatch / Parquet.
//!
//! Enables batch results (thousands of toys, scan points) to be written as
//! compact, queryable Parquet files instead of large JSON arrays.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, Float64Array, StringBuilder, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use ns_core::types::FitResult;

/// Error type for results export.
#[derive(Debug, thiserror::Error)]
pub enum ResultsExportError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Parquet error: {0}")]
    Parquet(#[from] super::parquet::ParquetError),

    #[error("no results to export")]
    Empty,
}

/// Convert a single [`FitResult`] with parameter names to an Arrow RecordBatch.
///
/// Schema: `name (Utf8)`, `value (Float64)`, `uncertainty (Float64)`.
pub fn fit_result_to_record_batch(
    result: &FitResult,
    param_names: &[String],
) -> Result<RecordBatch, ResultsExportError> {
    let n = result.parameters.len();
    let mut name_builder = StringBuilder::with_capacity(n, n * 20);
    let mut value_arr = Vec::with_capacity(n);
    let mut unc_arr = Vec::with_capacity(n);

    for i in 0..n {
        let name = param_names.get(i).map(|s| s.as_str()).unwrap_or("?");
        name_builder.append_value(name);
        value_arr.push(result.parameters[i]);
        unc_arr.push(result.uncertainties[i]);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("uncertainty", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(name_builder.finish()) as ArrayRef,
            Arc::new(Float64Array::from(value_arr)) as ArrayRef,
            Arc::new(Float64Array::from(unc_arr)) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

/// Convert a batch of toy [`FitResult`]s to an Arrow RecordBatch.
///
/// Schema: `toy_index (UInt64)`, `nll (Float64)`, `converged (Boolean)`,
/// `par_0 (Float64)`, `par_1 (Float64)`, ...
///
/// One row per toy. Parameter columns are named from `param_names`.
pub fn toy_results_to_record_batch(
    results: &[FitResult],
    param_names: &[String],
) -> Result<RecordBatch, ResultsExportError> {
    if results.is_empty() {
        return Err(ResultsExportError::Empty);
    }

    let n_toys = results.len();
    let n_params = results[0].parameters.len();

    // Build columns.
    let mut toy_idx = Vec::with_capacity(n_toys);
    let mut nll_arr = Vec::with_capacity(n_toys);
    let mut conv_arr = Vec::with_capacity(n_toys);
    let mut param_cols: Vec<Vec<f64>> = (0..n_params).map(|_| Vec::with_capacity(n_toys)).collect();

    for (i, r) in results.iter().enumerate() {
        toy_idx.push(i as u64);
        nll_arr.push(r.nll);
        conv_arr.push(r.converged);
        for (j, col) in param_cols.iter_mut().enumerate() {
            col.push(if j < r.parameters.len() { r.parameters[j] } else { f64::NAN });
        }
    }

    // Build schema.
    let mut fields = vec![
        Field::new("toy_index", DataType::UInt64, false),
        Field::new("nll", DataType::Float64, false),
        Field::new("converged", DataType::Boolean, false),
    ];
    for i in 0..n_params {
        let name = param_names.get(i).map(|s| s.as_str()).unwrap_or("?");
        fields.push(Field::new(name, DataType::Float64, false));
    }

    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(toy_idx)),
        Arc::new(Float64Array::from(nll_arr)),
        Arc::new(BooleanArray::from(conv_arr)),
    ];
    for col in param_cols {
        arrays.push(Arc::new(Float64Array::from(col)));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)?;
    Ok(batch)
}

/// Convert profile likelihood scan points to an Arrow RecordBatch.
///
/// Schema: `mu (Float64)`, `nll (Float64)`, `delta_nll (Float64)`.
pub fn scan_points_to_record_batch(
    mu_values: &[f64],
    nll_values: &[f64],
) -> Result<RecordBatch, ResultsExportError> {
    if mu_values.is_empty() {
        return Err(ResultsExportError::Empty);
    }

    let nll_min = nll_values.iter().cloned().fold(f64::INFINITY, f64::min);

    let delta_nll: Vec<f64> = nll_values.iter().map(|&v| v - nll_min).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("mu", DataType::Float64, false),
        Field::new("nll", DataType::Float64, false),
        Field::new("delta_nll", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(mu_values.to_vec())) as ArrayRef,
            Arc::new(Float64Array::from(nll_values.to_vec())) as ArrayRef,
            Arc::new(Float64Array::from(delta_nll)) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

/// Write toy results to a Parquet file.
pub fn toy_results_to_parquet(
    results: &[FitResult],
    param_names: &[String],
    path: &Path,
) -> Result<(), ResultsExportError> {
    let batch = toy_results_to_record_batch(results, param_names)?;
    super::parquet::write_parquet(path, &[batch]).map_err(ResultsExportError::Parquet)
}

/// Write scan points to a Parquet file.
pub fn scan_points_to_parquet(
    mu_values: &[f64],
    nll_values: &[f64],
    path: &Path,
) -> Result<(), ResultsExportError> {
    let batch = scan_points_to_record_batch(mu_values, nll_values)?;
    super::parquet::write_parquet(path, &[batch]).map_err(ResultsExportError::Parquet)
}
