//! HistFactoryModel → Arrow RecordBatch export.
//!
//! Exports model metadata, expected yields, and parameter information
//! as Arrow RecordBatches, serialized to IPC bytes for Python consumption.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, ListBuilder, StringBuilder, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;

use crate::pyhf::HistFactoryModel;

/// Error type for Arrow export.
#[derive(Debug, thiserror::Error)]
pub enum ArrowExportError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Export model expected yields as an Arrow RecordBatch.
///
/// Schema:
/// - `channel` (Utf8) — channel name
/// - `sample` (Utf8) — sample name
/// - `yields` (List<Float64>) — expected yields at given parameters
///
/// One row per (channel, sample) pair.
pub fn yields_to_record_batch(
    model: &HistFactoryModel,
    params: &[f64],
) -> Result<RecordBatch, ArrowExportError> {
    let channel_names = model.channel_names();
    let expected = model.expected_data(params).map_err(|e| {
        ArrowExportError::Arrow(arrow::error::ArrowError::ComputeError(e.to_string()))
    })?;

    let mut channel_builder = StringBuilder::new();
    let mut sample_builder = StringBuilder::new();
    let mut yields_builder = ListBuilder::new(Float64Builder::new());

    let mut offset = 0;
    for (ch_idx, ch_name) in channel_names.iter().enumerate() {
        let n_bins = model.channel_bin_count(ch_idx).map_err(|e| {
            ArrowExportError::Arrow(arrow::error::ArrowError::ComputeError(e.to_string()))
        })?;

        channel_builder.append_value(ch_name);
        sample_builder.append_value("total");

        let values_builder = yields_builder.values();
        for b in 0..n_bins {
            if offset + b < expected.len() {
                values_builder.append_value(expected[offset + b]);
            }
        }
        yields_builder.append(true);

        offset += n_bins;
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("channel", DataType::Utf8, false),
        Field::new("sample", DataType::Utf8, false),
        Field::new(
            "yields",
            DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true))),
            false,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(channel_builder.finish()) as ArrayRef,
            Arc::new(sample_builder.finish()) as ArrayRef,
            Arc::new(yields_builder.finish()) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

/// Export model parameters as an Arrow RecordBatch.
///
/// Schema:
/// - `name` (Utf8) — parameter name
/// - `index` (UInt32) — parameter index
/// - `value` (Float64) — parameter value
/// - `bound_lo` (Float64) — lower bound
/// - `bound_hi` (Float64) — upper bound
/// - `init` (Float64) — initial value
pub fn parameters_to_record_batch(
    model: &HistFactoryModel,
    params: Option<&[f64]>,
) -> Result<RecordBatch, ArrowExportError> {
    let param_meta = model.parameters();
    let init_values: Vec<f64> = param_meta.iter().map(|p| p.init).collect();
    let bounds: Vec<(f64, f64)> = param_meta.iter().map(|p| p.bounds).collect();

    let n = param_meta.len();

    let mut name_builder = StringBuilder::with_capacity(n, n * 20);
    let mut index_arr = Vec::with_capacity(n);
    let mut value_arr = Vec::with_capacity(n);
    let mut bound_lo_arr = Vec::with_capacity(n);
    let mut bound_hi_arr = Vec::with_capacity(n);
    let mut init_arr = Vec::with_capacity(n);

    for (i, p) in param_meta.iter().enumerate() {
        name_builder.append_value(&p.name);
        index_arr.push(i as u32);
        value_arr.push(params.map_or(init_values[i], |pv| pv[i]));
        bound_lo_arr.push(bounds[i].0);
        bound_hi_arr.push(bounds[i].1);
        init_arr.push(init_values[i]);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("index", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
        Field::new("bound_lo", DataType::Float64, false),
        Field::new("bound_hi", DataType::Float64, false),
        Field::new("init", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(name_builder.finish()) as ArrayRef,
            Arc::new(UInt32Array::from(index_arr)) as ArrayRef,
            Arc::new(Float64Array::from(value_arr)) as ArrayRef,
            Arc::new(Float64Array::from(bound_lo_arr)) as ArrayRef,
            Arc::new(Float64Array::from(bound_hi_arr)) as ArrayRef,
            Arc::new(Float64Array::from(init_arr)) as ArrayRef,
        ],
    )?;

    Ok(batch)
}

/// Serialize a RecordBatch to Arrow IPC stream bytes.
pub fn record_batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>, ArrowExportError> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Export model yields as Arrow IPC bytes.
///
/// This is the main entry point for Python: returns bytes that can be
/// deserialized with `pyarrow.ipc.open_stream(bytes).read_all()`.
pub fn yields_to_ipc(
    model: &HistFactoryModel,
    params: &[f64],
) -> Result<Vec<u8>, ArrowExportError> {
    let batch = yields_to_record_batch(model, params)?;
    record_batch_to_ipc(&batch)
}

/// Export model parameters as Arrow IPC bytes.
pub fn parameters_to_ipc(
    model: &HistFactoryModel,
    params: Option<&[f64]>,
) -> Result<Vec<u8>, ArrowExportError> {
    let batch = parameters_to_record_batch(model, params)?;
    record_batch_to_ipc(&batch)
}
