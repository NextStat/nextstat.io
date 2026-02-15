//! Arrow RecordBatch → pyhf Workspace ingestion.
//!
//! Converts an Arrow table (via IPC bytes) into a pyhf-style [`Workspace`]
//! suitable for fitting with NextStat. The table must follow the histogram
//! schema documented in the [`super`] module.

use std::collections::HashMap;
use std::io::Cursor;

use arrow::array::{
    Array, ArrayRef, Float64Array, LargeListArray, LargeStringArray, ListArray, StringArray,
};
use arrow::datatypes::DataType;
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;

use crate::pyhf::schema::{
    Channel, Measurement, MeasurementConfig, Modifier, Observation, Sample, Workspace,
};

/// Configuration for Arrow → Workspace conversion.
#[derive(Debug, Clone)]
pub struct ArrowIngestConfig {
    /// Parameter of interest name (default: `"mu"`).
    pub poi: String,
    /// Observed data per channel: `{channel_name: [bin_counts]}`.
    /// If `None`, Asimov data (sum of all sample yields) is used.
    pub observations: Option<HashMap<String, Vec<f64>>>,
    /// Measurement name (default: `"NormalMeasurement"`).
    pub measurement_name: String,
}

impl Default for ArrowIngestConfig {
    fn default() -> Self {
        Self {
            poi: "mu".to_string(),
            observations: None,
            measurement_name: "NormalMeasurement".to_string(),
        }
    }
}

/// Error type for Arrow ingestion.
#[derive(Debug, thiserror::Error)]
pub enum ArrowIngestError {
    #[error("Arrow IPC deserialization failed: {0}")]
    Ipc(#[from] arrow::error::ArrowError),

    #[error("missing required column: {0}")]
    MissingColumn(String),

    #[error("column '{col}' has wrong type: expected {expected}, got {actual}")]
    WrongType { col: String, expected: String, actual: String },

    #[error("row {row}: yields list is null or empty for channel={channel}, sample={sample}")]
    EmptyYields { row: usize, channel: String, sample: String },

    #[error(
        "inconsistent bin count in channel '{channel}': expected {expected}, got {actual} (sample '{sample}')"
    )]
    InconsistentBins { channel: String, sample: String, expected: usize, actual: usize },
}

/// Deserialize Arrow IPC bytes into a list of [`RecordBatch`].
pub fn read_ipc_batches(ipc_bytes: &[u8]) -> Result<Vec<RecordBatch>, ArrowIngestError> {
    let cursor = Cursor::new(ipc_bytes);
    let reader = StreamReader::try_new(cursor, None)?;
    let batches: Result<Vec<_>, _> = reader.collect();
    Ok(batches?)
}

/// Convert Arrow IPC bytes into a pyhf [`Workspace`].
///
/// The IPC stream must contain a table with the histogram schema:
/// `channel (Utf8)`, `sample (Utf8)`, `yields (List<Float64>)`,
/// optionally `stat_error (List<Float64>)`.
pub fn from_arrow_ipc(
    ipc_bytes: &[u8],
    config: &ArrowIngestConfig,
) -> Result<Workspace, ArrowIngestError> {
    let batches = read_ipc_batches(ipc_bytes)?;
    from_record_batches(&batches, config)
}

/// Convert a slice of [`RecordBatch`] into a pyhf [`Workspace`].
pub fn from_record_batches(
    batches: &[RecordBatch],
    config: &ArrowIngestConfig,
) -> Result<Workspace, ArrowIngestError> {
    // Collect all rows across batches into (channel, sample, yields, stat_error)
    let mut rows: Vec<RowData> = Vec::new();

    for batch in batches {
        extract_rows(batch, &mut rows)?;
    }

    build_workspace(&rows, config)
}

/// Convert yields + modifiers [`RecordBatch`]es into a pyhf [`Workspace`].
///
/// `yield_batches` must follow the histogram schema (`channel`, `sample`, `yields`, `stat_error`).
/// `modifier_batches` must follow the modifiers schema (`channel`, `sample`, `modifier_name`,
/// `modifier_type`, `data_hi`, `data_lo`, `data`).
pub fn from_record_batches_with_modifiers(
    yield_batches: &[RecordBatch],
    modifier_batches: &[RecordBatch],
    config: &ArrowIngestConfig,
) -> Result<Workspace, ArrowIngestError> {
    let mut rows: Vec<RowData> = Vec::new();
    for batch in yield_batches {
        extract_rows(batch, &mut rows)?;
    }

    let mut mod_rows: Vec<ModifierRow> = Vec::new();
    for batch in modifier_batches {
        extract_modifier_rows(batch, &mut mod_rows)?;
    }

    build_workspace_with_modifiers(&rows, &mod_rows, config)
}

/// Intermediate row representation.
struct RowData {
    channel: String,
    sample: String,
    yields: Vec<f64>,
    stat_error: Option<Vec<f64>>,
}

/// Intermediate modifier row representation.
struct ModifierRow {
    channel: String,
    sample: String,
    modifier_name: String,
    modifier_type: String,
    data_hi: Option<Vec<f64>>,
    data_lo: Option<Vec<f64>>,
    data: Option<Vec<f64>>,
}

/// Extract typed rows from a single RecordBatch.
fn extract_rows(batch: &RecordBatch, rows: &mut Vec<RowData>) -> Result<(), ArrowIngestError> {
    let schema = batch.schema();

    // Required columns
    let channel_idx = schema
        .index_of("channel")
        .map_err(|_| ArrowIngestError::MissingColumn("channel".into()))?;
    let sample_idx =
        schema.index_of("sample").map_err(|_| ArrowIngestError::MissingColumn("sample".into()))?;
    let yields_idx =
        schema.index_of("yields").map_err(|_| ArrowIngestError::MissingColumn("yields".into()))?;

    // Optional columns
    let stat_error_idx = schema.index_of("stat_error").ok();

    // Validate types
    validate_utf8(batch, channel_idx, "channel")?;
    validate_utf8(batch, sample_idx, "sample")?;
    validate_list_f64(batch, yields_idx, "yields")?;
    if let Some(idx) = stat_error_idx {
        validate_list_f64(batch, idx, "stat_error")?;
    }

    let channel_col = batch.column(channel_idx);
    let sample_col = batch.column(sample_idx);
    let yields_col = batch.column(yields_idx);
    let stat_error_col: Option<&ArrayRef> = stat_error_idx.map(|idx| batch.column(idx));

    for i in 0..batch.num_rows() {
        let channel = string_row_value(channel_col, i, "channel")?;
        let sample = string_row_value(sample_col, i, "sample")?;

        let yields_values = list_row_values(yields_col, i, "yields")?;
        let yields_f64 =
            yields_values.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                ArrowIngestError::WrongType {
                    col: "yields".into(),
                    expected: "Float64".into(),
                    actual: format!("{:?}", yields_values.data_type()),
                }
            })?;

        if yields_f64.is_empty() {
            return Err(ArrowIngestError::EmptyYields {
                row: i,
                channel: channel.clone(),
                sample: sample.clone(),
            });
        }

        let yields: Vec<f64> = yields_f64.values().to_vec();

        let stat_error = if let Some(se_col) = stat_error_col {
            if se_col.is_null(i) {
                None
            } else {
                let se_values = list_row_values(se_col, i, "stat_error")?;
                let se_f64 =
                    se_values.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                        ArrowIngestError::WrongType {
                            col: "stat_error".into(),
                            expected: "Float64".into(),
                            actual: format!("{:?}", se_values.data_type()),
                        }
                    })?;
                Some(se_f64.values().to_vec())
            }
        } else {
            None
        };

        rows.push(RowData { channel, sample, yields, stat_error });
    }

    Ok(())
}

fn validate_utf8(batch: &RecordBatch, idx: usize, name: &str) -> Result<(), ArrowIngestError> {
    let dt = batch.column(idx).data_type();
    if !matches!(dt, DataType::Utf8 | DataType::LargeUtf8) {
        return Err(ArrowIngestError::WrongType {
            col: name.into(),
            expected: "Utf8".into(),
            actual: format!("{dt:?}"),
        });
    }
    Ok(())
}

fn validate_list_f64(batch: &RecordBatch, idx: usize, name: &str) -> Result<(), ArrowIngestError> {
    let dt = batch.column(idx).data_type();
    match dt {
        DataType::List(field) | DataType::LargeList(field) => {
            if !matches!(field.data_type(), DataType::Float64) {
                return Err(ArrowIngestError::WrongType {
                    col: name.into(),
                    expected: "List<Float64>".into(),
                    actual: format!("List<{:?}>", field.data_type()),
                });
            }
        }
        _ => {
            return Err(ArrowIngestError::WrongType {
                col: name.into(),
                expected: "List<Float64>".into(),
                actual: format!("{dt:?}"),
            });
        }
    }
    Ok(())
}

/// Extract modifier rows from a single RecordBatch.
fn extract_modifier_rows(
    batch: &RecordBatch,
    rows: &mut Vec<ModifierRow>,
) -> Result<(), ArrowIngestError> {
    let schema = batch.schema();

    let channel_idx = schema
        .index_of("channel")
        .map_err(|_| ArrowIngestError::MissingColumn("channel (modifiers)".into()))?;
    let sample_idx = schema
        .index_of("sample")
        .map_err(|_| ArrowIngestError::MissingColumn("sample (modifiers)".into()))?;
    let mod_name_idx = schema
        .index_of("modifier_name")
        .map_err(|_| ArrowIngestError::MissingColumn("modifier_name".into()))?;
    let mod_type_idx = schema
        .index_of("modifier_type")
        .map_err(|_| ArrowIngestError::MissingColumn("modifier_type".into()))?;

    validate_utf8(batch, channel_idx, "channel")?;
    validate_utf8(batch, sample_idx, "sample")?;
    validate_utf8(batch, mod_name_idx, "modifier_name")?;
    validate_utf8(batch, mod_type_idx, "modifier_type")?;

    let data_hi_idx = schema.index_of("data_hi").ok();
    let data_lo_idx = schema.index_of("data_lo").ok();
    let data_idx = schema.index_of("data").ok();

    if let Some(idx) = data_hi_idx {
        validate_list_f64(batch, idx, "data_hi")?;
    }
    if let Some(idx) = data_lo_idx {
        validate_list_f64(batch, idx, "data_lo")?;
    }
    if let Some(idx) = data_idx {
        validate_list_f64(batch, idx, "data")?;
    }

    let channel_col = batch.column(channel_idx);
    let sample_col = batch.column(sample_idx);
    let mod_name_col = batch.column(mod_name_idx);
    let mod_type_col = batch.column(mod_type_idx);

    let data_hi_col: Option<&ArrayRef> = data_hi_idx.map(|i| batch.column(i));
    let data_lo_col: Option<&ArrayRef> = data_lo_idx.map(|i| batch.column(i));
    let data_col: Option<&ArrayRef> = data_idx.map(|i| batch.column(i));

    for i in 0..batch.num_rows() {
        let extract_list = |arr: Option<&ArrayRef>,
                            row: usize,
                            col_name: &str|
         -> Result<Option<Vec<f64>>, ArrowIngestError> {
            let a = match arr {
                Some(a) => a,
                None => return Ok(None),
            };
            if a.is_null(row) {
                return Ok(None);
            }
            let values = list_row_values(a, row, col_name)?;
            let f64_arr = values.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                ArrowIngestError::WrongType {
                    col: col_name.into(),
                    expected: "Float64".into(),
                    actual: format!("{:?}", values.data_type()),
                }
            })?;
            Ok(Some(f64_arr.values().to_vec()))
        };

        rows.push(ModifierRow {
            channel: string_row_value(channel_col, i, "channel")?,
            sample: string_row_value(sample_col, i, "sample")?,
            modifier_name: string_row_value(mod_name_col, i, "modifier_name")?,
            modifier_type: string_row_value(mod_type_col, i, "modifier_type")?,
            data_hi: extract_list(data_hi_col, i, "data_hi")?,
            data_lo: extract_list(data_lo_col, i, "data_lo")?,
            data: extract_list(data_col, i, "data")?,
        });
    }

    Ok(())
}

/// Build a pyhf Workspace from yields + modifiers rows.
fn build_workspace_with_modifiers(
    rows: &[RowData],
    mod_rows: &[ModifierRow],
    config: &ArrowIngestConfig,
) -> Result<Workspace, ArrowIngestError> {
    // Index modifiers by (channel, sample).
    let mut mod_map: HashMap<(String, String), Vec<&ModifierRow>> = HashMap::new();
    for mr in mod_rows {
        mod_map.entry((mr.channel.clone(), mr.sample.clone())).or_default().push(mr);
    }

    let mut channel_map: HashMap<String, Vec<&RowData>> = HashMap::new();
    for row in rows {
        channel_map.entry(row.channel.clone()).or_default().push(row);
    }

    let mut channels: Vec<Channel> = Vec::new();
    let mut observations: Vec<Observation> = Vec::new();

    for (channel_name, channel_rows) in &channel_map {
        let mut expected_bins: Option<usize> = None;
        let mut samples: Vec<Sample> = Vec::new();

        for row in channel_rows {
            let n_bins = row.yields.len();
            if let Some(eb) = expected_bins {
                if n_bins != eb {
                    return Err(ArrowIngestError::InconsistentBins {
                        channel: channel_name.clone(),
                        sample: row.sample.clone(),
                        expected: eb,
                        actual: n_bins,
                    });
                }
            } else {
                expected_bins = Some(n_bins);
            }

            let mut modifiers: Vec<Modifier> = Vec::new();

            // Modifiers from the modifiers table.
            let key = (channel_name.clone(), row.sample.clone());
            if let Some(mods) = mod_map.get(&key) {
                for mr in mods {
                    if let Some(m) = modifier_row_to_modifier(mr) {
                        modifiers.push(m);
                    }
                }
            }

            // Fallback: if no modifiers table provided, add POI normfactor + staterror.
            if mod_rows.is_empty() {
                if row.sample == config.poi || row.sample == "signal" {
                    modifiers.push(Modifier::NormFactor { name: config.poi.clone(), data: None });
                }
                if let Some(ref se) = row.stat_error {
                    modifiers.push(Modifier::StatError {
                        name: format!("staterror_{}", channel_name),
                        data: se.clone(),
                    });
                }
            }

            samples.push(Sample { name: row.sample.clone(), data: row.yields.clone(), modifiers });
        }

        let obs_data = if let Some(ref obs_map) = config.observations {
            obs_map
                .get(channel_name.as_str())
                .cloned()
                .unwrap_or_else(|| asimov_from_samples(&samples))
        } else {
            asimov_from_samples(&samples)
        };

        observations.push(Observation { name: channel_name.clone(), data: obs_data });
        channels.push(Channel { name: channel_name.clone(), samples });
    }

    channels.sort_by(|a, b| a.name.cmp(&b.name));
    observations.sort_by(|a, b| a.name.cmp(&b.name));

    let workspace = Workspace {
        channels,
        observations,
        measurements: vec![Measurement {
            name: config.measurement_name.clone(),
            config: MeasurementConfig { poi: config.poi.clone(), parameters: vec![] },
        }],
        version: Some("1.0.0".to_string()),
    };

    Ok(workspace)
}

/// Convert a modifier row to a pyhf Modifier.
fn modifier_row_to_modifier(mr: &ModifierRow) -> Option<Modifier> {
    use crate::pyhf::schema::{HistoSysData, NormSysData};
    match mr.modifier_type.as_str() {
        "normfactor" => Some(Modifier::NormFactor { name: mr.modifier_name.clone(), data: None }),
        "normsys" => {
            let hi = mr.data_hi.as_ref().and_then(|v| v.first().copied())?;
            let lo = mr.data_lo.as_ref().and_then(|v| v.first().copied())?;
            Some(Modifier::NormSys { name: mr.modifier_name.clone(), data: NormSysData { hi, lo } })
        }
        "histosys" => {
            let hi_data = mr.data_hi.clone()?;
            let lo_data = mr.data_lo.clone()?;
            Some(Modifier::HistoSys {
                name: mr.modifier_name.clone(),
                data: HistoSysData { hi_data, lo_data },
            })
        }
        "shapesys" => {
            let data = mr.data.clone()?;
            Some(Modifier::ShapeSys { name: mr.modifier_name.clone(), data })
        }
        "shapefactor" => Some(Modifier::ShapeFactor { name: mr.modifier_name.clone(), data: None }),
        "staterror" => {
            let data = mr.data.clone()?;
            Some(Modifier::StatError { name: mr.modifier_name.clone(), data })
        }
        "lumi" => Some(Modifier::Lumi { name: mr.modifier_name.clone(), data: None }),
        _ => None,
    }
}

/// Build a pyhf Workspace from extracted rows.
fn build_workspace(
    rows: &[RowData],
    config: &ArrowIngestConfig,
) -> Result<Workspace, ArrowIngestError> {
    // Group rows by channel → samples
    let mut channel_map: HashMap<String, Vec<&RowData>> = HashMap::new();
    for row in rows {
        channel_map.entry(row.channel.clone()).or_default().push(row);
    }

    // Track expected bin count per channel for consistency check
    let mut channels: Vec<Channel> = Vec::new();
    let mut observations: Vec<Observation> = Vec::new();

    for (channel_name, channel_rows) in &channel_map {
        let mut expected_bins: Option<usize> = None;
        let mut samples: Vec<Sample> = Vec::new();

        for row in channel_rows {
            let n_bins = row.yields.len();

            if let Some(eb) = expected_bins {
                if n_bins != eb {
                    return Err(ArrowIngestError::InconsistentBins {
                        channel: channel_name.clone(),
                        sample: row.sample.clone(),
                        expected: eb,
                        actual: n_bins,
                    });
                }
            } else {
                expected_bins = Some(n_bins);
            }

            let mut modifiers: Vec<Modifier> = Vec::new();

            // If this sample has the POI name, add a normfactor modifier
            if row.sample == config.poi || row.sample == "signal" {
                modifiers.push(Modifier::NormFactor { name: config.poi.clone(), data: None });
            }

            // If stat_error is provided, add staterror modifier
            if let Some(ref se) = row.stat_error {
                modifiers.push(Modifier::StatError {
                    name: format!("staterror_{}", channel_name),
                    data: se.clone(),
                });
            }

            samples.push(Sample { name: row.sample.clone(), data: row.yields.clone(), modifiers });
        }

        // Build observation: user-provided or Asimov (sum of yields)
        let obs_data = if let Some(ref obs_map) = config.observations {
            obs_map.get(channel_name.as_str()).cloned().unwrap_or_else(|| {
                // Fallback to Asimov
                asimov_from_samples(&samples)
            })
        } else {
            asimov_from_samples(&samples)
        };

        observations.push(Observation { name: channel_name.clone(), data: obs_data });

        channels.push(Channel { name: channel_name.clone(), samples });
    }

    // Sort channels deterministically
    channels.sort_by(|a, b| a.name.cmp(&b.name));
    observations.sort_by(|a, b| a.name.cmp(&b.name));

    let workspace = Workspace {
        channels,
        observations,
        measurements: vec![Measurement {
            name: config.measurement_name.clone(),
            config: MeasurementConfig { poi: config.poi.clone(), parameters: vec![] },
        }],
        version: Some("1.0.0".to_string()),
    };

    Ok(workspace)
}

/// Compute Asimov data: sum of all sample yields per bin.
fn asimov_from_samples(samples: &[Sample]) -> Vec<f64> {
    if samples.is_empty() {
        return vec![];
    }
    let n_bins = samples[0].data.len();
    let mut asimov = vec![0.0; n_bins];
    for s in samples {
        for (i, &v) in s.data.iter().enumerate() {
            if i < n_bins {
                asimov[i] += v;
            }
        }
    }
    asimov
}

fn string_row_value(
    col: &ArrayRef,
    row: usize,
    col_name: &str,
) -> Result<String, ArrowIngestError> {
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                ArrowIngestError::WrongType {
                    col: col_name.into(),
                    expected: "Utf8".into(),
                    actual: format!("{:?}", col.data_type()),
                }
            })?;
            Ok(arr.value(row).to_string())
        }
        DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
                ArrowIngestError::WrongType {
                    col: col_name.into(),
                    expected: "LargeUtf8".into(),
                    actual: format!("{:?}", col.data_type()),
                }
            })?;
            Ok(arr.value(row).to_string())
        }
        _ => Err(ArrowIngestError::WrongType {
            col: col_name.into(),
            expected: "Utf8".into(),
            actual: format!("{:?}", col.data_type()),
        }),
    }
}

fn list_row_values(
    col: &ArrayRef,
    row: usize,
    col_name: &str,
) -> Result<ArrayRef, ArrowIngestError> {
    match col.data_type() {
        DataType::List(_) => {
            let arr = col.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                ArrowIngestError::WrongType {
                    col: col_name.into(),
                    expected: "List<Float64>".into(),
                    actual: format!("{:?}", col.data_type()),
                }
            })?;
            Ok(arr.value(row))
        }
        DataType::LargeList(_) => {
            let arr = col.as_any().downcast_ref::<LargeListArray>().ok_or_else(|| {
                ArrowIngestError::WrongType {
                    col: col_name.into(),
                    expected: "LargeList<Float64>".into(),
                    actual: format!("{:?}", col.data_type()),
                }
            })?;
            Ok(arr.value(row))
        }
        _ => Err(ArrowIngestError::WrongType {
            col: col_name.into(),
            expected: "List<Float64>".into(),
            actual: format!("{:?}", col.data_type()),
        }),
    }
}
