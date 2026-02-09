//! Arrow RecordBatch → pyhf Workspace ingestion.
//!
//! Converts an Arrow table (via IPC bytes) into a pyhf-style [`Workspace`]
//! suitable for fitting with NextStat. The table must follow the histogram
//! schema documented in the [`super`] module.

use std::collections::HashMap;
use std::io::Cursor;

use arrow::array::{Array, AsArray, Float64Array, ListArray};
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

/// Intermediate row representation.
struct RowData {
    channel: String,
    sample: String,
    yields: Vec<f64>,
    stat_error: Option<Vec<f64>>,
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

    let channel_arr = batch.column(channel_idx).as_string::<i32>();
    let sample_arr = batch.column(sample_idx).as_string::<i32>();
    let yields_arr = batch.column(yields_idx).as_list::<i32>();

    let stat_error_arr: Option<&ListArray> = stat_error_idx.map(|idx| batch.column(idx).as_list());

    for i in 0..batch.num_rows() {
        let channel = channel_arr.value(i).to_string();
        let sample = sample_arr.value(i).to_string();

        let yields_values = yields_arr.value(i);
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

        let stat_error = if let Some(se_arr) = stat_error_arr {
            if se_arr.is_null(i) {
                None
            } else {
                let se_values = se_arr.value(i);
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
